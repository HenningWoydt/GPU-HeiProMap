/*******************************************************************************
 * MIT License
 *
 * This file is part of GPU-HeiProMap.
 *
 * Copyright (C) 2025 Henning Woydt <henning.woydt@informatik.uni-heidelberg.de>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#ifndef GPU_HEIPROMAP_SOLVER_H
#define GPU_HEIPROMAP_SOLVER_H

#include <iomanip>
#include <utility>

#include "../utility/definitions.h"
#include "../utility/configuration.h"
#include "data_structures/distance_oracle.h"
#include "matching/heavy_edge_matching.h"
#include "refinement/jet_label_propagation.h"
#include "data_structures/partition_manager.h"
#include "utility/gpu_assert.h"
#include "partitioning/kaffpa_initial_partition.h"
#include "../utility/profiler.h"


namespace GPU_HeiProMap {
    class IM_Solver {
    public:
        Configuration config;
        weight_t lmax = 0;

        HostGraph host_g;
        std::vector<Graph> device_graphs;
        std::vector<Matching> matchings;

        PartitionManager p_manager;

        DistanceOracle d_oracle;


        explicit IM_Solver(Configuration t_config) : config(std::move(t_config)) {
            io();
            initialize();
        }

        HostPartition solve() {
            partition_t c = 128;

            // first coarsening
            u32 level = 0;
            while (device_graphs.back().n > c * config.k) {
                matching();

                u32 n_matched = n_matched_v(matchings.back());
                if ((f64) n_matched < HeavyEdgeMatcher().threshold * (f64) device_graphs.back().n) {
                    matchings.pop_back();
                    break;
                }

                coarsening();

                level += 1;
            }

            // initial partitioning
            u32 max_level = level;
            initial_partitioning();

            // v cycles
            u32 max_cycle = 0;
            for (u32 cycle = 0; cycle < max_cycle; ++cycle) {
                while (!matchings.empty()) {
                    level -= 1;
                    uncoarsening();
                    refinement(max_level, level);
                }

                while (device_graphs.back().n > 8 * config.k) {
                    matching();
                    coarsening();
                    level += 1;
                }
                std::cout << "V-Cycle " << cycle << " " << comm_cost(device_graphs.back(), p_manager, d_oracle) << " " << max_weight(p_manager) << " " << lmax << std::endl;
            }

            // final refinement
            while (!matchings.empty()) {
                level -= 1;
                uncoarsening();
                refinement(max_level, level);

                std::cout << "Level  " << level << " " << device_graphs.back().n << " " << comm_cost(device_graphs.back(), p_manager, d_oracle) << " " << max_weight(p_manager) << " " << lmax << std::endl;
            }

            TIME("io", "io", "write_partition",
                 HostPartition host_partition = HostPartition("host_partition", device_graphs.back().n);
                 Kokkos::deep_copy(host_partition, p_manager.partition);

                 write_partition(host_partition, device_graphs.back().n, config.mapping_out);
            );

            std::string config_JSON = config.to_JSON();
            std::string profile_JSON = Profiler::instance().to_JSON();

            // Combine manually into a single JSON string
            std::string combined_JSON = "{\n";
            combined_JSON += "  \"config\": " + config_JSON + ",\n";
            combined_JSON += "  \"profile\": " + profile_JSON + "\n";
            combined_JSON += "}";

            std::cout << combined_JSON << std::endl;

            // Save to file
            if (config.is_set("--statistics")) {
                std::ofstream outFile(config.statistics_out);
                if (outFile.is_open()) {
                    outFile << combined_JSON;
                    outFile.close();
                } else {
                    std::cerr << "Error: Could not open " << config.statistics_out << " to write statistics!" << std::endl;
                }
            }

            return host_partition;
        }

    private:
        void io() {
            TIME("io", "HostGraph", "load",
                 host_g = HostGraph(config.graph_in);
            );
        }

        void initialize() {
            lmax = (weight_t) std::ceil((1.0 + config.imbalance) * ((f64) host_g.g_weight / (f64) config.k));

            device_graphs.emplace_back(initialize_device_g(host_g));
            free_host_graph(host_g);

            p_manager = initialize_p_manager(device_graphs.back().n, config.k, lmax);
            d_oracle = initialize_d_oracle(config.k, config.hierarchy, config.distance);

            assert_state_pre_partition(device_graphs.back(), d_oracle, config.hierarchy, config.distance);
        }

        void matching() {
            HeavyEdgeMatcher hem = initialize_hem(device_graphs.back().n, lmax);
            matchings.emplace_back(match(hem, device_graphs.back(), p_manager));
        }

        void coarsening() {
            device_graphs.emplace_back(initialize_device_g(device_graphs.back(), matchings.back()));
            contract(p_manager, matchings.back());
            assert_state_pre_partition(device_graphs.back(), d_oracle, config.hierarchy, config.distance);
        }

        void initial_partitioning() {
            kaffpa_initial_partition(device_graphs.back(), config.hierarchy, config.distance, config.k, config.imbalance, 0, p_manager);
            recalculate_weights(p_manager, device_graphs.back());

            assert_state_after_partition(device_graphs.back(), p_manager, config.k, d_oracle, config.hierarchy, config.distance);
        }

        void uncoarsening() {
            uncontract(p_manager, matchings.back());

            device_graphs.pop_back();
            matchings.pop_back();

            assert_state_after_partition(device_graphs.back(), p_manager, config.k, d_oracle, config.hierarchy, config.distance);
        }

        void refinement(u32 max_level, u32 level) {
            LabelPropagationStruct lp_struct = initialize_lp(device_graphs.back().n, device_graphs.back().m, config.k, lmax);

            lp_struct.n_max_iterations = 12 + level;
            if (level == 0) {
                lp_struct.max_weak_iterations = 10;
            }

            lp_struct.sigma_percent = lp_struct.sigma_percent_min + (level == 0 ? 0.0 : lp_struct.sigma_percent * ((f64) level / (f64) max_level));
            lp_struct.sigma = lp_struct.lmax - (weight_t) ((f64) lp_struct.lmax * lp_struct.sigma_percent);

            refine(lp_struct, device_graphs.back(), p_manager, d_oracle, level);
            assert_state_after_partition(device_graphs.back(), p_manager, config.k, d_oracle, config.hierarchy, config.distance);
        }
    };
}

#endif //GPU_HEIPROMAP_SOLVER_H
