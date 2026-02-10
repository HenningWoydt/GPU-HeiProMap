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

        std::vector<Graph> device_graphs;
        std::vector<Matching> matchings;

        PartitionManager p_manager;
        DistanceOracle d_oracle;

        f64 io_ms = 0.0;
        f64 misc_ms = 0.0;
        f64 coarsening_ms = 0.0;
        f64 contraction_ms = 0.0;
        f64 initial_partitioning_ms = 0.0;
        f64 uncontraction_ms = 0.0;
        f64 refinement_ms = 0.0;

        explicit IM_Solver(Configuration t_config) : config(std::move(t_config)) {

        }


        HostPartition solve(HostGraph &host_g) {
            auto sp = get_time_point();

            initialize(host_g);

            partition_t c = 128;

            // first coarsening
            u32 level = 0;
            while (device_graphs.back().n > c * config.k) {
                coarsening();

                u32 n_matched = n_matched_v(matchings.back());
                if ((f64) n_matched < HeavyEdgeMatcher().threshold * (f64) device_graphs.back().n) {
                    matchings.pop_back();
                    break;
                }

                contraction();

                level += 1;
            }

            // initial partitioning
            u32 max_level = level;
            initial_partitioning();

            weight_t initial_comm_cost = comm_cost(device_graphs.back(), p_manager, d_oracle);
            weight_t initial_max_block_weight = max_weight(p_manager);

            // final refinement
            while (!matchings.empty()) {
                level -= 1;
                uncontraction();
                refinement(max_level, level);
            }

            ScopedTimer _t_write("io", "IM_Solver", "download_partition");
            auto p = get_time_point();
            HostPartition host_partition = HostPartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_partition"), device_graphs.back().n);
            Kokkos::deep_copy(host_partition, p_manager.partition);
            Kokkos::fence();

            misc_ms += get_milli_seconds(p, get_time_point());

            _t_write.stop();

            auto ep = get_time_point();
            f64 duration = get_seconds(sp, ep);

            std::cout << "Total time        : " << duration << std::endl;
            std::cout << "#Nodes            : " << device_graphs.back().n << std::endl;
            std::cout << "#Edges            : " << device_graphs.back().m << std::endl;
            std::cout << "k                 : " << config.k << std::endl;
            std::cout << "Lmax              : " << lmax << std::endl;
            // std::cout << "Init. comm-cost   : " << initial_comm_cost << std::endl;
            // std::cout << "Init. max block w : " << initial_max_block_weight << std::endl;
            // std::cout << "Final comm-cost   : " << comm_cost(device_graphs.back(), p_manager, d_oracle) << std::endl;
            // std::cout << "Final max block w : " << max_weight(p_manager) << std::endl;

            // size_t n_empty_partitions = 0;
            // size_t n_overloaded_partitions = 0;
            // weight_t sum_too_much = 0;
            // HostPartitionManager partition_host = to_host_p_manager(p_manager);
            // for (partition_t id = 0; id < config.k; ++id) {
            //     n_empty_partitions += partition_host.bweights(id) == 0;
            //     n_overloaded_partitions += partition_host.bweights(id) > lmax;
            //     sum_too_much += std::max((weight_t) 0, partition_host.bweights(id) - lmax);
            // }
            // std::cout << "#empty partitions : " << n_empty_partitions << std::endl;
            // std::cout << "#oload partitions : " << n_overloaded_partitions << std::endl;
            // std::cout << "Sum oload weights : " << sum_too_much << std::endl;
            std::cout << "IO            : " << io_ms << std::endl;
            std::cout << "Misc          : " << misc_ms << std::endl;
            std::cout << "Coarsening    : " << coarsening_ms << std::endl;
            std::cout << "Contraction   : " << contraction_ms << std::endl;
            std::cout << "Init. Part.   : " << initial_partitioning_ms << std::endl;
            std::cout << "Uncontraction : " << uncontraction_ms << std::endl;
            std::cout << "Refinement    : " << refinement_ms << std::endl;

            return host_partition;
        }

    private:
        void initialize(HostGraph &host_g) {
            auto p = get_time_point();

            io_ms += get_milli_seconds(p, get_time_point());

            auto p1 = get_time_point();

            lmax = (weight_t) std::ceil((1.0 + config.imbalance) * ((f64) host_g.g_weight / (f64) config.k));

            device_graphs.emplace_back(initialize_device_g(host_g));

            p_manager = initialize_p_manager(device_graphs.back().n, config.k, lmax);
            d_oracle = initialize_d_oracle(config.k, config.hierarchy, config.distance);

            misc_ms += get_milli_seconds(p1, get_time_point());

            assert_state_pre_partition(device_graphs.back(), d_oracle, config.hierarchy, config.distance);
        }

        void coarsening() {
            auto p = get_time_point();

            HeavyEdgeMatcher hem = initialize_hem(device_graphs.back().n, lmax);
            matchings.emplace_back(match(hem, device_graphs.back(), p_manager));

            coarsening_ms += get_milli_seconds(p, get_time_point());
        }

        void contraction() {
            auto p = get_time_point();

            device_graphs.emplace_back(initialize_device_g(device_graphs.back(), matchings.back()));
            contract(p_manager, matchings.back());

            contraction_ms += get_milli_seconds(p, get_time_point());

            assert_state_pre_partition(device_graphs.back(), d_oracle, config.hierarchy, config.distance);
        }

        void initial_partitioning() {
            auto p = get_time_point();

            kaffpa_initial_partition(device_graphs.back(), config.hierarchy, config.distance, config.k, config.imbalance, 0, p_manager);
            recalculate_weights(p_manager, device_graphs.back());

            initial_partitioning_ms += get_milli_seconds(p, get_time_point());

            assert_state_after_partition(device_graphs.back(), p_manager, config.k, d_oracle, config.hierarchy, config.distance);
        }

        void uncontraction() {
            auto p = get_time_point();

            uncontract(p_manager, matchings.back());

            device_graphs.pop_back();
            matchings.pop_back();

            uncontraction_ms += get_milli_seconds(p, get_time_point());

            assert_state_after_partition(device_graphs.back(), p_manager, config.k, d_oracle, config.hierarchy, config.distance);
        }

        void refinement(u32 max_level, u32 level) {
            auto p = get_time_point();

            JetLabelPropagation lp_struct = initialize_lp(device_graphs.back().n, device_graphs.back().m, config.k, lmax);

            lp_struct.n_max_iterations = 12 + level;
            if (level == 0) {
                lp_struct.max_weak_iterations = 10;
            }

            lp_struct.sigma_percent = lp_struct.sigma_percent_min + (level == 0 ? 0.0 : lp_struct.sigma_percent * ((f64) level / (f64) max_level));
            lp_struct.sigma = lp_struct.lmax - (weight_t) ((f64) lp_struct.lmax * lp_struct.sigma_percent);

            refine(lp_struct, device_graphs.back(), p_manager, d_oracle, level);

            refinement_ms += get_milli_seconds(p, get_time_point());

            assert_state_after_partition(device_graphs.back(), p_manager, config.k, d_oracle, config.hierarchy, config.distance);
        }
    };
}

#endif //GPU_HEIPROMAP_SOLVER_H
