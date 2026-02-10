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

#ifndef GPU_HEIPROMAP_HM_SOLVER_H
#define GPU_HEIPROMAP_HM_SOLVER_H

#include <fstream>
#include <ostream>

#include "datastructures/hm_device_graph.h"
#include "datastructures/hm_host_graph.h"
#include "partitioning/partition.h"
#include "../utility/configuration.h"
#include "../utility/profiler.h"

namespace GPU_HeiProMap {
    class HM_Solver {
        Configuration config;

    public:
        explicit HM_Solver(Configuration t_config) : config(std::move(t_config)) {
        }

        JetHostPartition solve(HM_HostGraph &g) {
            // solve problem
            JetHostPartition partition = internal_solve(g);

            return partition;
        }

        JetHostPartition internal_solve(HM_HostGraph &host_g) {
            ScopedTimer _t_allocate("partitioning", "HM_Solver", "allocate");
            JetHostPartition final_host_partition = JetHostPartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "final_host_partition"), (size_t) host_g.n);
            JetDevicePartition final_device_partition = JetDevicePartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "final_device_partition"), (size_t) host_g.n);

            // references for better code readability
            const int l = (int) config.hierarchy.size();
            std::vector<int> hierarchy((u64) l);
            for (int i = 0; i < l; ++i) { hierarchy[(size_t) i] = (int) config.hierarchy[(size_t) i]; }

            std::vector<int> index_vec; // index vector to correctly offset all resulting graphs
            index_vec = {1};
            for (size_t i = 0; i < hierarchy.size() - 1; ++i) {
                index_vec.push_back(index_vec[i] * hierarchy[i]);
            }

            std::vector<int> k_rem_vec; // remaining k vector
            k_rem_vec.resize(hierarchy.size());
            int p = 1;
            for (size_t i = 0; i < hierarchy.size(); ++i) {
                k_rem_vec[i] = p * hierarchy[i];
                p *= hierarchy[i];
            }

            const f64 global_imbalance = config.imbalance;
            const int global_g_weight = host_g.graph_weight;
            const int global_k = (int) config.k;
            const bool use_ultra = config.config == "HM-ultra";

            _t_allocate.stop();
            ScopedTimer _t_item("partitioning", "HM_Solver", "first_item");

            std::vector<HM_Item> stack;
            JetDeviceEntries device_n_to_o(Kokkos::view_alloc(Kokkos::WithoutInitializing, "n_to_o"), (size_t) host_g.n);
            Kokkos::parallel_for("init_mappings", (size_t) host_g.n, KOKKOS_LAMBDA(int u) {
                device_n_to_o(u) = u;
            });
            Kokkos::fence();
            _t_item.stop();

            stack.push_back({HM_DeviceGraph(host_g), {}, device_n_to_o});

            while (!stack.empty()) {
                HM_Item item = std::move(stack.back());
                stack.pop_back();

                // get depth info
                const int depth = l - 1 - (int) item.identifier.size();
                const int local_k = hierarchy[(size_t) depth];
                const int local_k_rem = k_rem_vec[(size_t) depth];
                const f64 local_imbalance = determine_adaptive_imbalance(global_imbalance, global_g_weight, global_k, item.device_g.graph_weight, local_k_rem, depth + 1);

                JetDevicePartition device_partition = jet_partition(item.device_g, local_k, local_imbalance, (int) config.seed, use_ultra);

                if (depth == 0) {
                    // insert solution
                    ScopedTimer _t("partitioning", "HM_Solver", "insert_solution");
                    int offset = 0;
                    for (size_t i = 0; i < item.identifier.size(); ++i) { offset += item.identifier[i] * index_vec[index_vec.size() - 1 - i]; }

                    auto n_to_o = item.n_to_o;
                    Kokkos::parallel_for("insert_solution", (size_t) item.device_g.n, KOKKOS_LAMBDA(const int u) {
                        const int original_u = n_to_o(u);
                        final_device_partition(original_u) = offset + device_partition(u);
                    });
                } else {
                    // create the subgraphs and place them in the next stack
                    create_subgraphs(item.device_g, item.n_to_o, local_k, device_partition, item.identifier, host_g.n, stack);
                }
            }
            Kokkos::fence();

            ScopedTimer _t("io", "HM_Solver", "download_partition");
            Kokkos::deep_copy(final_host_partition, final_device_partition);
            Kokkos::fence();

            return final_host_partition;
        }
    };
}

#endif //GPU_HEIPROMAP_HM_SOLVER_H
