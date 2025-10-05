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

#ifndef GPU_HEIPROMAP_PARTITION_H
#define GPU_HEIPROMAP_PARTITION_H

#include "../datastructures/hm_device_graph.h"
#include "../../utility/definitions.h"

namespace GPU_HeiProMap {
    struct HM_Item {
        HM_DeviceGraph device_g;
        std::vector<int> identifier; // host-only container
        JetDeviceEntries o_to_n;
        JetDeviceEntries n_to_o;
    };

    inline JetDevicePartition jet_partition(HM_DeviceGraph &device_g,
                                            int k,
                                            f64 imbalance,
                                            int seed,
                                            bool use_ultra) {
        if (k == 1) {
            ScopedTimer _t("partitioning", "jet_partition", "k=1");
            JetDevicePartition partition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "k=1 partition"), (size_t) device_g.n);
            Kokkos::deep_copy(partition, 0);
            Kokkos::fence();
            return partition;
        }
        ScopedTimer _t_mtx("partitioning", "jet_partition", "get_mtx");

        jet_partitioner::config_t config;
        config.max_imb_ratio = 1.0 + imbalance;
        config.num_parts = k;
        config.ultra_settings = use_ultra;
        config.verbose = false;
        config.num_iter = 1;
        config.coarsening_alg = 0;

        jet_partitioner::value_t edge_cut;
        jet_partitioner::experiment_data<jet_partitioner::value_t> data;

        jet_partitioner::matrix_t mtx = device_g.get_mtx();
        _t_mtx.stop();

        ScopedTimer _t("partitioning", "jet_partition", "k>1");
        JetDevicePartition partition = jet_partitioner::partition(edge_cut,
                                                                  config,
                                                                  mtx,
                                                                  device_g.vertex_weights,
                                                                  false,
                                                                  data);
        Kokkos::fence();

        return partition;
    }

    inline f64 determine_adaptive_imbalance(const f64 global_imbalance,
                                            const int global_g_weight,
                                            const int global_k,
                                            const int local_g_weight,
                                            const int local_k_rem,
                                            const int depth) {
        f64 local_imbalance = (1.0 + global_imbalance) * ((f64) (local_k_rem * global_g_weight) / (f64) (global_k * local_g_weight));
        local_imbalance = std::pow(local_imbalance, (f64) 1 / (f64) depth) - 1.0;

        return local_imbalance;
    }

    inline void create_subgraphs(const HM_DeviceGraph &device_g,
                                 const JetDeviceEntries &n_to_o, // map local->original for current graph
                                 const int k,
                                 const jet_partitioner::part_vt &device_partition,
                                 const std::vector<int> &identifier,
                                 const int global_n,
                                 std::vector<HM_Item> &stack) {
        stack.reserve(stack.size() + (size_t) k);

        std::vector<int> sub_ns((size_t) k, 0);
        std::vector<int> sub_ms((size_t) k, 0);
        std::vector<int> sub_ws((size_t) k, 0);

        ScopedTimer _t("partitioning", "create_subgraphs", "get_n_m_weight");
        for (int id = 0; id < k; ++id) {
            Kokkos::parallel_reduce("SubN", (size_t) device_g.n, KOKKOS_LAMBDA(const int u, int &lsum) {
                if (device_partition(u) == id) lsum += 1;
            }, sub_ns[(size_t) id]);

            Kokkos::parallel_reduce("SubWeight", (size_t) device_g.n, KOKKOS_LAMBDA(const int u, int &lsum) {
                if (device_partition(u) == id) lsum += device_g.vertex_weights(u);
            }, sub_ws[(size_t) id]);

            Kokkos::parallel_reduce("SubM", (size_t) device_g.n, KOKKOS_LAMBDA(const int u, int &lsum) {
                if (device_partition(u) == id) {
                    int cnt = 0;
                    for (int i = device_g.neighborhood(u); i < device_g.neighborhood(u + 1); ++i) {
                        const int v = device_g.edges_v(i);
                        if (device_partition(v) == id) ++cnt;
                    }
                    lsum += cnt;
                }
            }, sub_ms[(size_t) id]);
        }
        Kokkos::fence();
        _t.stop();

        for (int id = 0; id < k; ++id) {
            int sub_n = sub_ns[(size_t) id];
            int sub_m = sub_ms[(size_t) id];
            int sub_weight = sub_ws[(size_t) id];

            HM_DeviceGraph device_sub_g(sub_n, sub_m, sub_weight);

            ScopedTimer _t_translate("partitioning", "create_subgraphs", "translate");
            JetDeviceEntries o_to_n_sub(Kokkos::view_alloc(Kokkos::WithoutInitializing, "o_to_n"), (size_t) global_n);
            JetDeviceEntries n_to_o_sub(Kokkos::view_alloc(Kokkos::WithoutInitializing, "n_to_o"), (size_t) sub_n);

            Kokkos::parallel_scan("AssignLocalIndex", (size_t) device_g.n, KOKKOS_LAMBDA(const int u, int &prefix, const bool final) {
                if (device_partition(u) == id) {
                    const int my_idx = prefix; // exclusive prefix
                    if (final) {
                        const int old_u = n_to_o(u);
                        o_to_n_sub(old_u) = my_idx;
                        n_to_o_sub(my_idx) = old_u;
                        device_sub_g.vertex_weights(my_idx) = device_g.vertex_weights(u);
                    }
                    prefix += 1;
                }
            });
            _t_translate.stop();

            ScopedTimer _t_graph("partitioning", "create_subgraphs", "build_graph");
            Kokkos::parallel_for("InitNeighborhood0", 1, KOKKOS_LAMBDA(const int) {
                device_sub_g.neighborhood(0) = 0;
            });

            Kokkos::parallel_scan("FillEdges", (size_t) device_g.n, KOKKOS_LAMBDA(const int u, int &edge_prefix, const bool final) {
                if (device_partition(u) == id) {
                    int start = edge_prefix;
                    int cnt = 0;

                    // Count + (in final phase) write edges for this vertex
                    for (int i = device_g.neighborhood(u); i < device_g.neighborhood(u + 1); ++i) {
                        const int v = device_g.edges_v(i);
                        if (device_partition(v) == id) {
                            if (final) {
                                const int sub_v = o_to_n_sub(n_to_o(v));
                                device_sub_g.edges_v(start) = sub_v;
                                device_sub_g.edges_w(start) = device_g.edges_w(i);
                            }
                            ++start; // advance write cursor for this vertex
                            ++cnt; // degree in subgraph
                        }
                    }

                    if (final) {
                        const int sub_u = o_to_n_sub(n_to_o(u));
                        device_sub_g.neighborhood(sub_u + 1) = edge_prefix + cnt; // end offset
                    }

                    edge_prefix += cnt; // contribute to global (subgraph) edge prefix
                }
            });
            _t_graph.stop();

            ScopedTimer _t_push("partitioning", "create_subgraphs", "push");
            stack.push_back({device_sub_g, identifier, o_to_n_sub, n_to_o_sub});
            stack.back().identifier.push_back(id);
            _t_push.stop();
        }
    }
}

#endif //GPU_HEIPROMAP_PARTITION_H
