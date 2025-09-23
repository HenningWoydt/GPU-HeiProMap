/*******************************************************************************
 * MIT License
 *
 * This file is part of GPU_HeiProMap.
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

#ifndef GPU_HEIPROMAP_ASSERT_H
#define GPU_HEIPROMAP_ASSERT_H

#include <unordered_set>

#include "definitions.h"
#include "distance_oracle.h"
#include "host_graph.h"

namespace GPU_HeiProMap {
    inline void assert_no_loops(const HostGraph &host_g) {
        for (vertex_t u = 0; u < host_g.n; ++u) {
            u32 begin = host_g.neighborhood[u];
            u32 end = host_g.neighborhood[u + 1];

            for (u32 i = begin; i < end; ++i) {
                vertex_t v = host_g.edges_v[i];
                ASSERT(v != u);
            }
        }
    }

    inline void assert_no_double_edges(const HostGraph &host_g) {
        for (vertex_t u = 0; u < host_g.n; ++u) {
            u32 begin = host_g.neighborhood[u];
            u32 end = host_g.neighborhood[u + 1];

            std::unordered_set<vertex_t> seen;

            for (u32 i = begin; i < end; ++i) {
                vertex_t v = host_g.edges_v[i];
                if (!seen.insert(v).second) {
                    throw std::runtime_error("Graph assertion failed: duplicate edge " + std::to_string(u) + " → " + std::to_string(v));
                }
            }
        }
    }

    inline void assert_positive_edges(const HostGraph &host_g) {
        for (vertex_t u = 0; u < host_g.n; ++u) {
            u32 begin = host_g.neighborhood(u);
            u32 end = host_g.neighborhood(u + 1);

            for (u32 i = begin; i < end; ++i) {
                weight_t w = host_g.edges_w(i);

                ASSERT(w > 0);
            }
        }
    }

    inline void assert_edges_u(const HostGraph &host_g,
                               const HostVertex &host_edges_u) {
        u32 i = 0;
        for (vertex_t u = 0; u < host_g.n; ++u) {
            u32 begin = host_g.neighborhood[u];
            u32 end = host_g.neighborhood[u + 1];

            for (u32 j = begin; j < end; ++j) {
                ASSERT(host_edges_u(i) == u);
                i += 1;
            }
        }
    }

    inline void assert_partition(const HostGraph &g,
                                 const HostPartitionManager &p_manager,
                                 const partition_t k) {
        for (vertex_t u = 0; u < g.n; ++u) {
            partition_t u_id = p_manager.partition(u);

            ASSERT(u_id < k);
        }
    }

    inline void assert_bweights(const HostGraph &g,
                                const HostPartitionManager &p_manager,
                                const partition_t k) {
        std::vector<weight_t> weights(k, 0);

        for (vertex_t u = 0; u < g.n; ++u) {
            partition_t u_id = p_manager.partition(u);

            weights[u_id] += g.weights(u);
        }

        for (partition_t id = 0; id < k; ++id) {
            ASSERT(weights[id] == p_manager.bweights(id));
        }
    }

    inline void assert_d_oracle(const HostDistanceOracle &host_d_oracle,
                                const std::vector<partition_t> &hierarchy,
                                const std::vector<weight_t> &distances) {
        for (partition_t id1 = 0; id1 < host_d_oracle.k; ++id1) {
            for (partition_t id2 = 0; id2 < host_d_oracle.k; ++id2) {
                weight_t d_fast = get(host_d_oracle, id1, id2);

                weight_t d_slow = 0;

                if (id1 != id2) {
                    u64 group_size = 1;
                    for (size_t i = 0; i < distances.size(); ++i) {
                        group_size *= hierarchy[i];
                        if (id1 / group_size == id2 / group_size) {
                            d_slow = distances[i];
                            break;
                        }
                    }
                }

                ASSERT(d_fast >= 0);
                ASSERT(d_fast == d_slow);
            }
        }
    }

    inline void assert_state_pre_partition(const Graph &device_g,
                                           const DistanceOracle &d_oracle,
                                           const std::vector<partition_t> &hierarchy,
                                           const std::vector<weight_t> &distances) {
#if !ASSERT_ENABLED
        return;
#endif
        HostGraph host_g = to_host_graph(device_g);
        HostVertex host_edges_u = HostVertex("edges_u", host_g.m);
        Kokkos::deep_copy(host_edges_u, device_g.edges_u);
        HostDistanceOracle host_d_oracle = to_host_d_oracle(d_oracle);
        Kokkos::fence();

        assert_no_loops(host_g);
        assert_no_double_edges(host_g);
        assert_positive_edges(host_g);
        assert_edges_u(host_g, host_edges_u);

        assert_d_oracle(host_d_oracle, hierarchy, distances);
    }

    inline void assert_state_after_partition(const Graph &device_g,
                                             const PartitionManager &p_manager,
                                             const partition_t k,
                                             const DistanceOracle &d_oracle,
                                             const std::vector<partition_t> &hierarchy,
                                             const std::vector<weight_t> &distances) {
#if !ASSERT_ENABLED
        return;
#endif
        HostGraph host_g = to_host_graph(device_g);
        HostVertex host_edges_u = HostVertex("edges_u", host_g.m);
        Kokkos::deep_copy(host_edges_u, device_g.edges_u);
        HostPartitionManager host_p_manager = to_host_p_manager(p_manager);
        HostDistanceOracle host_d_oracle = to_host_d_oracle(d_oracle);

        assert_no_loops(host_g);
        assert_no_double_edges(host_g);
        assert_positive_edges(host_g);
        assert_edges_u(host_g, host_edges_u);

        assert_partition(host_g, host_p_manager, k);
        assert_bweights(host_g, host_p_manager, k);

        assert_d_oracle(host_d_oracle, hierarchy, distances);
    }
}

#endif //GPU_HEIPROMAP_GRAPH_ASSERT_H
