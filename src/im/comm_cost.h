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

#ifndef GPU_HEIPROMAP_COMM_COST_H
#define GPU_HEIPROMAP_COMM_COST_H

#include "definitions.h"
#include "device_graph.h"
#include "distance_oracle.h"
#include "large_vertex_partition_csr.h"
#include "partition_manager.h"

namespace GPU_HeiProMap {
    inline weight_t comm_cost(const Graph &device_g,
                              const PartitionManager &p_manager,
                              const DistanceOracle &d_oracle) {
        weight_t total_comm_cost = 0;

        Kokkos::parallel_reduce("comm_cost", device_g.m, KOKKOS_LAMBDA(const u32 i, weight_t &local_comm_cost) {
                                    vertex_t u = device_g.edges_u(i);
                                    vertex_t v = device_g.edges_v(i);
                                    weight_t w = device_g.edges_w(i);

                                    partition_t u_id = p_manager.partition(u);
                                    partition_t v_id = p_manager.partition(v);

                                    local_comm_cost += w * get(d_oracle, u_id, v_id);
                                },
                                total_comm_cost);
        Kokkos::fence();

        return total_comm_cost;
    }

    inline weight_t comm_cost(const Graph &device_g,
                              const PartitionManager &p_manager,
                              const LargeVertexPartitionCSR &csr,
                              const DistanceOracle &d_oracle) {
        weight_t total_comm_cost = 0;

        Kokkos::parallel_reduce("comm_cost", device_g.n, KOKKOS_LAMBDA(const vertex_t u, weight_t &local_comm_cost) {
                                    partition_t u_id = p_manager.partition(u);
                                    weight_t sum = 0;

                                    for (u32 i = csr.row(u); i < csr.row(u + 1); ++i) {
                                        partition_t id = csr.ids(i);
                                        weight_t w = csr.weights(i);

                                        if (id == u_id) { continue; }
                                        if (id == p_manager.k) { continue; }

                                        sum += w * get(d_oracle, u_id, id);
                                    }

                                    local_comm_cost += sum;
                                },
                                total_comm_cost);
        Kokkos::fence();

        return total_comm_cost;
    }

    inline weight_t comm_cost_host(const Graph &device_g,
                                   const PartitionManager &p_manager,
                                   const DistanceOracle &d_oracle) {
        HostGraph host_g = to_host_graph(device_g);
        HostPartitionManager host_p_manager = to_host_p_manager(p_manager);
        HostDistanceOracle host_d_oracle = to_host_d_oracle(d_oracle);

        weight_t total_comm_cost = 0;

        for (u32 u = 0; u < host_g.n; ++u) {
            partition_t u_id = host_p_manager.partition(u);

            weight_t local_comm_cost = 0;
            for (u32 i = host_g.neighborhood(u); i < host_g.neighborhood(u + 1); ++i) {
                vertex_t v = host_g.edges_v(i);
                weight_t w = host_g.edges_w(i);

                partition_t v_id = host_p_manager.partition(v);

                local_comm_cost += w * get(host_d_oracle, u_id, v_id);
            }

            total_comm_cost += local_comm_cost;
        }

        return total_comm_cost;
    }

    KOKKOS_INLINE_FUNCTION
    weight_t comm_cost_delta(const Graph &device_g,
                             const PartitionManager &p_manager,
                             const DistanceOracle &d_oracle,
                             const vertex_t u,
                             const partition_t old_id,
                             const partition_t new_id) {
        weight_t comm_cost = 0;

        for (vertex_t i = device_g.neighborhood(u); i < device_g.neighborhood(u + 1); ++i) {
            vertex_t v = device_g.edges_v(i);
            weight_t w = device_g.edges_w(i);
            partition_t v_id = p_manager.partition(v);

            comm_cost += w * get_diff(d_oracle, old_id, new_id, v_id);
        }

        return comm_cost;
    }
}

#endif //GPU_HEIPROMAP_COMM_COST_H
