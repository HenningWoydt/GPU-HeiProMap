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

#ifndef GPU_HEIPROMAP_FAILSAFE_REBALANCER_H
#define GPU_HEIPROMAP_FAILSAFE_REBALANCER_H

#include "comm_cost.h"
#include "definitions.h"
#include "device_graph.h"
#include "partition_manager.h"

namespace GPU_HeiProMap {
    inline void rebalance(const Graph &device_g,
                          const DistanceOracle &d_oracle,
                          PartitionManager &p_manager) {
        if (n_empty_partitions(p_manager) > 0) {

        }

        vertex_t n = device_g.n;
        // vertex_t m = device_g.m;
        partition_t k = d_oracle.k;
        weight_t lmax = p_manager.lmax;

        DeviceWeight gain = DeviceWeight("gain", n);
        DevicePartition id = DevicePartition("id", n);;

        DeviceWeight send_gain = DeviceWeight("send_gain", k);
        DeviceVertex send_vertex = DeviceVertex("send_vertex", k);
        DeviceWeight recieve_gain = DeviceWeight("recieve_gain", k);
        DeviceVertex recieve_vertex = DeviceVertex("recieve_vertex", k);

        while (max_weight(p_manager) > lmax) {
            std::cout << max_weight(p_manager) << " " << lmax << std::endl;
            // reset vars
            Kokkos::deep_copy(gain, -max_sentinel<weight_t>());
            Kokkos::parallel_for("reset", n, KOKKOS_LAMBDA(const vertex_t u) {
                id(u) = p_manager.partition(u);
            });
            Kokkos::deep_copy(send_gain, -max_sentinel<weight_t>());
            Kokkos::deep_copy(recieve_gain, -max_sentinel<weight_t>());
            Kokkos::deep_copy(send_vertex, max_sentinel<vertex_t>());
            Kokkos::deep_copy(recieve_vertex, max_sentinel<vertex_t>());
            Kokkos::fence();

            Kokkos::parallel_for("max_gain", n * k, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = i / k;
                weight_t u_w = device_g.weights(u);
                partition_t u_id = p_manager.partition(u);
                if (p_manager.bweights(u_id) <= lmax) { return; }

                partition_t v_id = i % k;
                if (u_id == v_id) { return; }
                if (p_manager.bweights(v_id) + u_w > lmax) { return; }

                const weight_t delta = comm_cost_delta(device_g, p_manager, d_oracle, u, u_id, v_id);
                Kokkos::atomic_max(&gain(u), delta);
            });
            Kokkos::fence();

            Kokkos::parallel_for("preferred_id", n * k, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = i / k;
                weight_t u_w = device_g.weights(u);
                partition_t u_id = p_manager.partition(u);
                if (p_manager.bweights(u_id) <= lmax) { return; }

                partition_t v_id = i % k;
                if (u_id == v_id) { return; }
                if (p_manager.bweights(v_id) + u_w > lmax) { return; }

                const weight_t delta = comm_cost_delta(device_g, p_manager, d_oracle, u, u_id, v_id);
                if (delta == gain(u)) {
                    Kokkos::atomic_store(&id(u), v_id);
                }
            });
            Kokkos::fence();

            Kokkos::parallel_for("max_send_recieve_gain", n, KOKKOS_LAMBDA(const vertex_t u) {
                partition_t u_id = p_manager.partition(u);
                partition_t v_id = id(u);
                if (u_id == v_id) { return; }

                Kokkos::atomic_max(&send_gain(u_id), gain(u));
                Kokkos::atomic_max(&recieve_gain(v_id), gain(u));
            });
            Kokkos::fence();

            Kokkos::parallel_for("max_send_recieve_vertex", n, KOKKOS_LAMBDA(const vertex_t u) {
                partition_t u_id = p_manager.partition(u);
                partition_t v_id = id(u);
                if (u_id == v_id) { return; }

                if (send_gain(u_id) == gain(u) && recieve_gain(v_id) == gain(u)) {
                    Kokkos::atomic_min(&send_vertex(u_id), u);
                    Kokkos::atomic_min(&recieve_vertex(v_id), u);
                }
            });
            Kokkos::fence();

            Kokkos::parallel_for("move", k, KOKKOS_LAMBDA(const partition_t u_id) {
                if (p_manager.bweights(u_id) <= lmax) { return; }
                if (send_vertex(u_id) == max_sentinel<vertex_t>()) { return; }

                vertex_t u = send_vertex(u_id);
                partition_t v_id = id(u);

                if (send_vertex(u_id) != recieve_vertex(v_id)) { return; }

                p_manager.partition(u) = v_id;
            });
            Kokkos::fence();

            recalculate_weights(p_manager, device_g);

            // std::cout << max_weight(p_manager) << " " << lmax << " " << n_overloaded(p_manager) << " " << weight_overloaded(p_manager) << " " << comm_cost(device_g, p_manager, d_oracle) << std::endl;
        }
    }
}

#endif //GPU_HEIPROMAP_FAILSAFE_REBALANCER_H
