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

#ifndef GPU_HEIPROMAP_SWAP_REFINEMENT_H
#define GPU_HEIPROMAP_SWAP_REFINEMENT_H

#include "comm_cost.h"
#include "definitions.h"
#include "device_graph.h"
#include "distance_oracle.h"
#include "partition_manager.h"
#include "profiler.h"


namespace GPU_HeiProMap {
    inline void swap_refinement(Graph &device_g,
                                PartitionManager &p_manager,
                                DistanceOracle &d_oracle) {
        for (u32 it = 0; it < 100; ++it) {
            DeviceWeight gain("gain", device_g.m);
            Kokkos::deep_copy(gain, -max_sentinel<weight_t>());
            Kokkos::fence();

            Kokkos::parallel_for("compute_gains", device_g.m, KOKKOS_LAMBDA(const u32 i) {
                vertex_t u = device_g.edges_u(i);
                vertex_t v = device_g.edges_v(i);

                partition_t u_id = p_manager.partition(u);
                partition_t v_id = p_manager.partition(v);

                if (u_id == v_id) { return; }

                weight_t u_w = device_g.weights(u);
                weight_t v_w = device_g.weights(v);

                if (u_w != v_w) { return; }

                weight_t u_delta = 0;
                for (u32 j = device_g.neighborhood(u); j < device_g.neighborhood(u + 1); ++j) {
                    vertex_t vv = device_g.edges_v(j);
                    weight_t vw = device_g.edges_w(j);
                    partition_t vv_id = p_manager.partition(vv);

                    if (v == vv) {
                        vv_id = u_id;
                    }

                    u_delta += vw * get_diff(d_oracle, u_id, v_id, vv_id);
                }

                weight_t v_delta = 0;
                for (u32 j = device_g.neighborhood(v); j < device_g.neighborhood(v + 1); ++j) {
                    vertex_t uu = device_g.edges_v(j);
                    weight_t uw = device_g.edges_w(j);
                    partition_t uu_id = p_manager.partition(uu);

                    if (u == uu) {
                        uu_id = v_id;
                    }

                    v_delta += uw * get_diff(d_oracle, v_id, u_id, uu_id);
                }

                gain(i) = u_delta + v_delta;
            });
            Kokkos::fence();

            DevicePartition target("target", device_g.n);
            DeviceVertex partner("partner", device_g.n);
            Kokkos::fence();

            Kokkos::parallel_for("determine_partner", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                vertex_t best_v;
                weight_t best_delta = -max_sentinel<weight_t>();

                for (u32 j = device_g.neighborhood(u); j < device_g.neighborhood(u + 1); ++j) {
                    vertex_t v = device_g.edges_v(j);
                    weight_t delta = gain(j);

                    if (delta > best_delta) {
                        best_v = v;
                        best_delta = delta;
                    }
                }

                partner(u) = u;
                if (best_delta >= 0) {
                    partner(u) = best_v;
                    target(u) = p_manager.partition(best_v);
                }
            });
            Kokkos::fence();

            Kokkos::parallel_for("set_matchings", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                vertex_t v = partner(u);
                if (v == u) { return; } // no partner found

                vertex_t uu = partner(v);
                if (u != uu) {
                    partner(u) = u;
                    return;
                } // partner wants other partner
            });
            Kokkos::fence();

            DeviceU32 active("active", device_g.n);
            Kokkos::deep_copy(active, 0);
            Kokkos::fence();

            Kokkos::parallel_for("set_active", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                vertex_t v = partner(u);
                if (v == u) { return; } // no partner found

                active(u) = 1;
            });
            Kokkos::fence();

            Kokkos::parallel_for("check_neighborhood", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                if (active(u) == 0) { return; }

                vertex_t u_partner = partner(u);
                vertex_t u_max = u > u_partner ? u : u_partner;
                vertex_t u_min = u <= u_partner ? u : u_partner;

                for (u32 j = device_g.neighborhood(u); j < device_g.neighborhood(u + 1); ++j) {
                    vertex_t v = device_g.edges_v(j);
                    if (v == u_partner) { continue; }
                    if (active(v) == 0) { continue; }

                    vertex_t v_partner = partner(v);
                    vertex_t v_max = v > v_partner ? v : v_partner;
                    vertex_t v_min = v <= v_partner ? v : v_partner;

                    // only allow this pair if it is smaller than the other pair
                    if (!(u_min < v_min || ((u_min == v_min) && (u_max < v_max)))) {
                        active(u) = 0;
                        return;
                    }
                }

                for (u32 j = device_g.neighborhood(u_partner); j < device_g.neighborhood(u_partner + 1); ++j) {
                    vertex_t v = device_g.edges_v(j);
                    if (v == u) { continue; }
                    if (active(v) == 0) { continue; }

                    vertex_t v_partner = partner(v);
                    vertex_t v_max = v > v_partner ? v : v_partner;
                    vertex_t v_min = v <= v_partner ? v : v_partner;

                    // only allow this pair if it is smaller than the other pair
                    if (!(u_min < v_min || ((u_min == v_min) && (u_max < v_max)))) {
                        active(u) = 0;
                        return;;
                    }
                }
            });
            Kokkos::fence();

            Kokkos::parallel_for("move_active", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                if (active(u) == 0) { return; }
                weight_t u_w = device_g.weights(u);

                partition_t u_id = p_manager.partition(u);
                partition_t v_id = target(u);

                p_manager.partition(u) = v_id;
                Kokkos::atomic_add(&p_manager.bweights(u_id), -u_w);
                Kokkos::atomic_add(&p_manager.bweights(v_id), u_w);
            });
            Kokkos::fence();
        }
    }
}

#endif //GPU_HEIPROMAP_SWAP_REFINEMENT_H
