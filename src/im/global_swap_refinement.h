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

#ifndef GPU_HEIPROMAP_GLOBAL_SWAP_REFINEMENT_H
#define GPU_HEIPROMAP_GLOBAL_SWAP_REFINEMENT_H


#include "comm_cost.h"
#include "definitions.h"
#include "device_graph.h"
#include "distance_oracle.h"
#include "partition_manager.h"
#include "profiler.h"


namespace GPU_HeiProMap {
    struct MaxDeltaInfo {
        weight_t delta = -max_sentinel<weight_t>();
        vertex_t u = max_sentinel<partition_t>();
        vertex_t v = max_sentinel<partition_t>();
    };

    // Custom reducer
    struct MaxDeltaReducer {
        using value_type = MaxDeltaInfo;

        KOKKOS_INLINE_FUNCTION
        static void init(value_type &dst) {
            dst.delta = -max_sentinel<weight_t>(); // or your sentinel
            dst.u = dst.v = max_sentinel<partition_t>();
        }

        KOKKOS_INLINE_FUNCTION
        static void join(value_type &dst, const value_type &src) {
            if (src.delta > dst.delta) {
                dst = src;
            }
        }
    };

    inline void global_swap_refinement(Graph &device_g,
                                       PartitionManager &p_manager,
                                       DistanceOracle &d_oracle) {
        for (u32 it = 0; it < 100; ++it) {
            using MaxLoc = Kokkos::MaxLoc<weight_t, u32>;
            MaxLoc::value_type best;
            Kokkos::parallel_reduce("compute_gains_max", Kokkos::RangePolicy<u32>(0, device_g.m), KOKKOS_LAMBDA(const u32 i, MaxLoc::value_type &local) {
                                        vertex_t u = device_g.edges_u(i);
                                        vertex_t v = device_g.edges_v(i);

                                        partition_t u_id = p_manager.partition(u);
                                        partition_t v_id = p_manager.partition(v);
                                        if (u_id == v_id) return;

                                        weight_t u_w = device_g.weights(u);
                                        weight_t v_w = device_g.weights(v);

                                        weight_t u_id_w = p_manager.bweights(u_id);
                                        weight_t v_id_w = p_manager.bweights(v_id);

                                        bool is_overloaded = u_id_w > p_manager.lmax || v_id_w > p_manager.lmax;
                                        weight_t max_weight = u_id_w > v_id_w ? u_id_w : v_id_w;

                                        weight_t new_u_id_w = p_manager.bweights(u_id) - u_w + v_w;
                                        weight_t new_v_id_w = p_manager.bweights(v_id) - v_w + u_w;

                                        bool new_is_overloaded = new_u_id_w > p_manager.lmax || new_v_id_w > p_manager.lmax;
                                        weight_t new_max_weight = new_u_id_w > new_v_id_w ? new_u_id_w : new_v_id_w;

                                        if (!is_overloaded && new_is_overloaded) { return; } // dont go from balanced to unbalanced
                                        if (is_overloaded && new_max_weight > max_weight) { return; } // dont worsen balance

                                        weight_t u_delta = 0;
                                        for (u32 j = device_g.neighborhood(u); j < device_g.neighborhood(u + 1); ++j) {
                                            vertex_t vv = device_g.edges_v(j);
                                            weight_t vw = device_g.edges_w(j);
                                            partition_t vv_id = p_manager.partition(vv);
                                            if (v == vv) vv_id = u_id;
                                            u_delta += vw * get_diff(d_oracle, u_id, v_id, vv_id);
                                        }

                                        weight_t v_delta = 0;
                                        for (u32 j = device_g.neighborhood(v); j < device_g.neighborhood(v + 1); ++j) {
                                            vertex_t uu = device_g.edges_v(j);
                                            weight_t uw = device_g.edges_w(j);
                                            partition_t uu_id = p_manager.partition(uu);
                                            if (u == uu) uu_id = v_id;
                                            v_delta += uw * get_diff(d_oracle, v_id, u_id, uu_id);
                                        }

                                        const weight_t gain = u_delta + v_delta;

                                        // Update the MaxLoc accumulator
                                        if (gain > local.val) {
                                            local.val = gain; // max value
                                            local.loc = i; // its "location" (edge index)
                                        }
                                    },
                                    MaxLoc(best) // pass reducer constructed with the output reference
            );

            // std::cout << "best " << best.val << " " << best.loc << std::endl;
            if (best.val <= 0) {
                return;
            }

            Kokkos::parallel_for("move", 1, KOKKOS_LAMBDA(const u32 _) {
                u32 i = best.loc;
                vertex_t u = device_g.edges_u(i);
                vertex_t v = device_g.edges_v(i);

                partition_t u_id = p_manager.partition(u);
                partition_t v_id = p_manager.partition(v);

                weight_t u_w = device_g.weights(u);
                weight_t v_w = device_g.weights(v);

                p_manager.bweights(u_id) -= u_w;
                p_manager.bweights(u_id) += v_w;
                p_manager.bweights(v_id) -= v_w;
                p_manager.bweights(v_id) += u_w;

                p_manager.partition(u) = v_id;
                p_manager.partition(v) = u_id;
            });
            Kokkos::fence();
        }
    }
}

#endif //GPU_HEIPROMAP_GLOBAL_SWAP_REFINEMENT_H
