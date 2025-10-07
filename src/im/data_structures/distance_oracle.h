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

#ifndef GPU_HEIPROMAP_DISTANCE_ORACLE_H
#define GPU_HEIPROMAP_DISTANCE_ORACLE_H

#include "../../utility/definitions.h"
#include "../../utility/profiler.h"

namespace GPU_HeiProMap {
    struct DistanceOracle {
        partition_t k = 0;

        DeviceWeight w_mtx;
        DevicePartition h_mtx;
    };

    inline DistanceOracle initialize_d_oracle(partition_t t_k,
                                              std::vector<partition_t> &t_hierarchy,
                                              std::vector<weight_t> &t_distance) {
        ScopedTimer _t("io", "DistanceOracle", "allocate");

        DistanceOracle d_oracle;

        d_oracle.k = t_k;
        d_oracle.w_mtx = DeviceWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "w_mtx"), t_k * t_k);
        d_oracle.h_mtx = DevicePartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_mtx"), t_k * t_k);

        // Copy hierarchy and distances to device views
        const size_t levels = t_hierarchy.size();
        HostPartition h_hierarchy(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_hierarchy"), levels);
        HostWeight h_distance(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_distance"), levels);
        for (size_t i = 0; i < levels; ++i) {
            h_hierarchy(i) = t_hierarchy[i];
            h_distance(i) = t_distance[i];
        }

        // Copy to device
        DevicePartition d_hierarchy(Kokkos::view_alloc(Kokkos::WithoutInitializing, "d_hierarchy"), levels);
        DeviceWeight d_distance(Kokkos::view_alloc(Kokkos::WithoutInitializing, "d_distance"), levels);
        Kokkos::deep_copy(d_hierarchy, h_hierarchy);
        Kokkos::deep_copy(d_distance, h_distance);
        Kokkos::fence();

        // Fill w_mtx and h_mtx in parallel
        Kokkos::parallel_for("build_distance_oracle", t_k * t_k, KOKKOS_LAMBDA(const u32 idx) {
            partition_t i = idx / t_k;
            partition_t j = idx % t_k;
            if (i == j) {
                d_oracle.w_mtx(idx) = 0;
                d_oracle.h_mtx(idx) = 0;
                return;
            }

            partition_t level = 0;
            partition_t group_size = 1;
            for (; level < levels; ++level) {
                group_size *= d_hierarchy(level);
                if ((i / group_size) == j / group_size) {
                    break;
                }
            }

            d_oracle.w_mtx(idx) = d_distance(level);
            d_oracle.h_mtx(idx) = level;
        });
        Kokkos::fence();

        return d_oracle;
    }

    KOKKOS_INLINE_FUNCTION
    weight_t get(const DistanceOracle &d_oracle, const partition_t u_id, const partition_t v_id) {
        return d_oracle.w_mtx(u_id * d_oracle.k + v_id);
    }

    KOKKOS_INLINE_FUNCTION
    weight_t get_diff(const DistanceOracle &d_oracle, const partition_t old_id, const partition_t new_id, const partition_t v_id) {
        return d_oracle.w_mtx(v_id * d_oracle.k + old_id) - d_oracle.w_mtx(v_id * d_oracle.k + new_id);
    }

    KOKKOS_INLINE_FUNCTION
    partition_t lowest_nid(const partition_t u_id, const partition_t a0) {
        return (u_id / a0) * a0;
    }

    KOKKOS_INLINE_FUNCTION
    partition_t greatest_nid(const partition_t u_id, const partition_t a0) {
        return ((u_id / a0) + 1) * a0;
    }

    struct HostDistanceOracle {
        partition_t k = 0;

        HostWeight w_mtx;
        HostPartition h_mtx;
    };

    inline HostDistanceOracle to_host_d_oracle(const DistanceOracle &d_oracle) {
        HostDistanceOracle host_d_oracle;

        host_d_oracle.k = d_oracle.k;

        host_d_oracle.w_mtx = HostWeight("partition", d_oracle.k * d_oracle.k);
        host_d_oracle.h_mtx = HostPartition("b_weights", d_oracle.k * d_oracle.k);
        Kokkos::deep_copy(host_d_oracle.w_mtx, d_oracle.w_mtx);
        Kokkos::deep_copy(host_d_oracle.h_mtx, d_oracle.h_mtx);
        Kokkos::fence();


        return host_d_oracle;
    }

    inline weight_t get(const HostDistanceOracle &host_d_oracle, const partition_t u_id, const partition_t v_id) {
        return host_d_oracle.w_mtx(u_id * host_d_oracle.k + v_id);
    }
}

#endif //GPU_HEIPROMAP_DISTANCE_ORACLE_H
