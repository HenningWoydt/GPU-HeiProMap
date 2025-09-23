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

#ifndef GPU_HEIPROMAP_PARTITION_MANAGER_H
#define GPU_HEIPROMAP_PARTITION_MANAGER_H

#include "definitions.h"

namespace GPU_HeiProMap {
    struct PartitionManager {
        vertex_t n = 0;
        partition_t k = 0;
        weight_t lmax = 0;

        DevicePartition partition;
        DeviceWeight bweights;
    };

    inline PartitionManager initialize_p_manager(const vertex_t t_n,
                                                 const partition_t t_k,
                                                 const weight_t t_lmax) {
        PartitionManager p_manager;

        p_manager.n = t_n;
        p_manager.k = t_k;
        p_manager.lmax = t_lmax;

        p_manager.partition = DevicePartition("partition", t_n);
        p_manager.bweights = DeviceWeight("b_weights", t_k);

        Kokkos::deep_copy(p_manager.partition, 0);
        Kokkos::deep_copy(p_manager.bweights, 0);
        Kokkos::fence();

        return p_manager;
    }

    inline void copy_into(PartitionManager &dst, const PartitionManager &src, u32 n) {
        dst.n = src.n;
        dst.k = src.k;
        dst.lmax = src.lmax;

        auto rN = std::make_pair<size_t, size_t>(0, n);
        Kokkos::deep_copy(Kokkos::subview(dst.partition, rN), Kokkos::subview(src.partition, rN));
        Kokkos::deep_copy(dst.bweights, src.bweights);

        Kokkos::fence();
    }

    inline void contract(PartitionManager &p_manager,
                           Matching &matching) {
        // reset activity
        DevicePartition temp_device_partition = DevicePartition("partition", p_manager.n);
        Kokkos::parallel_for("initialize", matching.n, KOKKOS_LAMBDA(const vertex_t u) {
            if (u == matching.matching(u)) {
                vertex_t u_new = matching.o_to_n(u);
                temp_device_partition(u_new) = p_manager.partition(u);
            }

            if (u < matching.matching(u)) {
                vertex_t u_new = matching.o_to_n(u);
                temp_device_partition(u_new) = p_manager.partition(u);
            }
        });
        Kokkos::fence();

        std::swap(p_manager.partition, temp_device_partition);
    }

    inline void uncontract(PartitionManager &p_manager,
                           Matching &matching) {
        // reset activity
        DevicePartition temp_device_partition = DevicePartition("device_partition", p_manager.n);
        Kokkos::parallel_for("initialize", matching.n, KOKKOS_LAMBDA(const vertex_t u) {
            vertex_t v = matching.matching(u);
            vertex_t u_new = matching.o_to_n(u);
            vertex_t v_new = matching.o_to_n(v);

            if (u == v || u < v) {
                temp_device_partition(u) = p_manager.partition(u_new);
                temp_device_partition(v) = p_manager.partition(v_new);
            }
        });
        Kokkos::fence();

        std::swap(p_manager.partition, temp_device_partition);
    }

    inline void recalculate_weights(PartitionManager &p_manager,
                                    const Graph &device_g) {
        // reset weights
        Kokkos::deep_copy(p_manager.bweights, 0);
        Kokkos::fence();

        // set weights
        Kokkos::parallel_for("set_block_weights", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            partition_t u_id = p_manager.partition(u);
            Kokkos::atomic_add(&p_manager.bweights(u_id), device_g.weights(u));
        });
        Kokkos::fence();
    }

    inline weight_t max_weight(const PartitionManager &p_manager) {
        weight_t max_val = 0;

        Kokkos::parallel_reduce("compute_max_weight", p_manager.k, KOKKOS_LAMBDA(const partition_t i, weight_t &local_max) {
                                    if (p_manager.bweights(i) > local_max) {
                                        local_max = p_manager.bweights(i);
                                    }
                                },
                                Kokkos::Max<weight_t>(max_val)
        );
        Kokkos::fence();

        return max_val;
    }

    inline void print_block_weights(const PartitionManager &p_manager) {
        // 1) Mirror device view to host
        auto host_weights = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p_manager.bweights);

        // 2) Iterate and print
        const partition_t k = p_manager.k;
        std::cout << "[ ";
        for (partition_t i = 0; i < k; ++i) {
            std::cout << host_weights(i) << ", ";
        }
        std::cout << "]" << std::endl;
    }

    inline void print_partition(const PartitionManager &p_manager, const vertex_t max_n) {
        // 1) Mirror device view to host
        auto host_partition = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p_manager.partition);

        // 2) Iterate and print
        const vertex_t n = p_manager.n;
        std::cout << "[ ";
        for (vertex_t i = 0; i < std::min(n, max_n); ++i) {
            std::cout << host_partition(i) << ", ";
        }
        std::cout << "]" << std::endl;
    }

    inline f64 max_balance(const PartitionManager &p_manager) {
        weight_t max_val = max_weight(p_manager);

        return (f64) max_val / (f64) p_manager.lmax;
    }

    inline vertex_t n_overloaded(const PartitionManager &p_manager) {
        vertex_t n_oload = 0;

        Kokkos::parallel_reduce("compute_max_weight", p_manager.k, KOKKOS_LAMBDA(const partition_t i, vertex_t &local_n) {
                                    if (p_manager.bweights(i) > p_manager.lmax) {
                                        local_n += 1;
                                    }
                                },
                                Kokkos::Sum<vertex_t>(n_oload)
        );
        Kokkos::fence();

        return n_oload;
    }

    inline partition_t n_empty_partitions(const PartitionManager &p_manager) {
        partition_t n = 0;

        Kokkos::parallel_reduce("compute", p_manager.k, KOKKOS_LAMBDA(const partition_t i, partition_t &local_n) {
                                    if (p_manager.bweights(i) == 0) {
                                        local_n += 1;
                                    }
                                },
                                Kokkos::Sum<partition_t>(n)
        );
        Kokkos::fence();

        return n;
    }

    inline weight_t weight_overloaded(const PartitionManager &p_manager) {
        weight_t w = 0;

        Kokkos::parallel_reduce("compute_max_weight", p_manager.k, KOKKOS_LAMBDA(const partition_t i, weight_t &local_weight) {
                                    if (p_manager.bweights(i) > p_manager.lmax) {
                                        local_weight += p_manager.bweights(i) - p_manager.lmax;
                                    }
                                },
                                Kokkos::Sum<weight_t>(w)
        );
        Kokkos::fence();

        return w;
    }

    struct HostPartitionManager {
        vertex_t n = 0;
        partition_t k = 0;
        weight_t lmax = 0;

        HostPartition partition;
        HostWeight bweights;
    };

    inline HostPartitionManager to_host_p_manager(const PartitionManager &p_manager) {
        HostPartitionManager host_p_manager;

        host_p_manager.n = p_manager.n;
        host_p_manager.k = p_manager.k;
        host_p_manager.lmax = p_manager.lmax;

        host_p_manager.partition = HostPartition("partition", p_manager.n);
        host_p_manager.bweights = HostWeight("b_weights", p_manager.k);
        Kokkos::deep_copy(host_p_manager.partition, p_manager.partition);
        Kokkos::deep_copy(host_p_manager.bweights, p_manager.bweights);
        Kokkos::fence();


        return host_p_manager;
    }
}

#endif //GPU_HEIPROMAP_PARTITION_MANAGER_H
