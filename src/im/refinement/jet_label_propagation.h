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

#ifndef GPU_HEIPROMAP_JET_LABEL_PROPAGATION_H
#define GPU_HEIPROMAP_JET_LABEL_PROPAGATION_H

#include "../utility/comm_cost.h"
#include "../../utility/definitions.h"
#include "../data_structures/graph.h"
#include "../data_structures/distance_oracle.h"
#include "../data_structures/large_vertex_partition_csr.h"
#include "../data_structures/partition_manager.h"
#include "../../utility/profiler.h"

namespace GPU_HeiProMap {
    struct JetLabelPropagation {
        vertex_t n = 0;
        vertex_t m = 0;
        partition_t k = 0;
        weight_t lmax = 0;

        u32 n_max_iterations = 12;
        u32 max_weak_iterations = 2;
        f64 phi = 0.999;
        f64 heavy_alpha = 10.0; // smaller - less vertices moved, larger more vertices moved
        f64 sigma_percent = 0.06;
        f64 sigma_percent_min = 0.005;
        weight_t sigma = 10;
        PartitionManager p_manager;

        DeviceWeight gain;
        DevicePartition preferred;
        DeviceWeight gain2;
        DeviceU32 locked;
        DeviceU32 in_X;
        DeviceU32 to_move;

        u32 max_slots = 32;
        u32 rho = 2;
        DeviceU32 bucket_counts;
        DeviceU32 bucket_offsets;
        DeviceU32 bucket_cursor;
        DeviceVertex flat_buckets;
    };

    inline JetLabelPropagation initialize_lp(const vertex_t t_n,
                                             const vertex_t t_m,
                                             const partition_t t_k,
                                             const weight_t t_lmax) {
        ScopedTimer _t("refinement", "JetLabelPropagation", "allocate");

        JetLabelPropagation lp;

        lp.n = t_n;
        lp.m = t_m;
        lp.k = t_k;
        lp.lmax = t_lmax;
        lp.sigma = lp.lmax - (weight_t) ((f64) lp.lmax * lp.sigma_percent);

        lp.p_manager = initialize_p_manager(t_n, t_k, t_lmax);
        lp.gain = DeviceWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gain"), lp.n);
        lp.preferred = DevicePartition(Kokkos::view_alloc(Kokkos::WithoutInitializing, "preferred"), lp.n);
        lp.gain2 = DeviceWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gain2"), lp.n);
        lp.locked = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "locked"), lp.n);
        lp.in_X = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "in_X"), lp.n);
        lp.to_move = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "to_move"), lp.n);

        lp.bucket_counts = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "bucket_counts"), lp.k * lp.max_slots * lp.rho);
        lp.bucket_offsets = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "bucket_offsets"), lp.k * lp.max_slots * lp.rho);
        lp.bucket_cursor = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "bucket_cursor"), lp.k * lp.max_slots * lp.rho);
        lp.flat_buckets = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "flat_buckets"), lp.n);
        return lp;
    }

    KOKKOS_INLINE_FUNCTION
    u32 floor_log2_u32(u32 v) {
        // precondition: v > 0
#if defined(__CUDA_ARCH__)
        // CUDA device: count-leading-zeros (32-bit)
        return 31u - (u32) __clz(v);
#elif defined(__HIP_DEVICE_COMPILE__)
        // HIP device: same builtin
        return 31u - (u32) __clz(v);
#elif defined(_MSC_VER)
        // MSVC host
        unsigned long idx;
        _BitScanReverse(&idx, v);
        return (u32) idx;
#elif defined(__GNUC__) || defined(__clang__)
        // GCC/Clang host
        return 31u - (u32) __builtin_clz(v);
#else
        // portable fallback
        u32 r = 0;
        while (v >>= 1) ++r;
        return r;
#endif
    }

    KOKKOS_INLINE_FUNCTION
    u32 floor_log2_u64(u64 x) {
        if (x == 0ull) return 0u;
#if defined(__CUDA_ARCH__)
        return 63u - (u32) __clzll((unsigned long long) x);
#elif defined(__HIP_DEVICE_COMPILE__)
        return 63u - (u32) __clzll((unsigned long long) x);
#else
        return 63u - (u32) __builtin_clzll((unsigned long long) x);
#endif
    }

    KOKKOS_INLINE_FUNCTION
    u64 abs_s64_to_u64(s64 x) {
        return (x >= 0)
                   ? (u64) x
                   : (x == (s64) 0x8000000000000000LL ? (1ull << 63) : (u64) (-x));
    }

    KOKKOS_INLINE_FUNCTION
    u32 loss_slot(const weight_t gain, const JetLabelPropagation &lp) {
        if (gain > 0) return 0; // best: positive gain
        if (gain == 0) return 1; // tie

        u32 s = 2 + floor_log2_u64((u64) -gain); // negative gain: log2 buckets
        return s < lp.max_slots ? s : lp.max_slots - 1;
    }

    KOKKOS_INLINE_FUNCTION
    u32 loss_slot_decades(weight_t gain, const JetLabelPropagation &lp) {
        const u32 S = lp.max_slots;
        if (S < 2u) return 0u; // degenerate safety
        if (gain > 0) return 0u; // positives
        if (gain == 0) return 1u; // ties

        const u64 a = abs_s64_to_u64(gain); // |negative gain|

        // Base index after {pos, zero}
        const u32 BASE = 2u;

        // −1..−10 → 10 singletons → slots 2..11
        if (a <= 10ull) {
            const u32 s = BASE + (u32) (a - 1ull);
            return (s < S) ? s : (S - 1u);
        }

        // −11..−99 → 9 tens-bins: [11–19], [20–29], …, [90–99] → slots 12..20
        if (a < 100ull) {
            const u32 s = BASE + 10u + (u32) (a / 10ull - 1ull); // a/10 in 1..9 → 0..8
            return (s < S) ? s : (S - 1u);
        }

        // −100..−999 → 9 hundreds-bins: [100–199], …, [900–999] → slots 21..29
        if (a < 1000ull) {
            const u32 s = BASE + 10u + 9u + (u32) (a / 100ull - 1ull); // a/100 in 1..9 → 0..8
            return (s < S) ? s : (S - 1u);
        }

        // ≥ 1000 in magnitude → clamp to last slot
        return S - 1u;
    }

    KOKKOS_INLINE_FUNCTION
    u32 idx_psm(partition_t p, u32 s, u32 m, const JetLabelPropagation &lp) {
        return (p * lp.max_slots + s) * lp.rho + m;
    }

    KOKKOS_INLINE_FUNCTION
    u32 idx_dsm(partition_t d, u32 s, u32 m, const JetLabelPropagation &lp) {
        return (d * lp.max_slots + s) * lp.rho + m;
    }

    KOKKOS_INLINE_FUNCTION
    bool ord_smaller(const JetLabelPropagation &lp, vertex_t u, vertex_t v) {
        weight_t gain_u = lp.gain(u);
        weight_t gain_v = lp.gain(v);

        if (gain_u > gain_v) { return true; }
        if (gain_u == gain_v && u < v) { return true; }
        return false;
    }

    KOKKOS_INLINE_FUNCTION
    partition_t random_partition(vertex_t u, u32 seed, u32 prime, u32 xor_const, partition_t k) {
        // Mix in the seed with a 32-bit hash style formula
        u32 key = (u * prime) ^ (xor_const + seed * 0x9e3779b9u); // 0x9e3779b9u is 32-bit golden ratio
        key ^= key >> 16;
        key *= 0x85ebca6bu; // Murmur3 finalizer constants
        key ^= key >> 13;
        key *= 0xc2b2ae35u;
        key ^= key >> 16;

        return key % k;
    }

    inline void move(JetLabelPropagation &lp,
                     Graph &device_g,
                     LargeVertexPartitionCSR &large_csr) {
        move(large_csr, device_g, lp.p_manager, lp.to_move, lp.preferred);

        Kokkos::parallel_for("move", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            if (lp.to_move(u) == 1) {
                weight_t u_w = device_g.weights(u);
                partition_t u_id = lp.p_manager.partition(u);
                partition_t v_id = lp.preferred(u);

                lp.p_manager.partition(u) = v_id;
                Kokkos::atomic_add(&lp.p_manager.bweights(u_id), -u_w);
                Kokkos::atomic_add(&lp.p_manager.bweights(v_id), u_w);
            }
        });
        Kokkos::fence();
    }

    inline void jetlp(JetLabelPropagation &lp,
                      Graph &device_g,
                      DistanceOracle &d_oracle,
                      LargeVertexPartitionCSR &csr) {
        ScopedTimer _t("refinement", "JetLabelPropagation", "jetlp");

        Kokkos::deep_copy(lp.to_move, 0);
        Kokkos::deep_copy(lp.gain2, -max_sentinel<weight_t>());
        Kokkos::deep_copy(lp.gain, -max_sentinel<weight_t>());
        Kokkos::deep_copy(lp.preferred, lp.k);
        Kokkos::deep_copy(lp.in_X, 0);
        Kokkos::fence();

        f64 avg_weight = (f64) device_g.g_weight / (f64) lp.k;
        Kokkos::parallel_for("determine_max_delta", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                                 partition_t old_id = lp.p_manager.partition(u);

                                 if (lp.locked(u) == 1) { return; }
                                 if ((f64) device_g.weights(u) > lp.heavy_alpha * ((f64) lp.p_manager.bweights(old_id) - avg_weight)) { return; } // vertex is too heavy

                                 const u32 row_begin = csr.row(u);
                                 const u32 row_end = csr.row(u + 1);

                                 weight_t best_delta = -max_sentinel<weight_t>();
                                 partition_t best_id = lp.k;
                                 for (u32 j = row_begin; j < row_end; ++j) {
                                     partition_t new_id = csr.ids(j);
                                     if (old_id == new_id) { continue; }
                                     if (new_id == lp.k) { continue; }

                                     weight_t delta = 0;
                                     for (u32 l = row_begin; l < row_end; ++l) {
                                         partition_t vv_id = csr.ids(l);
                                         weight_t w = csr.weights(l);

                                         if (vv_id == lp.k) { continue; }

                                         delta += w * get_diff(d_oracle, old_id, new_id, vv_id);
                                     }

                                     if (delta > best_delta) {
                                         best_delta = delta;
                                         best_id = new_id;
                                     }
                                 }

                                 if (best_delta >= 0) {
                                     lp.gain(u) = best_delta;
                                     lp.preferred(u) = best_id;

                                     lp.in_X(u) = 1;
                                     lp.gain2(u) = 0;
                                 }
                             }
        );
        Kokkos::fence();

        Kokkos::parallel_for("afterburner", device_g.m, KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = device_g.edges_u(i);
            if (lp.in_X(u) == 0) { return; }

            vertex_t v = device_g.edges_v(i);
            weight_t w = device_g.edges_w(i);

            partition_t old_id = lp.p_manager.partition(u);
            partition_t new_id = lp.preferred(u);

            // decide assumed part of neighbor v
            partition_t v_id = lp.p_manager.partition(v);
            if (lp.in_X(v) == 1 && ord_smaller(lp, u, v)) {
                v_id = lp.preferred(v); // assume v was moved
            }

            weight_t edge_gain = w * get_diff(d_oracle, old_id, new_id, v_id);
            Kokkos::atomic_add(&lp.gain2(u), edge_gain);
        });
        Kokkos::fence();

        Kokkos::parallel_for("nonnegative_filter", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            lp.to_move(u) = lp.gain2(u) >= 0;
        });
        Kokkos::fence();

        Kokkos::deep_copy(lp.locked, lp.to_move);
        Kokkos::fence();

        move(lp, device_g, csr);
    }


    inline void jetrw(JetLabelPropagation &lp,
                      Graph &device_g,
                      DistanceOracle &d_oracle,
                      LargeVertexPartitionCSR &csr) {
        ScopedTimer _t("refinement", "JetLabelPropagation", "jetrw");

        Kokkos::deep_copy(lp.bucket_counts, 0);
        Kokkos::deep_copy(lp.to_move, 0);
        Kokkos::deep_copy(lp.bucket_cursor, 0);
        Kokkos::deep_copy(lp.gain, -max_sentinel<weight_t>());
        Kokkos::deep_copy(lp.preferred, lp.k);
        Kokkos::fence();

        partition_t n_destinations = 0;
        Kokkos::parallel_reduce("count_open_blocks", lp.p_manager.k, KOKKOS_LAMBDA(const partition_t id, partition_t &local) {
                                    local += lp.p_manager.bweights(id) < lp.sigma;
                                },
                                n_destinations);
        Kokkos::fence();

        Kokkos::parallel_for("determine_min_delta", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            partition_t old_id = lp.p_manager.partition(u);
            if (lp.p_manager.bweights(old_id) <= lp.lmax) { return; } // vertex does not need to be moved
            if ((f64) device_g.weights(u) > lp.heavy_alpha * (f64) (lp.p_manager.bweights(old_id) - lp.lmax)) { return; } // vertex is too heavy

            const u32 row_begin = csr.row(u);
            const u32 row_end = csr.row(u + 1);

            weight_t best_delta = -max_sentinel<weight_t>();
            partition_t best_id = lp.k;
            for (u32 j = row_begin; j < row_end; ++j) {
                partition_t new_id = csr.ids(j);
                if (old_id == new_id) { continue; } // dont move to same partition
                if (new_id == lp.k) { continue; }
                if (lp.p_manager.bweights(new_id) > lp.sigma) { continue; } // dont move to overloaded block

                weight_t delta = 0;
                for (u32 l = row_begin; l < row_end; ++l) {
                    partition_t vv_id = csr.ids(l);
                    weight_t w = csr.weights(l);

                    if (vv_id == lp.k) { continue; }

                    delta += w * get_diff(d_oracle, old_id, new_id, vv_id);
                }

                if (delta > best_delta) {
                    best_delta = delta;
                    best_id = new_id;
                }
            }

            if (best_delta == lp.k) {
                // failsafe, choose a random valid destination
                u32 x = (u * 1664525u + 1013904223u) % n_destinations;
                u32 y = 0;
                for (partition_t id = 0; id < lp.k; ++id) {
                    if (lp.p_manager.bweights(id) < lp.sigma) {
                        if (y == x) {
                            best_id = id;
                            break;
                        }
                        ++y;
                    }
                }

                weight_t delta = 0;
                for (u32 l = row_begin; l < row_end; ++l) {
                    partition_t vv_id = csr.ids(l);
                    weight_t w = csr.weights(l);

                    delta += w * get_diff(d_oracle, old_id, best_id, vv_id);
                }
                best_delta = delta;
            }

            if (best_delta != -max_sentinel<weight_t>()) {
                lp.gain(u) = best_delta;
                lp.preferred(u) = best_id;
            }
        });
        Kokkos::fence();

        Kokkos::parallel_for("count_buckets", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            if (lp.preferred(u) == lp.k) { return; } // no move found

            partition_t u_id = lp.p_manager.partition(u);
            u32 s = loss_slot_decades(lp.gain(u), lp);
            u32 mini = u % lp.rho;

            Kokkos::atomic_inc(&lp.bucket_counts(idx_psm(u_id, s, mini, lp)));
        });
        Kokkos::fence();

        // prefix sum for bucket offsets
        Kokkos::parallel_scan("bucket_offsets", lp.k * lp.max_slots * lp.rho,KOKKOS_LAMBDA(const u32 i, u32 &upd, const bool final_pass) {
            u32 val = lp.bucket_counts(i);
            if (final_pass) lp.bucket_offsets(i) = upd; // write old prefix
            upd += val;
        });
        Kokkos::fence();

        // fill the buckets
        Kokkos::deep_copy(lp.bucket_cursor, lp.bucket_offsets);
        Kokkos::fence();

        Kokkos::parallel_for("fill_buckets", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            if (lp.preferred(u) == lp.k) { return; } // no move found

            partition_t u_id = lp.p_manager.partition(u);
            u32 s = loss_slot_decades(lp.gain(u), lp);
            u32 mini = u % lp.rho;

            u32 pos = Kokkos::atomic_fetch_inc(&lp.bucket_cursor(idx_psm(u_id, s, mini, lp)));
            lp.flat_buckets(pos) = u;
        });
        Kokkos::fence();

        Kokkos::parallel_for("pick_prefix", lp.k,KOKKOS_LAMBDA(const partition_t id) {
            if (lp.p_manager.bweights(id) <= lp.lmax) { return; }

            weight_t min_to_lose = lp.p_manager.bweights(id) - lp.sigma;
            weight_t moved = 0;

            // slots in increasing loss order
            for (u32 s_idx = 0; s_idx < lp.max_slots && moved < min_to_lose; ++s_idx) {
                for (u32 r_idx = 0; r_idx < lp.rho && moved < min_to_lose; ++r_idx) {
                    u32 b_idx = idx_psm(id, s_idx, r_idx, lp);
                    u32 beg = lp.bucket_offsets(b_idx);
                    u32 end = lp.bucket_offsets(b_idx) + lp.bucket_counts(b_idx);

                    for (u32 pos = beg; pos < end && moved < min_to_lose; ++pos) {
                        vertex_t u = lp.flat_buckets(pos);
                        weight_t wu = device_g.weights(u);

                        if (moved < min_to_lose) {
                            lp.to_move(u) = 1; // mark selection
                            moved += wu;
                        } else {
                            break;
                        }
                    }
                }
            }
        });
        Kokkos::fence();

        move(lp, device_g, csr);
    }

    inline void jetrs(JetLabelPropagation &lp,
                      Graph &device_g,
                      DistanceOracle &d_oracle,
                      u32 seed,
                      LargeVertexPartitionCSR &csr) {
        ScopedTimer _t("refinement", "JetLabelPropagation", "jetrs");

        Kokkos::deep_copy(lp.bucket_counts, 0);
        Kokkos::deep_copy(lp.to_move, 0);
        Kokkos::deep_copy(lp.gain, -max_sentinel<weight_t>());
        Kokkos::deep_copy(lp.preferred, lp.p_manager.partition);
        Kokkos::fence();

        Kokkos::parallel_for("determine_min_delta", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            partition_t u_id = lp.p_manager.partition(u);

            if (lp.p_manager.bweights(u_id) <= lp.lmax) { return; } // u_id not overloaded
            if ((f64) device_g.weights(u) > lp.heavy_alpha * ((f64) lp.p_manager.bweights(u_id) - ((f64) device_g.g_weight / (f64) lp.k))) { return; } // vertex is too heavy

            const u32 row_begin = csr.row(u);
            const u32 row_end = csr.row(u + 1);

            weight_t best_delta = -max_sentinel<weight_t>();
            partition_t best_id = lp.k;
            for (u32 j = row_begin; j < row_end; ++j) {
                partition_t v_id = csr.ids(j);
                if (u_id == v_id) { continue; } // dont move to same partition
                if (v_id == lp.k) { continue; }
                if (lp.p_manager.bweights(v_id) >= lp.lmax) { continue; } // dont move to overloaded block

                weight_t delta = 0;
                for (u32 l = row_begin; l < row_end; ++l) {
                    partition_t vv_id = csr.ids(l);
                    weight_t w = csr.weights(l);

                    if (vv_id == lp.k) { continue; }

                    delta += w * get_diff(d_oracle, u_id, v_id, vv_id);
                }

                if (delta > best_delta) {
                    best_delta = delta;
                    best_id = v_id;
                }
            }

            if (best_delta != -max_sentinel<weight_t>()) {
                lp.gain(u) = best_delta;
                lp.preferred(u) = best_id;
            }
        });
        Kokkos::fence();

        // 2) For each vertex u in overloaded sources: pick best destination d with cap>0
        Kokkos::parallel_for("count_buckets", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            partition_t u_id = lp.p_manager.partition(u);
            partition_t v_id = lp.preferred(u);

            if (lp.p_manager.bweights(u_id) <= lp.lmax) { return; } // u_id not overloaded
            if ((f64) device_g.weights(u) > lp.heavy_alpha * ((f64) lp.p_manager.bweights(u_id) - ((f64) device_g.g_weight / (f64) lp.k))) { return; } // vertex too heavy
            if (u_id == v_id) { return; }

            u32 s = loss_slot(lp.gain(u), lp);
            u32 mini = u % lp.rho;
            Kokkos::atomic_inc(&lp.bucket_counts(idx_dsm(v_id, s, mini, lp)));
        });
        Kokkos::fence();

        // prefix sum
        Kokkos::parallel_scan("bucket_offsets", lp.k * lp.max_slots * lp.rho, KOKKOS_LAMBDA(const u32 i, u32 &upd, const bool final_pass) {
            u32 val = lp.bucket_counts(i);
            if (final_pass) lp.bucket_offsets(i) = upd;
            upd += val;
        });
        Kokkos::fence();

        // fill
        Kokkos::deep_copy(lp.bucket_cursor, lp.bucket_offsets);
        Kokkos::fence();

        Kokkos::parallel_for("fill_buckets", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            partition_t u_id = lp.p_manager.partition(u);
            partition_t v_id = lp.preferred(u);
            if (u_id == v_id) { return; } // not a candidate

            u32 s = loss_slot(lp.gain(u), lp);
            u32 mini = u % lp.rho;
            u32 pos = Kokkos::atomic_fetch_inc(&lp.bucket_cursor(idx_dsm(v_id, s, mini, lp)));
            lp.flat_buckets(pos) = u;
        });
        Kokkos::fence();

        // 4) Per-destination prefix selection: honor cap[d]
        Kokkos::parallel_for("pick_prefix", lp.k, KOKKOS_LAMBDA(const partition_t d) {
            weight_t w = lp.p_manager.bweights(d);
            weight_t cap_d = (w < lp.lmax) ? (lp.lmax - w) : 0;
            if (cap_d <= 0) return;

            weight_t acquired = 0;
            for (u32 s = 0; s < lp.max_slots && acquired < cap_d; ++s) {
                for (u32 mini = 0; mini < lp.rho && acquired < cap_d; ++mini) {
                    u32 ridx = idx_dsm(d, s, mini, lp);
                    u32 beg = lp.bucket_offsets(ridx);
                    u32 end = lp.bucket_offsets(ridx) + lp.bucket_counts(ridx);

                    for (u32 pos = beg; pos < end && acquired < cap_d; ++pos) {
                        vertex_t u = lp.flat_buckets(pos);

                        // partition_t pu = lp.p_manager.partition(u);
                        // if (lp.p_manager.bweights(pu) <= lp.lmax) continue;

                        weight_t wu = device_g.weights(u);
                        if (acquired + wu <= cap_d) {
                            lp.to_move(u) = 1;
                            acquired += wu;
                        } else {
                            break;
                        }
                    }
                }
            }
        });
        Kokkos::fence();

        move(lp, device_g, csr);
    }

    inline void refine(JetLabelPropagation &lp,
                       Graph &device_g,
                       PartitionManager &p_manager,
                       DistanceOracle &d_oracle,
                       u32 level) {
        copy_into(lp.p_manager, p_manager, device_g.n);

        ScopedTimer _t("refinement", "LargeVertexPartitionCSR", "build_scratch");
        LargeVertexPartitionCSR large_csr = rebuild_scratch(device_g, lp.p_manager);
        _t.stop();

        ScopedTimer _t_comm_cost("refinement", "JetLabelPropagation", "get_comm_cost");
        weight_t best_comm_cost = comm_cost(device_g, lp.p_manager, large_csr, d_oracle);
        weight_t best_weight = max_weight(lp.p_manager);
        _t_comm_cost.stop();

        weight_t curr_comm_cost = best_comm_cost;
        weight_t curr_weight = best_weight;

        bool executed_rw = false;
        weight_t last_rw_comm_cost = max_sentinel<weight_t>();
        weight_t last_rw_weight = max_sentinel<weight_t>();

        ScopedTimer _t_reset_lock("refinement", "JetLabelPropagation", "reset_lock");
        Kokkos::deep_copy(lp.locked, 0);
        Kokkos::fence();
        _t_reset_lock.stop();

        u32 weak_iterations = 0;
        u32 iteration = 0;
        u32 seed = 0;
        while (iteration < lp.n_max_iterations) {
            seed += 1;
            executed_rw = false;
            if (curr_weight <= lp.lmax) {
                jetlp(lp, device_g, d_oracle, large_csr);
                weak_iterations = 0;
            } else {
                ScopedTimer _t_reset_lock2("refinement", "JetLabelPropagation", "reset_lock");
                Kokkos::deep_copy(lp.locked, 0);
                Kokkos::fence();
                _t_reset_lock2.stop();
                if (weak_iterations < lp.max_weak_iterations) {
                    jetrw(lp, device_g, d_oracle, large_csr);
                    weak_iterations++;
                    executed_rw = true;
                } else {
                    jetrs(lp, device_g, d_oracle, seed, large_csr);
                    weak_iterations = 0;
                }
            }

            ScopedTimer _t_comm_cost2("refinement", "JetLabelPropagation", "get_comm_cost");
            curr_comm_cost = comm_cost(device_g, lp.p_manager, large_csr, d_oracle);
            curr_weight = max_weight(lp.p_manager);
            _t_comm_cost2.stop();

            if (executed_rw) {
                if (curr_comm_cost == last_rw_comm_cost && last_rw_weight == curr_weight) {
                    // we executed rw, but it stayed the same, another rw call will change nothing
                    weak_iterations = lp.max_weak_iterations;
                } else {
                    last_rw_comm_cost = curr_comm_cost;
                    last_rw_weight = curr_weight;
                }
            } else {
                last_rw_comm_cost = max_sentinel<weight_t>();
                last_rw_weight = max_sentinel<weight_t>();
            }

            if (curr_weight <= lp.lmax) {
                if (curr_comm_cost < best_comm_cost) {
                    if ((f64) curr_comm_cost < lp.phi * (f64) best_comm_cost) { iteration = 0; }
                    copy_into(p_manager, lp.p_manager, device_g.n);
                    best_comm_cost = curr_comm_cost;
                    best_weight = curr_weight;
                }
            } else if (curr_weight < best_weight) {
                copy_into(p_manager, lp.p_manager, device_g.n);
                best_comm_cost = curr_comm_cost;
                best_weight = curr_weight;
                iteration = 0;
            }

            iteration += 1;
        }
    }
}

#endif //GPU_HEIPROMAP_JET_LABEL_PROPAGATION_H
