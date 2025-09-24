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

#ifndef GPU_HEIPROMAP_LARGE_VERTEX_PARTITION_CSR_H
#define GPU_HEIPROMAP_LARGE_VERTEX_PARTITION_CSR_H

#include <unordered_map>
#include <unordered_set>
#include <iostream>

#include "../../utility/definitions.h"
#include "device_graph.h"
#include "partition_manager.h"
#include "../../utility/profiler.h"

namespace GPU_HeiProMap {
    struct CSRCheckResult {
        std::uint64_t bad_vertices = 0;
        std::uint64_t mismatched_weight = 0; // entries whose weight != recomputed
        std::uint64_t missing_entry = 0; // in recomputed, not present in CSR
        std::uint64_t extra_entry = 0; // present in CSR, not in recomputed (non-sentinel)
        std::uint64_t duplicate_id = 0; // duplicate partition IDs in a row
        std::uint64_t negative_weight = 0; // CSR has negative weight
    };

    template<class VP> // works for VertexPartitionCSR and LargeVertexPartitionCSR
    inline CSRCheckResult check_vertex_partition_csr(const Graph &G,
                                                     const PartitionManager &PM,
                                                     const VP &csr,
                                                     std::size_t max_reports = 8) {
        CSRCheckResult res{};

        // Mirror to host
        auto h_row = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), csr.row);
        auto h_ids = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), csr.ids);
        auto h_weights = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), csr.weights);

        auto h_neigh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), G.neighborhood);
        auto h_ev = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), G.edges_v);
        auto h_ew = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), G.edges_w);
        auto h_part = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), PM.partition);

        const partition_t sentinel = PM.k;
        const vertex_t n = G.n;

        for (vertex_t u = 0; u < n; ++u) {
            // 1) recompute map<partition, sum_w> for this vertex
            std::unordered_map<partition_t, long long> want; {
                const u32 e_beg = h_neigh(u);
                const u32 e_end = h_neigh(u + 1);
                for (u32 e = e_beg; e < e_end; ++e) {
                    const vertex_t v = h_ev(e);
                    const partition_t pid = h_part(v);
                    const long long w = static_cast<long long>(h_ew(e));
                    want[pid] += w;
                }
            }

            // 2) scan CSR row: check duplicates, negative weights, and match against "want"
            std::unordered_set<partition_t> seen_pid;
            bool bad = false;

            const u32 r_beg = h_row(u);
            const u32 r_end = h_row(u + 1);
            for (u32 j = r_beg; j < r_end; ++j) {
                const partition_t pid = h_ids(j);
                const weight_t w = h_weights(j);

                if (pid == sentinel) {
                    // unused slot; ignore
                    if (w != 0) {
                        // extra safety
                        ++res.extra_entry;
                        bad = true;
                        if (res.bad_vertices < max_reports)
                            std::cerr << "[u=" << u << "] sentinel slot has nonzero weight " << w << "\n";
                    }
                    continue;
                }

                if (w < 0) {
                    ++res.negative_weight;
                    bad = true;
                }

                // duplicate?
                if (!seen_pid.insert(pid).second) {
                    ++res.duplicate_id;
                    bad = true;
                    if (res.bad_vertices < max_reports)
                        std::cerr << "[u=" << u << "] duplicate partition id " << pid << " in CSR row\n";
                }

                auto it = want.find(pid);
                if (it == want.end()) {
                    ++res.extra_entry;
                    bad = true;
                    if (res.bad_vertices < max_reports)
                        std::cerr << "[u=" << u << "] CSR has extra id " << pid
                                << " (weight=" << w << ") not present in recomputed map\n";
                } else {
                    const long long expect = it->second;
                    if (static_cast<long long>(w) != expect) {
                        ++res.mismatched_weight;
                        bad = true;
                        if (res.bad_vertices < max_reports)
                            std::cerr << "[u=" << u << "] weight mismatch for id " << pid
                                    << ": csr=" << w << " recomputed=" << expect << "\n";
                    }
                    want.erase(it); // matched
                }
            }

            // 3) any leftover in "want" are missing entries in CSR
            if (!want.empty()) {
                res.missing_entry += want.size();
                bad = true;
                if (res.bad_vertices < max_reports) {
                    std::cerr << "[u=" << u << "] missing " << want.size()
                            << " ids in CSR; e.g. first missing id=" << want.begin()->first
                            << " weight=" << want.begin()->second << "\n";
                }
            }

            if (bad) ++res.bad_vertices;
            if (res.bad_vertices == max_reports && max_reports > 0) {
                std::cerr << "(… further errors suppressed …)\n";
            }
        }

        return res;
    }

    // Convenience: return bool and optionally print a summary
    template<class VP>
    inline bool assert_vertex_partition_csr(const Graph &G,
                                            const PartitionManager &PM,
                                            const VP &csr,
                                            bool verbose = true) {
        auto r = check_vertex_partition_csr(G, PM, csr, /*max_reports=*/16);
        if (verbose) {
            std::cout << "CSR check: bad_vertices=" << r.bad_vertices
                    << " mismatched=" << r.mismatched_weight
                    << " missing=" << r.missing_entry
                    << " extra=" << r.extra_entry
                    << " duplicate=" << r.duplicate_id
                    << " negative=" << r.negative_weight
                    << std::endl;
        }
        return !(r.bad_vertices == 0 &&
                 r.mismatched_weight == 0 &&
                 r.missing_entry == 0 &&
                 r.extra_entry == 0 &&
                 r.duplicate_id == 0 &&
                 r.negative_weight == 0);
    }


    constexpr u32 WORD_BITS = 64;

    // portable popcount32
    KOKKOS_INLINE_FUNCTION
    u32 popcount32(u32 x) {
#if defined(__CUDA_ARCH__)
        return __popc(x);
#elif defined(__HIP_DEVICE_COMPILE__)
        return __popcnt_u32(x);
#else
        return (u32) __builtin_popcount(x);
#endif
    }

    // --- 64-bit popcount helper (from earlier) ---
    KOKKOS_INLINE_FUNCTION
    u32 popcount64(u64 x) {
#if defined(__CUDA_ARCH__)
        return (u32) __popcll((unsigned long long) x);
#elif defined(__HIP_DEVICE_COMPILE__)
        return (u32) __popcll((unsigned long long) x);
#else
#if defined(_MSC_VER) && !defined(__clang__)
        return (u32) __popcnt64((unsigned long long) x);
#else
        return (u32) __builtin_popcountll((unsigned long long) x);
#endif
#endif
    }

    KOKKOS_INLINE_FUNCTION
    u32 hash32(u32 x) {
        // Knuth multiplicative hash
        return x * 2654435761u;
    }

    struct LargeVertexPartitionCSR {
        DeviceU32 row; // length n
        DevicePartition ids;
        DeviceWeight weights;
        u32 size = 0;

        DeviceU64 bit_array;
    };

    inline LargeVertexPartitionCSR rebuild_scratch(const Graph &device_g,
                                                   const PartitionManager &p_manager) {
        const u32 words = (p_manager.k + WORD_BITS - 1) / WORD_BITS;

        LargeVertexPartitionCSR csr;
        csr.row = DeviceU32("row", device_g.n + 1);
        csr.bit_array = DeviceU64("bit_aray", device_g.n * words);
        DeviceU32 counts("counts", device_g.n);
        Kokkos::deep_copy(csr.bit_array, 0);
        Kokkos::deep_copy(csr.row, 0);
        Kokkos::deep_copy(counts, 0);
        Kokkos::fence();

        Kokkos::parallel_for("mark", device_g.m, KOKKOS_LAMBDA(const u32 e) {
            vertex_t u = device_g.edges_u(e);
            vertex_t v = device_g.edges_v(e);
            partition_t p = p_manager.partition(v);

            u32 widx = p / WORD_BITS;
            u32 b = p % WORD_BITS;
            u64 mask = (u64(1) << b); // 64-bit mask

            u32 idx = u * words + widx;
            Kokkos::atomic_or(&csr.bit_array(idx), mask);
        });
        Kokkos::fence();

        Kokkos::parallel_for("count", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            u32 total = 0;
            u32 base = u * words;
            for (u32 w = 0; w < words; ++w) {
                total += popcount64(csr.bit_array(base + w));
            }
            counts(u) = total + 2;
        });
        Kokkos::fence();

        Kokkos::parallel_scan("set_rows", device_g.n, KOKKOS_LAMBDA(const vertex_t u, u32 &carry, const bool final) {
            const u32 c = counts(u);
            if (final) csr.row(u + 1) = carry + c; // write inclusive, row[0] already 0
            carry += c;
        });
        Kokkos::fence();

        csr.size = 0;
        Kokkos::deep_copy(csr.size, Kokkos::subview(csr.row, device_g.n));

        csr.ids = DevicePartition("ids", csr.size);
        csr.weights = DeviceWeight("weights", csr.size);
        Kokkos::deep_copy(csr.ids, p_manager.k);
        Kokkos::deep_copy(csr.weights, 0);
        Kokkos::fence();

        Kokkos::parallel_for("fill", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            u32 n_beg = device_g.neighborhood(u);
            u32 n_end = device_g.neighborhood(u + 1);
            // u32 n_len = n_end - n_beg;

            u32 r_beg = csr.row(u);
            u32 r_end = csr.row(u + 1);
            u32 r_len = r_end - r_beg;

            for (u32 i = n_beg; i < n_end; ++i) {
                vertex_t v = device_g.edges_v(i);
                weight_t w = device_g.edges_w(i);

                partition_t v_id = p_manager.partition(v);

                u32 j = r_beg + hash32(v_id) % r_len;
                for (u32 t = 0; t < r_len; t++) {
                    if (j == r_end) { j = r_beg; }
                    if (csr.ids(j) == p_manager.k) { csr.ids(j) = v_id; }
                    if (csr.ids(j) == v_id) {
                        csr.weights(j) += w;
                        break;
                    }
                    j += 1;
                }
            }
        });
        Kokkos::fence();

        return csr;
    }

    inline LargeVertexPartitionCSR rebuild(const Graph &device_g,
                                           const PartitionManager &p_manager,
                                           const DeviceU32 &counts,
                                           const DeviceU32 &to_move,
                                           const DevicePartition &preferred) {
        LargeVertexPartitionCSR csr;
        csr.row = DeviceU32("row", device_g.n + 1);
        Kokkos::deep_copy(csr.row, 0);
        Kokkos::fence();

        Kokkos::parallel_scan("set_rows", device_g.n, KOKKOS_LAMBDA(const vertex_t u, u32 &carry, const bool final) {
            const u32 c = counts(u);
            if (final) csr.row(u + 1) = carry + c; // write inclusive, row[0] already 0
            carry += c;
        });
        Kokkos::fence();

        csr.size = 0;
        Kokkos::deep_copy(csr.size, Kokkos::subview(csr.row, device_g.n));

        const u32 words = (p_manager.k + WORD_BITS - 1) / WORD_BITS;

        csr.ids = DevicePartition("ids", csr.size);
        csr.weights = DeviceWeight("weights", csr.size);
        csr.bit_array = DeviceU64("bit_array", device_g.n * words);
        Kokkos::deep_copy(csr.ids, p_manager.k);
        Kokkos::deep_copy(csr.weights, 0);
        Kokkos::fence();

        Kokkos::parallel_for("fill", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            u32 n_beg = device_g.neighborhood(u);
            u32 n_end = device_g.neighborhood(u + 1);
            // u32 n_len = n_end - n_beg;

            u32 r_beg = csr.row(u);
            u32 r_end = csr.row(u + 1);
            u32 r_len = r_end - r_beg;

            for (u32 i = n_beg; i < n_end; ++i) {
                vertex_t v = device_g.edges_v(i);
                weight_t w = device_g.edges_w(i);

                partition_t v_id = p_manager.partition(v);

                if (to_move(v) == 1) {
                    v_id = preferred(v);
                }

                u32 j = r_beg + hash32(v_id) % r_len;
                for (u32 t = 0; t < r_len; t++) {
                    if (j == r_end) { j = r_beg; }
                    if (csr.ids(j) == p_manager.k) {
                        csr.ids(j) = v_id;
                        csr.weights(j) = w;
                        break ;
                    }
                    if (csr.ids(j) == v_id) {
                        csr.weights(j) += w;
                        break ;
                    }
                    j += 1;
                }
            }
        });
        Kokkos::fence();

        return csr;
    }

    inline void move(LargeVertexPartitionCSR &csr,
                     const Graph &device_g,
                     const PartitionManager &p_manager,
                     const DeviceU32 &to_move,
                     const DevicePartition &preferred) {
        const u32 words = (p_manager.k + WORD_BITS - 1) / WORD_BITS;

        Kokkos::deep_copy(csr.bit_array, 0);

        Kokkos::parallel_for("remove_weight", device_g.m, KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = device_g.edges_u(i);
            if (to_move(u) == 0) { return; }

            vertex_t v = device_g.edges_v(i);
            weight_t w = device_g.edges_w(i);

            partition_t u_id = p_manager.partition(u);

            u32 r_beg = csr.row(v);
            u32 r_end = csr.row(v + 1);
            u32 r_len = r_end - r_beg;

            u32 j = r_beg + hash32(u_id) % r_len;
            for (u32 t = 0; t < r_len; t++) {
                if (j == r_end) { j = r_beg; }
                if (csr.ids(j) == u_id) {
                    weight_t old_w = Kokkos::atomic_fetch_add(&csr.weights(j), -w);
                    if (old_w == w) {
                        csr.ids(j) = p_manager.k;
                        return;
                    }
                }
                j += 1;
            }
        });
        Kokkos::fence();

        Kokkos::parallel_for("add_conn", device_g.m, KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = device_g.edges_u(i);
            if (to_move(u) == 0) { return; }

            vertex_t v = device_g.edges_v(i);

            u32 r_beg = csr.row(v);
            u32 r_end = csr.row(v + 1);
            u32 r_len = r_end - r_beg;

            partition_t new_u_id = preferred(u);

            u32 widx = new_u_id / WORD_BITS;
            u32 b = new_u_id % WORD_BITS;
            u64 mask = (u64(1) << b);
            u64 rev_mask = ~mask;

            u32 idx = v * words + widx;

            // only the 0->1 flipper inserts
            const u64 old = Kokkos::atomic_fetch_or(&csr.bit_array(idx), mask);
            bool flipped = (old & mask) == 0;

            if (!flipped) { return; }


            // first pass check if new_u_id exists anywhere
            u32 j = r_beg + hash32(new_u_id) % r_len;
            for (u32 t = 0; t < r_len; t++) {
                if (j == r_end) { j = r_beg; }
                if (csr.ids(j) == new_u_id) {
                    // flip the bit again so we do not count it later
                    Kokkos::atomic_and(&csr.bit_array(idx), rev_mask);
                    return;
                }
                j += 1;
            }

            // new_u_id does not exist, no start from the front and search first empty spot
            j = r_beg + hash32(new_u_id) % r_len;
            for (u32 t = 0; t < r_len; t++) {
                if (j == r_end) { j = r_beg; }
                partition_t val = Kokkos::atomic_compare_exchange(&csr.ids(j), p_manager.k, new_u_id);
                if (val == p_manager.k) {
                    // we inserted, flip the bit again so we do not count it later
                    Kokkos::atomic_and(&csr.bit_array(idx), rev_mask);
                    return;
                }
                j += 1;
            }

            // no empty spot found, bit_array already set to 1 at that spot
        });
        Kokkos::fence();

        DeviceU32 counts("counts", device_g.n);
        u32 sum = 0;

        Kokkos::parallel_reduce("count_and_sum", device_g.n, KOKKOS_LAMBDA(const vertex_t u, u32 &lsum) {
                                    u32 total = 0;
                                    u32 base = u * words;
                                    for (u32 w = 0; w < words; ++w) { total += popcount64(csr.bit_array(base + w)); }
                                    counts(u) = total;
                                    lsum += total;
                                },
                                Kokkos::Sum<u32>(sum)
        );
        Kokkos::fence();

        if (sum > 0) {
            // not enough space, we need to rebuild
            Kokkos::parallel_for("sum_counts", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                counts(u) += (csr.row(u + 1) - csr.row(u)) + (counts(u) > 0 ? 2 : 0);
            });
            Kokkos::fence();

            csr.row = DeviceU32("", 0);
            csr.ids = DevicePartition("", 0);
            csr.weights = DeviceWeight("", 0);
            csr.bit_array = DeviceU64("", 0);

            LargeVertexPartitionCSR new_csr = rebuild(device_g, p_manager, counts, to_move, preferred);
            std::swap(csr, new_csr);
            return;
        }

        Kokkos::parallel_for("add_weight", device_g.m, KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = device_g.edges_u(i);
            if (to_move(u) == 0) { return; }

            vertex_t v = device_g.edges_v(i);
            weight_t w = device_g.edges_w(i);

            partition_t new_u_id = preferred(u);

            u32 r_beg = csr.row(v);
            u32 r_end = csr.row(v + 1);
            u32 r_len = r_end - r_beg;

            u32 j = r_beg + hash32(new_u_id) % r_len;
            for (u32 t = 0; t < r_len; t++) {
                if (j == r_end) { j = r_beg; }
                if (csr.ids(j) == new_u_id) {
                    Kokkos::atomic_add(&csr.weights(j), w);
                    return;
                }
                j += 1;
            }
        });
        Kokkos::fence();
    }
}

#endif //GPU_HEIPROMAP_LARGE_VERTEX_PARTITION_CSR_H
