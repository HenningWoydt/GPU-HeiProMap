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

#ifndef GPU_HEIPROMAP_DEVICE_GRAPH_H
#define GPU_HEIPROMAP_DEVICE_GRAPH_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include "../../utility/definitions.h"
#include "../../utility/profiler.h"
#include "host_graph.h"
#include "../matching/matching.h"

namespace GPU_HeiProMap {
    struct Graph {
        vertex_t n = 0;
        vertex_t m = 0;
        weight_t g_weight = 0;

        DeviceWeight weights;
        DeviceU32 neighborhood;
        DeviceVertex edges_u;
        DeviceVertex edges_v;
        DeviceWeight edges_w;
    };

    inline weight_t max_weight(const Graph &device_g) {
        weight_t max_val = 0;
        Kokkos::parallel_reduce("compute_max_vertex_weight", device_g.n, KOKKOS_LAMBDA(const vertex_t i, weight_t &local_max) {
                                    if (device_g.weights(i) > local_max) {
                                        local_max = device_g.weights(i);
                                    }
                                },
                                Kokkos::Max<weight_t>(max_val)
        );
        Kokkos::fence();
        return max_val;
    }

    inline Graph initialize_device_g(const HostGraph &host_g) {
        TIME("io", "initialize_device_g", "copy",
             Graph device_g;

             device_g.n = host_g.n;
             device_g.m = host_g.m;
             device_g.g_weight = host_g.g_weight;

             device_g.weights = DeviceWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vertex_weights"), host_g.n);
             device_g.neighborhood = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighborhood"), host_g.n + 1);
             device_g.edges_u = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_u"), host_g.m);
             device_g.edges_v = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_v"), host_g.m);
             device_g.edges_w = DeviceWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_w"), host_g.m);

             Kokkos::deep_copy(device_g.weights, host_g.weights);
             Kokkos::deep_copy(device_g.neighborhood, host_g.neighborhood);
             Kokkos::deep_copy(device_g.edges_v, host_g.edges_v);
             Kokkos::deep_copy(device_g.edges_w, host_g.edges_w);
             Kokkos::fence();
        );

        TIME("io", "initialize_device_g", "fill_edges_u",
             Kokkos::parallel_for("fill_edges_u", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                 u32 begin = device_g.neighborhood(u);
                 u32 end = device_g.neighborhood(u + 1);
                 for (u32 i = begin; i < end; ++i) {
                 device_g.edges_u(i) = u;
                 }
                 });
             Kokkos::fence();
        );

        return device_g;
    }

    inline Graph initialize_device_g(const Graph &device_g,
                                     const Matching &matching) {
        TIME("coarsening", "initialize_device_g", "init_vars",
             Graph coarse_device_g;

             // n, m, weights, offsets
             coarse_device_g.n = device_g.n - (n_matched_v(matching) / 2);
             coarse_device_g.m = device_g.m; // upper bound, will recompute exactly later
             coarse_device_g.g_weight = device_g.g_weight;

             coarse_device_g.weights = DeviceWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vertex_weights"), coarse_device_g.n);
             coarse_device_g.neighborhood = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighborhood"), coarse_device_g.n + 1);
             DeviceVertex real_neighborhood_sizes = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "real_neighborhood_sizes"), coarse_device_g.n);
             Kokkos::deep_copy(real_neighborhood_sizes, 0);
             Kokkos::fence();
        );

        TIME("coarsening", "initialize_device_g", "max_neighborhood_sizes",
        // 1) Per-coarse vertex: max table size and vertex weights
        DeviceU32 max_neighborhood_sizes = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "max_neighborhood_sizes"), coarse_device_g.n);
        Kokkos::parallel_for("max_neighborhood_size", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            vertex_t v = matching.matching(u);
            if (u > v) { return; }

            vertex_t new_u = matching.o_to_n(u);

            u32 deg_u = device_g.neighborhood(u + 1) - device_g.neighborhood(u);
            u32 deg_v = u == v ? 0 : (device_g.neighborhood(v + 1) - device_g.neighborhood(v));
            max_neighborhood_sizes(new_u) = deg_u + deg_v;

            weight_t w_u = device_g.weights(u);
            weight_t w_v = u == v ? 0 : device_g.weights(v);
            coarse_device_g.weights(new_u) = w_u + w_v;
        });
        Kokkos::fence();
        );

        TIME("coarsening", "initialize_device_g", "hash_offsets",
        // 2) Build per-vertex hash ranges (offsets) and get total slots H
        //    hash_offsets has length n+1 already
        DeviceU32 hash_offsets = DeviceU32(Kokkos::view_alloc(Kokkos::WithoutInitializing, "hash_offsets"), coarse_device_g.n + 1);
        Kokkos::deep_copy(hash_offsets, 0);
        );

        TIME("coarsening", "initialize_device_g", "prefix_sum_offsets",
        Kokkos::parallel_scan("prefix_sum_offsets", coarse_device_g.n + 1, KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
            u32 cnt = i < coarse_device_g.n ? max_neighborhood_sizes(i) : 0;
            if (final) hash_offsets(i) = running;
            running += cnt;
        });
        );

        TIME("coarsening", "initialize_device_g", "init_hash_keys_values",
        // 3) Allocate hash tables to exact size H (NOT m)
        DeviceVertex hash_keys(Kokkos::view_alloc(Kokkos::WithoutInitializing, "hash_keys"), device_g.m);
        DeviceWeight hash_vals(Kokkos::view_alloc(Kokkos::WithoutInitializing, "hash_vals"), device_g.m);

        // sentinel for “empty”: coarse_device_g.n (fits in vertex_t)
        Kokkos::deep_copy(hash_keys, coarse_device_g.n);
        Kokkos::deep_copy(hash_vals, 0);
        Kokkos::fence();
        );

        TIME("coarsening", "initialize_device_g", "hash_edges",
        // 5) Insert edges into per-vertex hash tables (linear probing within each vertex range)
        Kokkos::parallel_for("hash_edges", device_g.m, KOKKOS_LAMBDA(const u32 i) {
            vertex_t u = device_g.edges_u(i);
            vertex_t v = device_g.edges_v(i);
            weight_t w = device_g.edges_w(i);

            vertex_t u_new = matching.o_to_n(u);
            vertex_t v_new = matching.o_to_n(v);
            if (matching.matching(u) == v) { return; } // matched edge vanishes

            u32 beg = hash_offsets(u_new);
            u32 end = hash_offsets(u_new + 1);
            u32 len = end - beg;
            if (len == 0) { return; }

            // 32-bit multiplicative mix on v_new; keep in range [0,size)
            u32 x = v_new;
            x ^= x >> 16;
            x *= 0x7feb352dU;
            x ^= x >> 15;
            x *= 0x846ca68bU;
            x ^= x >> 16;

            uint32_t idx = beg + x % len;
            for (u32 j = 0; j < len; ++j) {
                if (idx == end) { idx = beg; }
                vertex_t old = Kokkos::atomic_compare_exchange(&hash_keys(idx), coarse_device_g.n, v_new);
                if (old == coarse_device_g.n || old == v_new) {
                    Kokkos::atomic_add(&hash_vals(idx), w);
                    if (old == coarse_device_g.n) { Kokkos::atomic_inc(&real_neighborhood_sizes(u_new)); }
                    break;
                }
                idx += 1;
            }
        });
        Kokkos::fence();
        );

        TIME("coarsening", "initialize_device_g", "scan_real_deg",
        // 6) Build CSR offsets for coarse graph
        Kokkos::parallel_scan("scan_real_deg", coarse_device_g.n + 1, KOKKOS_LAMBDA(const u32 i, u32 &running, const bool final) {
            const u32 cnt = i < coarse_device_g.n ? real_neighborhood_sizes(i) : 0;
            if (final) coarse_device_g.neighborhood(i) = running;
            running += cnt;
        });
        Kokkos::fence();

        Kokkos::deep_copy(coarse_device_g.m, Kokkos::subview(coarse_device_g.neighborhood, coarse_device_g.n));
        Kokkos::fence();
        );

        TIME("coarsening", "initialize_device_g", "allocate_edge_memory",
        coarse_device_g.edges_u = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_u"), coarse_device_g.m);
        coarse_device_g.edges_v = DeviceVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_v"), coarse_device_g.m);
        coarse_device_g.edges_w = DeviceWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_w"), coarse_device_g.m);
        );

        TIME("coarsening", "initialize_device_g", "materialize_edges",
        // 7) Fill CSR without atomics: iterate each coarse vertex’s hash range and stream its occupied slots into its CSR segment
        Kokkos::parallel_for("materialize_edges", coarse_device_g.n, KOKKOS_LAMBDA(const vertex_t u_new) {
            u32 out = coarse_device_g.neighborhood(u_new);
            u32 end = coarse_device_g.neighborhood(u_new + 1);

            u32 off = hash_offsets(u_new);
            u32 size = hash_offsets(u_new + 1) - off;

            for (u32 idx = 0; idx < size && out < end; ++idx) {
                u32 pos = off + idx;
                vertex_t v = hash_keys(pos);
                if (v != coarse_device_g.n) {
                    coarse_device_g.edges_v(out) = v;
                    coarse_device_g.edges_w(out) = hash_vals(pos);
                    ++out;
                }
            }
        });
        Kokkos::fence();
        );

        TIME("coarsening", "initialize_device_g", "fill_edges_u",
        Kokkos::parallel_for("fill_edges_u", coarse_device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
            u32 begin = coarse_device_g.neighborhood(u);
            u32 end = coarse_device_g.neighborhood(u + 1);
            for (u32 i = begin; i < end; ++i) {
                coarse_device_g.edges_u(i) = u;
            }
        });
        Kokkos::fence();
        );

        return coarse_device_g;
    }

    inline HostGraph to_host_graph(const Graph &device_g) {
        HostGraph host_g;

        host_g.n = device_g.n;
        host_g.m = device_g.m;
        host_g.g_weight = device_g.g_weight;

        host_g.weights = HostWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "weights_host"), device_g.n);
        host_g.neighborhood = HostVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighborhood_host"), device_g.n + 1);
        host_g.edges_v = HostVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_v_host"), device_g.m);
        host_g.edges_w = HostWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_w_host"), device_g.m);

        Kokkos::deep_copy(host_g.weights, device_g.weights);
        Kokkos::deep_copy(host_g.neighborhood, device_g.neighborhood);
        Kokkos::deep_copy(host_g.edges_v, device_g.edges_v);
        Kokkos::deep_copy(host_g.edges_w, device_g.edges_w);
        Kokkos::fence();

        return host_g;
    }
}

#endif //GPU_HEIPROMAP_DEVICE_GRAPH_H
