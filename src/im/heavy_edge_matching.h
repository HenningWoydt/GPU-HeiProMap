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
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#ifndef GPU_HEIPROMAP_HEAVY_EDGE_MATCHING_H
#define GPU_HEIPROMAP_HEAVY_EDGE_MATCHING_H

#include "definitions.h"
#include "device_graph.h"
#include "profiler.h"

namespace GPU_HeiProMap {
    KOKKOS_INLINE_FUNCTION
    f32 edge_noise(vertex_t u, vertex_t v) {
        // make (u,v) order-independent
        uint32_t a = (u < v) ? u : v;
        uint32_t b = (u < v) ? v : u;

        // combine the pair (boost::hash_combine style)
        uint32_t x = a;
        x ^= b + 0x9e3779b9u + (x << 6) + (x >> 2);

        // Murmur3 32-bit finalizer (good avalanche, cheap)
        x ^= x >> 16;
        x *= 0x85ebca6bu;
        x ^= x >> 13;
        x *= 0xc2b2ae35u;
        x ^= x >> 16;

        // map to [0,1): use top 24 bits to match float mantissa width
        uint32_t mant = x >> 8; // top 24 bits
        f32 noise = (f32) mant * (1.0f / 16777216.0f);
        return noise * 0.000001f; // 1% amplitude
    }

    // Helper: bitcast float->u32 without UB
    KOKKOS_INLINE_FUNCTION
    u32 f32_bits(f32 x) {
        union {
            f32 f;
            u32 u;
        } u;
        u.f = x;
        return u.u;
    }

    // Helper: pack rating+tie into one u64
    KOKKOS_INLINE_FUNCTION
    u64 pack_pair(f32 rating, vertex_t v) {
        const u32 r = f32_bits(rating); // monotonic if rating >= 0
        const u32 tie = 0xFFFFFFFFu - (u32) v; // prefer smaller v on ties
        return (u64(r) << 32) | u64(tie);
    }

    // Helper: unpack vertex from packed pair
    KOKKOS_INLINE_FUNCTION
    vertex_t unpack_vertex(u64 packed) {
        if (packed == 0ull) return UINT32_MAX;
        const u32 inv_v = (u32) (packed & 0xFFFFFFFFu);
        return (vertex_t) (0xFFFFFFFFu - inv_v);
    }

    struct HashVertex {
        u32 hash;
        vertex_t v;

        KOKKOS_INLINE_FUNCTION
        bool operator<(const HashVertex &other) const {
            return (hash < other.hash) || (hash == other.hash && v < other.v);
        }
    };

    struct HeavyEdgeMatcher {
        vertex_t n = 0;
        weight_t lmax = 0;
        f64 threshold = 0.4;

        DeviceU32 neighborhood_hash;
        DeviceVertex preferred_neighbor;
        DeviceF32 scratch_rating;
        DeviceU64 u64_helper;

        Kokkos::View<HashVertex *> hash_vertex_array;
        DeviceU32 vertex_to_index; // index into hash_vertex_array
        DeviceU32 index_to_group_id; // for each vertex, which hash group it belongs
        DeviceU32 is_head;

        u32 n_hash_groups = 0; // number of hash groups
        DeviceU32 group_n_vertices; // sizes of each hash group
        DeviceU32 group_begin; // start index of each hash group

        u32 max_iterations_heavy = 3;
        u32 max_iterations_leaf = 3;
        u32 max_iterations_twins = 3;
        u32 max_iterations_relatives = 3;
    };

    inline HeavyEdgeMatcher initialize_hem(const vertex_t t_n,
                                           const weight_t t_lmax) {
        TIME("matching", "misc", "initialiazation",
             HeavyEdgeMatcher hem;
             hem.n = t_n;
             hem.lmax = t_lmax;

             hem.neighborhood_hash = DeviceU32("neighborhood_hash", t_n);
             hem.preferred_neighbor = DeviceVertex("preferred_neighbors", t_n);
             hem.scratch_rating = DeviceF32("scratch_rating", t_n);
             hem.u64_helper = DeviceU64("u64_helper", t_n);

             hem.vertex_to_index = DeviceU32("vertex_to_index", t_n);
             hem.hash_vertex_array = Kokkos::View<HashVertex *>("hash_vertex_array", t_n);
             hem.index_to_group_id = DeviceU32("index_to_group_id", t_n);
             hem.is_head = DeviceU32("is_head", t_n);

             hem.group_n_vertices = DeviceU32("group_n_vertices", t_n + 1);
             hem.group_begin = DeviceU32("group_begin", t_n + 2);
        );
        return hem;
    }

    inline void heavy_edge_matching(HeavyEdgeMatcher &hem,
                                    Graph &device_g,
                                    Matching &matching,
                                    PartitionManager &p_manager) {
        for (u32 iteration = 0; iteration < hem.max_iterations_heavy; ++iteration) {
            if ((f64) n_matched_v(matching) >= hem.threshold * (f64) device_g.n) { return; }

            TIME("matching", "heavy_edge_matching", "reset",
                 Kokkos::parallel_for("reset", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                     hem.u64_helper(u) = pack_pair(0.0f, u);
                     });
                 Kokkos::fence();
            );

            TIME("matching", "heavy_edge_matching", "pick_neighbor",
                 Kokkos::parallel_for("pick_neighbor", device_g.m, KOKKOS_LAMBDA(const u32 i) {
                     vertex_t u = device_g.edges_u(i);
                     vertex_t v = device_g.edges_v(i);

                     if (p_manager.partition(u) != p_manager.partition(v)) { return ; }
                     if (matching.matching(u) != u || matching.matching(v) != v) { return ; }
                     if (device_g.weights(u) + device_g.weights(v) > hem.lmax) { return ; }

                     weight_t w = device_g.edges_w(i);
                     // f32 rating = (f32) w;
                     // f32 rating = (f32) (w) / (f32) (device_g.weights(u) + device_g.weights(v));
                     f32 rating = (f32) (w * w) / (f32) (device_g.weights(u) * device_g.weights(v));
                     // f32 rating = (f32) (1) / (f32) (device_g.weights(u) * device_g.weights(v));
                     rating += edge_noise(u, v);

                     Kokkos::atomic_max(&hem.u64_helper(u), pack_pair(rating, v));
                     });
                 Kokkos::fence();
            );

            TIME("matching", "heavy_edge_matching", "apply_matching",
                 Kokkos::parallel_for("apply_matching", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                     if (matching.matching(u) != u) { return; }

                     vertex_t v = unpack_vertex(hem.u64_helper(u));
                     if (v >= device_g.n || v == u || matching.matching(v) != v) { return; }

                     vertex_t pref_v = unpack_vertex(hem.u64_helper(v));
                     if (pref_v == u && u < v) {
                     matching.matching(u) = v;
                     matching.matching(v) = u;
                     }
                     });
                 Kokkos::fence();
            );
        }
    }

    inline void leaf_matching(HeavyEdgeMatcher &hem,
                              const Graph &device_g,
                              Matching &matching,
                              PartitionManager &p_manager) {
        for (u32 iteration = 0; iteration < hem.max_iterations_leaf; ++iteration) {
            if ((f64) n_matched_v(matching) >= hem.threshold * (f64) device_g.n) { return; }

            TIME("matching", "leaf_matching", "reset",
                 Kokkos::parallel_for("reset", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                     hem.preferred_neighbor(u) = u;
                     });
                 Kokkos::fence();
            );

            TIME("matching", "leaf_matching", "pick_neighbor",
                 Kokkos::parallel_for("pick_neighbor", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                     if (matching.matching(u) != u) { return; } // already matched
                     if (device_g.neighborhood(u + 1) - device_g.neighborhood(u) != 1) { return; }

                     // locate u’s bucket
                     u32 idx = hem.vertex_to_index(u);
                     u32 gid = hem.index_to_group_id(idx);
                     u32 b = hem.group_begin(gid);
                     u32 e = hem.group_begin(gid + 1);

                     constexpr u32 LIMIT = 1024;
                     if ((e - b) > LIMIT) {
                     const u32 half = LIMIT >> 1;
                     const u32 e_minus_LIMIT = e - LIMIT;
                     u32 start = idx > half ? (idx - half) : b;
                     if (start < b) start = b;
                     if (start > e_minus_LIMIT) start = e_minus_LIMIT;
                     b = start;
                     e = start + LIMIT;
                     }

                     f32 best_rating = -max_sentinel<f32>();
                     vertex_t best_v = u;
                     for (u32 i = b; i < e; ++i) {
                     vertex_t v = hem.hash_vertex_array(i).v;
                     if (v == u) { continue ; }
                     if (matching.matching(v) != v) { continue ; }
                     if (p_manager.partition(u) != p_manager.partition(v)) { continue ; }
                     if (device_g.weights(u) + device_g.weights(v) > hem.lmax) { continue ; }

                     f32 rating = (f32) (1) / (f32) (device_g.weights(u) * device_g.weights(v));
                     rating += edge_noise(u, v);
                     if (rating > best_rating) {
                     best_rating = rating;
                     best_v = v;
                     }
                     }

                     hem.preferred_neighbor(u) = best_v;
                     });
                 Kokkos::fence();
            );

            TIME("matching", "leaf_matching", "apply_matching",
                 Kokkos::parallel_for("apply_matching", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                     if (u == hem.preferred_neighbor(u)) { return; }

                     vertex_t v = hem.preferred_neighbor(u);
                     vertex_t preferred_v = hem.preferred_neighbor(v);

                     if (matching.matching(u) != u || matching.matching(v) != v) { return; }

                     if (u == preferred_v && u < v) {
                     matching.matching(u) = v;
                     matching.matching(v) = u;
                     }
                     });
                 Kokkos::fence();
            );
        }
    }

    inline void twin_matching(HeavyEdgeMatcher &hem,
                              const Graph &device_g,
                              Matching &matching,
                              PartitionManager &p_manager) {
        for (u32 iteration = 0; iteration < hem.max_iterations_twins; ++iteration) {
            if ((f64) n_matched_v(matching) >= hem.threshold * (f64) device_g.n) { return; }

            TIME("matching", "twin_matching", "reset",
                 Kokkos::parallel_for("reset", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                     hem.preferred_neighbor(u) = u;
                     });
                 Kokkos::fence();
            );

            TIME("matching", "twin_matching", "pick_neighbor",
                 Kokkos::parallel_for("pick_neighbor", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                     if (matching.matching(u) != u) { return; } // already matched

                     // locate u’s bucket
                     u32 idx = hem.vertex_to_index(u);
                     u32 gid = hem.index_to_group_id(idx);
                     u32 b = hem.group_begin(gid);
                     u32 e = hem.group_begin(gid + 1);

                     constexpr u32 LIMIT = 1024;
                     if ((e - b) > LIMIT) {
                     const u32 half = LIMIT >> 1;
                     const u32 e_minus_LIMIT = e - LIMIT;
                     u32 start = idx > half ? (idx - half) : b;
                     if (start < b) start = b;
                     if (start > e_minus_LIMIT) start = e_minus_LIMIT;
                     b = start;
                     e = start + LIMIT;
                     }

                     f32 best_rating = -max_sentinel<f32>();
                     vertex_t best_v = u;
                     for (u32 i = b; i < e; ++i) {
                     vertex_t v = hem.hash_vertex_array(i).v;
                     if (v == u) { continue ; }
                     if (matching.matching(v) != v) { continue ; }
                     if (p_manager.partition(u) != p_manager.partition(v)) { continue ; }
                     if (device_g.weights(u) + device_g.weights(v) > hem.lmax) { continue ; }

                     f32 rating = (f32) (1) / (f32) (device_g.weights(u) * device_g.weights(v));
                     rating += edge_noise(u, v);
                     if (rating > best_rating) {
                     best_rating = rating;
                     best_v = v;
                     }
                     }

                     hem.preferred_neighbor(u) = best_v;
                     });
                 Kokkos::fence();
            );

            TIME("matching", "twin_matching", "apply_matching",
                 Kokkos::parallel_for("apply_matching", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                     if (u == hem.preferred_neighbor(u)) { return; }

                     vertex_t v = hem.preferred_neighbor(u);
                     vertex_t preferred_v = hem.preferred_neighbor(v);

                     if (matching.matching(u) != u || matching.matching(v) != v) { return; }

                     if (u == preferred_v && u < v) {
                     matching.matching(u) = v;
                     matching.matching(v) = u;
                     }
                     });
                 Kokkos::fence();
            );
        }
    }

    inline void relative_matching(HeavyEdgeMatcher &hem,
                                  Graph &device_g,
                                  Matching &matching,
                                  PartitionManager &p_manager) {
        for (u32 iteration = 0; iteration < hem.max_iterations_relatives; ++iteration) {
            if ((f64) n_matched_v(matching) >= hem.threshold * (f64) device_g.n) { return; }

            TIME("matching", "twin_matching", "reset",
                 Kokkos::parallel_for("reset", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                     hem.preferred_neighbor(u) = u;
                     });
                 Kokkos::fence();
            );

            TIME("matching", "relative_matching", "pick_neighbor",
                 Kokkos::parallel_for("pick_neighbor", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                     if (matching.matching(u) != u) { return; } // ignore already matched

                     vertex_t best_v = u;
                     f32 best_rating = 0;

                     for (u32 i = device_g.neighborhood(u); i < device_g.neighborhood(u + 1); ++i) {
                     vertex_t mid_v = device_g.edges_v(i);
                     weight_t mid_e_w = device_g.edges_w(i);
                     vertex_t mid_deg = device_g.neighborhood(mid_v + 1) - device_g.neighborhood(mid_v);
                     if (mid_deg > 10) { continue; } // avoid matchmaker with high degree

                     for (u32 j = device_g.neighborhood(mid_v); j < device_g.neighborhood(mid_v + 1); ++j) {
                     vertex_t v = device_g.edges_v(j);
                     weight_t v_e_w = device_g.edges_w(j);

                     if (u == v) { continue; }
                     if (matching.matching(v) != v) { continue; } // ignore already matched
                     if (p_manager.partition(u) != p_manager.partition(v)) { continue ; }
                     if (device_g.weights(u) + device_g.weights(v) > hem.lmax) { continue; } // resulting weight to large

                     f32 rating = (f32) ((mid_e_w + v_e_w) * (mid_e_w + v_e_w)) / (f32) (device_g.weights(u) * device_g.weights(v));
                     rating += edge_noise(u, v);

                     if (rating > best_rating) {
                     best_v = v;
                     best_rating = rating;
                     }
                     }
                     }

                     hem.preferred_neighbor(u) = best_v;
                     });
                 Kokkos::fence();
            );

            TIME("matching", "relative_matching", "apply_matching",
                 Kokkos::parallel_for("apply_matching", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                     vertex_t v = hem.preferred_neighbor(u);
                     vertex_t preferred_v = hem.preferred_neighbor(v);

                     if (matching.matching(u) != u || matching.matching(v) != v) { return; }

                     if (u == preferred_v && u < v) {
                     matching.matching(u) = v;
                     matching.matching(v) = u;
                     }
                     });
                 Kokkos::fence();
            );
        }
    }

    inline void build_hash_vertex_array(HeavyEdgeMatcher &hem,
                                        Graph &device_g) {
        TIME("matching", "build_hash_vertex_array", "reset",
             Kokkos::deep_copy(hem.neighborhood_hash, 0);
             Kokkos::fence();
        );

        TIME("matching", "build_hash_vertex_array", "hash",
             Kokkos::parallel_for("hash", device_g.m, KOKKOS_LAMBDA(const u32 i) {
                 vertex_t u = device_g.edges_u(i);
                 vertex_t v = device_g.edges_v(i);

                 u32 h = v * 2654435761u; // 2654435761 = 2^32 / golden ratio

                 Kokkos::atomic_fetch_add(&hem.neighborhood_hash(u), h);
                 });
             Kokkos::fence();
        );

        TIME("matching", "build_hash_vertex_array", "fill_array",
             // build the hash array
             Kokkos::parallel_for("fill_array", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                 hem.hash_vertex_array(u).hash = hem.neighborhood_hash(u);
                 hem.hash_vertex_array(u).v = u;
                 });
             Kokkos::fence();
        );

        TIME("matching", "build_hash_vertex_array", "sort",
             // sort the hash array
             Kokkos::sort(hem.hash_vertex_array);
             Kokkos::fence();
        );

        TIME("matching", "build_hash_vertex_array", "index_map",
             // build index map
             Kokkos::parallel_for("index_map", device_g.n, KOKKOS_LAMBDA(const u32 i) {
                 vertex_t u = hem.hash_vertex_array(i).v;
                 hem.vertex_to_index(u) = i;
                 });
             Kokkos::fence();
        );

        TIME("matching", "build_hash_vertex_array", "mark_and_map_groups",
             // mark the start of each hash group
             Kokkos::parallel_scan("mark_and_map_groups", device_g.n, KOKKOS_LAMBDA(u32 i, u32 &running, bool final) {
                 bool head = (i == 0 || hem.hash_vertex_array(i).hash != hem.hash_vertex_array(i - 1).hash);
                 u32 inc = head ? 1 : 0;

                 if (final) {
                 hem.is_head(i) = inc;
                 hem.index_to_group_id(i) = running;
                 }

                 running += inc;
                 },
                 hem.n_hash_groups
             );
             Kokkos::fence();
             hem.n_hash_groups += 1;
        );

        TIME("matching", "build_hash_vertex_array", "set_to_0",
             // set the size of each group
             Kokkos::parallel_for("set_to_0", hem.n_hash_groups, KOKKOS_LAMBDA(u32 g) {
                 hem.group_n_vertices(g) = 0;
                 });
             Kokkos::fence();
        );

        TIME("matching", "build_hash_vertex_array", "count",
             Kokkos::parallel_for("count", device_g.n, KOKKOS_LAMBDA(u32 i) {
                 Kokkos::atomic_fetch_add(&hem.group_n_vertices(hem.index_to_group_id(i)), 1);
                 });
        );

        TIME("matching", "build_hash_vertex_array", "prefix_sum_groups",
             // set the start of each group
             Kokkos::parallel_scan("prefix_sum_groups", hem.n_hash_groups + 1, KOKKOS_LAMBDA(const u32 g, u32 &running, const bool final) {
                 // for g < n_groups use the count, else (the last slot) use 0
                 u32 cnt = (g < hem.n_hash_groups ? hem.group_n_vertices(g) : 0);
                 if (final) {
                 hem.group_begin(g) = running;
                 }
                 running += cnt;
                 }
             );
             Kokkos::fence();
        );
    }

    inline Matching match(HeavyEdgeMatcher &hem,
                          Graph &device_g,
                          PartitionManager &p_manager) {
        Matching matching = initialize_matching(device_g.n);
        vertex_t n_matched = 0;

        heavy_edge_matching(hem, device_g, matching, p_manager);
        n_matched = n_matched_v(matching);

        if ((f64) n_matched < hem.threshold * (f64) device_g.n) {
            build_hash_vertex_array(hem, device_g);
        }

        if ((f64) n_matched < hem.threshold * (f64) device_g.n) {
            leaf_matching(hem, device_g, matching, p_manager);
            n_matched = n_matched_v(matching);
        }

        if ((f64) n_matched < hem.threshold * (f64) device_g.n) {
            twin_matching(hem, device_g, matching, p_manager);
            n_matched = n_matched_v(matching);
        }

        if ((f64) n_matched < hem.threshold * (f64) device_g.n) {
            relative_matching(hem, device_g, matching, p_manager);
        }

        TIME("matching", "misc", "determine_translation",
             determine_translation(matching);
        );

        return matching;
    }
} // namespace GPU_HeiProMap

#endif // GPU_HEIPROMAP_HEAVY_EDGE_MATCHING_H
