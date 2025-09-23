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

#ifndef GPU_HEIPROMAP_MATCHING_H
#define GPU_HEIPROMAP_MATCHING_H

#include "definitions.h"
#include "macros.h"

namespace GPU_HeiProMap {
    struct Matching {
        vertex_t n = 0;
        DeviceVertex matching;
        DeviceVertex o_to_n;
    };

    inline Matching initialize_matching(const vertex_t t_n) {
        Matching matching;
        matching.n = t_n;

        matching.matching = DeviceVertex("matching", t_n);
        matching.o_to_n = DeviceVertex("o_to_n", t_n);

        Kokkos::parallel_for("init_matching", t_n, KOKKOS_LAMBDA(const vertex_t u) {
            matching.matching(u) = u;
            matching.o_to_n(u) = u;
        });
        Kokkos::fence();

        return matching;
    }

    inline vertex_t n_matched_v(const Matching &matching) {
        vertex_t n = 0;

        Kokkos::parallel_reduce("count_matches", matching.n, KOKKOS_LAMBDA(const vertex_t u, vertex_t &local_count) {
                                    vertex_t v = matching.matching(u);
                                    vertex_t uu = matching.matching(v);
                                    if (u != v && u == uu) { local_count += 1; }
                                },
                                n);
        Kokkos::fence();

        return n;
    }

    inline void determine_translation(Matching &matching) {
        DeviceU32 needs_id("needs_id", matching.n);
        DeviceVertex assigned_id("assigned_id", matching.n);
        Kokkos::fence();

        // First, mark which vertices need IDs
        Kokkos::parallel_for("mark_needs_id", matching.n, KOKKOS_LAMBDA(const vertex_t u) {
            vertex_t v = matching.matching(u);
            if (v == u || u < v) {
                needs_id(u) = 1; // assign only one ID for matched pair
            } else {
                needs_id(u) = 0;
            }
        });
        Kokkos::fence();

        Kokkos::parallel_scan("assign_ids", matching.n,
                              KOKKOS_LAMBDA(const vertex_t u, vertex_t &update, const bool final) {
                                  if (needs_id(u)) {
                                      if (final) assigned_id(u) = update;
                                      update += 1;
                                  }
                              });
        Kokkos::fence();

        // Assign translation
        Kokkos::parallel_for("assign_o_to_n", matching.n, KOKKOS_LAMBDA(const vertex_t u) {
            vertex_t v = matching.matching(u);
            if (v == u || u < v) {
                matching.o_to_n(u) = assigned_id(u);
                matching.o_to_n(v) = assigned_id(u); // v gets same new ID
            }
        });
        Kokkos::fence();
    }
}

#endif //GPU_HEIPROMAP_MATCHING_H
