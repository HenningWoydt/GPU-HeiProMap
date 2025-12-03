/*******************************************************************************
 * MIT License
 *
 * This file is part of GPU-HeiProMap.
 *
 * Copyright (C) 2025 Henning Woydt <henninwoydt@informatik.uni-heidelberde>
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

#ifndef GPU_HEIPROMAP_HOST_GRAPH_H
#define GPU_HEIPROMAP_HOST_GRAPH_H

#include <iostream>
#include <regex>

#include "../../utility/definitions.h"
#include "../../utility/util.h"

namespace GPU_HeiProMap {
    class HostGraph {
    public:
        vertex_t n;
        vertex_t m;
        weight_t g_weight;

        HostWeight weights;
        HostVertex neighborhood;
        HostVertex edges_v;
        HostWeight edges_w;

        HostGraph() {
            n = 0;
            m = 0;
            g_weight = 0;
        }

        explicit HostGraph(const std::string &file_path) {
            ScopedTimer _t_allocate("io", "CSRGraph", "allocate");
            if (!file_exists(file_path)) {
                std::cerr << "File " << file_path << " does not exist!" << std::endl;
                exit(EXIT_FAILURE);
            }

            // mmap the whole file
            MMap mm = mmap_file_ro(file_path);
            char *p = mm.data;
            const char *end = mm.data + mm.size;

            _t_allocate.stop();
            ScopedTimer _t_read_header("io", "CSRGraph", "read_header");

            // skip comment lines
            while (*p == '%') {
                while (*p != '\n') { ++p; }
                ++p;
            }

            // skip whitespace
            while (*p == ' ') { ++p; }

            // read number of vertices - optimized parsing
            n = 0;
            while (*p != ' ' && *p != '\n') {
                n = n * 10 + (vertex_t) (*p - '0');
                ++p;
            }

            // skip whitespace
            while (*p == ' ') { ++p; }

            // read number of edges - optimized parsing  
            m = 0;
            while (*p != ' ' && *p != '\n') {
                m = m * 10 + (vertex_t) (*p - '0');
                ++p;
            }
            m *= 2;

            // search end of line or fmt
            std::string fmt = "000";
            bool has_v_weights = false;
            bool has_e_weights = false;
            while (*p == ' ') { ++p; }
            if (*p != '\n') {
                // found fmt
                fmt[0] = *p;
                ++p;
                if (*p != '\n') {
                    // found fmt
                    fmt[1] = *p;
                    ++p;
                    if (*p != '\n') {
                        // found fmt
                        fmt[2] = *p;
                        ++p;
                    }
                }
                // skip whitespaces
                while (*p == ' ') { ++p; }
            }
            g_weight = 0;
            weights = HostWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "weights"), n);
            neighborhood = HostVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighborhood"), n + 1);
            neighborhood(0) = 0;
            edges_v = HostVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_v"), m);
            edges_w = HostWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_w"), m);
            has_v_weights = fmt[1] == '1';
            has_e_weights = fmt[2] == '1';

            _t_read_header.stop();
            ScopedTimer _t_read_edges("io", "CSRGraph", "read_edges");

            ++p;
            vertex_t u = 0;
            size_t curr_m = 0;

            // Pre-fetch data pointers for better cache performance
            vertex_t *edges_v_ptr = edges_v.data();
            weight_t *edges_w_ptr = edges_w.data();
            weight_t *weights_ptr = weights.data();
            vertex_t *neighborhood_ptr = neighborhood.data();

            while (p < end) {
                // skip comment lines
                while (*p == '%') {
                    while (*p != '\n') { ++p; }
                    ++p;
                }

                // skip whitespaces
                while (*p == ' ') { ++p; }

                // read in vertex weight - optimized
                weight_t vw = 1;
                if (has_v_weights) {
                    vw = 0;
                    while (*p != ' ' && *p != '\n') {
                        vw = vw * 10 + (weight_t) (*p - '0');
                        ++p;
                    }
                    // skip whitespaces
                    while (*p == ' ') { ++p; }
                }
                weights_ptr[u] = vw;
                g_weight += vw;

                // read in edges - optimized inner loop
                while (*p != '\n' && p < end) {
                    vertex_t v = 0;
                    while (*p != ' ' && *p != '\n') {
                        v = v * 10 + (vertex_t) (*p - '0');
                        ++p;
                    }

                    // skip whitespaces
                    while (*p == ' ') { ++p; }

                    weight_t w = 1;
                    if (has_e_weights) {
                        w = 0;
                        while (*p != ' ' && *p != '\n') {
                            w = w * 10 + (weight_t) (*p - '0');
                            ++p;
                        }
                        // skip whitespaces
                        while (*p == ' ') { ++p; }
                    }

                    edges_v_ptr[curr_m] = v - 1;
                    edges_w_ptr[curr_m] = w;
                    ++curr_m;
                }
                neighborhood_ptr[u + 1] = (vertex_t) curr_m;
                ++u;
                ++p;
            }

            if (curr_m != m) {
                std::cerr << "Number of expected edges " << m << " not equal to number edges " << curr_m << " found!\n";
                munmap_file(mm);
                exit(EXIT_FAILURE);
            }

            _t_read_edges.stop();
            // done with the file
            munmap_file(mm);
        }
    };
}

#endif //GPU_HEIPROMAP_HOST_GRAPH_H
