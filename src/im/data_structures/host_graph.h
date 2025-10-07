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
            ScopedTimer _t_allocate("io", "HostGraph", "allocate");
            if (!file_exists(file_path)) {
                std::cerr << "File " << file_path << " does not exist!" << std::endl;
                exit(EXIT_FAILURE);
            }

            std::ifstream file(file_path, std::ios::binary);
            std::vector<char> big_buf(32u << 20);
            file.rdbuf()->pubsetbuf(big_buf.data(), (long) big_buf.size());

            std::string line;
            line.reserve(1u << 20);
            bool has_v_weights = false;
            bool has_e_weights = false;

            _t_allocate.stop();
            ScopedTimer _t_read_header("io", "HostGraph", "read_header");

            // read in header
            while (std::getline(file, line)) {
                if (line[0] == '%') { continue; }

                // read in header
                std::vector<std::string> header = split_ws(line);
                n = (vertex_t) std::stoul(header[0]);
                m = (vertex_t) std::stoul(header[1]) * 2;

                // allocate space
                g_weight = 0;
                weights = HostWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vertex_weights"), n);
                neighborhood = HostVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighborhood"), n + 1);
                neighborhood(0) = 0;
                edges_v = HostVertex(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_v"), m);
                edges_w = HostWeight(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_w"), m);

                // read in header
                std::string fmt = "000";
                if (header.size() == 3 && header[2].size() == 3) {
                    fmt = header[2];
                }
                has_v_weights = fmt[1] == '1';
                has_e_weights = fmt[2] == '1';

                break;
            }

            _t_read_header.stop();
            ScopedTimer _t_read_edges("io", "HostGraph", "read_edges");

            // read in edges
            std::vector<vertex_t> ints(n);
            vertex_t curr_m = 0;

            vertex_t u = 0;
            while (std::getline(file, line)) {
                if (line[0] == '%') { continue; }
                // convert the lines into ints
                size_t size = str_to_ints(line, ints);

                size_t i = 0;

                // check if vertex weights
                weight_t w = 1;
                if (has_v_weights) { w = ints[i++]; }
                weights(u) = w;
                g_weight += w;

                if (has_e_weights) {
                    for (; i < size; i += 2) {
                        edges_v(curr_m) = ints[i] - 1;
                        edges_w(curr_m) = ints[i + 1];
                        curr_m += 1;
                    }
                } else {
                    for (; i < size; ++i) {
                        edges_v(curr_m) = ints[i] - 1;
                        edges_w(curr_m) = 1;
                        curr_m += 1;
                    }
                }
                neighborhood(u + 1) = curr_m;

                u += 1;
            }

            if (curr_m != m) {
                std::cerr << "Number of expected edges " << m << " not equal to number edges " << curr_m << " found!" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    };
}

#endif //GPU_HEIPROMAP_HOST_GRAPH_H
