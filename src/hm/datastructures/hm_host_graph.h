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

#ifndef GPU_HEIPROMAP_HM_HOST_GRAPH_H
#define GPU_HEIPROMAP_HM_HOST_GRAPH_H

#include <string>
#include <vector>
#include <iostream>

#include "../../utility/definitions.h"
#include "../../utility/util.h"

namespace GPU_HeiProMap {
    class HM_HostGraph {
    public:
        int n;
        int m;
        int graph_weight;

        JetHostWeights vertex_weights;
        JetHostRowMap neighborhood;
        JetHostEntries edges_v;
        JetHostValues edges_w;

        explicit HM_HostGraph(const std::string &file_path) noexcept {
            ScopedTimer _t_allocate("io", "HM_HostGraph", "allocate");
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
            ScopedTimer _t_read_header("io", "HM_HostGraph", "read_header");

            // read in header
            while (std::getline(file, line)) {
                if (line[0] == '%') { continue; }

                // read in header
                std::vector<std::string> header = split_ws(line);
                n = std::stoi(header[0]);
                m = std::stoi(header[1]) * 2;

                // allocate space
                graph_weight = 0;
                vertex_weights = JetHostWeights(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vertex_weights"), (size_t) n);
                neighborhood = JetHostRowMap(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighborhood"), (size_t) n + 1);
                neighborhood(0) = 0;
                edges_v = JetHostEntries(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_v"), (size_t) m);
                edges_w = JetHostValues(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_w"), (size_t) m);

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
            ScopedTimer _t_read_edges("io", "HM_HostGraph", "read_edges");

            // read in edges
            int curr_m = 0;
            std::vector<int> ints((size_t) n);

            int u = 0;
            while (std::getline(file, line)) {
                if (line[0] == '%') { continue; }
                // convert the lines into ints
                size_t size = str_to_ints(line, ints);

                size_t i = 0;

                // check if vertex weights
                int w = 1;
                if (has_v_weights) { w = ints[i++]; }
                vertex_weights(u) = w;
                graph_weight += w;

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

#endif //GPU_HEIPROMAP_HM_HOST_GRAPH_H
