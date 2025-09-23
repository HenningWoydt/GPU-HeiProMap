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

#ifndef GPU_HEIPROMAP_HOST_GRAPH_H
#define GPU_HEIPROMAP_HOST_GRAPH_H

#include <iostream>
#include <regex>

#include "definitions.h"
#include "util.h"

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

        HostGraph(const vertex_t t_n, const vertex_t t_m, const weight_t t_weight) {
            n = t_n;
            m = t_m;
            g_weight = t_weight;

            weights = HostWeight("vertex_weights", n);
            neighborhood = HostVertex("neighborhood", n + 1);
            edges_v = HostVertex("edges_v", m);
            edges_w = HostWeight("edges_w", m);
        }

        explicit HostGraph(const std::string &file_path) {
            if (!file_exists(file_path)) {
                std::cerr << "File " << file_path << " does not exist!" << std::endl;
                exit(EXIT_FAILURE);
            }

            std::vector<char> buffer = load_file_to_buffer(file_path);
            size_t size = buffer.size();
            size_t i = 0;

            bool has_v_weights = false;
            bool has_e_weights = false;

            // read header
            n = 0;
            m = 0;
            g_weight = 0;
            while (i < size) {
                if (buffer[i] == '%') {
                    // skip this line
                    while (buffer[i] != '\n') { i += 1; }
                    i += 1;
                    continue;
                }

                // header line
                while (buffer[i] == ' ') { i += 1; } // skip leading whitespaces

                // read n
                n = 0;
                while (buffer[i] != ' ') {
                    n = n * 10 + (vertex_t) (buffer[i] - '0');
                    i += 1;
                }
                while (buffer[i] == ' ') { i += 1; } // skip whitespaces

                // read m
                m = 0;
                while (buffer[i] != ' ' && buffer[i] != '\n') {
                    m = m * 10 + (vertex_t) (buffer[i] - '0');
                    i += 1;
                }
                m *= 2;
                while (buffer[i] == ' ') { i += 1; } // skip whitespaces

                // read fmt
                size_t fmt = 0;
                while (buffer[i] != ' ' && buffer[i] != '\n') {
                    fmt = fmt * 10 + (vertex_t) (buffer[i] - '0');
                    i += 1;
                }

                has_e_weights = fmt % 10 == 1;
                has_v_weights = (fmt / 10) % 10 == 1;

                while (buffer[i] == ' ') { i += 1; }
                i += 1;

                break;
            }

            // allocate space
            g_weight = 0;
            weights = HostWeight("vertex_weights", n);
            neighborhood = HostVertex("neighborhood", n + 1);
            neighborhood(0) = 0;
            edges_v = HostVertex("edges_v", m);
            edges_w = HostWeight("edges_w", m);

            // read body
            vertex_t u = 0;
            size_t idx = 0;
            while (i < size) {
                if (buffer[i] == '%') {
                    // skip this line
                    while (buffer[i] != '\n') { i += 1; }
                    i += 1;
                    continue;
                }

                while (buffer[i] == ' ') { i += 1; } // skip leading whitespaces

                // read in the vertex weight
                if (has_v_weights) {
                    weight_t w = 0;
                    while (buffer[i] != ' ' && buffer[i] != '\n') {
                        w = w * 10 + (buffer[i] - '0');
                        i += 1;
                    }
                    while (buffer[i] == ' ') { i += 1; } // skip whitespaces

                    weights(u) = w;
                    g_weight += w;
                } else {
                    weights(u) = 1;
                    g_weight += 1;
                }

                // read in the edges
                while (i < size && buffer[i] != '\n') {
                    vertex_t v = 0;
                    weight_t w = 1;
                    while (buffer[i] != ' ' && buffer[i] != '\n') {
                        v = v * 10 + (vertex_t) (buffer[i] - '0');
                        i += 1;
                    }
                    while (buffer[i] == ' ') { i += 1; } // skip whitespaces
                    if (has_e_weights) {
                        w = 0;
                        while (buffer[i] != ' ' && buffer[i] != '\n') {
                            w = w * 10 + (buffer[i] - '0');
                            i += 1;
                        }
                        while (buffer[i] == ' ') { i += 1; } // skip whitespaces
                    }

                    edges_v(idx) = v - 1;
                    edges_w(idx) = w;
                    idx += 1;
                }
                i += 1;

                neighborhood(u + 1) = (u32) idx;
                u += 1;
            }

            if (idx != m) {
                std::cerr << "Number of expected edges " << m << " not equal to number edges " << idx << " found!" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    };

    inline void free_host_graph(HostGraph &g) {
        g.n = 0;
        g.m = 0;
        g.g_weight = 0;

        // Reassign empty views to release device/host allocations
        g.weights = HostWeight();
        g.neighborhood = HostVertex();
        g.edges_v = HostVertex();
        g.edges_w = HostWeight();
    }

    inline void write_graph_to_metis(const HostGraph &graph, const std::string &output_file) {
        std::ofstream out(output_file);
        if (!out.is_open()) {
            std::cerr << "Failed to open output file: " << output_file << std::endl;
            exit(EXIT_FAILURE);
        }

        // Write header: n m fmt
        out << graph.n << " " << graph.m / 2 << " 011\n";

        for (vertex_t u = 0; u < graph.n; ++u) {
            out << graph.weights(u);

            for (uint64_t i = graph.neighborhood(u); i < graph.neighborhood(u + 1); ++i) {
                vertex_t v = graph.edges_v(i) + 1;
                weight_t w = graph.edges_w(i);
                out << " " << v << " " << w;
            }

            out << "\n";
        }

        out.close();
    }
}

#endif //GPU_HEIPROMAP_HOST_GRAPH_H
