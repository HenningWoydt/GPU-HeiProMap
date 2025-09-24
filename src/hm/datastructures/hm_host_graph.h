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
#include <regex>
#include <unordered_set>

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

        HM_HostGraph() {
            n            = 0;
            m            = 0;
            graph_weight = 0;
        }

        HM_HostGraph(const int t_n, const int t_m, const int t_weight) {
            n            = t_n;
            m            = t_m;
            graph_weight = t_weight;

            vertex_weights = JetHostWeights("vertex_weights", n);
            neighborhood   = JetHostRowMap("neighborhood", n + 1);
            edges_v        = JetHostEntries("edges_v", m);
            edges_w        = JetHostValues("edges_w", m);
        }

        explicit HM_HostGraph(const std::string& file_path) {
            if (!file_exists(file_path)) {
                std::cerr << "File " << file_path << " does not exist!" << std::endl;
                exit(EXIT_FAILURE);
            }

            std::ifstream file(file_path);
            if (!file.is_open()) {
                std::cerr << "Could not open file " << file_path << "!" << std::endl;
                exit(EXIT_FAILURE);
            }

            std::string line(64, ' ');
            bool has_v_weights = false;
            bool has_e_weights = false;

            // read in header
            while (std::getline(file, line)) {
                if (line[0] == '%') { continue; }

                // read in header
                std::vector<std::string> header = split_ws(line);
                n                               = std::stoi(header[0]);
                m                               = std::stoi(header[1]) * 2;

                // allocate space
                graph_weight    = 0;
                vertex_weights  = JetHostWeights("vertex_weights", n);
                neighborhood    = JetHostRowMap("neighborhood", n + 1);
                neighborhood(0) = 0;
                edges_v         = JetHostEntries("edges_v", m);
                edges_w         = JetHostValues("edges_w", m);

                // read in header
                std::string fmt = "000";
                if (header.size() == 3 && header[2].size() == 3) {
                    fmt = header[2];
                }
                has_v_weights = fmt[1] == '1';
                has_e_weights = fmt[2] == '1';

                break;
            }

            // read in edges
            int u = 0;
            std::vector<int> ints;
            int curr_m = 0;

            while (std::getline(file, line)) {
                if (line[0] == '%') { continue; }
                // convert the lines into ints
                str_to_ints(line, ints);

                size_t i = 0;

                // check if vertex weights
                int w = 1;
                if (has_v_weights) { w = ints[i++]; }
                vertex_weights(u) = w;
                graph_weight += w;

                while (i < ints.size()) {
                    int v = ints[i++] - 1;

                    // check if edge weights
                    w = 1;
                    if (has_e_weights) { w = ints[i++]; }
                    edges_v(curr_m) = v;
                    edges_w(curr_m) = w;
                    curr_m += 1;
                }
                neighborhood(u + 1) = curr_m;

                u += 1;
            }

            if (curr_m != m) {
                std::cerr << "Number of expected edges " << m << " not equal to number edges " << curr_m << " found!" << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        size_t size(const int u) const { return neighborhood(u + 1) - neighborhood(u); }

        int neighbor(const int u, const size_t i) const { return edges_v(neighborhood(u) + i); }

        int weight(const int u) const { return vertex_weights(u); }

        int weight(const int u, const size_t i) const { return edges_w(neighborhood(u) + i); }
    };

    inline bool validate_host_graph(const HM_HostGraph& g) {
        using edge_offset_t = jet_partitioner::edge_offset_t;
        using ordinal_t     = jet_partitioner::ordinal_t;
        using value_t       = jet_partitioner::value_t;

        // 1) Basic size checks
        if (g.n < 0) {
            std::cerr << "Invalid n: " << g.n << "\n";
            return false;
        }
        if (g.m < 0) {
            std::cerr << "Invalid m: " << g.m << "\n";
            return false;
        }
        if ((int)g.vertex_weights.extent(0) != g.n) {
            std::cerr << "vertex_weights size " << g.vertex_weights.extent(0) << " != n (" << g.n << ")\n";
            return false;
        }
        if ((int)g.neighborhood.extent(0) != g.n + 1) {
            std::cerr << "neighborhood size " << g.neighborhood.extent(0) << " != n+1 (" << (g.n + 1) << ")\n";
            return false;
        }
        if ((int)g.edges_v.extent(0) != g.m) {
            std::cerr << "edges_v size " << g.edges_v.extent(0) << " != m (" << g.m << ")\n";
            return false;
        }
        if ((int)g.edges_w.extent(0) != g.m) {
            std::cerr << "edges_w size " << g.edges_w.extent(0) << " != m (" << g.m << ")\n";
            return false;
        }

        // 2) CSR row‐map sanity
        if (g.neighborhood(0) != 0) {
            std::cerr << "neighborhood[0] = " << g.neighborhood(0) << " (must be 0)\n";
            return false;
        }
        if (g.neighborhood(g.n) != g.m) {
            std::cerr << "neighborhood[n] = " << g.neighborhood(g.n) << " != m (" << g.m << ")\n";
            return false;
        }
        for (int i = 0; i < g.n; ++i) {
            edge_offset_t a = g.neighborhood(i);
            edge_offset_t b = g.neighborhood(i + 1);
            if (a > b) {
                std::cerr << "neighborhood[" << i << "]=" << a << " > neighborhood[" << (i + 1) << "]=" << b << "\n";
                return false;
            }
        }

        // 3) Edge‐index bounds
        for (int e = 0; e < g.m; ++e) {
            ordinal_t v = g.edges_v(e);
            if (v < 0 || v >= g.n) {
                std::cerr << "edges_v[" << e << "]=" << v << " out of [0," << g.n << ")\n";
                return false;
            }
        }

        // 4) Weight sum
        value_t sum = 0;
        for (int u = 0; u < g.n; ++u) {
            value_t w = g.vertex_weights(u);
            if (w <= 0) {
                std::cerr << "vertex_weights[" << u << "]=" << w << " is negative\n";
                return false;
            }
            sum += w;
        }
        if (sum != g.graph_weight) {
            std::cerr << "sum(vertex_weights)=" << sum << " != graph_weight (" << g.graph_weight << ")\n";
            return false;
        }

        // all checks passed
        return true;
    }

    inline bool validate_no_self_loops_and_duplicates(const HM_HostGraph& g) {
        // Basic structure sanity (optional, rely on previous validate_host_graph)
        if ((int)g.neighborhood.extent(0) != g.n + 1 ||
            (int)g.edges_v.extent(0) != g.m) {
            std::cerr << "CSR structure invalid: neighborhood size " << g.neighborhood.extent(0) << " (expected " << (g.n + 1) << "), edges_v size " << g.edges_v.extent(0) << " (expected " << g.m << ")\n";
            return false;
        }

        for (int u = 0; u < g.n; ++u) {
            int row_begin = g.neighborhood(u);
            int row_end   = g.neighborhood(u + 1);

            std::unordered_set<int> seen;
            seen.reserve(row_end - row_begin);

            for (int idx = row_begin; idx < row_end; ++idx) {
                int v = g.edges_v(idx);

                // Bounds check
                if (v < 0 || v >= g.n) {
                    std::cerr << "Out‐of‐bounds neighbor: vertex " << u << " has neighbor " << v << " (valid range [0," << g.n << "))\n";
                    return false;
                }

                // Self‐loop check
                if (v == u) {
                    std::cerr << "Self‐loop detected at vertex " << u << "\n";
                    return false;
                }

                // Duplicate‐edge check
                auto result = seen.insert(v);
                if (!result.second) {
                    std::cerr << "Duplicate edge detected in adjacency of vertex " << u << ": neighbor " << v << " appears multiple times\n";
                    return false;
                }
            }
        }

        // check connected both ways
        for (int u = 0; u < g.n; ++u) {
            int row_begin = g.neighborhood(u);
            int row_end   = g.neighborhood(u + 1);

            for (int i = row_begin; i < row_end; ++i) {
                int v = g.edges_v(i);

                int row_begin_v = g.neighborhood(v);
                int row_end_v   = g.neighborhood(v + 1);
                bool found = false;
                for (int j = row_begin_v; j < row_end_v; ++j) {
                    int uu = g.edges_v(j);
                    if (uu == u) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std::cerr << "Edge " << u << " " << v << " not connected both ways\n";
                    return false;
                }

            }
        }

        // If we reach here, no self‐loops or duplicates were found
        return true;
    }


    inline void write_metis(const HM_HostGraph& g, const std::string& filename) {
        // sanity check
        assert(g.n >= 0 && g.m >= 0);
        assert((int)g.neighborhood.extent(0) == g.n + 1);
        assert((int)g.edges_v.extent(0) == g.m);
        assert((int)g.edges_w.extent(0) == g.m);

        // 2) Open file and write header
        std::ofstream out(filename);
        if (!out) {
            throw std::runtime_error("Could not open file " + filename);
        }
        out << g.n << " " << g.m << "\n";

        // 3) Emit adjacency lists (1-based indices)
        for (int u = 0; u < g.n; ++u) {
            int rowStart = g.neighborhood(u);
            int rowEnd   = g.neighborhood(u + 1);
            for (int e = rowStart; e < rowEnd; ++e) {
                int v = g.edges_v(e);
                out << (v + 1) << " ";
            }
            out << "\n";
        }

        out.close();
    }
}

#endif //GPU_HEIPROMAP_HM_HOST_GRAPH_H
