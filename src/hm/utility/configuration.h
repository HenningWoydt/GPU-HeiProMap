/*******************************************************************************
 * MIT License
 *
 * This file is part of SharedMap_GPU.
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

#ifndef SHAREDMAP_GPU_CONFIGURATION_H
#define SHAREDMAP_GPU_CONFIGURATION_H

#include <iostream>
#include <string>
#include <vector>

#include "definitions.h"
#include "JSON_utils.h"
#include "util.h"

namespace SharedMap_GPU {

    class Configuration {
    public:
        // graph information
        std::string graph_in;
        std::string mapping_out;

        // hierarchy information
        std::string hierarchy_string;
        std::vector<int> hierarchy;
        int k;

        // distance information
        std::string distance_string;
        std::vector<int> distance;

        // info for correctly identifying subgraphs
        std::vector<int> index_vec; // index vector to correctly offset all resulting graphs
        std::vector<int> k_rem_vec; // remaining k vector

        // balancing information
        f64 imbalance;

        // partitioning algorithm
        std::string config;
        bool use_ultra;

        // random initialization
        int seed;

        // device space info
        std::string device_space;

        Configuration(const std::string &t_graph_in,
                      const std::string &t_mapping_out,
                      const std::string &t_hierarchy_string,
                      const std::string &t_distance_string,
                      const f64 t_imbalance,
                      const std::string &t_config,
                      const int t_seed) {
            graph_in = t_graph_in;
            mapping_out = t_mapping_out;

            hierarchy_string = t_hierarchy_string;
            hierarchy = convert<int>(split(hierarchy_string, ':'));
            k = product(hierarchy);

            // distance information
            distance_string = t_distance_string;
            distance = convert<int>(split(distance_string, ':'));

            // info for correctly identifying subgraphs
            index_vec = {1};
            for (size_t i = 0; i < hierarchy.size() - 1; ++i) {
                index_vec.push_back(index_vec[i] * hierarchy[i]);
            }

            k_rem_vec.resize(hierarchy.size());
            int p = 1;
            for (size_t i = 0; i < hierarchy.size(); ++i) {
                k_rem_vec[i] = p * hierarchy[i];
                p *= hierarchy[i];
            }

            // balancing information
            imbalance = t_imbalance;

            // partitioning algorithm
            config = t_config;
            use_ultra = config == "ultra";

            // random initialization
            seed = t_seed;

            if (hierarchy.size() != distance.size()) {
                std::cout << "Hierarchy (size " << hierarchy.size() << ") and Distance (size " << distance.size() << ") are not equal!" << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        /**
         * Converts algorithm configuration into a string in JSON format.
         *
         * @param n_tabs Number of tabs appended in front of each line (for visual purposes).
         * @return String in JSON format.
         */
        std::string to_JSON(const int n_tabs = 0) const {
            std::string tabs;
            for (int i = 0; i < n_tabs; ++i) { tabs.push_back('\t'); }

            std::string s = "{\n";

            s += tabs + to_JSON_MACRO(graph_in);
            s += tabs + to_JSON_MACRO(mapping_out);
            s += tabs + to_JSON_MACRO(hierarchy_string);
            s += tabs + to_JSON_MACRO(hierarchy);
            s += tabs + to_JSON_MACRO(k);
            s += tabs + to_JSON_MACRO(distance_string);
            s += tabs + to_JSON_MACRO(distance);
            s += tabs + to_JSON_MACRO(index_vec);
            s += tabs + to_JSON_MACRO(k_rem_vec);
            s += tabs + to_JSON_MACRO(imbalance);
            s += tabs + to_JSON_MACRO(config);
            s += tabs + to_JSON_MACRO(use_ultra);
            s += tabs + to_JSON_MACRO(seed);
            s += tabs + to_JSON_MACRO(device_space);

            s.pop_back();
            s.pop_back();
            s += "\n" + tabs + "}";
            return s;
        }
    };

}

#endif //SHAREDMAP_GPU_CONFIGURATION_H
