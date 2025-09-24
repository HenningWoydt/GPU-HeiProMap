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

#ifndef GPU_HEIPROMAP_CONFIGURATION_H
#define GPU_HEIPROMAP_CONFIGURATION_H

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "definitions.h"
#include "JSON_utils.h"
#include "util.h"

namespace GPU_HeiProMap {
    struct CommandLineOption {
        std::string large_key;
        std::string small_key;
        std::string description;
        std::string default_val;
        std::string input;
        bool is_set;
    };

    class Configuration {
        std::vector<CommandLineOption> options = {
            {"--help", "", "Produces the help message", "", "", false},
            {"--graph", "-g", "Filepath to the graph.", "", "", false},
            {"--mapping", "-m", "Output filepath to the generated mapping.", "", "", false},
            {"--statistics", "", "Output filepath to the statistics file.", "GPU-HeiProMap_stats.JSON", "", false},
            {"--hierarchy", "-h", "Hierarchy in the form a1:a2:...:al .", "", "", false},
            {"--distance", "-d", "Distance in the form d1:d2:...:dl .", "", "", false},
            {"--imbalance", "-e", "Allowed imbalance (for example 0.03).", "0.03", "", false},
            {"--config", "-c", "Algorithm to use {IM, HM, HM-ultra}.", "0.03", "", false},
            {"--seed", "-s", "Seed for more randomness.", "0", "", false}
        };

    public:
        // graph information
        std::string graph_in;
        std::string mapping_out;
        std::string statistics_out;

        // hierarchy information
        std::string hierarchy_string;
        std::vector<partition_t> hierarchy;
        partition_t k;

        // distance information
        std::string distance_string;
        std::vector<weight_t> distance;

        // balancing information
        f64 imbalance;

        // partitioning algorithm
        std::string config;

        // random initialization
        u64 seed;

        // device space info
        std::string device_space;

        Configuration(int argc, char *argv[]) {
            // read command lines into vector
            std::vector<std::string> args(argv, argv + argc);

            // check for a help message
            for (size_t i = 1; i < (size_t) argc; ++i) {
                if (args[i] == "--help") {
                    print_help_message();
                    exit(EXIT_SUCCESS);
                }
            }

            // read all command line args
            for (size_t i = 1; i < (size_t) argc; ++i) {
                for (auto &[large_key, small_key, description, default_val, input, is_set]: options) {
                    if (large_key == args[i] || small_key == args[i]) {
                        input = args[i + 1];
                        is_set = true;
                        i += 1;
                        break;
                    }
                }
            }

            // extract info
            graph_in = get("--graph");
            mapping_out = get("--mapping");
            statistics_out = get("--statistics");

            hierarchy_string = get("--hierarchy");
            hierarchy = convert<partition_t>(split(hierarchy_string, ':'));
            k = prod<partition_t>(hierarchy);

            distance_string = get("--distance");
            distance = convert<weight_t>(split(distance_string, ':'));

            imbalance = std::stod(get("--imbalance"));

            // partitioning algorithm
            config = get("--config");

            // random initialization
            if (is_set("--seed")) {
                seed = std::stoull(get("--seed"));
            } else {
                seed = std::random_device{}();
            }

            device_space = get_device_space();

            if (hierarchy.size() != distance.size()) {
                std::cout << "Hierarchy (size " << hierarchy.size() << ") and Distance (size " << distance.size() << ") are not equal!" << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        /**
         * Returns whether the option was entered.
         *
         * @param var The option in interest.
         * @return True if the option was entered, false else.
         */
        bool is_set(const std::string &var) {
            for (const auto &[large_key, small_key, description, default_val, input, is_set]: options) {
                if (large_key == var || small_key == var) {
                    return is_set;
                }
            }
            std::cout << "Command Line \"" << var << "\" is not an allowed name!" << std::endl;
            exit(EXIT_FAILURE);
        }

        /**
         * Gets the entered input as a string.
         *
         * @param var The option in interest.
         * @return The input.
         */
        std::string get(const std::string &var) {
            for (const auto &[large_key, small_key, description, default_val, input, is_set]: options) {
                if (large_key == var || small_key == var) {
                    if (input.empty() && default_val.empty()) {
                        std::cout << "Command Line \"" << var << "\" not set!" << std::endl;
                        exit(EXIT_FAILURE);
                    } else if (input.empty()) {
                        return default_val;
                    }
                    return input;
                }
            }
            std::cout << "Command Line \"" << var << "\" is not an allowed name!" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string get_device_space() {
            using ExecSpace = Kokkos::DefaultExecutionSpace;
#if defined(KOKKOS_ENABLE_CUDA)
            if (std::is_same<ExecSpace, Kokkos::Cuda>::value) { return "Cuda"; }
#endif

#if defined(KOKKOS_ENABLE_HIP)
            if (std::is_same<ExecSpace, Kokkos::HIP>::value) { return "HIP"; }
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
            if (std::is_same<ExecSpace, Kokkos::OpenMP>::value) { return "OpenMP"; }
#endif

#if defined(KOKKOS_ENABLE_THREADS)
            if (std::is_same<ExecSpace, Kokkos::Threads>::value) { return "Threads"; }
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
            if (std::is_same<ExecSpace, Kokkos::Serial>::value) { return "Serial"; }
#endif
        }

        /**
         * Prints the help message.
         */
        void print_help_message() {
            for (const auto &[large_key, small_key, description, default_val, input, is_set]: options) {
                if (small_key.empty()) {
                    std::cout << "[ " << large_key << "] - " << description << std::endl;
                } else {
                    std::cout << "[ " << large_key << ", " << small_key << "] - " << description << std::endl;
                }
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
            s += tabs + to_JSON_MACRO(statistics_out);
            s += tabs + to_JSON_MACRO(hierarchy_string);
            s += tabs + to_JSON_MACRO(hierarchy);
            s += tabs + to_JSON_MACRO(k);
            s += tabs + to_JSON_MACRO(distance_string);
            s += tabs + to_JSON_MACRO(distance);
            s += tabs + to_JSON_MACRO(imbalance);
            s += tabs + to_JSON_MACRO(config);
            s += tabs + to_JSON_MACRO(seed);
            s += tabs + to_JSON_MACRO(device_space);

            s.pop_back();
            s.pop_back();
            s += "\n" + tabs + "}";
            return s;
        }
    };
}

#endif //GPU_HEIPROMAP_CONFIGURATION_H
