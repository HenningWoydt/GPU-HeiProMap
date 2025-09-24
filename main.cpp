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

#include <iostream>

#include <Kokkos_Core.hpp>

#include "src/utility/configuration.h"
#include "src/im/im_solver.h"
#include "src/hm/hm_solver.h"


using namespace GPU_HeiProMap;

int main(int argc, char *argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    if (argc == 1) {
        Configuration config(argc, argv);
        config.print_help_message();
        exit(EXIT_FAILURE);
        {
            std::vector<std::pair<std::string, std::string> > input = {
                {"--graph", "../../graph_collection/mapping/rgg24.graph"}, // SharedMap comm cost 10 243 578
                {"--mapping", "../data/out/partition/rgg24.txt"},
                {"--statistics", "../data/out/statistics/rgg24.JSON"},
                {"--hierarchy", "4:8:6"},
                {"--distance", "1:10:100"},
                {"--imbalance", "0.03"},
                {"--config", "IM"},
                {"--seed", "0"},
            };

            std::vector<std::string> args = {"GPU-HeiProMap"};
            for (const auto &[key, val]: input) {
                args.push_back(key);
                args.push_back(val);
            }

            // Step 3: Prepare argc and argv.
            int argc_temp = (int) args.size();
            if (argc_temp < 0) {
                std::cerr << "Error: Invalid argc size" << std::endl;
                exit(EXIT_FAILURE);
            }

            // Allocate an array of char* for argv.
            char **argv_temp = new char *[(size_t) argc_temp];

            for (size_t i = 0; i < (size_t) argc_temp; ++i) {
                // Allocate enough space for the string plus the null terminator.
                argv_temp[i] = new char[args[i].size() + 1];
                std::strcpy(argv_temp[i], args[i].c_str());
            }

            Configuration config(argc_temp, argv_temp);

            if (config.config == "IM") {
                IM_Solver(config).solve();
            } else if (config.config == "HM" || config.config == "HM-Ultra") {
                HM_Solver(config).solve();
            } else {
                std::cerr << "Error: Invalid config" << std::endl;
                exit(EXIT_FAILURE);
            }

            for (int i = 0; i < argc_temp; ++i) { delete[] argv_temp[i]; }
            delete[] argv_temp;
        }
    } else {
        Configuration config(argc, argv);

        if (config.config == "IM") {
            IM_Solver(config).solve();
        } else if (config.config == "HM" || config.config == "HM-Ultra") {
            HM_Solver(config).solve();
        } else {
            std::cerr << "Error: Invalid config" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return 0;
}
