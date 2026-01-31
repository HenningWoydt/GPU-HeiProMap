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
    auto sp = get_time_point();
    std::ios::sync_with_stdio(false);
    std::cout.tie(nullptr);
    //
    {
        ScopedTimer _t("io", "main", "Kokkos::initialize");
        Kokkos::initialize();
    }

    Configuration config;
    if (argc == 1) {
        config.print_help_message();
        // return 0;
        //
        {
            ScopedTimer _t("io", "Configuration", "read_args");

            std::vector<std::pair<std::string, std::string> > input = {
                {"--graph", "../../ProMapRepo/data/mapping/rgg23.graph"}, // comm cost 9543754, 1098 ms
                {"--mapping", "../data/out/partition/rgg23.txt"},
                {"--statistics", "../data/out/statistics/rgg23.JSON"},
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

            config = Configuration(argc_temp, argv_temp);

            ScopedTimer _t_cleanup("io", "main", "cleanup");
            for (int i = 0; i < argc_temp; ++i) { delete[] argv_temp[i]; }
            delete[] argv_temp;
        }
    } else {
        ScopedTimer _t("io", "main", "read_args");
        config = Configuration(argc, argv);
    }

    if (config.config == "IM") {
        HostGraph g(config.graph_in);
        std::cout << "Read graph in     : " << get_milli_seconds(sp, get_time_point()) << std::endl;

        auto sp_solve = get_time_point();
        HostPartition host_partition = IM_Solver(config).solve(g);
        std::cout << "Solved in         : " << get_milli_seconds(sp_solve, get_time_point()) << std::endl;

        auto sp_write = get_time_point();
        write_partition(host_partition, g.n, config.mapping_out);
        std::cout << "Written in        : " << get_milli_seconds(sp_write, get_time_point()) << std::endl;
    } else if (config.config == "HM" || config.config == "HM-ultra") {
        HM_HostGraph g(config.graph_in);
        std::cout << "Read graph in     : " << get_milli_seconds(sp, get_time_point()) << std::endl;

        auto sp_solve = get_time_point();
        JetHostPartition jet_host_partition = HM_Solver(config).solve(g);
        std::cout << "Solved in         : " << get_milli_seconds(sp_solve, get_time_point()) << std::endl;

        auto sp_write = get_time_point();
        write_partition(jet_host_partition, (size_t) g.n, config.mapping_out);
        std::cout << "Written in        : " << get_milli_seconds(sp_write, get_time_point()) << std::endl;
    } else {
        std::cerr << "Error: Invalid config" << std::endl;
        exit(EXIT_FAILURE);
    }

    Kokkos::fence();
    //
    {
        ScopedTimer _t("io", "main", "Kokkos::finalize");
        Kokkos::finalize();
    }

    Profiler::instance().print_table_ascii_colored(std::cout);

    auto ep = get_time_point();
    std::cout << "Total Time in main.cpp : " << get_seconds(sp, ep) << " seconds." << std::endl;

    return 0;
}
