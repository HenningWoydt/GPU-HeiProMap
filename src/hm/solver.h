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

#ifndef SHAREDMAP_GPU_SOLVER_H
#define SHAREDMAP_GPU_SOLVER_H

#include <chrono>
#include <fstream>
#include <ostream>

#include "datastructures/device_graph.h"
#include "datastructures/host_graph.h"
#include "partitioning/partition.h"
#include "utility/configuration.h"

namespace SharedMap_GPU {
    class Solver {
        Configuration& m_configuration;

        f64 io_time              = 0.0;
        f64 solve_time           = 0.0;
        f64 time_partition       = 0.0;
        f64 time_subgraphs       = 0.0;
        f64 time_final_partition = 0.0;

    public:
        explicit Solver(Configuration& configuration) : m_configuration(configuration) {
            using ExecSpace = Kokkos::DefaultExecutionSpace;
            std::string space_name;

#if defined(KOKKOS_ENABLE_CUDA)
            if (std::is_same<ExecSpace, Kokkos::Cuda>::value) {
                m_configuration.device_space = "Cuda";
            }
#endif

#if defined(KOKKOS_ENABLE_HIP)
            if (std::is_same<ExecSpace, Kokkos::HIP>::value) {
                m_configuration.device_space = "HIP";
            }
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
            if (std::is_same<ExecSpace, Kokkos::OpenMP>::value) {
                m_configuration.device_space = "OpenMP";
            }
#endif

#if defined(KOKKOS_ENABLE_THREADS)
            if (std::is_same<ExecSpace, Kokkos::Threads>::value) {
                m_configuration.device_space = "Threads";
            }
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
            if (std::is_same<ExecSpace, Kokkos::Serial>::value) {
                m_configuration.device_space = "Serial";
            }
#endif
        }

        void solve() {
            auto sp = std::chrono::steady_clock::now();
            HostGraph g(m_configuration.graph_in);
            auto ep = std::chrono::steady_clock::now();
            io_time += (f64)std::chrono::duration_cast<std::chrono::nanoseconds>(ep - sp).count() / 1e9;

            // solve problem
            sp                         = std::chrono::steady_clock::now();
            std::vector<int> partition = internal_solve(g);
            ep                         = std::chrono::steady_clock::now();
            solve_time += (f64)std::chrono::duration_cast<std::chrono::nanoseconds>(ep - sp).count() / 1e9;

            // write output
            sp = std::chrono::steady_clock::now();
            write_solution(partition);
            ep = std::chrono::steady_clock::now();
            io_time += (f64)std::chrono::duration_cast<std::chrono::nanoseconds>(ep - sp).count() / 1e9;

            print_statistics();
        }

        std::vector<int> internal_solve(HostGraph& host_g) {
            DevicePartition final_device_partition = DevicePartition("final_device_partition", host_g.n);
            DeviceScratchMemory scratch_mem(host_g.n, max(m_configuration.hierarchy));

            // references for better code readability
            const std::vector<int>& hierarchy = m_configuration.hierarchy;
            const int l                       = (int)hierarchy.size();
            const std::vector<int>& index_vec = m_configuration.index_vec;
            const std::vector<int>& k_rem_vec = m_configuration.k_rem_vec;
            const f64 global_imbalance        = m_configuration.imbalance;
            const int global_g_weight         = host_g.graph_weight;
            const int global_k                = m_configuration.k;

            std::vector<Item> stack;
            HostEntries host_o_to_n("o_to_n", host_g.n);
            HostEntries host_n_to_o("n_to_o", host_g.n);
            for (int u = 0; u < host_g.n; ++u) {
                host_o_to_n(u) = u;
                host_n_to_o(u) = u;
            }
            DeviceEntries device_o_to_n("o_to_n", host_g.n);
            DeviceEntries device_n_to_o("n_to_o", host_g.n);
            Kokkos::deep_copy(device_o_to_n, host_o_to_n);
            Kokkos::deep_copy(device_n_to_o, host_n_to_o);
            Kokkos::fence();

            stack.push_back({DeviceGraph(host_g), {}, device_o_to_n, device_n_to_o});

            while (!stack.empty()) {
                Item item = std::move(stack.back());
                stack.pop_back();

                // get depth info
                const int depth           = l - 1 - (int)item.identifier.size();
                const int local_k         = hierarchy[depth];
                const int local_k_rem     = k_rem_vec[depth];
                const f64 local_imbalance = determine_adaptive_imbalance(global_imbalance, global_g_weight, global_k, item.device_g.graph_weight, local_k_rem, depth + 1);

                auto sp_partition                = std::chrono::high_resolution_clock::now();
                DevicePartition device_partition = jet_partition(item.device_g, local_k, local_imbalance, m_configuration.seed, m_configuration.use_ultra);
                auto ep_partition                = std::chrono::high_resolution_clock::now();
                time_partition += get_seconds(sp_partition, ep_partition);

                if (depth == 0) {
                    // insert solution
                    auto sp_insert = std::chrono::high_resolution_clock::now();
                    int offset     = 0;
                    for (size_t i = 0; i < item.identifier.size(); ++i) {
                        offset += item.identifier[i] * index_vec[index_vec.size() - 1 - i];
                    }
                    Kokkos::parallel_for("AssignFinalPartition", Kokkos::RangePolicy<>(0, item.device_g.n), KOKKOS_LAMBDA(int u) {
                                             int original_u                     = item.n_to_o(u);
                                             final_device_partition(original_u) = offset + device_partition(u);
                                         }
                                        );
                    auto ep_insert = std::chrono::high_resolution_clock::now();
                    time_final_partition += get_seconds(sp_insert, ep_insert);
                } else {
                    // create the subgraphs and place them in the next stack
                    auto sp_subgraphs = std::chrono::high_resolution_clock::now();
                    create_subgraphs(item.device_g, item.n_to_o, local_k, device_partition, item.identifier, host_g.n, stack, scratch_mem);
                    auto ep_subgraphs = std::chrono::high_resolution_clock::now();
                    time_subgraphs += get_seconds(sp_subgraphs, ep_subgraphs);
                }
            }

            HostPartition final_host_partition = Kokkos::create_mirror(final_device_partition);
            Kokkos::deep_copy(final_host_partition, final_device_partition);

            std::vector<int> partition(host_g.n);
            for (int u = 0; u < host_g.n; ++u) {
                partition[u] = final_host_partition(u);
            }

            return partition;
        }

        void write_solution(std::vector<int>& partition) {
            std::stringstream ss;

            for (int i : partition) {
                ss << i << "\n";
            }
            std::ofstream out(m_configuration.mapping_out);
            out << ss.rdbuf();
            out.close();
        }

        void print_statistics() {
            std::string s = "{\n";

            s += to_JSON_MACRO(io_time);
            s += to_JSON_MACRO(solve_time);
            s += to_JSON_MACRO(time_partition);
            s += to_JSON_MACRO(time_subgraphs);
            s += to_JSON_MACRO(time_final_partition);
            s += "\"algorithm-configuration\" : " + m_configuration.to_JSON(2) + ",\n";

            s.pop_back();
            s.pop_back();
            s += "\n}";
            std::cout << s << std::endl;
        }
    };
}

#endif //SHAREDMAP_GPU_SOLVER_H
