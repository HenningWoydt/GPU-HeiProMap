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

#ifndef SHAREDMAP_GPU_PARTITION_H
#define SHAREDMAP_GPU_PARTITION_H

#include "../datastructures/device_graph.h"
#include "../utility/definitions.h"

namespace SharedMap_GPU {
    struct Item {
        DeviceGraph device_g;
        std::vector<int> identifier;
        DeviceEntries o_to_n;
        DeviceEntries n_to_o;
    };

    class DeviceScratchMemory {
    public:
        Kokkos::View<int *> sub_ns;
        Kokkos::View<int *> sub_ms;
        Kokkos::View<int *> sub_weights;
        Kokkos::View<int *> edge_counts;
        Kokkos::View<int *> edge_offsets; // +1 for final offset
        Kokkos::View<int *> vertices_per_id;

        DeviceScratchMemory(int global_n, int max_k) {
            sub_ns = Kokkos::View<int *>("sub_n", max_k);
            sub_ms = Kokkos::View<int *>("sub_n", max_k);
            sub_weights = Kokkos::View<int *>("sub_n", max_k);
            edge_counts = Kokkos::View<int *>("edge_counts", global_n);
            edge_offsets = Kokkos::View<int *>("edge_offsets", global_n + 1); // +1 for final offset
            vertices_per_id = Kokkos::View<int *>("vertices_per_id", global_n * max_k);
        }
    };

    inline DevicePartition jet_partition_serial(DeviceGraph &device_g,
                                                int k,
                                                f64 imbalance,
                                                int seed,
                                                bool use_ultra) {
        jet_partitioner::config_t config;
        config.max_imb_ratio = 1.0 + imbalance;
        config.num_parts = k;
        config.ultra_settings = use_ultra;
        config.verbose = false;
        config.num_iter = 10;
        config.coarsening_alg = 0;

        jet_partitioner::value_t edge_cut;
        jet_partitioner::experiment_data<jet_partitioner::value_t> data;

        jet_partitioner::part_mt partition = jet_partitioner::partition_serial(edge_cut,
                                                                               config,
                                                                               get_serial_mtx(device_g),
                                                                               get_serial_vertex_weights(device_g),
                                                                               false,
                                                                               data);

        // Allocate and copy to device
        DevicePartition device_partition("device_partition", partition.extent(0));
        Kokkos::deep_copy(device_partition, partition);

        return device_partition;
    }

    inline DevicePartition jet_partition(DeviceGraph &device_g,
                                         int k,
                                         f64 imbalance,
                                         int seed,
                                         bool use_ultra) {
        if (k == 1) {
            DevicePartition partition("k=1 partition", device_g.n);
            Kokkos::deep_copy(partition, 0);
            return partition;
        }

        jet_partitioner::config_t config;
        config.max_imb_ratio = 1.0 + imbalance;
        config.num_parts = k;
        config.ultra_settings = use_ultra;
        config.verbose = false;
        config.num_iter = 10;
        config.coarsening_alg = 0;

        jet_partitioner::value_t edge_cut;
        jet_partitioner::experiment_data<jet_partitioner::value_t> data;

        jet_partitioner::matrix_t mtx = device_g.get_mtx();
        DevicePartition partition = jet_partitioner::partition(edge_cut,
                                                               config,
                                                               mtx,
                                                               device_g.vertex_weights,
                                                               false,
                                                               data);

        return partition;
    }

    inline f64 determine_adaptive_imbalance(const f64 global_imbalance,
                                            const int global_g_weight,
                                            const int global_k,
                                            const int local_g_weight,
                                            const int local_k_rem,
                                            const int depth) {
        f64 local_imbalance = (1.0 + global_imbalance) * ((f64) (local_k_rem * global_g_weight) / (f64) (global_k * local_g_weight));
        local_imbalance = std::pow(local_imbalance, (f64) 1 / (f64) depth) - 1.0;

        return local_imbalance;
    }

    inline void create_one_subgraph_host(DeviceGraph &device_g,
                                         DeviceEntries &n_to_o, // maps local to global
                                         int id, // target partition
                                         DevicePartition &device_partition,
                                         std::vector<int> &identifier,
                                         int global_n,
                                         std::vector<Item> &stack) {
        // Copy the device to the host
        HostGraph host_g = convert(device_g);

        HostPartition host_partition = Kokkos::create_mirror_view(device_partition);
        Kokkos::deep_copy(host_partition, device_partition);

        HostEntries host_n_to_o = Kokkos::create_mirror_view(n_to_o);
        Kokkos::deep_copy(host_n_to_o, n_to_o);
        Kokkos::fence();

        int n = host_g.n;

        // first pass, count the number of new vertices, edges and the new graph weight
        int sub_n = 0, sub_m = 0, sub_weight = 0;
        for (int u = 0; u < n; ++u) {
            if (host_partition(u) != id) { continue; }

            sub_n += 1;
            sub_weight += host_g.vertex_weights(u);

            for (int i = host_g.neighborhood(u); i < host_g.neighborhood(u + 1); ++i) {
                int v = host_g.edges_v(i);
                if (device_partition(v) == id) { sub_m++; }
            }
        }

        // second pass, build translation table
        HostEntries o_to_n_sub = HostEntries("o_to_n", global_n);
        HostEntries n_to_o_sub = HostEntries("n_to_o", sub_n);

        int counter = 0;
        for (int u = 0; u < n; ++u) {
            if (host_partition(u) != id) { continue; }
            int old_u = host_n_to_o(u);
            int new_u = counter;

            o_to_n_sub(old_u) = new_u;
            n_to_o_sub(new_u) = old_u;

            counter += 1;
        }

        // third pass, build the graph
        HostGraph host_sub_g(sub_n, sub_m, sub_weight);
        host_sub_g.neighborhood(0) = 0;

        int idx = 0;
        for (int u = 0; u < n; ++u) {
            if (host_partition(u) != id) { continue; }

            int sub_u = o_to_n_sub(n_to_o(u));

            // get the weight
            host_sub_g.vertex_weights(sub_u) = host_g.vertex_weights(u);

            // fill in edges
            for (int i = host_g.neighborhood(u); i < host_g.neighborhood(u + 1); ++i) {
                int v = host_g.edges_v(i);
                int w = host_g.edges_w(i);
                if (device_partition(v) == id) {
                    int sub_v = o_to_n_sub(n_to_o(v));

                    host_sub_g.edges_v(idx) = sub_v;
                    host_sub_g.edges_w(idx) = w;
                    idx += 1;
                }
            }
            host_sub_g.neighborhood(sub_u + 1) = idx;
        }

        // upload from host to device
        DeviceGraph device_sub_g(host_sub_g);

        DeviceEntries device_n_to_o_sub("device_n_to_o", n_to_o_sub.extent(0));
        Kokkos::deep_copy(device_n_to_o_sub, n_to_o_sub);

        DeviceEntries device_o_to_n_sub("device_o_to_n", o_to_n_sub.extent(0));
        Kokkos::deep_copy(device_o_to_n_sub, o_to_n_sub);
        Kokkos::fence();

        // add to stack
        stack.push_back({
                                device_sub_g,
                                identifier,
                                device_o_to_n_sub,
                                device_n_to_o_sub
                        });
        stack.back().identifier.push_back(id);
    }

    inline void create_one_subgraph_device(DeviceGraph &device_g,
                                           DeviceEntries &n_to_o, // maps local to global
                                           int id, // target partition
                                           DevicePartition &device_partition,
                                           std::vector<int> &identifier,
                                           int global_n,
                                           std::vector<Item> &stack,
                                           DeviceScratchMemory &device_scratch_mem) {
        int n = device_g.n;

        // Assuming device_partition, vertex_weights, neighborhood, and edges_v are Kokkos::Views
        int sub_n = 0, sub_m = 0, sub_weight = 0;
        Kokkos::parallel_reduce("CountSubgraph", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int u, int &local_n, int &local_m, int &local_weight) {
                                    if (device_partition(u) != id) {
                                        device_scratch_mem.edge_counts(u) = 0;
                                        return;
                                    }

                                    local_n += 1;
                                    local_weight += device_g.vertex_weights(u);

                                    int count = 0;
                                    for (int i = device_g.neighborhood(u); i < device_g.neighborhood(u + 1); ++i) {
                                        int v = device_g.edges_v(i);
                                        if (device_partition(v) == id) {
                                            local_m += 1;
                                            count += 1;
                                        }
                                    }
                                    device_scratch_mem.edge_counts(u) = count;
                                },
                                sub_n, sub_m, sub_weight
        );

        // second pass, build translation table
        DeviceEntries o_to_n_sub = DeviceEntries("o_to_n", global_n);
        DeviceEntries n_to_o_sub = DeviceEntries("n_to_o", sub_n);

        Kokkos::View<int *> new_indices("new_indices", n);
        Kokkos::parallel_scan("PrefixSum", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int u, int &update, const bool final) {
            if (final) {
                new_indices(u) = update;
            }
            update += device_partition(u) == id;
        });

        Kokkos::parallel_for("BuildTranslationTables", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int u) {
            if (device_partition(u) != id) return;

            int old_u = n_to_o(u);
            int new_u = new_indices(u);

            o_to_n_sub(old_u) = new_u;
            n_to_o_sub(new_u) = old_u;
        });

        // third pass, build the graph
        Kokkos::parallel_scan("EdgeOffsetScan", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i, int &update, const bool final) {
            if (final) {
                device_scratch_mem.edge_offsets(i) = update;
            }
            update += device_scratch_mem.edge_counts(i);
        });

        // Set final offset manually
        Kokkos::parallel_for("FinalEdgeOffset", Kokkos::RangePolicy<>(n, n + 1), KOKKOS_LAMBDA(int i) {
            device_scratch_mem.edge_offsets(i) = device_scratch_mem.edge_offsets(i - 1) + device_scratch_mem.edge_counts(i - 1);
        });


        DeviceGraph device_sub_g(sub_n, sub_m, sub_weight);

        Kokkos::parallel_for("BuildSubgraph", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int u) {
            if (device_partition(u) != id) return;

            int sub_u = o_to_n_sub(n_to_o(u));
            device_sub_g.vertex_weights(sub_u) = device_g.vertex_weights(u);

            int write_idx = device_scratch_mem.edge_offsets(u);
            for (int i = device_g.neighborhood(u); i < device_g.neighborhood(u + 1); ++i) {
                int v = device_g.edges_v(i);
                int w = device_g.edges_w(i);
                if (device_partition(v) == id) {
                    int sub_v = o_to_n_sub(n_to_o(v));
                    device_sub_g.edges_v(write_idx) = sub_v;
                    device_sub_g.edges_w(write_idx) = w;
                    write_idx++;
                }
            }

            // Store neighborhood offset
            device_sub_g.neighborhood(sub_u + 1) = device_scratch_mem.edge_offsets(u + 1);
        });

        // Set neighborhood(0) = 0
        Kokkos::parallel_for("SetNeighborhoodZero", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {
            device_sub_g.neighborhood(0) = 0;
        });


        // add to stack
        stack.push_back({device_sub_g, identifier, o_to_n_sub, n_to_o_sub});
        stack.back().identifier.push_back(id);
    }

    inline void create_subgraphs_device(DeviceGraph &device_g,
                                        DeviceEntries &n_to_o, // maps local to global
                                        int k,
                                        DevicePartition &device_partition,
                                        std::vector<int> &identifier,
                                        int global_n,
                                        std::vector<Item> &stack,
                                        DeviceScratchMemory &device_scratch_mem) {
        int n = device_g.n;

        Kokkos::deep_copy(device_scratch_mem.sub_ns, 0);
        Kokkos::deep_copy(device_scratch_mem.sub_ms, 0);
        Kokkos::deep_copy(device_scratch_mem.sub_weights, 0);
        Kokkos::fence();

        Kokkos::parallel_for("CountSubgraphPerPartition", n, KOKKOS_LAMBDA(int u) {
            int id = device_partition(u);

            int offset = Kokkos::atomic_fetch_inc(&device_scratch_mem.sub_ns(id)); // count number of vertices
            device_scratch_mem.vertices_per_id(id * global_n + offset) = u; // save which vertex is in each block

            Kokkos::atomic_add(&device_scratch_mem.sub_weights(id), device_g.vertex_weights(u)); // sum weight

            int count = 0; // count number of edges
            for (int i = device_g.neighborhood(u); i < device_g.neighborhood(u + 1); ++i) {
                int v = device_g.edges_v(i);
                if (device_partition(v) == id) {
                    count += 1;
                }
            }

            device_scratch_mem.edge_counts(u) = count;
            Kokkos::atomic_add(&device_scratch_mem.sub_ms(id), count); // add edges
        });
        Kokkos::fence();

        for (int id = 0; id < k; ++id) {
            int sub_n;
            int sub_m;
            int sub_weight;
            Kokkos::deep_copy(sub_n, Kokkos::subview(device_scratch_mem.sub_ns, id));
            Kokkos::deep_copy(sub_m, Kokkos::subview(device_scratch_mem.sub_ms, id));
            Kokkos::deep_copy(sub_weight, Kokkos::subview(device_scratch_mem.sub_weights, id));
            Kokkos::fence();

            // second pass, build translation table
            DeviceEntries o_to_n_sub = DeviceEntries("o_to_n", global_n);
            DeviceEntries n_to_o_sub = DeviceEntries("n_to_o", sub_n);

            Kokkos::parallel_for("BuildTranslationTables", Kokkos::RangePolicy<>(0, sub_n), KOKKOS_LAMBDA(int idx) {
                int u = device_scratch_mem.vertices_per_id(id * global_n + idx);

                int old_u = n_to_o(u);
                int new_u = idx;

                o_to_n_sub(old_u) = new_u;
                n_to_o_sub(new_u) = old_u;
            });
            Kokkos::fence();

            Kokkos::parallel_scan("EdgeOffsetScan", Kokkos::RangePolicy<>(0, sub_n), KOKKOS_LAMBDA(const int i, int &update, const bool final) {
                int u = device_scratch_mem.vertices_per_id(id * global_n + i);
                if (final) {
                    device_scratch_mem.edge_offsets(u) = update;  // STORE START
                }
                update += device_scratch_mem.edge_counts(u);
            });
            Kokkos::fence();


            DeviceGraph device_sub_g(sub_n, sub_m, sub_weight);

            Kokkos::parallel_for("BuildSubgraph", Kokkos::RangePolicy<>(0, sub_n), KOKKOS_LAMBDA(int idx) {
                int u = device_scratch_mem.vertices_per_id(id * global_n + idx);

                int sub_u = o_to_n_sub(n_to_o(u));
                device_sub_g.vertex_weights(sub_u) = device_g.vertex_weights(u);

                int write_idx = device_scratch_mem.edge_offsets(u);
                for (int i = device_g.neighborhood(u); i < device_g.neighborhood(u + 1); ++i) {
                    int v = device_g.edges_v(i);
                    int w = device_g.edges_w(i);
                    if (device_partition(v) == id) {
                        int sub_v = o_to_n_sub(n_to_o(v));
                        device_sub_g.edges_v(write_idx) = sub_v;
                        device_sub_g.edges_w(write_idx) = w;
                        write_idx++;
                    }
                }

                // Store neighborhood offset
                device_sub_g.neighborhood(sub_u + 1) = write_idx;
            });
            Kokkos::fence();

            // Set neighborhood(0) = 0
            Kokkos::parallel_for("SetNeighborhoodZero", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {
                device_sub_g.neighborhood(0) = 0;
            });
            Kokkos::fence();

            // add to stack
            stack.push_back({device_sub_g, identifier, o_to_n_sub, n_to_o_sub});
            stack.back().identifier.push_back(id);
        }
    }

    inline void create_subgraphs(DeviceGraph &device_g,
                                 DeviceEntries &n_to_o,
                                 int k,
                                 jet_partitioner::part_vt &device_partition,
                                 std::vector<int> &identifier,
                                 int global_n,
                                 std::vector<Item> &stack,
                                 DeviceScratchMemory &device_scratch_mem) {
        create_subgraphs_device(device_g, n_to_o, k, device_partition, identifier, global_n, stack, device_scratch_mem);
    }
}

#endif //SHAREDMAP_GPU_PARTITION_H
