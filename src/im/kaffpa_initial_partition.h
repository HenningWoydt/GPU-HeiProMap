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

#ifndef GPU_HEIPROMAP_KAFFPA_INITIAL_PARTITION_H
#define GPU_HEIPROMAP_KAFFPA_INITIAL_PARTITION_H

#include "interface/kaHIP_interface.h"

#include "definitions.h"
#include "device_graph.h"
#include "partition_manager.h"

namespace GPU_HeiProMap {
    inline f64 determine_adaptive_imbalance(const f64 global_imbalance,
                                            const weight_t global_g_weight,
                                            const partition_t global_k,
                                            const weight_t local_g_weight,
                                            const partition_t local_k_rem,
                                            const u64 depth) {
        f64 local_imbalance = (1.0 + global_imbalance) * ((f64) (local_k_rem * global_g_weight) / (f64) (global_k * local_g_weight));
        local_imbalance = std::pow(local_imbalance, (f64) 1 / (f64) depth) - 1.0;
        return local_imbalance;
    }

    struct Item {
        // variables needed for KaFFPa
        vertex_t n = 0;
        vertex_t m = 0;
        int *vwgt = nullptr;
        int *xadj = nullptr;
        int *adjcwgt = nullptr;
        int *adjncy = nullptr;
        partition_t nparts = 0;
        f64 imbalance = 0;
        bool suppress_output = true;
        u32 seed = 0;
        int mode = FAST;
        int edge_cut_temp = 0;
        int *part_temp = nullptr;
        int edge_cut = 0;
        int *part = nullptr;

        // variables needed for multisection
        weight_t total_weight = 0;
        std::vector<vertex_t> o_to_n;
        std::vector<vertex_t> n_to_o;
        std::vector<partition_t> identifier;

        Item(vertex_t max_n, vertex_t t_n, vertex_t t_m, weight_t t_w) {
            n = t_n;
            m = t_m;
            total_weight = t_w;
            vwgt = (int *) malloc(n * sizeof(int));
            xadj = (int *) malloc((n + 1) * sizeof(int));
            xadj[0] = 0;
            adjcwgt = (int *) malloc(m * sizeof(int));
            adjncy = (int *) malloc(m * sizeof(int));
            part_temp = (int *) malloc(n * sizeof(int));
            part = (int *) malloc(n * sizeof(int));

            o_to_n.resize(max_n);
            n_to_o.resize(max_n);
        }

        ~Item() {
            free(vwgt);
            free(xadj);
            free(adjcwgt);
            free(adjncy);
            free(part_temp);
            free(part);
        }
    };

    inline void kaffpa_initial_partition(Graph &device_g,
                                         const std::vector<partition_t> &hierarchy,
                                         const std::vector<weight_t> &distance,
                                         partition_t k,
                                         f64 imbalance,
                                         u32 seed,
                                         PartitionManager &p_manager) {
        // Convert device graph to simple CSR arrays on host
        HostGraph host_g = to_host_graph(device_g);
        HostPartition host_partition("host_partition", device_g.n);

        partition_t l = (partition_t) hierarchy.size();

        std::vector<partition_t> index_vec = {1};
        for (partition_t i = 0; i < l - 1; ++i) { index_vec.push_back(index_vec[i] * hierarchy[i]); }

        std::vector<partition_t> k_rem_vec(l);
        u32 p = 1;

        for (partition_t i = 0; i < l; ++i) {
            k_rem_vec[i] = p * hierarchy[i];
            p *= hierarchy[i];
        }

        const f64 global_imbalance = imbalance;
        const weight_t global_g_weight = host_g.g_weight;
        const partition_t global_k = k;

        // initialize stack;
        std::vector<Item *> stack = {new Item(host_g.n, host_g.n, host_g.m, global_g_weight)}; {
            TIME("partitioning", "kaffpa_initial_partition", "create_first_graph",
            // create the first graph
            Item *first_graph = stack[0];

            // initialize the translation table of the first graph
            vertex_t new_u = 0;
            for (vertex_t old_u = 0; old_u < host_g.n; ++old_u) {
                first_graph->o_to_n[old_u] = new_u;
                first_graph->n_to_o[new_u] = old_u;
                new_u += 1;
            }

            // write the first graph into pointers
            first_graph->xadj[0] = 0;
            first_graph->total_weight = global_g_weight;
            for (vertex_t old_u = 0; old_u < host_g.n; ++old_u) {
                first_graph->vwgt[old_u] = (int) host_g.weights(old_u);
                first_graph->xadj[old_u + 1] = first_graph->xadj[old_u];

                for (u32 i = host_g.neighborhood(old_u); i < host_g.neighborhood(old_u + 1); ++i) {
                    vertex_t v = host_g.edges_v(i);
                    weight_t w = host_g.edges_w(i);

                    first_graph->adjcwgt[first_graph->xadj[old_u + 1]] = (int) w;
                    first_graph->adjncy[first_graph->xadj[old_u + 1]] = (int) v;
                    first_graph->xadj[old_u + 1] += 1;
                }
            }

            // fill in other information
            first_graph->nparts = hierarchy.back();
            first_graph->imbalance = determine_adaptive_imbalance(global_imbalance, global_g_weight, global_k, first_graph->total_weight, k_rem_vec[l - 1], l);
            first_graph->suppress_output = true;
            first_graph->seed = seed;
            );
        }

        // process the stack
        while (!stack.empty()) {
            Item *item = stack.back(); // process first item
            stack.pop_back(); // remove top item

            item->edge_cut = std::numeric_limits<int>::max();
            if (item->n > 0) {
                if (item->nparts == 1) {
                    for (vertex_t u = 0; u < item->n; ++u) {
                        item->part[u] = 0;
                    }
                } else {
                    for (int i = 0; i < 1; ++i) {
                        int n = (int) item->n;
                        int nparts = (int) item->nparts;
                        int local_seed = (int) item->seed + i;
                        TIME("partitioning", "kaffpa_initial_partition", "kaffpa",
                             kaffpa(&n, item->vwgt, item->xadj, item->adjcwgt, item->adjncy, &nparts, &item->imbalance, item->suppress_output, local_seed, item->mode, &item->edge_cut_temp, item->part_temp);
                             // kaffpa_balance(&n, item->vwgt, item->xadj, item->adjcwgt, item->adjncy, &nparts, &item->imbalance, false, item->suppress_output, local_seed, item->mode, &item->edge_cut_temp, item->part_temp);
                        );
                        if (item->edge_cut_temp < item->edge_cut) {
                            item->edge_cut = item->edge_cut_temp;
                            std::swap(item->part_temp, item->part);
                        }
                    }
                }
            }

            if (item->identifier.size() == l - 1) {
                TIME("partitioning", "kaffpa_initial_partition", "insert",
                // insert solution
                partition_t offset = 0;
                for (partition_t i = 0; i < l - 1; ++i) { offset += item->identifier[i] * index_vec[index_vec.size() - 1 - i]; }
                for (vertex_t u = 0; u < item->n; ++u) { host_partition(item->n_to_o[u]) = offset + (partition_t) item->part[u]; }
                );
            } else {
                TIME("partitioning", "kaffpa_initial_partition", "create_subgraphs",
                // create the subgraphs and place them in the next stack

                // collect the number of vertices and edges for each new subgraph
                std::vector<vertex_t> new_n(item->nparts, 0);
                std::vector<vertex_t> new_m(item->nparts, 0);
                std::vector<weight_t> new_w(item->nparts, 0);
                for (vertex_t u = 0; u < item->n; ++u) {
                    partition_t u_id = (partition_t) item->part[u];
                    new_n[u_id] += 1; // increase number of vertices
                    new_w[u_id] += (weight_t) item->vwgt[u];

                    for (int i = item->xadj[u]; i < item->xadj[u + 1]; ++i) {
                        vertex_t v = (vertex_t) item->adjncy[i];
                        partition_t v_id = (partition_t) item->part[v];
                        if (u_id == v_id) { new_m[u_id] += 1; }
                    }
                }

                // create the new subgraphs on the stack
                for (partition_t i = 0; i < item->nparts; ++i) {
                    Item *new_item = new Item(host_g.n, new_n[i], new_m[i], new_w[i]);
                    new_item->identifier = item->identifier;
                    new_item->identifier.push_back(i);
                    stack.push_back(new_item);
                }

                // fill the translation tables
                std::vector<vertex_t> new_us(item->nparts, 0);
                for (vertex_t old_u = 0; old_u < item->n; ++old_u) {
                    partition_t u_id = (partition_t) item->part[old_u];
                    size_t idx = stack.size() - (item->nparts - u_id);

                    vertex_t original_u = item->n_to_o[old_u];
                    vertex_t new_u = new_us[u_id];

                    stack[idx]->o_to_n[original_u] = new_u;
                    stack[idx]->n_to_o[new_u] = original_u;
                    new_us[u_id] += 1;
                }

                // create the graphs
                for (vertex_t old_u = 0; old_u < item->n; ++old_u) {
                    partition_t u_id = (partition_t) item->part[old_u];
                    size_t idx = stack.size() - (item->nparts - u_id);
                    Item &new_item = *stack[idx];

                    vertex_t original_u = item->n_to_o[old_u];
                    vertex_t new_u = stack[idx]->o_to_n[original_u]; // vertex in new graph

                    // set the weight
                    new_item.vwgt[new_u] = item->vwgt[old_u];
                    new_item.xadj[new_u + 1] = new_item.xadj[new_u];

                    // set the edges
                    for (int i = item->xadj[old_u]; i < item->xadj[old_u + 1]; ++i) {
                        vertex_t old_v = (vertex_t) item->adjncy[i];
                        weight_t w = (weight_t) item->adjcwgt[i];

                        if (u_id == (partition_t) item->part[old_v]) {
                            // add the edge
                            vertex_t original_v = item->n_to_o[old_v];
                            vertex_t new_v = stack[idx]->o_to_n[original_v]; // vertex in new graph
                            int edge_idx = new_item.xadj[new_u + 1];

                            new_item.adjncy[edge_idx] = (int) new_v;
                            new_item.adjcwgt[edge_idx] = (int) w;
                            new_item.xadj[new_u + 1] += 1;
                        }
                    }
                }

                // fill in other information
                for (partition_t i = 0; i < item->nparts; ++i) {
                    size_t idx = stack.size() - 1 - i;
                    Item &new_item = *stack[idx];

                    new_item.nparts = hierarchy[l - 1 - new_item.identifier.size()];
                    new_item.imbalance = determine_adaptive_imbalance(global_imbalance, global_g_weight, global_k, new_item.total_weight, k_rem_vec[l - 1 - new_item.identifier.size()], l - new_item.identifier.size());
                    new_item.suppress_output = true;
                    new_item.seed = seed;
                }
                );
            }
            delete item;
        }
        auto device_subview = Kokkos::subview(p_manager.partition, std::pair<size_t, size_t>(0, host_partition.extent(0)));
        Kokkos::deep_copy(device_subview, host_partition);
        Kokkos::fence();
    }
}

#endif //GPU_HEIPROMAP_KAFFPA_INITIAL_PARTITION_H
