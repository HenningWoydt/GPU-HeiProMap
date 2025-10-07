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

#ifndef GPU_HEIPROMAP_HM_DEVICE_GRAPH_H
#define GPU_HEIPROMAP_HM_DEVICE_GRAPH_H

#include <jet.h>

#include "hm_host_graph.h"

namespace GPU_HeiProMap {
    class HM_DeviceGraph {
    public:
        int n;
        int m;
        int graph_weight;

        JetDeviceWeights vertex_weights;
        JetDeviceRowMap neighborhood;
        JetDeviceEntries edges_v;
        JetDeviceValues edges_w;

        explicit HM_DeviceGraph(const HM_HostGraph &host_g) noexcept {
            ScopedTimer _t_allocate("io", "HM_DeviceGraph", "allocate");
            n = host_g.n;
            m = host_g.m;
            graph_weight = host_g.graph_weight;

            vertex_weights = JetDeviceWeights(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vertex_weights"), (size_t) n);
            neighborhood = JetDeviceRowMap(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighborhood"), (size_t) n + 1);
            edges_v = JetDeviceEntries(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_v"), (size_t) m);
            edges_w = JetDeviceValues(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_w"), (size_t) m);

            _t_allocate.stop();
            ScopedTimer _t_copy("io", "HM_DeviceGraph", "copy");

            auto exec = Kokkos::DefaultExecutionSpace{};
            Kokkos::deep_copy(exec, vertex_weights, host_g.vertex_weights);
            Kokkos::deep_copy(exec, neighborhood, host_g.neighborhood);
            Kokkos::deep_copy(exec, edges_v, host_g.edges_v);
            Kokkos::deep_copy(exec, edges_w, host_g.edges_w);
            exec.fence();
        }

        HM_DeviceGraph(const int t_n, const int t_m, const int t_weight) noexcept {
            ScopedTimer _t_allocate("io", "HM_DeviceGraph", "allocate");
            n = t_n;
            m = t_m;
            graph_weight = t_weight;

            vertex_weights = JetDeviceWeights(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vertex_weights"), (size_t) n);
            neighborhood = JetDeviceRowMap(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighborhood"), (size_t) n + 1);
            edges_v = JetDeviceEntries(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_v"), (size_t) m);
            edges_w = JetDeviceValues(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_w"), (size_t) m);
        }

        jet_partitioner::matrix_t get_mtx() const noexcept {
            jet_partitioner::matrix_t mtx = jet_partitioner::matrix_t("mtx",
                                                                      n,
                                                                      n,
                                                                      m,
                                                                      edges_w,
                                                                      neighborhood,
                                                                      edges_v
            );
            Kokkos::fence();
            return mtx;
        }
    };
}

#endif //GPU_HEIPROMAP_HM_DEVICE_GRAPH_H
