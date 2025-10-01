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

        HM_DeviceGraph() {
            n = 0;
            m = 0;
            graph_weight = 0;
        }

        explicit HM_DeviceGraph(const HM_HostGraph &host_g) {
            TIME("io", "initialize_device_g", "copy",
                 n = host_g.n;
                 m = host_g.m;
                 graph_weight = host_g.graph_weight;

                 vertex_weights = JetDeviceWeights(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vertex_weights"), host_g.vertex_weights.extent(0));
                 neighborhood = JetDeviceRowMap(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighborhood"), host_g.neighborhood.extent(0));
                 edges_v = JetDeviceEntries(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_v"), host_g.edges_v.extent(0));
                 edges_w = JetDeviceValues(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_w"), host_g.edges_w.extent(0));

                 Kokkos::deep_copy(vertex_weights, host_g.vertex_weights);
                 Kokkos::deep_copy(neighborhood, host_g.neighborhood);
                 Kokkos::deep_copy(edges_v, host_g.edges_v);
                 Kokkos::deep_copy(edges_w, host_g.edges_w);
                 Kokkos::fence();
            );
        }

        HM_DeviceGraph(const int t_n, const int t_m, const int t_weight) {
            n = t_n;
            m = t_m;
            graph_weight = t_weight;

            vertex_weights = JetDeviceWeights(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vertex_weights"), n);
            neighborhood = JetDeviceRowMap(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighborhood"), n + 1);
            edges_v = JetDeviceEntries(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_v"), m);
            edges_w = JetDeviceValues(Kokkos::view_alloc(Kokkos::WithoutInitializing, "edges_w"), m);
        }

        jet_partitioner::matrix_t get_mtx() const {
            jet_partitioner::matrix_t mtx = jet_partitioner::matrix_t("mtx",
                                                                      n,
                                                                      n,
                                                                      m,
                                                                      edges_w,
                                                                      neighborhood,
                                                                      edges_v
            );
            Kokkos::fence(); // optional: ensure mtx is fully ready
            return mtx;
        }

        size_t size(const int u) const { return neighborhood(u + 1) - neighborhood(u); }

        int neighbor(const int u, const size_t i) const { return edges_v(neighborhood(u) + i); }
    };

    inline HM_HostGraph convert(HM_DeviceGraph &device_g) {
        HM_HostGraph host_g;
        host_g.n = device_g.n;
        host_g.m = device_g.m;
        host_g.graph_weight = device_g.graph_weight;

        // Allocate host views with matching sizes
        host_g.vertex_weights = JetHostWeights(Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_vertex_weights"), host_g.n);
        host_g.neighborhood = JetHostRowMap(Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_neighborhood"), host_g.n + 1);
        host_g.edges_v = JetHostEntries(Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_edges_v"), host_g.m);
        host_g.edges_w = JetHostValues(Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_edges_w"), host_g.m);

        // Deep copy from device to host
        Kokkos::deep_copy(host_g.vertex_weights, device_g.vertex_weights);
        Kokkos::deep_copy(host_g.neighborhood, device_g.neighborhood);
        Kokkos::deep_copy(host_g.edges_v, device_g.edges_v);
        Kokkos::deep_copy(host_g.edges_w, device_g.edges_w);
        Kokkos::fence();

        return host_g;
    }

    inline jet_partitioner::serial_matrix_t get_serial_mtx(const HM_DeviceGraph &device_g) {
        // Mirror the CRS components to host
        auto h_row_map = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device_g.neighborhood);
        auto h_entries = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device_g.edges_v);
        auto h_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device_g.edges_w);

        // Now create Serial-space mirrors
        JetDeviceRowMap::HostMirror serial_row_map(Kokkos::view_alloc(Kokkos::WithoutInitializing, "serial_row_map"), h_row_map.extent(0));
        JetDeviceEntries::HostMirror serial_entries(Kokkos::view_alloc(Kokkos::WithoutInitializing, "serial_entries"), h_entries.extent(0));
        JetDeviceValues::HostMirror serial_values(Kokkos::view_alloc(Kokkos::WithoutInitializing, "serial_values"), h_values.extent(0));

        Kokkos::deep_copy(serial_row_map, h_row_map);
        Kokkos::deep_copy(serial_entries, h_entries);
        Kokkos::deep_copy(serial_values, h_values);

        // Construct the Serial CRS matrix
        jet_partitioner::serial_matrix_t serial_mtx("serial_mtx",
                                                    device_g.n,
                                                    device_g.n,
                                                    device_g.m,
                                                    serial_values,
                                                    serial_row_map,
                                                    serial_entries);

        Kokkos::fence(); // Ensure everything is ready before returning
        return serial_mtx;
    }

    inline jet_partitioner::wgt_serial_vt get_serial_vertex_weights(const HM_DeviceGraph &device_g) {
        // Mirror the vertex weights from device to host
        auto h_vertex_weights = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device_g.vertex_weights);

        // Create a serial mirror of the host view
        jet_partitioner::wgt_serial_vt serial_vertex_weights(Kokkos::view_alloc(Kokkos::WithoutInitializing, "serial_vertex_weights"), h_vertex_weights.extent(0));

        // Copy host weights into the serial view
        Kokkos::deep_copy(serial_vertex_weights, h_vertex_weights);

        return serial_vertex_weights;
    }
}

#endif //GPU_HEIPROMAP_HM_DEVICE_GRAPH_H
