#ifndef GPU_HEIPROMAP_VERTEX_PARTITION_CSR_H
#define GPU_HEIPROMAP_VERTEX_PARTITION_CSR_H

#include "comm_cost.h"
#include "definitions.h"
#include "device_graph.h"
#include "distance_oracle.h"
#include "partition_manager.h"
#include "profiler.h"

namespace GPU_HeiProMap {
    struct VertexPartitionCSR {
        DeviceU32 row; // length n+1
        DevicePartition ids; // length nnz (partition IDs)
        DeviceWeight weights; // length nnz (sum of weights)
    };

    static f64 t_init = 0;
    static f64 t_size = 0;
    static f64 t_scan = 0;
    static f64 t_fill = 0;

    // portable popcount32
    KOKKOS_INLINE_FUNCTION
    u32 popcount32(u32 x) {
#if defined(__CUDA_ARCH__)
        return __popc(x);
#elif defined(__HIP_DEVICE_COMPILE__)
        return __popcnt_u32(x);
#else
        return (u32) __builtin_popcount(x);
#endif
    }

    // --- 64-bit popcount helper (from earlier) ---
    KOKKOS_INLINE_FUNCTION
    u32 popcount64(u64 x) {
#if defined(__CUDA_ARCH__)
        return (u32) __popcll((unsigned long long) x);
#elif defined(__HIP_DEVICE_COMPILE__)
        return (u32) __popcll((unsigned long long) x);
#else
#if defined(_MSC_VER) && !defined(__clang__)
        return (u32) __popcnt64((unsigned long long) x);
#else
        return (u32) __builtin_popcountll((unsigned long long) x);
#endif
#endif
    }

    KOKKOS_INLINE_FUNCTION
    u32 hash32(u32 x) {
        // Knuth multiplicative hash
        return x * 2654435761u;
    }


    inline VertexPartitionCSR build_vertex_partition_csr(const Graph &device_g,
                                                         const PartitionManager &p_manager) {
        TIME("VertexPartitionCSR", "build_vertex_partition_csr", "init",
             // 64-bit words
             constexpr u32 WORD_BITS = 64;
             const u32 words = (p_manager.k + WORD_BITS - 1) / WORD_BITS;

             VertexPartitionCSR csr;
             csr.row = DeviceU32("row", device_g.n + 1);
             DeviceU32 counts("counts", device_g.n);
             DeviceU64 seen("seen", device_g.n * words);
             Kokkos::deep_copy(seen, 0);
             Kokkos::deep_copy(csr.row, 0);
             Kokkos::deep_copy(counts, 0);
             Kokkos::fence();
        );

        TIME("VertexPartitionCSR", "build_vertex_partition_csr", "mark",
             // mark (u, partition(v)) as seen
             Kokkos::parallel_for("mark", device_g.m, KOKKOS_LAMBDA(const u32 e) {
                 const vertex_t u = device_g.edges_u(e);
                 const vertex_t v = device_g.edges_v(e);
                 const partition_t p = p_manager.partition(v);

                 const u32 widx = p / WORD_BITS;
                 const u32 b = p % WORD_BITS;
                 const u64 mask = (u64(1) << b); // 64-bit mask

                 const size_t idx = u * words + widx;
                 Kokkos::atomic_or(&seen(idx), mask);
                 });
             Kokkos::fence();
        );

        TIME("VertexPartitionCSR", "build_vertex_partition_csr", "popcount",
             Kokkos::parallel_for("popcount", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                 u32 total = 0;
                 const size_t base = static_cast<size_t>(u) * words;
                 for (int w = 0; w < words; ++w) {
                 total += popcount64(seen(base + w));
                 }
                 counts(u) = total;
                 });
             Kokkos::fence();
        );

        TIME("VertexPartitionCSR", "build_vertex_partition_csr", "exclusive_scan_row",
             Kokkos::parallel_scan("exclusive_scan_row", device_g.n, KOKKOS_LAMBDA(const vertex_t u, u32 &carry, const bool final) {
                 const u32 c = counts(u);
                 if (final) csr.row(u + 1) = carry + c; // write inclusive, row[0] already 0
                 carry += c;
                 });
             Kokkos::fence();

             u32 nnz = 0;
             Kokkos::deep_copy(nnz, Kokkos::subview(csr.row, device_g.n));
        );

        TIME("VertexPartitionCSR", "build_vertex_partition_csr", "init2",
             csr.ids = DevicePartition("ids", nnz);
             csr.weights = DeviceWeight("weights", nnz);
             Kokkos::deep_copy(csr.ids, p_manager.k);
             Kokkos::deep_copy(csr.weights, 0);
             Kokkos::fence();
        );

        TIME("VertexPartitionCSR", "build_vertex_partition_csr", "fill",
             /*
              Kokkos::parallel_for("fill", device_g.m, KOKKOS_LAMBDA(const u32 i) {
                  vertex_t u = device_g.edges_u(i);
                  vertex_t v = device_g.edges_v(i);
                  weight_t w = device_g.edges_w(i);

                  partition_t v_id = p_manager.partition(v);

                  u32 beg = csr.row(u);
                  u32 end = csr.row(u + 1);
                  u32 len = end - beg;
                  if (len == 0) { return; }
                  if (len == 1) {
                  Kokkos::atomic_store(&csr.ids(beg), v_id);
                  Kokkos::atomic_add(&csr.weights(beg), w);
                  }

                  u32 j = beg + hash32(v_id) % len;
                  for (u32 t = 0; t < len; t++) {
                  if (j == end) { j = beg; }
                  partition_t res = Kokkos::atomic_compare_exchange(&csr.ids(j), p_manager.k, v_id);
                  if (res == p_manager.k || res == v_id) {
                  Kokkos::atomic_add(&csr.weights(j), w);
                  break;
                  }
                  j += 1;
                  }
                  });
              Kokkos::fence();
              */

             Kokkos::parallel_for("fill", device_g.n, KOKKOS_LAMBDA(const vertex_t u) {
                 u32 n_beg = device_g.neighborhood(u);
                 u32 n_end = device_g.neighborhood(u + 1);
                 u32 n_len = n_end - n_beg;

                 u32 r_beg = csr.row(u);
                 u32 r_end = csr.row(u + 1);
                 u32 r_len = r_end - r_beg;

                 if (n_len == 1) {
                 vertex_t v = device_g.edges_v(n_beg);
                 weight_t w = device_g.edges_w(n_beg);
                 partition_t v_id = p_manager.partition(v);
                 csr.ids(r_beg) = v_id;
                 csr.weights(r_beg) = w;
                 return;
                 }

                 if (r_len == 1) {
                     for (u32 i = n_beg; i < n_end; ++i) {
                         vertex_t v = device_g.edges_v(i);
                         weight_t w = device_g.edges_w(i);

                         partition_t v_id = p_manager.partition(v);

                         csr.ids(r_beg) = v_id;
                         csr.weights(r_beg) += w;
                     }
                 return;
                 }

                 for (u32 i = n_beg; i < n_end; ++i) {
                 vertex_t v = device_g.edges_v(i);
                 weight_t w = device_g.edges_w(i);

                 partition_t v_id = p_manager.partition(v);

                 u32 j = r_beg + hash32(v_id) % r_len;
                 for (u32 t = 0; t < r_len; t++) {
                 if (j == r_end) { j = r_beg; }
                 if (csr.ids(j) == p_manager.k) {
                 csr.ids(j) = v_id;
                 }
                 if (csr.ids(j) == v_id) {
                 csr.weights(j) += w;
                 break;
                 }
                 j += 1;
                 }
                 }

                 });
             Kokkos::fence();
        );

        return csr;
    }
}


#endif //GPU_HEIPROMAP_VERTEX_PARTITION_CSR_H
