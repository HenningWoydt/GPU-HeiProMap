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

#ifndef GPU_HEIPROMAP_DEFINITIONS_H
#define GPU_HEIPROMAP_DEFINITIONS_H

#include <cstdint>

#include <Kokkos_Core.hpp>

namespace GPU_HeiProMap {
    typedef int8_t s8;
    typedef int16_t s16;
    typedef int32_t s32;
    typedef int64_t s64;

    typedef uint8_t u8;
    typedef uint16_t u16;
    typedef uint32_t u32;
    typedef uint64_t u64;

    typedef float f32;
    typedef double f64;

    typedef u32 vertex_t;
    typedef s64 weight_t;
    typedef u32 partition_t;

    using HostVertex    = Kokkos::View<vertex_t*, Kokkos::HostSpace>;
    using HostWeight    = Kokkos::View<weight_t*, Kokkos::HostSpace>;
    using HostPartition = Kokkos::View<partition_t*, Kokkos::HostSpace>;
    using HostU8        = Kokkos::View<u8*, Kokkos::HostSpace>;
    using HostU64       = Kokkos::View<u64*, Kokkos::HostSpace>;

    using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

    using DeviceVertex    = Kokkos::View<vertex_t*, DeviceMemorySpace>;
    using DeviceWeight    = Kokkos::View<weight_t*, DeviceMemorySpace>;
    using DevicePartition = Kokkos::View<partition_t*, DeviceMemorySpace>;
    using DeviceU8        = Kokkos::View<u8*, DeviceMemorySpace>;
    using DeviceU32       = Kokkos::View<u32*, DeviceMemorySpace>;
    using DeviceU64       = Kokkos::View<u64*, DeviceMemorySpace>;
    using DeviceF32       = Kokkos::View<f32*, DeviceMemorySpace>;
    using DeviceF64       = Kokkos::View<f64*, DeviceMemorySpace>;
}

#endif //GPU_HEIPROMAP_DEFINITIONS_H
