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

#ifndef SHAREDMAP_GPU_DEFINITIONS_H
#define SHAREDMAP_GPU_DEFINITIONS_H

#include <cstdint>

#include <jet.h>

namespace SharedMap_GPU {
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

    using DeviceRowMap = Kokkos::View<jet_partitioner::edge_offset_t *, jet_partitioner::Device>;
    using DeviceEntries = Kokkos::View<jet_partitioner::ordinal_t *, jet_partitioner::Device>;
    using DeviceValues = Kokkos::View<jet_partitioner::value_t *, jet_partitioner::Device>;
    using DeviceWeights = Kokkos::View<jet_partitioner::value_t *, jet_partitioner::Device>;
    using DevicePartition = Kokkos::View<jet_partitioner::part_t *, jet_partitioner::Device>;
    using HostRowMap = Kokkos::View<jet_partitioner::edge_offset_t *, Kokkos::HostSpace>;
    using HostEntries = Kokkos::View<jet_partitioner::ordinal_t *, Kokkos::HostSpace>;
    using HostValues = Kokkos::View<jet_partitioner::value_t *, Kokkos::HostSpace>;
    using HostWeights = Kokkos::View<jet_partitioner::value_t *, Kokkos::HostSpace>;
    using HostPartition = Kokkos::View<jet_partitioner::part_t *, Kokkos::HostSpace>;
}

#endif //SHAREDMAP_GPU_DEFINITIONS_H
