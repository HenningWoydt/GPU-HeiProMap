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

#ifndef GPU_HEIPROMAP_KOKKOS_UTIL_H
#define GPU_HEIPROMAP_KOKKOS_UTIL_H

#include <charconv>
#include <fstream>
#include <iomanip>
#include <istream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

#include "definitions.h"

namespace GPU_HeiProMap {
    template<class View>
    KOKKOS_INLINE_FUNCTION
    size_t view_bytes(const View &v) {
        using T = typename View::value_type;
        // span() is the allocated number of elements (safe even if extents are 0)
        return v.span() * sizeof(T);
    }

    template<class View>
    double view_megabytes(const View &v) {
        return static_cast<double>(view_bytes(v)) / (1024.0 * 1024.0);
    }

    template<typename T>
    u64 count_occurrences(Kokkos::View<T *, DeviceMemorySpace> &view,
                          T needle) {
        u64 result = 0;
        Kokkos::parallel_reduce("count_occurrences", view.size(), KOKKOS_LAMBDA(const u64 i, u64 &local) {
                                    if (view(i) == needle) local += 1ULL;
                                },
                                result);
        Kokkos::fence();
        return result;
    }

    template<typename T>
    u64 count_neq_occurrences(Kokkos::View<T *, DeviceMemorySpace> &view,
                              T needle) {
        u64 result = 0;
        Kokkos::parallel_reduce("count_occurrences", view.size(), KOKKOS_LAMBDA(const u64 i, u64 &local) {
                                    if (view(i) != needle) local += 1ULL;
                                },
                                result);
        Kokkos::fence();
        return result;
    }

    template<typename T>
    u64 greater_occurrences(Kokkos::View<T *, DeviceMemorySpace> &view,
                            T needle) {
        u64 result = 0;
        Kokkos::parallel_reduce("count_occurrences", view.size(), KOKKOS_LAMBDA(const u64 i, u64 &local) {
                                    if (view(i) > needle) local += 1ULL;
                                },
                                result);
        Kokkos::fence();
        return result;
    }

    template<typename T>
    u64 smaller_occurrences(Kokkos::View<T *, DeviceMemorySpace> &view,
                            T needle) {
        u64 result = 0;
        Kokkos::parallel_reduce("count_occurrences", view.size(), KOKKOS_LAMBDA(const u64 i, u64 &local) {
                                    if (view(i) < needle) local += 1ULL;
                                },
                                result);
        Kokkos::fence();
        return result;
    }
}

#endif //GPU_HEIPROMAP_KOKKOS_UTIL_H
