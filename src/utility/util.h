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

#ifndef GPU_HEIPROMAP_UTIL_H
#define GPU_HEIPROMAP_UTIL_H

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
    inline std::vector<std::string> split(const std::string &str, char c) {
        std::vector<std::string> splits;

        std::istringstream iss(str);
        std::string token;

        while (std::getline(iss, token, c)) {
            splits.push_back(token);
        }

        return splits;
    }

    inline std::vector<std::string> split_ws(const std::string &str) {
        std::vector<std::string> result;
        std::istringstream iss(str);
        std::string token;
        while (iss >> token) {
            // skips all whitespace automatically
            result.push_back(token);
        }
        return result;
    }

    inline bool file_exists(const std::string &path) {
        std::ifstream f(path.c_str());
        return f.good();
    }

    inline std::vector<char> load_file_to_buffer(const std::string &file_path) {
        std::ifstream file(file_path, std::ios::binary | std::ios::ate); // open at end
        if (!file) {
            std::cerr << "Could not open file " << file_path << std::endl;
            exit(EXIT_FAILURE);
        }

        std::streamsize size = file.tellg(); // get size
        file.seekg(0, std::ios::beg); // rewind

        std::vector<char> buffer((size_t) size);
        if (!file.read(buffer.data(), size)) {
            std::cerr << "Failed to read file into buffer" << std::endl;
            exit(EXIT_FAILURE);
        }

        return buffer;
    }

    inline std::vector<std::string> read_header(const std::string &str) {
        std::vector<std::string> header;

        size_t i = 0;
        while (i < str.size()) {
            // Skip leading spaces
            while (i < str.size() && str[i] == ' ') { ++i; }

            size_t start = i;

            while (i < str.size() && str[i] != ' ') { ++i; }

            if (start < i) { header.emplace_back(str.substr(start, i - start)); }
        }

        return header;
    }

    inline void str_to_ints(const std::string &str,
                            std::vector<u64> &ints) {
        ints.resize(str.size());

        size_t idx = 0;
        u64 curr_number = 0;

        for (const char c: str) {
            if (c == ' ') {
                ints[idx] = curr_number;
                idx += curr_number != 0;
                curr_number = 0;
            } else {
                curr_number = curr_number * 10 + (u64) (c - '0');
            }
        }

        ints[idx] = curr_number;
        idx += curr_number != 0;
        ints.resize(idx);
    }

    inline void str_to_ints(const std::string &str,
                            std::vector<int> &ints) {
        ints.resize(str.size());

        size_t idx = 0;
        int curr_number = 0;

        for (const char c: str) {
            if (c == ' ') {
                ints[idx] = curr_number;
                idx += curr_number != 0;
                curr_number = 0;
            } else {
                curr_number = curr_number * 10 + (int) (c - '0');
            }
        }

        ints[idx] = curr_number;
        idx += curr_number != 0;
        ints.resize(idx);
    }

    inline void str_to_ints(const std::string &str,
                            std::vector<vertex_t> &ints) {
        ints.resize(str.size());

        size_t idx = 0;
        vertex_t curr_number = 0;

        for (const char c: str) {
            if (c == ' ') {
                ints[idx] = curr_number;
                idx += curr_number != 0;
                curr_number = 0;
            } else {
                curr_number = curr_number * 10 + (vertex_t) (c - '0');
            }
        }

        ints[idx] = curr_number;
        idx += curr_number != 0;
        ints.resize(idx);
    }

    inline auto get_time_point() {
        return std::chrono::high_resolution_clock::now();
    }

    inline f64 get_seconds(std::chrono::high_resolution_clock::time_point sp, std::chrono::high_resolution_clock::time_point ep) {
        return (f64) std::chrono::duration_cast<std::chrono::nanoseconds>(ep - sp).count() / 1e9;
    }

    inline f64 get_milli_seconds(std::chrono::high_resolution_clock::time_point sp, std::chrono::high_resolution_clock::time_point ep) {
        return (f64) std::chrono::duration_cast<std::chrono::nanoseconds>(ep - sp).count() / 1e6;
    }

    template<typename ViewType>
    void print_view(const ViewType &view, const std::string &label = "") {
        using ValueType = typename ViewType::value_type;
        auto size = view.extent(0);
        auto host_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);

        if (!label.empty()) {
            std::cout << label << " (" << size << "):\n";
        }

        for (std::size_t i = 0; i < size; ++i) {
            std::cout << std::fixed << std::setprecision(3) << static_cast<ValueType>(host_view(i)) << " ";
        }
        std::cout << std::endl;
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION
    constexpr T min_sentinel() {
        if constexpr (std::is_same<T, f32>::value) {
            return -1e30f;
        } else if constexpr (std::is_same<T, f64>::value) {
            return -1e300;
        } else if constexpr (std::is_same<T, s8>::value) {
            return -128;
        } else if constexpr (std::is_same<T, s16>::value) {
            return -32768;
        } else if constexpr (std::is_same<T, s32>::value) {
            return -2147483647 - 1;
        } else if constexpr (std::is_same<T, s64>::value) {
            return static_cast<s64>(-9223372036854775807LL - 1);
        } else if constexpr (std::is_same<T, u8>::value ||
                             std::is_same<T, u16>::value ||
                             std::is_same<T, u32>::value ||
                             std::is_same<T, u64>::value) {
            return 0; // No negative sentinel for unsigned
        } else {
            return static_cast<T>(-1); // fallback for unknown types
        }
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION
    constexpr T max_sentinel() {
        if constexpr (std::is_same<T, f32>::value) {
            return 1e30f;
        } else if constexpr (std::is_same<T, f64>::value) {
            return 1e300;
        } else if constexpr (std::is_same<T, s8>::value) {
            return 127;
        } else if constexpr (std::is_same<T, s16>::value) {
            return 32767;
        } else if constexpr (std::is_same<T, s32>::value) {
            return 2147483647;
        } else if constexpr (std::is_same<T, s64>::value) {
            return static_cast<s64>(9223372036854775807LL);
        } else if constexpr (
            std::is_same<T, u8>::value ||
            std::is_same<T, u16>::value ||
            std::is_same<T, u32>::value ||
            std::is_same<T, u64>::value
        ) {
            // all bits 1 for unsigned gives the maximum
            return static_cast<T>(-1);
        } else {
            // fallback for other types
            return std::numeric_limits<T>::max();
        }
    }

    /**
     * Converts a string into the specified datatype. Conversion is done via
     * string stream and ">>" operator.
     *
     * @tparam T The desired datatype.
     * @param str The string.
     * @return The converted string.
     */
    template<typename T>
    T convert_to(const std::string &str) {
        T result;
        std::istringstream iss(str);
        iss >> result;
        return result;
    }

    /**
     * Converts the vector of strings into a vector of T's.
     *
     * @tparam T Type of conversion.
     * @param vec The vector.
     * @return Vector of transformed T's.
     */
    template<typename T>
    std::vector<T> convert(const std::vector<std::string> &vec) {
        std::vector<T> v;

        for (auto &s: vec) {
            v.push_back(convert_to<T>(s));
        }

        return v;
    }

    /**
     * Multiplies all elements in the vector.
     *
     * @tparam T1 Resulting type.
     * @tparam T2 Type of vector elements.
     * @param vec The vector.
     * @return The product.
     */
    template<typename T1, typename T2>
    T1 prod(const std::vector<T2> &vec) {
        T1 p = (T1) 1;

        for (auto &x: vec) {
            p *= (T1) x;
        }

        return p;
    }

    template<typename T>
    T max(const std::vector<T> &vec) {
        T p = vec[0];

        for (auto &x: vec) {
            p = std::max(p, x);
        }

        return p;
    }

    inline void write_partition(const std::vector<partition_t> &partition, const std::string &file_path) {
        std::ofstream out(file_path, std::ios::binary); // Open file in binary mode for faster writing
        if (!out.is_open()) {
            std::cerr << "Error: Could not open " << file_path << " to write partition!" << std::endl;
            return;
        }

        // Write each partition element directly to the file, separating them by newlines
        for (const auto &i: partition) {
            out << i << '\n'; // Write each element followed by a newline
        }

        out.close();
    }

    inline void write_partition_slow(const HostPartition &partition, vertex_t n, const std::string &file_path) {
        std::ofstream out(file_path, std::ios::binary); // Open file in binary mode for faster writing
        if (!out.is_open()) {
            std::cerr << "Error: Could not open " << file_path << " to write partition!" << std::endl;
            return;
        }

        // Write each partition element directly to the file, separating them by newlines
        for (vertex_t u = 0; u < n; ++u) {
            out << partition(u) << '\n'; // Write each element followed by a newline
        }

        out.close();
    }

    inline void write_partition(const HostPartition &partition,
                                vertex_t n,
                                const std::string &file_path) {
        std::ofstream out(file_path, std::ios::binary);
        if (!out) {
            std::cerr << "Error: Could not open " << file_path << " to write partition!\n";
            return;
        }

        // 1) Enlarge the stream's internal buffer (optional but helps).
        std::vector<char> stream_buf(1 << 20); // 1 MB
        out.rdbuf()->pubsetbuf(stream_buf.data(), stream_buf.size());

        // 2) Our own aggregation buffer for batched writes.
        std::string buf;
        buf.reserve(1 << 20); // 1 MB; tune as needed

        for (vertex_t u = 0; u < n; ++u) {
            // Convert integer to text without iostream overhead.
            char tmp[32]; // enough for signed 64-bit
            auto val = partition(u);
            auto res = std::to_chars(std::begin(tmp), std::end(tmp), val);
            if (res.ec != std::errc{}) {
                std::cerr << "Error: to_chars failed while writing.\n";
                return;
            }
            const size_t len = static_cast<size_t>(res.ptr - tmp);

            // Flush our buffer if adding this line would overflow capacity.
            if (buf.size() + len + 1 > buf.capacity()) {
                out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
                buf.clear();
            }
            buf.append(tmp, len);
            buf.push_back('\n');
        }

        if (!buf.empty())
            out.write(buf.data(), static_cast<std::streamsize>(buf.size()));

        out.flush(); // optional
    }

    inline void write_partition(const std::vector<int> &partition,
                                const std::string &file_path) {
        vertex_t n = (vertex_t) partition.size();
        std::ofstream out(file_path, std::ios::binary);
        if (!out) {
            std::cerr << "Error: Could not open " << file_path << " to write partition!\n";
            return;
        }

        // 1) Enlarge the stream's internal buffer (optional but helps).
        std::vector<char> stream_buf(1 << 20); // 1 MB
        out.rdbuf()->pubsetbuf(stream_buf.data(), stream_buf.size());

        // 2) Our own aggregation buffer for batched writes.
        std::string buf;
        buf.reserve(1 << 20); // 1 MB; tune as needed

        for (vertex_t u = 0; u < n; ++u) {
            // Convert integer to text without iostream overhead.
            char tmp[32]; // enough for signed 64-bit
            auto val = partition[u];
            auto res = std::to_chars(std::begin(tmp), std::end(tmp), val);
            if (res.ec != std::errc{}) {
                std::cerr << "Error: to_chars failed while writing.\n";
                return;
            }
            const size_t len = static_cast<size_t>(res.ptr - tmp);

            // Flush our buffer if adding this line would overflow capacity.
            if (buf.size() + len + 1 > buf.capacity()) {
                out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
                buf.clear();
            }
            buf.append(tmp, len);
            buf.push_back('\n');
        }

        if (!buf.empty())
            out.write(buf.data(), static_cast<std::streamsize>(buf.size()));

        out.flush(); // optional
    }

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

#endif //GPU_HEIPROMAP_UTIL_H
