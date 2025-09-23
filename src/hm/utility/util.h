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

#ifndef SHAREDMAP_GPU_UTIL_H
#define SHAREDMAP_GPU_UTIL_H

#include <vector>
#include <string>
#include <istream>
#include <sstream>
#include <fstream>

namespace SharedMap_GPU {

    /**
     * Splits a string into multiple sub-strings. The specified character will
     * serve as the delimiter and will not be present in any string.
     *
     * @param str The string.
     * @param c The character.
     * @return Vector of sub-strings.
     */
    inline std::vector<std::string> split(const std::string &str, char c) {
        std::vector<std::string> splits;

        std::istringstream iss(str);
        std::string token;

        while (std::getline(iss, token, c)) {
            splits.push_back(token);
        }

        return splits;
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
    inline T convert_to(const std::string &str) {
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
    inline std::vector<T> convert(const std::vector<std::string> &&vec) {
        std::vector<T> v;

        for (auto &s: vec) {
            v.push_back(convert_to<T>(s));
        }

        return v;
    }

    /**
     * Multiplies all elements in the vector.
     *
     * @tparam T Vector type.
     * @param vec The vector.
     * @return The product.
     */
    template<typename T>
    inline T product(const std::vector<T> &vec) {
        T p = (T) 1;
        for (auto &x: vec) { p *= x; }
        return p;
    }

    template<typename T>
    inline T max(const std::vector<T> &vec) {
        T m = vec[0];
        for (auto &x: vec) { m = std::max(m, x); }
        return m;
    }

    inline bool file_exists(const std::string &path) {
        std::ifstream f(path.c_str());
        return f.good();
    }

    inline void str_to_ints(const std::string &str,
                            std::vector<int> &ints) {
        ints.resize(str.size());

        int idx = 0;
        int curr_number = 0;

        for (const char c: str) {
            if (c == ' ') {
                ints[idx] = curr_number;
                idx += curr_number != 0;
                curr_number = 0;
            } else {
                curr_number = curr_number * 10 + (c - '0');
            }
        }

        ints[idx] = curr_number;
        idx += curr_number != 0;
        ints.resize(idx);
    }

    inline f64 get_seconds(std::chrono::high_resolution_clock::time_point sp, std::chrono::high_resolution_clock::time_point ep) {
        return (f64) std::chrono::duration_cast<std::chrono::nanoseconds>(ep - sp).count() / 1e9;
    }
}

#endif //SHAREDMAP_GPU_UTIL_H
