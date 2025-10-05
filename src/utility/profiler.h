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

#ifndef GPU_HEIPROMAP_PROFILER_H
#define GPU_HEIPROMAP_PROFILER_H

#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <sstream>

#include "util.h"

#ifndef ENABLE_PROFILER
#define ENABLE_PROFILER 1
#endif

namespace GPU_HeiProMap {
    // ---------- simple stats -----------
    struct KTStat {
        double total_ms = 0.0;
        unsigned long calls = 0;

        void add(double ms) {
            total_ms += ms;
            ++calls;
        }

        double avg() const { return calls ? total_ms / (f64) calls : 0.0; }
    };

    struct KTKernels {
        std::unordered_map<std::string, KTStat> kernels; // kernel name -> stat
        KTStat agg; // aggregate of kernels
    };

    struct KTGroup {
        std::unordered_map<std::string, KTKernels> functions; // function name -> kernels
        KTStat agg; // aggregate of functions
    };

    // ========== Profiler (hierarchical) ==========
    class Profiler {
    public:
        static Profiler &instance() {
            static Profiler R;
            return R;
        }

        // Add a timing sample
        void add(const char *group, const char *function, const char *kernel, double ms) {
            auto &g = groups_[group];
            auto &f = g.functions[function];
            f.kernels[kernel].add(ms);
            f.agg.add(ms);
            g.agg.add(ms);
            total_.add(ms);
        }

        // --------- JSON export (nested, pretty-printed) ----------
        std::string to_JSON() const {
#if !ENABLE_PROFILER
            return "{}";
#endif
            auto esc = [](const std::string &s) {
                std::ostringstream e;
                for (char c: s) {
                    if (c == '\"' || c == '\\') e << '\\' << c;
                    else if (c == '\n') e << "\\n";
                    else e << c;
                }
                return e.str();
            };

            auto indent = [](std::ostringstream &oss, int level) {
                for (int i = 0; i < level; ++i) oss << '\t'; // tabs for indentation
            };

            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss.precision(6);

            int lvl = 0;
            oss << "{\n";

            // overall total (only total_ms)
            indent(oss, ++lvl);
            oss << "\"total\": {\n";
            indent(oss, ++lvl);
            oss << "\"total_ms\": " << total_.total_ms << "\n";
            indent(oss, --lvl);
            oss << "},\n";

            // groups (sorted by total time desc)
            indent(oss, lvl);
            oss << "\"groups\": {\n";
            ++lvl;

            std::vector<std::pair<std::string, KTGroup const *> > gs;
            for (auto &kv: groups_) gs.emplace_back(kv.first, &kv.second);
            std::sort(gs.begin(), gs.end(),
                      [](auto &a, auto &b) { return a.second->agg.total_ms > b.second->agg.total_ms; });

            bool first_g = true;
            for (auto &[gname, gptr]: gs) {
                if (!first_g) oss << ",\n";
                first_g = false;

                indent(oss, lvl);
                oss << "\"" << esc(gname) << "\": {\n";
                ++lvl;

                // group total
                indent(oss, lvl);
                oss << "\"total_ms\": " << gptr->agg.total_ms << ",\n";

                // functions
                indent(oss, lvl);
                oss << "\"functions\": {\n";
                ++lvl;

                std::vector<std::pair<std::string, KTKernels const *> > fs;
                for (auto &fk: gptr->functions) fs.emplace_back(fk.first, &fk.second);
                std::sort(fs.begin(), fs.end(),
                          [](auto &a, auto &b) { return a.second->agg.total_ms > b.second->agg.total_ms; });

                bool first_f = true;
                for (auto &[fname, fptr]: fs) {
                    if (!first_f) oss << ",\n";
                    first_f = false;

                    indent(oss, lvl);
                    oss << "\"" << esc(fname) << "\": {\n";
                    ++lvl;

                    // function total
                    indent(oss, lvl);
                    oss << "\"total_ms\": " << fptr->agg.total_ms << ",\n";

                    // kernels
                    indent(oss, lvl);
                    oss << "\"kernels\": {\n";
                    ++lvl;

                    std::vector<std::pair<std::string, KTStat const *> > ks;
                    for (auto &kk: fptr->kernels) ks.emplace_back(kk.first, &kk.second);
                    std::sort(ks.begin(), ks.end(),
                              [](auto &a, auto &b) { return a.second->total_ms > b.second->total_ms; });

                    bool first_k = true;
                    for (auto &[kname, kstat]: ks) {
                        if (!first_k) oss << ",\n";
                        first_k = false;

                        indent(oss, lvl);
                        oss << "\"" << esc(kname) << "\": {\n";
                        ++lvl;
                        indent(oss, lvl);
                        oss << "\"calls\": " << kstat->calls << ",\n";
                        indent(oss, lvl);
                        oss << "\"total_ms\": " << kstat->total_ms << ",\n";
                        indent(oss, lvl);
                        oss << "\"avg_ms\": " << kstat->avg() << "\n";
                        --lvl;
                        indent(oss, lvl);
                        oss << "}";
                    }
                    oss << "\n";
                    --lvl;
                    indent(oss, lvl);
                    oss << "}\n"; // end kernels

                    --lvl;
                    indent(oss, lvl);
                    oss << "}";
                }
                oss << "\n";
                --lvl;
                indent(oss, lvl);
                oss << "}\n"; // end functions

                --lvl;
                indent(oss, lvl);
                oss << "}";
            }
            oss << "\n";
            --lvl;
            indent(oss, lvl);
            oss << "}\n"; // end groups

            --lvl;
            oss << "}";
            return oss.str();
        }

    private:
        Profiler() = default;

        std::unordered_map<std::string, KTGroup> groups_;
        KTStat total_;
    };

    struct ScopedTimer {
#if ENABLE_PROFILER
        const char *group;
        const char *function;
        const char *kernel;
        std::chrono::time_point<std::chrono::system_clock> t0;
        bool stopped = false;
#endif

        ScopedTimer(const char *g, const char *f, const char *k) {
#if ENABLE_PROFILER
            group = g;
            function = f;
            kernel = k;
            t0 = get_time_point();
            stopped = false;
#endif
        }

        void stop() {
#if ENABLE_PROFILER
            if (!stopped) {
                Profiler::instance().add(group, function, kernel, get_milli_seconds(t0, get_time_point()));
                stopped = true;
            }
#endif
        }

        ~ScopedTimer() { stop(); }
    };
} // namespace GPU_HeiProMap

#endif // GPU_HEIPROMAP_PROFILER_H
