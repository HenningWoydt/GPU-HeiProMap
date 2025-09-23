/*******************************************************************************
 * MIT License
 * (c) 2025 Henning Woydt
 ******************************************************************************/

#ifndef GPU_HEIPROMAP_PROFILER_H
#define GPU_HEIPROMAP_PROFILER_H

#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <cstdio>
#include <cstdlib>

#include "util.h"

#ifndef ENABLE_PROFILER
#define ENABLE_PROFILER 1
#endif

namespace GPU_HeiProMap {
    // ---------- simple stats -----------
    struct KTStat {
        double total_ms = 0.0;
        unsigned long calls = 0;

        inline void add(double ms) {
            total_ms += ms;
            ++calls;
        }

        inline double avg() const { return calls ? total_ms / (f64) calls : 0.0; }
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

        // --------- Pretty print (sorted by total time desc at each level) ----------
        void print(FILE *out = stderr) const {
#if !ENABLE_PROFILER
            return;
#endif
            auto pct = [](double part, double whole) -> double {
                return (whole > 0.0) ? (part * 100.0 / whole) : 0.0;
            };

            std::fprintf(out, "\n=== Kernel Timing (group → function → kernel) ===\n");
            std::fprintf(out, "%-50s %10s %12s %12s %8s %8s\n",
                         "Name", "Calls", "Total(ms)", "Avg(ms)", "%Func", "%Group");

            // Sort groups by total time
            std::vector<std::pair<std::string, KTGroup const *> > gs;
            gs.reserve(groups_.size());
            for (auto &kv: groups_) gs.emplace_back(kv.first, &kv.second);
            std::sort(gs.begin(), gs.end(),
                      [](auto &a, auto &b) { return a.second->agg.total_ms > b.second->agg.total_ms; });

            for (auto &[gname, gptr]: gs) {
                const auto &g = *gptr;
                // Group header
                std::fprintf(out, "%-50s %10lu %12.3f %12.3f %8s %8.1f\n",
                             gname.c_str(), g.agg.calls, g.agg.total_ms, g.agg.avg(),
                             "", pct(g.agg.total_ms, total_.total_ms));

                // Sort functions by total time
                std::vector<std::pair<std::string, KTKernels const *> > fs;
                fs.reserve(g.functions.size());
                for (auto &fk: g.functions) fs.emplace_back(fk.first, &fk.second);
                std::sort(fs.begin(), fs.end(),
                          [](auto &a, auto &b) { return a.second->agg.total_ms > b.second->agg.total_ms; });

                for (auto &[fname, fptr]: fs) {
                    const auto &f = *fptr;
                    // Function row (1-tab indent)
                    std::fprintf(out, "    %-46s %10lu %12.3f %12.3f %8s %8.1f\n",
                                 fname.c_str(), f.agg.calls, f.agg.total_ms, f.agg.avg(),
                                 "", pct(f.agg.total_ms, g.agg.total_ms));

                    // Sort kernels by total time
                    std::vector<std::pair<std::string, KTStat const *> > ks;
                    ks.reserve(f.kernels.size());
                    for (auto &kk: f.kernels) ks.emplace_back(kk.first, &kk.second);
                    std::sort(ks.begin(), ks.end(),
                              [](auto &a, auto &b) { return a.second->total_ms > b.second->total_ms; });

                    for (auto &[kname, kstat]: ks) {
                        // Kernel row (2-tab indent)
                        std::fprintf(out, "        %-42s %10lu %12.3f %12.3f %8.1f %8.1f\n",
                                     kname.c_str(), kstat->calls, kstat->total_ms, kstat->avg(),
                                     pct(kstat->total_ms, f.agg.total_ms),
                                     pct(kstat->total_ms, g.agg.total_ms));
                    }
                }
                std::fprintf(out, "\n"); // blank line after each group
            }

            // Total row at end
            std::fprintf(out, "-------------------------------------------------\n");
            std::fprintf(out, "%-50s %10lu %12.3f %12.3f %8s %8s\n",
                         "TOTAL", total_.calls, total_.total_ms, total_.avg(), "", "100.0");
            std::fprintf(out, "=================================================\n");
        }

        // --------- JSON export (nested) ----------
        std::string to_JSON() const {
            auto esc = [](const std::string &s) {
                std::ostringstream e;
                for (char c: s) {
                    if (c == '\"' || c == '\\') e << '\\' << c;
                    else if (c == '\n') e << "\\n";
                    else e << c;
                }
                return e.str();
            };

            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss.precision(6);
            oss << "{";

            // total
            oss << "\"total\":{"
                    << "\"calls\":" << total_.calls << ","
                    << "\"total_ms\":" << total_.total_ms << ","
                    << "\"avg_ms\":" << total_.avg() << "},";

            // groups (sorted)
            oss << "\"groups\":{";
            bool first_g = true;
            std::vector<std::pair<std::string, KTGroup const *> > gs;
            for (auto &kv: groups_) gs.emplace_back(kv.first, &kv.second);
            std::sort(gs.begin(), gs.end(),
                      [](auto &a, auto &b) { return a.second->agg.total_ms > b.second->agg.total_ms; });

            for (auto &[gname, gptr]: gs) {
                if (!first_g) oss << ",";
                first_g = false;
                const auto &g = *gptr;
                oss << "\"" << esc(gname) << "\":{";
                oss << "\"total\":{"
                        << "\"calls\":" << g.agg.calls << ","
                        << "\"total_ms\":" << g.agg.total_ms << ","
                        << "\"avg_ms\":" << g.agg.avg() << "},";

                // functions (sorted)
                oss << "\"functions\":{";
                bool first_f = true;
                std::vector<std::pair<std::string, KTKernels const *> > fs;
                for (auto &fk: g.functions) fs.emplace_back(fk.first, &fk.second);
                std::sort(fs.begin(), fs.end(),
                          [](auto &a, auto &b) { return a.second->agg.total_ms > b.second->agg.total_ms; });

                for (auto &[fname, fptr]: fs) {
                    if (!first_f) oss << ",";
                    first_f = false;
                    const auto &f = *fptr;
                    oss << "\"" << esc(fname) << "\":{";
                    oss << "\"total\":{"
                            << "\"calls\":" << f.agg.calls << ","
                            << "\"total_ms\":" << f.agg.total_ms << ","
                            << "\"avg_ms\":" << f.agg.avg() << "},";

                    // kernels (sorted)
                    oss << "\"kernels\":{";
                    bool first_k = true;
                    std::vector<std::pair<std::string, KTStat const *> > ks;
                    for (auto &kk: f.kernels) ks.emplace_back(kk.first, &kk.second);
                    std::sort(ks.begin(), ks.end(),
                              [](auto &a, auto &b) { return a.second->total_ms > b.second->total_ms; });

                    for (auto &[kname, kstat]: ks) {
                        if (!first_k) oss << ",";
                        first_k = false;
                        oss << "\"" << esc(kname) << "\":{"
                                << "\"calls\":" << kstat->calls << ","
                                << "\"total_ms\":" << kstat->total_ms << ","
                                << "\"avg_ms\":" << kstat->avg() << "}";
                    }
                    oss << "}"; // kernels
                    oss << "}"; // function
                }
                oss << "}"; // functions
                oss << "}"; // group
            }
            oss << "}"; // groups

            oss << "}";
            return oss.str();
        }

    private:
        Profiler() = default;

        std::unordered_map<std::string, KTGroup> groups_;
        KTStat total_;
    };

    // ===== Convenience macro =====
    // Helper macros to concatenate tokens safely
#define CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) CONCAT_IMPL(x, y)

    // Pick a unique name per call using __LINE__
#define UNIQUE_NAME(base) CONCAT(base, __LINE__)

#if ENABLE_PROFILER
#define TIME(group, function, kernel, kernel_stmt) \
        std::chrono::time_point<std::chrono::system_clock> UNIQUE_NAME(_p) = get_time_point(); \
        kernel_stmt; \
        Profiler::instance().add(group, function, kernel, get_milli_seconds(UNIQUE_NAME(_p), get_time_point()));
#else
#define TIME(group, function, kernel, kernel_stmt) kernel_stmt
#endif
} // namespace GPU_HeiProMap

#endif // GPU_HEIPROMAP_PROFILER_H
