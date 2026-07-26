#include "vknn/profiler.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

namespace vknn {

    double Profiler::totalCpuMs() const {
        double s = 0;
        for (auto &r: records_)
        {
            s += r.cpuMs;
        }
        return s;
    }
    double Profiler::totalGpuMs() const {
        double s = 0;
        for (auto &r: records_)
        {
            // A negative gpuMs marks a record with no GPU measurement (e.g. a CPU fallback); skip it.
            if (r.gpuMs >= 0)
            {
                s += r.gpuMs;
            }
        }
        return s;
    }
    uint64_t Profiler::totalDispatches() const {
        uint64_t s = 0;
        for (auto &r: records_)
        {
            s += r.dispatches;
        }
        return s;
    }

    void Profiler::printTable() const {
        // Width of the horizontal rule bracketing the per-node table; spans the fixed-width columns
        // of the header/row format strings below.
        constexpr int kTableRuleWidth = 81;
        if (records_.empty())
        {
            printf("(profiler: no records)\n");
            return;
        }
        // Aggregate per op type for the summary, but print per-node too.
        printf("\n%-28s %-10s %-7s %10s %10s  %10s\n", "op (node)", "backend", "type", "cpu(ms)", "gpu(ms)", "dispatches");
        printf("%s\n", std::string(kTableRuleWidth, '-').c_str());
        double tcpu = 0, tgpu = 0;
        // Per-op-type totals for the summary below.
        struct TypeTotals {
            double   cpuMs = 0, gpuMs = 0;
            uint64_t dispatches = 0;
        };
        std::map<std::string, TypeTotals> byType;
        uint64_t                          tdisp = 0;
        for (const auto &r: records_)
        {
            char gpu[16] = "-";
            if (r.gpuMs >= 0)
            {
                snprintf(gpu, sizeof(gpu), "%.3f", r.gpuMs);
            }
            printf("%-28.28s %-10s %-7s %10.3f %10s  %10u%s\n", r.name.c_str(), r.backend.c_str(), opTypeName(r.type), r.cpuMs, gpu, r.dispatches, r.fellBack ? "  [FALLBACK]" : "");
            tcpu += r.cpuMs;
            if (r.gpuMs >= 0)
            {
                tgpu += r.gpuMs;
            }
            tdisp += r.dispatches;
            auto &a = byType[opTypeName(r.type)];
            a.cpuMs += r.cpuMs;
            a.gpuMs += (r.gpuMs >= 0 ? r.gpuMs : 0);
            a.dispatches += r.dispatches;
        }
        printf("%s\n", std::string(kTableRuleWidth, '-').c_str());
        printf("%-28s %-10s %-7s %10.3f %10.3f  %10llu\n", "TOTAL", "", "", tcpu, tgpu, (unsigned long long) tdisp);
        // The op records cover the graph's nodes only; a segment's boundary converts, resident-link
        // copies, decode-chain feedback, and argmax epilogues dispatch outside any node, so the
        // segment's own "segment dispatches" log line is at or above this total.
        printf("%-28s %zu record(s), %llu dispatch(es) over the nodes\n", "", records_.size(), (unsigned long long) tdisp);
        printf("\nPer op-type (cpu ms / gpu ms / dispatches):\n");
        // byType is keyed on op-type name; copy to a vector so the summary can be reordered by cost.
        std::vector<std::pair<std::string, TypeTotals>> v(byType.begin(), byType.end());
        // Sort descending by aggregate GPU time so the hottest op types lead.
        std::sort(v.begin(), v.end(), [](auto &a, auto &b) {
            return a.second.gpuMs > b.second.gpuMs;
        });
        for (auto &kv: v)
        {
            printf("  %-22s %8.3f / %8.3f / %8llu\n", kv.first.c_str(), kv.second.cpuMs, kv.second.gpuMs, (unsigned long long) kv.second.dispatches);
        }
    }

    std::string Profiler::toJson() const {
        std::ostringstream os;
        os << "[";
        for (size_t i = 0; i < records_.size(); ++i)
        {
            const auto &r = records_[i];
            os << "{\"name\":\"" << r.name << "\",\"type\":\"" << opTypeName(r.type) << "\",\"backend\":\"" << r.backend << "\",\"cpuMs\":" << r.cpuMs << ",\"gpuMs\":" << r.gpuMs << ",\"dispatches\":" << r.dispatches << ",\"fellBack\":" << (r.fellBack ? "true" : "false") << "}";
            if (i + 1 < records_.size())
            {
                os << ",";
            }
        }
        os << "]";
        return os.str();
    }

    void Profiler::writeChromeTrace(const std::string &path) const {
        // chrome://tracing JSON: one complete event per op, sequential on a single track.
        std::ofstream f(path);
        if (!f)
        {
            return;
        }
        f << "{\"traceEvents\":[";
        double ts    = 0; // microseconds, synthetic timeline
        bool   first = true;
        for (const auto &r: records_)
        {
            // Event duration is the GPU time when measured, else CPU wall time; convert ms to us.
            double dur = (r.gpuMs >= 0 ? r.gpuMs : r.cpuMs) * 1000.0;
            if (!first)
            {
                f << ",";
            }
            first = false;
            f << "{\"name\":\"" << r.name << "\",\"cat\":\"" << opTypeName(r.type) << "\",\"ph\":\"X\",\"pid\":1,\"tid\":1,\"ts\":" << ts << ",\"dur\":" << dur << ",\"args\":{\"backend\":\"" << r.backend << "\",\"cpuMs\":" << r.cpuMs << ",\"gpuMs\":" << r.gpuMs << ",\"dispatches\":" << r.dispatches << "}}";
            ts += dur;
        }
        f << "],\"displayTimeUnit\":\"ms\"}";
    }

} // namespace vknn
