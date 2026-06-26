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
            if (r.gpuMs >= 0)
            {
                s += r.gpuMs;
            }
        }
        return s;
    }

    void Profiler::printTable() const {
        if (records_.empty())
        {
            printf("(profiler: no records)\n");
            return;
        }
        // Aggregate per op type for the summary, but print per-node too.
        printf("\n%-28s %-10s %-7s %10s %10s  %s\n", "op (node)", "backend", "type", "cpu(ms)", "gpu(ms)", "dispatch");
        printf("%s\n", std::string(86, '-').c_str());
        double                                           tcpu = 0, tgpu = 0;
        std::map<std::string, std::pair<double, double>> byType;
        for (const auto &r: records_)
        {
            char disp[48] = "-";
            if (r.dispatch[0])
            {
                snprintf(disp, sizeof(disp), "%ux%ux%u", r.dispatch[0], r.dispatch[1], r.dispatch[2]);
            }
            char gpu[16] = "-";
            if (r.gpuMs >= 0)
            {
                snprintf(gpu, sizeof(gpu), "%.3f", r.gpuMs);
            }
            printf("%-28.28s %-10s %-7s %10.3f %10s  %s%s\n", r.name.c_str(), r.backend.c_str(), opTypeName(r.type), r.cpuMs, gpu, disp, r.fellBack ? "  [FALLBACK]" : "");
            tcpu += r.cpuMs;
            if (r.gpuMs >= 0)
            {
                tgpu += r.gpuMs;
            }
            auto &a = byType[opTypeName(r.type)];
            a.first += r.cpuMs;
            a.second += (r.gpuMs >= 0 ? r.gpuMs : 0);
        }
        printf("%s\n", std::string(86, '-').c_str());
        printf("%-28s %-10s %-7s %10.3f %10.3f\n", "TOTAL", "", "", tcpu, tgpu);
        printf("\nPer op-type (cpu ms / gpu ms):\n");
        std::vector<std::pair<std::string, std::pair<double, double>>> v(byType.begin(), byType.end());
        std::sort(v.begin(), v.end(), [](auto &a, auto &b) {
            return a.second.second > b.second.second;
        });
        for (auto &kv: v)
        {
            printf("  %-22s %8.3f / %8.3f\n", kv.first.c_str(), kv.second.first, kv.second.second);
        }
    }

    std::string Profiler::toJson() const {
        std::ostringstream os;
        os << "[";
        for (size_t i = 0; i < records_.size(); ++i)
        {
            const auto &r = records_[i];
            os << "{\"name\":\"" << r.name << "\",\"type\":\"" << opTypeName(r.type) << "\",\"backend\":\"" << r.backend << "\",\"cpuMs\":" << r.cpuMs << ",\"gpuMs\":" << r.gpuMs << ",\"dispatch\":[" << r.dispatch[0] << "," << r.dispatch[1] << "," << r.dispatch[2] << "],\"fellBack\":" << (r.fellBack ? "true" : "false") << "}";
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
            double dur = (r.gpuMs >= 0 ? r.gpuMs : r.cpuMs) * 1000.0; // us
            if (!first)
            {
                f << ",";
            }
            first = false;
            f << "{\"name\":\"" << r.name << "\",\"cat\":\"" << opTypeName(r.type) << "\",\"ph\":\"X\",\"pid\":1,\"tid\":1,\"ts\":" << ts << ",\"dur\":" << dur << ",\"args\":{\"backend\":\"" << r.backend << "\",\"cpuMs\":" << r.cpuMs << ",\"gpuMs\":" << r.gpuMs << "}}";
            ts += dur;
        }
        f << "],\"displayTimeUnit\":\"ms\"}";
    }

} // namespace vknn
