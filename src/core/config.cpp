#include "vknn/config.h"
#include "json.h"
#include "vknn/logging.h"
#include <fstream>
#include <sstream>

namespace vknn {

    const char *backendName(BackendKind k) {
        switch (k)
        {
            case BackendKind::Vulkan:
                return "VULKAN";
            case BackendKind::Cpu:
                return "CPU";
        }
        return "?";
    }
    BackendKind backendFromStr(const std::string &s) {
        if (s == "VULKAN" || s == "vulkan")
        {
            return BackendKind::Vulkan;
        }
        return BackendKind::Cpu;
    }
    static Precision precFromStr(const std::string &s) {
        if (s == "fp16" || s == "FP16" || s == "low")
        {
            return Precision::Fp16;
        }
        if (s == "auto")
        {
            return Precision::Auto;
        }
        return Precision::Fp32;
    }
    static const char *precStr(Precision p) {
        return p == Precision::Fp16 ? "fp16" : p == Precision::Auto ? "auto" : "fp32";
    }
    static TensorFormat fmtFromStr(const std::string &s) {
        if (s == "NHWC")
        {
            return TensorFormat::NHWC;
        }
        return TensorFormat::NCHW;
    }
    TuningLevel tuningFromStr(const std::string &s) {
        if (s == "off")
        {
            return TuningLevel::Off;
        }
        if (s == "thorough")
        {
            return TuningLevel::Thorough;
        }
        return TuningLevel::Fast;
    }
    static const char *tuneStr(TuningLevel t) {
        return t == TuningLevel::Off ? "off" : t == TuningLevel::Thorough ? "thorough" : "fast";
    }
    WinogradMode winogradFromStr(const std::string &s) {
        if (s == "on")
        {
            return WinogradMode::On;
        }
        if (s == "off")
        {
            return WinogradMode::Off;
        }
        return WinogradMode::Auto;
    }
    static const char *winoStr(WinogradMode w) {
        return w == WinogradMode::On ? "on" : w == WinogradMode::Off ? "off" : "auto";
    }

    Config Config::fromJsonFile(const std::string &path) {
        std::ifstream f(path);
        if (!f)
        {
            VKNN_WARN << "config file not found: " << path << " (using defaults)";
            return {};
        }
        std::stringstream ss;
        ss << f.rdbuf();
        return fromJsonString(ss.str());
    }

    Config Config::fromJsonString(const std::string &json) {
        Config    c;
        JsonValue v = JsonParser::parse(json);
        if (!v.isObject())
        {
            return c;
        }
        auto S = [&](const char *k, std::string &dst) {
            if (auto *j = v.get(k))
            {
                dst = j->asStr(dst);
            }
        };
        auto B = [&](const char *k, bool &dst) {
            if (auto *j = v.get(k))
            {
                dst = j->asBool(dst);
            }
        };
        auto I = [&](const char *k, int &dst) {
            if (auto *j = v.get(k))
            {
                dst = (int) j->asNum(dst);
            }
        };

        if (auto *j = v.get("backend"))
        {
            c.backend = backendFromStr(j->asStr("VULKAN"));
        }
        if (auto *j = v.get("fallback"))
        {
            c.fallback.clear();
            if (j->type == JsonValue::kArray)
            {
                for (auto &e: j->arr)
                {
                    c.fallback.push_back(backendFromStr(e.asStr()));
                }
            }
        }
        B("allowCpuFallback", c.allowCpuFallback);
        if (auto *j = v.get("precision"))
        {
            c.precision = precFromStr(j->asStr("fp16"));
        }
        if (auto *j = v.get("inputLayout"))
        {
            c.inputLayout = fmtFromStr(j->asStr("NCHW"));
        }
        if (auto *j = v.get("outputLayout"))
        {
            c.outputLayout = fmtFromStr(j->asStr("NCHW"));
        }
        I("cpuThreads", c.cpuThreads);
        I("maxSubmitNodes", c.maxSubmitNodes);
        S("cacheFile", c.cacheFile);
        S("cacheDir", c.cacheDir);
        B("cachePipeline", c.cachePipeline);
        B("cacheWeights", c.cacheWeights);
        B("cacheTuning", c.cacheTuning);
        B("profile", c.profile);
        I("verbosity", c.verbosity);
        B("layerDump", c.layerDump);
        S("layerDumpDir", c.layerDumpDir);
        if (auto *j = v.get("tuning"))
        {
            c.tuning = tuningFromStr(j->asStr("fast"));
        }
        if (auto *j = v.get("winograd"))
        {
            c.winograd = winogradFromStr(j->asStr("auto"));
        }
        B("debugSegments", c.debugSegments);
        S("disableVkOps", c.disableVkOps);
        S("dumpTensors", c.dumpTensors);
        if (auto *j = v.get("hints"))
        { // {"hints": [v0, v1, ...]} indexed by (int)Hint
            if (j->type == JsonValue::kArray)
            {
                for (auto &e: j->arr)
                {
                    c.hints.push_back((int) e.asNum(0));
                }
            }
        }
        if (auto *j = v.get("power"))
        {
            std::string p = j->asStr("normal");
            c.power       = p == "high" ? PowerHint::High : p == "low" ? PowerHint::Low : PowerHint::Normal;
        }
        return c;
    }

    std::string Config::toJson() const {
        std::ostringstream os;
        os << "{\n";
        os << "  \"backend\": \"" << backendName(backend) << "\",\n";
        os << "  \"fallback\": [";
        for (size_t i = 0; i < fallback.size(); ++i)
        {
            os << "\"" << backendName(fallback[i]) << "\"" << (i + 1 < fallback.size() ? ", " : "");
        }
        os << "],\n";
        os << "  \"allowCpuFallback\": " << (allowCpuFallback ? "true" : "false") << ",\n";
        os << "  \"precision\": \"" << precStr(precision) << "\",\n";
        os << "  \"power\": \"" << (power == PowerHint::High ? "high" : power == PowerHint::Low ? "low" : "normal") << "\",\n";
        os << "  \"cpuThreads\": " << cpuThreads << ",\n";
        os << "  \"maxSubmitNodes\": " << maxSubmitNodes << ",\n";
        os << "  \"inputLayout\": \"" << formatStr(inputLayout) << "\",\n";
        os << "  \"outputLayout\": \"" << formatStr(outputLayout) << "\",\n";
        os << "  \"cacheFile\": \"" << cacheFile << "\",\n";
        os << "  \"cacheDir\": \"" << cacheDir << "\",\n";
        os << "  \"cachePipeline\": " << (cachePipeline ? "true" : "false") << ",\n";
        os << "  \"cacheWeights\": " << (cacheWeights ? "true" : "false") << ",\n";
        os << "  \"cacheTuning\": " << (cacheTuning ? "true" : "false") << ",\n";
        os << "  \"profile\": " << (profile ? "true" : "false") << ",\n";
        os << "  \"verbosity\": " << verbosity << ",\n";
        os << "  \"layerDump\": " << (layerDump ? "true" : "false") << ",\n";
        os << "  \"layerDumpDir\": \"" << layerDumpDir << "\",\n";
        os << "  \"timing\": " << (timing ? "true" : "false") << ",\n";
        os << "  \"debugSegments\": " << (debugSegments ? "true" : "false") << ",\n";
        os << "  \"tuning\": \"" << tuneStr(tuning) << "\",\n";
        os << "  \"winograd\": \"" << winoStr(winograd) << "\"\n";
        os << "}\n";
        return os.str();
    }

    void Config::applyLogLevel() const {
        switch (verbosity)
        {
            case 0:
                Log::setLevel(LogLevel::Warn);
                break;
            case 1:
                Log::setLevel(LogLevel::Info);
                break;
            default:
                Log::setLevel(LogLevel::Debug);
                break;
        }
    }

} // namespace vknn
