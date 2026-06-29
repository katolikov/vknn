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
        I("maxSubmitNodes", c.maxSubmitNodes);
        S("cacheFile", c.cacheFile);
        S("cacheDir", c.cacheDir);
        B("cachePipeline", c.cachePipeline);
        B("cacheWeights", c.cacheWeights);
        B("cacheTuning", c.cacheTuning);
        B("freeWeightsAfterUpload", c.freeWeightsAfterUpload);
        B("noFlatOps", c.noFlatOps);
        B("timing", c.timing);
        B("profile", c.profile);
        I("verbosity", c.verbosity);
        B("layerDump", c.layerDump);
        S("layerDumpDir", c.layerDumpDir);
        B("debugSegments", c.debugSegments);
        S("disableVkOps", c.disableVkOps);
        S("dumpTensors", c.dumpTensors);
        if (auto *j = v.get("hints"))
        { // raw array indexed by (int)Hint; the named keys below override specific entries
            if (j->type == JsonValue::kArray)
            {
                for (size_t i = 0; i < j->arr.size(); ++i)
                {
                    c.setHint((Hint) i, (int) j->arr[i].asNum(0));
                }
            }
        }
        if (auto *j = v.get("winograd"))
        {
            c.setHint(Hint::Winograd, (int) winogradFromStr(j->asStr("auto")));
        }
        if (auto *j = v.get("tuning"))
        {
            c.setHint(Hint::Tuning, (int) tuningFromStr(j->asStr("fast")));
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
        os << "  \"maxSubmitNodes\": " << maxSubmitNodes << ",\n";
        os << "  \"cacheFile\": \"" << cacheFile << "\",\n";
        os << "  \"cacheDir\": \"" << cacheDir << "\",\n";
        os << "  \"cachePipeline\": " << (cachePipeline ? "true" : "false") << ",\n";
        os << "  \"cacheWeights\": " << (cacheWeights ? "true" : "false") << ",\n";
        os << "  \"cacheTuning\": " << (cacheTuning ? "true" : "false") << ",\n";
        os << "  \"freeWeightsAfterUpload\": " << (freeWeightsAfterUpload ? "true" : "false") << ",\n";
        os << "  \"noFlatOps\": " << (noFlatOps ? "true" : "false") << ",\n";
        os << "  \"timing\": " << (timing ? "true" : "false") << ",\n";
        os << "  \"profile\": " << (profile ? "true" : "false") << ",\n";
        os << "  \"verbosity\": " << verbosity << ",\n";
        os << "  \"layerDump\": " << (layerDump ? "true" : "false") << ",\n";
        os << "  \"layerDumpDir\": \"" << layerDumpDir << "\",\n";
        os << "  \"debugSegments\": " << (debugSegments ? "true" : "false") << ",\n";
        os << "  \"disableVkOps\": \"" << disableVkOps << "\",\n";
        os << "  \"dumpTensors\": \"" << dumpTensors << "\",\n";
        os << "  \"winograd\": \"" << winoStr((WinogradMode) hint(Hint::Winograd, (int) WinogradMode::Auto)) << "\",\n";
        os << "  \"tuning\": \"" << tuneStr((TuningLevel) hint(Hint::Tuning, (int) TuningLevel::Fast)) << "\"\n";
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
