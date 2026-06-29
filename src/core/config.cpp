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
    Precision precisionFromStr(const std::string &s) {
        if (s == "low" || s == "fp16" || s == "FP16")
        {
            return Precision::Low;
        }
        if (s == "normal" || s == "mixed")
        {
            return Precision::Normal;
        }
        if (s == "high" || s == "fp32" || s == "FP32")
        {
            return Precision::High;
        }
        if (s == "auto")
        {
            return Precision::Auto;
        }
        return Precision::Low;
    }
    static const char *precStr(Precision p) {
        switch (p)
        {
            case Precision::Low:
                return "low";
            case Precision::Normal:
                return "normal";
            case Precision::High:
                return "high";
            default:
                return "auto";
        }
    }
    const char *mixedPrecisionFp32Tensors() {
        // The geometry tail of a feed-forward-3DGS encoder (build_covariance matmuls, the world/means
        // einsum transforms, the scale/quaternion adapter chain, the camera feature MLP) lifted to fp32
        // storage. Excludes the camera-pose SVD (Newton-Schulz), which diverges to NaN in fp32 on marginal
        // poses. Matches by name substring, so it is a no-op for models without these tensors.
        return "/enc/MatMul_,/enc/Einsum,/enc/Mul_,/enc/Add_,/enc/Sub_,/enc/Reshape_,/enc/Slice_,"
               "/enc/Transpose_,/enc/Concat_,/enc/Squeeze,/enc/Split_,/enc/Clip,/enc/Softplus,/enc/Exp_,"
               "/enc/Neg,/enc/Reciprocal,/enc/ScatterND,/enc/camera_head/res_conv,/enc/camera_head/more_mlps";
    }
    Mode tuningFromStr(const std::string &s) {
        if (s == "off")
        {
            return Mode::NoTune;
        }
        if (s == "thorough")
        {
            return Mode::Thorough;
        }
        return Mode::Fast;
    }
    static const char *tuneStr(Mode t) {
        return t == Mode::NoTune ? "off" : t == Mode::Thorough ? "thorough" : "fast";
    }
    Mode winogradFromStr(const std::string &s) {
        if (s == "on")
        {
            return Mode::On;
        }
        if (s == "off")
        {
            return Mode::Off;
        }
        return Mode::Auto;
    }
    static const char *winoStr(Mode w) {
        return w == Mode::On ? "on" : w == Mode::Off ? "off" : "auto";
    }
    CacheMode cacheModeFromStr(const std::string &s) {
        if (s == "off")
        {
            return CacheMode::Off;
        }
        if (s == "tune")
        {
            return CacheMode::Tune;
        }
        return CacheMode::Full;
    }
    const char *cacheModeStr(CacheMode m) {
        return m == CacheMode::Off ? "off" : m == CacheMode::Tune ? "tune" : "full";
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
            c.precision = precisionFromStr(j->asStr("fp16"));
        }
        I("maxSubmitNodes", c.maxSubmitNodes);
        S("cacheFile", c.cacheFile);
        S("cacheDir", c.cacheDir);
        if (auto *j = v.get("cacheMode"))
        {
            c.cacheMode = cacheModeFromStr(j->asStr("full"));
        }
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
            c.setHint(Hint::Winograd, winogradFromStr(j->asStr("auto")));
        }
        if (auto *j = v.get("tuning"))
        {
            c.setHint(Hint::Tuning, tuningFromStr(j->asStr("fast")));
        }
        if (auto *j = v.get("winogradVariant"))
        {
            c.setHint(Hint::WinogradVariant, (int) j->asNum(0));
        }
        if (auto *j = v.get("winogradUnit"))
        {
            c.setHint(Hint::WinogradUnit, (int) j->asNum(0));
        }
        if (auto *j = v.get("directConv3x3"))
        {
            c.setHint(Hint::DirectConv3x3, (int) j->asNum(0));
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
        os << "  \"cacheMode\": \"" << cacheModeStr(cacheMode) << "\",\n";
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
        os << "  \"winograd\": \"" << winoStr((Mode) hint(Hint::Winograd, (int) Mode::Auto)) << "\",\n";
        os << "  \"tuning\": \"" << tuneStr((Mode) hint(Hint::Tuning, (int) Mode::Fast)) << "\",\n";
        os << "  \"winogradVariant\": " << hint(Hint::WinogradVariant, 0) << ",\n";
        os << "  \"winogradUnit\": " << hint(Hint::WinogradUnit, 0) << ",\n";
        os << "  \"directConv3x3\": " << hint(Hint::DirectConv3x3, 0) << "\n";
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
