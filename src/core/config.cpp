#include "vknn/config.h"
#include "json.h"
#include "vknn/logging.h"
#include <cctype>
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
        return Precision::Low;
    }
    static const char *precStr(Precision p) {
        switch (p)
        {
            case Precision::Normal:
                return "normal";
            case Precision::High:
                return "high";
            case Precision::Low:
                break;
        }
        return "low";
    }
    Priority priorityFromStr(const std::string &s) {
        if (s == "low")
        {
            return Priority::Low;
        }
        if (s == "high")
        {
            return Priority::High;
        }
        return Priority::Normal;
    }
    static const char *priorityStr(Priority p) {
        switch (p)
        {
            case Priority::Low:
                return "low";
            case Priority::High:
                return "high";
            case Priority::Normal:
                break;
        }
        return "normal";
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
    Tuning tuningFromStr(const std::string &s) {
        if (s == "none" || s == "off") // "off" = legacy alias for the former --tuning knob
        {
            return Tuning::None;
        }
        if (s == "heavy" || s == "thorough") // "thorough" = legacy alias
        {
            return Tuning::Heavy;
        }
        return Tuning::Fast;
    }
    static const char *tuningStr(Tuning t) {
        return t == Tuning::None ? "none" : t == Tuning::Heavy ? "heavy" : "fast";
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

    /// Overlays the keys present in @p json onto a default-constructed Config; any key not present
    /// leaves its field at the default (the S/B/I helpers only assign when the key exists). A JSON
    /// value that is not an object yields the all-defaults Config.
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
        if (auto *j = v.get("priority"))
        {
            c.priority = priorityFromStr(j->asStr("normal"));
        }
        I("maxSubmitNodes", c.maxSubmitNodes);
        I("maxSubmitBindings", c.maxSubmitBindings);
        I("decodeChainSteps", c.decodeChainSteps);
        S("cacheFile", c.cacheFile);
        S("cacheDir", c.cacheDir);
        B("noCache", c.noCache);
        if (v.get("cacheMode"))
        {
            VKNN_WARN << "config key 'cacheMode' is obsolete and ignored (caching is always on now; use 'tuning' for autotune effort)";
        }
        if (auto *j = v.get("tuning"))
        {
            c.tuning = tuningFromStr(j->asStr("fast"));
        }
        I("cpuThreads", c.cpuThreads);
        B("freeWeightsAfterUpload", c.freeWeightsAfterUpload);
        // Flat-layout pass + GPU-island folding are hints now (default On). Accept the new keys
        // (flatLayout / gpuIslandFold) and the legacy booleans. noFlatOps is the negation of
        // flatLayout: noFlatOps=true disables the pass (FlatLayout Off).
        if (auto *j = v.get("flatLayout"))
        {
            c.setHint(Hint::FlatLayout, j->asBool(true) ? (int) Mode::On : (int) Mode::Off);
        }
        if (auto *j = v.get("noFlatOps"))
        {
            c.setHint(Hint::FlatLayout, j->asBool(false) ? (int) Mode::Off : (int) Mode::On);
        }
        if (auto *j = v.get("gpuIslandFold"))
        {
            c.setHint(Hint::GpuIslandFold, j->asBool(true) ? (int) Mode::On : (int) Mode::Off);
        }
        if (auto *j = v.get("foldGpuIslands"))
        {
            c.setHint(Hint::GpuIslandFold, j->asBool(true) ? (int) Mode::On : (int) Mode::Off);
        }
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
        if (auto *j = v.get("kvCacheQuant"))
        { // auto/on/off; the parser is the shared Mode-triple reader (winogradFromStr)
            c.setHint(Hint::KvCacheQuant, winogradFromStr(j->asStr("auto")));
        }
        return c;
    }

    /// Serializes the effective config as JSON parseable by fromJsonString (round-trips). Emits the
    /// resolved hint-derived values (flatLayout(), the Winograd/conv hints) rather than the raw hints
    /// vector, and only the keys fromJsonString reads back.
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
        os << "  \"priority\": \"" << priorityStr(priority) << "\",\n";
        os << "  \"maxSubmitNodes\": " << maxSubmitNodes << ",\n";
        os << "  \"maxSubmitBindings\": " << maxSubmitBindings << ",\n";
        os << "  \"decodeChainSteps\": " << decodeChainSteps << ",\n";
        os << "  \"cacheFile\": \"" << cacheFile << "\",\n";
        os << "  \"cacheDir\": \"" << cacheDir << "\",\n";
        os << "  \"noCache\": " << (noCache ? "true" : "false") << ",\n";
        os << "  \"tuning\": \"" << tuningStr(tuning) << "\",\n";
        os << "  \"cpuThreads\": " << cpuThreads << ",\n";
        os << "  \"freeWeightsAfterUpload\": " << (freeWeightsAfterUpload ? "true" : "false") << ",\n";
        os << "  \"flatLayout\": " << (flatLayout() ? "true" : "false") << ",\n";
        os << "  \"gpuIslandFold\": " << (gpuIslandFold() ? "true" : "false") << ",\n";
        os << "  \"timing\": " << (timing ? "true" : "false") << ",\n";
        os << "  \"profile\": " << (profile ? "true" : "false") << ",\n";
        os << "  \"verbosity\": " << verbosity << ",\n";
        os << "  \"layerDump\": " << (layerDump ? "true" : "false") << ",\n";
        os << "  \"layerDumpDir\": \"" << layerDumpDir << "\",\n";
        os << "  \"debugSegments\": " << (debugSegments ? "true" : "false") << ",\n";
        os << "  \"disableVkOps\": \"" << disableVkOps << "\",\n";
        os << "  \"dumpTensors\": \"" << dumpTensors << "\",\n";
        os << "  \"winograd\": \"" << winoStr((Mode) hint(Hint::Winograd, (int) Mode::Auto)) << "\",\n";
        os << "  \"winogradVariant\": " << hint(Hint::WinogradVariant, 0) << ",\n";
        os << "  \"winogradUnit\": " << hint(Hint::WinogradUnit, 0) << ",\n";
        os << "  \"directConv3x3\": " << hint(Hint::DirectConv3x3, 0) << ",\n";
        os << "  \"kvCacheQuant\": \"" << winoStr((Mode) kvCacheQuantMode()) << "\"\n";
        os << "}\n";
        return os.str();
    }

    /// True when @p name appears as one whole comma-separated entry of @p list, ignoring surrounding
    /// whitespace on each entry (so "a, b ,c" contains "b"). Substring hits do not count.
    bool Config::listContains(const std::string &list, std::string_view name) {
        if (name.empty())
        {
            return false;
        }
        size_t pos = 0;
        while (pos <= list.size())
        {
            size_t comma = list.find(',', pos);
            size_t end   = comma == std::string::npos ? list.size() : comma;
            // Trim leading/trailing whitespace to get this entry's [tokBegin, tokEnd) span.
            size_t tokBegin = pos;
            size_t tokEnd   = end;
            while (tokBegin < tokEnd && std::isspace((unsigned char) list[tokBegin]))
            {
                ++tokBegin;
            }
            while (tokEnd > tokBegin && std::isspace((unsigned char) list[tokEnd - 1]))
            {
                --tokEnd;
            }
            if (tokEnd - tokBegin == name.size() && list.compare(tokBegin, tokEnd - tokBegin, name.data(), name.size()) == 0)
            {
                return true;
            }
            if (comma == std::string::npos)
            {
                break;
            }
            pos = comma + 1;
        }
        return false;
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
