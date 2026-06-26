#include "vx/config.h"

#include <fstream>
#include <sstream>

#include "json.h"
#include "vx/logging.h"

namespace vx {

const char* backendName(BackendKind k) {
  switch (k) {
    case BackendKind::kVulkan:
      return "VULKAN";
    case BackendKind::kCpu:
      return "CPU";
    case BackendKind::kEnn:
      return "ENN";
  }
  return "?";
}
BackendKind backendFromStr(const std::string& s) {
  if (s == "VULKAN" || s == "vulkan")
    return BackendKind::kVulkan;
  if (s == "ENN" || s == "enn")
    return BackendKind::kEnn;
  return BackendKind::kCpu;
}
static Precision precFromStr(const std::string& s) {
  if (s == "fp16" || s == "FP16" || s == "low")
    return Precision::kFp16;
  if (s == "auto")
    return Precision::kAuto;
  return Precision::kFp32;
}
static const char* precStr(Precision p) {
  return p == Precision::kFp16 ? "fp16" : p == Precision::kAuto ? "auto" : "fp32";
}
static TensorFormat fmtFromStr(const std::string& s) {
  if (s == "NHWC")
    return TensorFormat::kNHWC;
  return TensorFormat::kNCHW;
}
static TuningLevel tuneFromStr(const std::string& s) {
  if (s == "off")
    return TuningLevel::kOff;
  if (s == "thorough")
    return TuningLevel::kThorough;
  return TuningLevel::kFast;
}
static const char* tuneStr(TuningLevel t) {
  return t == TuningLevel::kOff ? "off" : t == TuningLevel::kThorough ? "thorough" : "fast";
}

Config Config::fromJsonFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    VX_WARN << "config file not found: " << path << " (using defaults)";
    return {};
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return fromJsonString(ss.str());
}

Config Config::fromJsonString(const std::string& json) {
  Config c;
  JsonValue v = JsonParser::parse(json);
  if (!v.isObject())
    return c;
  auto S = [&](const char* k, std::string& dst) {
    if (auto* j = v.get(k))
      dst = j->asStr(dst);
  };
  auto B = [&](const char* k, bool& dst) {
    if (auto* j = v.get(k))
      dst = j->asBool(dst);
  };
  auto I = [&](const char* k, int& dst) {
    if (auto* j = v.get(k))
      dst = (int)j->asNum(dst);
  };

  if (auto* j = v.get("backend"))
    c.backend = backendFromStr(j->asStr("VULKAN"));
  if (auto* j = v.get("fallback")) {
    c.fallback.clear();
    if (j->type == JsonValue::kArray)
      for (auto& e : j->arr)
        c.fallback.push_back(backendFromStr(e.asStr()));
  }
  B("allowCpuFallback", c.allowCpuFallback);
  if (auto* j = v.get("precision"))
    c.precision = precFromStr(j->asStr("fp16"));
  if (auto* j = v.get("inputLayout"))
    c.inputLayout = fmtFromStr(j->asStr("NCHW"));
  if (auto* j = v.get("outputLayout"))
    c.outputLayout = fmtFromStr(j->asStr("NCHW"));
  I("cpuThreads", c.cpuThreads);
  B("enableZeroCopy", c.enableZeroCopy);
  S("cacheDir", c.cacheDir);
  B("cachePipeline", c.cachePipeline);
  B("cacheWeights", c.cacheWeights);
  B("cacheTuning", c.cacheTuning);
  B("profile", c.profile);
  I("verbosity", c.verbosity);
  B("layerDump", c.layerDump);
  S("layerDumpDir", c.layerDumpDir);
  if (auto* j = v.get("tuning"))
    c.tuning = tuneFromStr(j->asStr("fast"));
  if (auto* j = v.get("power")) {
    std::string p = j->asStr("normal");
    c.power = p == "high" ? PowerHint::kHigh : p == "low" ? PowerHint::kLow : PowerHint::kNormal;
  }
  return c;
}

std::string Config::toJson() const {
  std::ostringstream os;
  os << "{\n";
  os << "  \"backend\": \"" << backendName(backend) << "\",\n";
  os << "  \"fallback\": [";
  for (size_t i = 0; i < fallback.size(); ++i)
    os << "\"" << backendName(fallback[i]) << "\"" << (i + 1 < fallback.size() ? ", " : "");
  os << "],\n";
  os << "  \"allowCpuFallback\": " << (allowCpuFallback ? "true" : "false") << ",\n";
  os << "  \"precision\": \"" << precStr(precision) << "\",\n";
  os << "  \"power\": \""
     << (power == PowerHint::kHigh  ? "high"
         : power == PowerHint::kLow ? "low"
                                    : "normal")
     << "\",\n";
  os << "  \"cpuThreads\": " << cpuThreads << ",\n";
  os << "  \"inputLayout\": \"" << formatStr(inputLayout) << "\",\n";
  os << "  \"outputLayout\": \"" << formatStr(outputLayout) << "\",\n";
  os << "  \"enableZeroCopy\": " << (enableZeroCopy ? "true" : "false") << ",\n";
  os << "  \"cacheDir\": \"" << cacheDir << "\",\n";
  os << "  \"cachePipeline\": " << (cachePipeline ? "true" : "false") << ",\n";
  os << "  \"cacheWeights\": " << (cacheWeights ? "true" : "false") << ",\n";
  os << "  \"cacheTuning\": " << (cacheTuning ? "true" : "false") << ",\n";
  os << "  \"profile\": " << (profile ? "true" : "false") << ",\n";
  os << "  \"verbosity\": " << verbosity << ",\n";
  os << "  \"layerDump\": " << (layerDump ? "true" : "false") << ",\n";
  os << "  \"layerDumpDir\": \"" << layerDumpDir << "\",\n";
  os << "  \"tuning\": \"" << tuneStr(tuning) << "\"\n";
  os << "}\n";
  return os.str();
}

void Config::applyLogLevel() const {
  switch (verbosity) {
    case 0:
      Log::setLevel(LogLevel::kWarn);
      break;
    case 1:
      Log::setLevel(LogLevel::kInfo);
      break;
    default:
      Log::setLevel(LogLevel::kDebug);
      break;
  }
}

}  // namespace vx
