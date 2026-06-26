// Runtime config (the knobs MNN exposes were the starting point). Plain struct + a JSON loader.
// Every field is documented in docs/CONFIG.md.
#pragma once
#include <string>
#include <vector>

#include "vknn/common.h"
#include "vknn/tensor_format.h"

namespace vknn {

enum class BackendKind { kVulkan = 0, kCpu = 1, kEnn = 2 };
enum class Precision { kFp32 = 0, kFp16 = 1, kAuto = 2 };
enum class PowerHint { kNormal = 0, kHigh = 1, kLow = 2 };
enum class TuningLevel { kOff = 0, kFast = 1, kThorough = 2 };

const char* backendName(BackendKind k);
BackendKind backendFromStr(const std::string& s);

struct Config {
  // Backend selection + ordered fallback list (CPU is always an implicit final fallback).
  BackendKind backend = BackendKind::kVulkan;
  std::vector<BackendKind> fallback = {BackendKind::kCpu};
  bool allowCpuFallback = true;

  Precision precision = Precision::kFp16;
  PowerHint power = PowerHint::kNormal;
  int cpuThreads = 4;

  // I/O layout the user supplies / wants back (engine converts internally).
  TensorFormat inputLayout = TensorFormat::kNCHW;
  TensorFormat outputLayout = TensorFormat::kNCHW;

  // Zero-copy (ION / dma-buf).
  bool enableZeroCopy = false;

  // Caches.
  std::string cacheDir = "/data/local/tmp/vxrt/cache";
  bool cachePipeline = true;
  bool cacheWeights = true;
  bool cacheTuning = true;

  // Free host weight buffers after they are uploaded to the device / decoded into the pool. run()
  // never reads graph initializers (it uses GPU buffers + the pool), so this is safe and reclaims
  // the full weight blob — needed to fit large (e.g. 965M-param fp16) models on-device.
  bool freeWeightsAfterUpload = true;

  // Optimization / debug (replace the old VKNN_* env vars; set from binary flags).
  int optLevel = 3;        // graph optimization level 0..3 (fusions). 0 = none.
  bool noFlatOps = false;  // disable the flat-layout GPU pass (was VKNN_NO_FLAT_OPS)
  bool timing = false;     // print pack/submit/unpack timing (was VKNN_TIMING)

  // Profiling / debug.
  bool profile = false;
  int verbosity = 1;  // maps to log level
  bool layerDump = false;
  std::string layerDumpDir = "/data/local/tmp/vxrt/dump";

  // Tuning.
  TuningLevel tuning = TuningLevel::kFast;

  static Config fromJsonFile(const std::string& path);
  static Config fromJsonString(const std::string& json);
  std::string toJson() const;
  void applyLogLevel() const;
};

}  // namespace vknn
