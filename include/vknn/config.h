// Runtime config (the knobs MNN exposes were the starting point). Plain struct + a JSON loader.
// Every field is documented in docs/CONFIG.md.
#pragma once
#include <string>
#include <vector>

#include "vknn/common.h"
#include "vknn/tensor_format.h"

namespace vknn {

enum class BackendKind { kVulkan = 0, kCpu = 1 };
enum class Precision { kFp32 = 0, kFp16 = 1, kAuto = 2 };
enum class PowerHint { kNormal = 0, kHigh = 1, kLow = 2 };
enum class TuningLevel { kOff = 0, kFast = 1, kThorough = 2 };
// How the 3x3-conv Winograd F(2,3) kernel is selected. kAuto (the recommended, best+fast default)
// measures the tiled-GEMM Winograd against the direct kernel per shape and keeps the faster; kOn
// forces Winograd on every eligible 3x3; kOff always uses the direct kernel.
enum class WinogradMode { kAuto = 0, kOn = 1, kOff = 2 };

// Advanced kernel-selection hints (MNN-style Config::setHint). The defaults (value 0) are the
// best/fast production kernels; non-zero values select experimental or research variants and are
// not needed for normal use. There are no environment variables — this is the only way to reach them.
enum class Hint {
  // Winograd matmul variant: 0 = tiled-GEMM (default/best), 1 = fused, 2 = fused-split, 3 = fully-fused.
  kWinogradVariant = 0,
  // Winograd output-tile size: 0 = F(2,3) (default), 4 = force F(4,3) (0.56x V/M traffic, 4x FLOP
  // saving, but register-heavy 6x6 transforms — usually slower on this GPU; for research).
  kWinogradUnit = 1,
  // Direct 3x3 kernel: 0 = autotuned (default), 1 = register-tiled (conv_reg), 2 = LDS input-halo.
  kDirectConv3x3 = 2,
};

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

  // Optimization / debug. These replace the old VKNN_* env vars entirely — there are NO environment
  // variables in the engine; everything is configured here (or via setHint above).
  int optLevel = 3;        // graph optimization level 0..3 (fusions). 0 = none.
  bool noFlatOps = false;  // disable the flat-layout GPU pass (was VKNN_NO_FLAT_OPS)
  bool timing = false;     // print pack/submit/unpack + per-stage timing (was VKNN_TIMING)
  bool debugSegments = false;            // trace per-segment + per-CPU-op execution (was VKNN_DEBUG_SEG)
  std::string disableVkOps;              // comma list of op types to force onto CPU (was VKNN_DISABLE_VK_OPS)
  std::string dumpTensors;               // comma list of tensor names to dump to disk (was VKNN_DUMP_NAMES)

  // Profiling / debug.
  bool profile = false;
  int verbosity = 1;  // maps to log level
  bool layerDump = false;
  std::string layerDumpDir = "/data/local/tmp/vxrt/dump";

  // Tuning. kFast (default) measures a few candidates per shape and caches the winner.
  TuningLevel tuning = TuningLevel::kFast;
  // 3x3 Winograd selection. kAuto picks the faster of tiled-GEMM Winograd vs the direct kernel per
  // shape (best+fast). Needs tuning != kOff to measure; with tuning off it stays on the direct kernel.
  WinogradMode winograd = WinogradMode::kAuto;

  // Advanced research/experimental kernel hints (MNN-style). Default = best production kernels.
  // e.g. cfg.setHint(Hint::kWinogradUnit, 4) to force F(4,3). Replaces the old VKNN_* env vars.
  std::vector<int> hints;  // indexed by (int)Hint; 0 = default. Use setHint()/hint().
  void setHint(Hint h, int value) {
    if ((int)h >= (int)hints.size())
      hints.resize((int)h + 1, 0);
    hints[(int)h] = value;
  }
  int hint(Hint h, int dflt = 0) const {
    return (int)h < (int)hints.size() ? hints[(int)h] : dflt;
  }

  static Config fromJsonFile(const std::string& path);
  static Config fromJsonString(const std::string& json);
  std::string toJson() const;
  void applyLogLevel() const;
};

}  // namespace vknn
