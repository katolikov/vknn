// Runtime config: a plain struct of knobs plus a JSON loader. Field set mirrors MNN's config.
// Every field is documented in docs/CONFIG.md.
#pragma once
#include "vknn/common.h"
#include "vknn/tensor_format.h"
#include <string>
#include <vector>

namespace vknn {

    enum class BackendKind { Vulkan = 0, Cpu = 1 };
    enum class Precision { Fp32 = 0, Fp16 = 1, Auto = 2 };
    enum class PowerHint { Normal = 0, High = 1, Low = 2 };
    enum class TuningLevel { Off = 0, Fast = 1, Thorough = 2 };
    // How the 3x3-conv Winograd F(2,3) kernel is selected. Auto measures the tiled-GEMM Winograd
    // against the direct kernel per shape and keeps the faster; On forces Winograd on every eligible
    // 3x3; Off always uses the direct kernel.
    enum class WinogradMode { Auto = 0, On = 1, Off = 2 };

    // Advanced kernel-selection hints (MNN-style Config::setHint). Default (value 0) selects the
    // production kernels; non-zero values select experimental variants.
    enum class Hint {
        // Winograd matmul variant: 0 = tiled-GEMM (default), 1 = fused, 2 = fused-split, 3 = fully-fused.
        WinogradVariant = 0,
        // Winograd output-tile size: 0 = F(2,3) (default), 4 = force F(4,3) (0.56x V/M traffic, 4x FLOP
        // saving, but register-heavy 6x6 transforms — usually slower on this GPU).
        WinogradUnit = 1,
        // Direct 3x3 kernel: 0 = autotuned (default), 1 = register-tiled (conv_reg), 2 = LDS input-halo.
        DirectConv3x3 = 2,
    };

    const char *backendName(BackendKind k);
    BackendKind backendFromStr(const std::string &s);
    // Parse the conv autotune knobs from a string. winograd: "auto" (measure per shape), "on" (force
    // Winograd), "off" (force the direct kernel). tuning: "off" / "fast" / "thorough". Forcing winograd
    // on or off makes the 3x3-conv kernel choice deterministic (no per-run timing measurement).
    WinogradMode winogradFromStr(const std::string &s);
    TuningLevel  tuningFromStr(const std::string &s);

    struct Config {
        // Backend selection + ordered fallback list (CPU is always an implicit final fallback).
        BackendKind              backend          = BackendKind::Vulkan;
        std::vector<BackendKind> fallback         = {BackendKind::Cpu};
        bool                     allowCpuFallback = true;

        Precision precision  = Precision::Fp16;
        PowerHint power      = PowerHint::Normal;
        int       cpuThreads = 4;

        // I/O layout the user supplies / wants back (engine converts internally).
        TensorFormat inputLayout  = TensorFormat::NCHW;
        TensorFormat outputLayout = TensorFormat::NCHW;

        // Caches. The unified per-model cache file bundles the compiled-pipeline blob and the
        // prepacked-weight + autotune blob; loading it skips shader compilation, conv autotuning, and
        // the Winograd weight transform on a warm start. Set via the Runtime::load() cacheFile argument
        // (empty there -> "<model>.cache" next to the model). cachePipeline/cacheWeights/cacheTuning
        // select what the file includes. cacheDir is the fallback location for sessions built from an
        // in-memory graph (no model path).
        std::string cacheFile; // unified cache path (resolved by Runtime::load; empty = no file cache)
        std::string cacheDir      = "/data/local/tmp/vxrt/cache";
        bool        cachePipeline = true;
        bool        cacheWeights  = true;
        bool        cacheTuning   = true;

        // Free host weight buffers after they are uploaded to the device / decoded into the pool. run()
        // never reads graph initializers (it uses GPU buffers + the pool), so this is safe and reclaims
        // the full weight blob — needed to fit large (e.g. 965M-param fp16) models on-device.
        bool freeWeightsAfterUpload = true;

        // Split a GPU segment whose recorded node count exceeds this into chunks of this many nodes,
        // each its own command-buffer submit, so no single submit runs long enough to trip the GPU
        // watchdog (an over-long submit is silently reset by the driver, zeroing its unexecuted tail
        // and corrupting the output). The submit fence between chunks is a full barrier, so buffer
        // reuse stays correct and results are numerically identical. Small graphs (every CNN) stay a
        // single submit. 0 disables chunking. Vulkan exposes no watchdog limit to auto-detect, so this
        // is a tunable knob; the default is conservative and forward-safe (a faster GPU runs each
        // chunk quicker, never slower). Only the very large YoNoSplat-class transformer needs it.
        int maxSubmitNodes = 500;

        // Optimization / debug.
        int         optLevel      = 3;     // graph optimization level 0..3 (fusions). 0 = none.
        bool        noFlatOps     = false; // disable the flat-layout GPU pass
        bool        timing        = false; // print pack/submit/unpack + per-stage timing
        bool        debugSegments = false; // trace per-segment + per-CPU-op execution
        std::string disableVkOps;          // comma list of op types to force onto CPU
        std::string dumpTensors;           // comma list of tensor names to dump to disk

        // Profiling / debug.
        bool        profile      = false;
        int         verbosity    = 1; // maps to log level
        bool        layerDump    = false;
        std::string layerDumpDir = "/data/local/tmp/vxrt/dump";

        // Tuning. Fast (default) measures a few candidates per shape and caches the winner.
        TuningLevel tuning = TuningLevel::Fast;
        // 3x3 Winograd selection. Auto picks the faster of tiled-GEMM Winograd vs the direct kernel per
        // shape. Needs tuning != Off to measure; with tuning off it stays on the direct kernel.
        WinogradMode winograd = WinogradMode::Auto;

        // Advanced experimental kernel hints (MNN-style). Default selects the production kernels.
        // e.g. cfg.setHint(Hint::WinogradUnit, 4) to force F(4,3).
        std::vector<int> hints; // indexed by (int)Hint; 0 = default. Use setHint()/hint().
        void             setHint(Hint h, int value) {
            if ((int) h >= (int) hints.size())
            {
                hints.resize((int) h + 1, 0);
            }
            hints[(int) h] = value;
        }
        int hint(Hint h, int dflt = 0) const {
            return (int) h < (int) hints.size() ? hints[(int) h] : dflt;
        }

        static Config fromJsonFile(const std::string &path);
        static Config fromJsonString(const std::string &json);
        std::string   toJson() const;
        void          applyLogLevel() const;
    };

} // namespace vknn
