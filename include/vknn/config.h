// Runtime config: a plain struct of knobs plus a JSON loader. Field set mirrors MNN's config.
// Every field is documented in docs/CONFIG.md.
#pragma once
#include "vknn/common.h"
#include "vknn/tensor_format.h"
#include <string>
#include <vector>

namespace vknn {

    enum class BackendKind { Vulkan = 0, Cpu = 1 };
    // Quality tiers (string tokens "low" / "normal" / "high"; "fp16" / "fp32" kept as aliases):
    //   Low    fp16 storage + fp32 accumulation everywhere.
    //   Normal fp16 storage, but a built-in geometry-tail set is kept fp32 (selective fp32).
    //   High   full fp32 storage.
    enum class Precision { High = 0, Low = 1, Auto = 2, Normal = 3 };

    // The default selective-fp32 set used by Precision::Normal ("normal") when Config::fp32Tensors is empty:
    // the comma-separated tensor-name substrings of the geometry tail that benefit from fp32 storage
    // without the NaN-fragile camera-pose SVD. A no-op for models without these names (e.g. CNNs).
    const char *mixedPrecisionFp32Tensors();
    // What a warm start reloads from the unified cache file, in increasing order. Off recomputes
    // everything every load; Tune keeps the cheap, deterministic blobs (compiled pipelines + the
    // conv autotune table) but recomputes/re-uploads weights; Full also keeps the prepacked-weight
    // blob for the fastest warm load (and the largest cache file).
    enum class CacheMode { Off = 0, Tune = 1, Full = 2 };
    // Which conv kernel knob a Mode value applies to (MNN-style Config::setHint).
    enum class Hint {
        Winograd        = 0, // 3x3 Winograd selection (Auto / On / Off)
        Tuning          = 1, // autotune effort (NoTune / Fast / Thorough)
        WinogradVariant = 2, // Winograd matmul impl (TiledGemm / Fused / FusedSplit / FullyFused / SubgroupGemm)
        WinogradUnit    = 3, // Winograd output tile (F23 / F43)
        DirectConv3x3   = 4, // direct 3x3 kernel (DirectAuto / RegisterTiled / LdsHalo)
    };

    // Every conv kernel-selection value, set uniformly via setHint(Hint, Mode). Each line is the set of
    // values valid for one Hint; the same underlying int recurs across groups (legal — the Hint picks the
    // knob, the Mode the value). Forcing Winograd On/Off skips per-shape timing, making the choice
    // deterministic run-to-run.
    enum class Mode {
        Auto = 0, On = 1, Off = 2,                                                  // Hint::Winograd
        NoTune = 0, Fast = 1, Thorough = 2,                                         // Hint::Tuning
        TiledGemm = 0, Fused = 1, FusedSplit = 2, FullyFused = 3, SubgroupGemm = 4, // Hint::WinogradVariant
        F23 = 0, F43 = 4,                                                           // Hint::WinogradUnit
        DirectAuto = 0, RegisterTiled = 1, LdsHalo = 2,                             // Hint::DirectConv3x3
    };

    const char *backendName(BackendKind k);
    BackendKind backendFromStr(const std::string &s);
    // Precision tier from a string: "low"/"fp16", "normal"/"mixed", "high"/"fp32", "auto" (unknown -> low).
    Precision precisionFromStr(const std::string &s);
    // Parse the conv autotune knobs from a string. winograd: "auto" (measure per shape), "on" (force
    // Winograd), "off" (force the direct kernel). tuning: "off" / "fast" / "thorough". Forcing winograd
    // on or off makes the 3x3-conv kernel choice deterministic (no per-run timing measurement).
    Mode winogradFromStr(const std::string &s); // "auto"/"on"/"off" -> Mode::Auto/On/Off
    Mode tuningFromStr(const std::string &s);   // "off"/"fast"/"thorough" -> Mode::NoTune/Fast/Thorough
    // Cache scope: "off" / "tune" / "full" -> CacheMode::Off/Tune/Full (and back).
    CacheMode   cacheModeFromStr(const std::string &s);
    const char *cacheModeStr(CacheMode m);

    struct Config {
        // Backend selection + ordered fallback list (CPU is always an implicit final fallback).
        BackendKind              backend          = BackendKind::Vulkan;
        std::vector<BackendKind> fallback         = {BackendKind::Cpu};
        bool                     allowCpuFallback = true;

        Precision precision = Precision::Low;

        // Caches. The unified per-model cache file bundles the compiled-pipeline blob and the
        // prepacked-weight + autotune blob; loading it skips shader compilation, conv autotuning, and
        // the Winograd weight transform on a warm start. Set via the Runtime::load() cacheFile argument
        // (empty there -> "<model>.cache" next to the model). cacheMode selects what the file includes
        // (Off / Tune / Full). cacheDir is the fallback location for sessions built from an in-memory
        // graph (no model path).
        std::string cacheFile; // unified cache path (resolved by Runtime::load; empty = no file cache)
        std::string cacheDir  = "/data/local/tmp/vxrt/cache";
        CacheMode   cacheMode = CacheMode::Full;

        // What the cacheMode retains, as predicates the backend reads directly.
        bool cachesPipeline() const { return cacheMode != CacheMode::Off; }
        bool cachesTuning() const { return cacheMode != CacheMode::Off; }
        bool cachesWeights() const { return cacheMode == CacheMode::Full; }

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
        bool        noFlatOps     = false; // disable the flat-layout GPU pass
        bool        timing        = false; // print pack/submit/unpack + per-stage timing
        bool        debugSegments = false; // trace per-segment + per-CPU-op execution
        std::string disableVkOps;          // comma list of op types to force onto CPU
        std::string dumpTensors;           // comma list of tensor names to dump to disk

        // Advanced override of the selective-fp32 set: comma list of tensor-name substrings (leading '-'
        // excludes) whose activations are kept in fp32 storage even when the segment runs fp16. Empty +
        // Precision::Normal uses the built-in mixedPrecisionFp32Tensors() preset; a non-empty value replaces
        // it (and also applies under Precision::Low). The markFp32 pass marks matching flat tensors and
        // bridges the fp16/fp32 frontier with convert_dtype nodes. Only affects accuracy/runtime.
        std::string fp32Tensors;

        // Profiling / debug.
        bool        profile      = false;
        int         verbosity    = 1; // maps to log level
        bool        layerDump    = false;
        std::string layerDumpDir = "/data/local/tmp/vxrt/dump";

        // Conv kernel selection + autotune effort, set via setHint(Hint, value) (see the Hint enum):
        // Hint::Winograd (auto/on/off), Hint::Tuning (off/fast/thorough), plus the experimental variant
        // hints. Forcing Winograd on/off makes the 3x3-conv choice deterministic.
        std::vector<int> hints; // indexed by (int)Hint; 0 = production default. Use setHint()/hint().
        void             setHint(Hint h, int value) {
            if ((int) h >= (int) hints.size())
            {
                hints.resize((int) h + 1, 0);
            }
            hints[(int) h] = value;
        }
        // Typed overload so a caller passes the enum directly: setHint(Hint::Winograd, Mode::Off).
        void setHint(Hint h, Mode v) {
            setHint(h, (int) v);
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
