// Runtime config: a plain struct of knobs plus a JSON loader. Field set mirrors MNN's config.
// Every field is documented in docs/CONFIG.md.
#pragma once
#include "vknn/backend_kind.h"
#include "vknn/cache_mode.h"
#include "vknn/hint.h"
#include "vknn/precision.h"
#include <string>
#include <vector>

namespace vknn {

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
