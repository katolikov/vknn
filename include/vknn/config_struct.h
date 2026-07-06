/// Runtime config: a plain struct of knobs plus a JSON loader. Field set mirrors MNN's config.
/// Every field is documented in docs/CONFIG.md.
#pragma once
#include "vknn/backend_kind.h"
#include "vknn/hint.h"
#include "vknn/precision.h"
#include "vknn/priority.h"
#include "vknn/tuning.h"
#include <string>
#include <string_view>
#include <vector>

namespace vknn {

    struct Config {
        /// Backend selection + ordered fallback list (CPU is always an implicit final fallback).
        BackendKind              backend          = BackendKind::Vulkan;
        std::vector<BackendKind> fallback         = {BackendKind::Cpu};
        bool                     allowCpuFallback = true;

        Precision precision = Precision::Low;

        /// GPU queue scheduling priority (Vulkan backend). Normal is the driver default and reproduces
        /// the default device-creation path exactly; Low/High request the matching queue global-priority
        /// tier (VK_KHR/EXT_global_priority). Scheduling only — never changes numerical output; inert on a
        /// device without a global-priority extension.
        Priority priority = Priority::Normal;

        /// Caches. Warm-start artifacts (compiled pipelines, prepacked/Winograd weights, the conv autotune
        /// table) are always saved to and reloaded from a per-model cache file, so a warm load skips shader
        /// compilation, weight prepacking, and autotuning. The file is self-validated (kernel hash + device
        /// + model) and multi-variant (one entry per cache-affecting config), so it auto-heals on a
        /// device/driver/model/code change. Set the path via Runtime::load()'s cacheFile argument (empty
        /// there -> "<model>.cache" next to the model); cacheDir is the fallback for a session built from an
        /// in-memory graph (no model path).
        std::string cacheFile;                                  ///< unified cache path (resolved by Runtime::load; empty = no file cache)
        std::string cacheDir  = "/data/local/tmp/vxrt/cache";   ///< fallback cache directory for an in-memory graph with no model path
        bool        noCache   = false;                          ///< debug: skip all cache read/write (cold compile every load)

        /// Load-time conv-kernel autotune effort (None / Fast / Heavy). Effort only — never changes
        /// numerical output; the chosen kernels are cached and reused on a warm start.
        Tuning tuning = Tuning::Fast;

        /// Free host weight buffers after they are uploaded to the device / decoded into the pool. run()
        /// never reads graph initializers (it uses GPU buffers + the pool), so this is safe and reclaims
        /// the full weight blob — needed to fit large (e.g. 965M-param fp16) models on-device.
        bool freeWeightsAfterUpload = true;

        /// Split a GPU segment whose recorded node count exceeds this into chunks of this many nodes,
        /// each its own command-buffer submit, so no single submit runs long enough to trip the GPU
        /// watchdog (an over-long submit is silently reset by the driver, zeroing its unexecuted tail
        /// and corrupting the output). The submit fence between chunks is a full barrier, so buffer
        /// reuse stays correct and results are numerically identical. Small graphs (every CNN) stay a
        /// single submit. 0 disables chunking. Vulkan exposes no watchdog limit to auto-detect, so this
        /// is a tunable knob; the default is conservative and forward-safe (a faster GPU runs each
        /// chunk quicker, never slower). Only the very large YoNoSplat-class transformer needs it.
        int maxSubmitNodes = 500;

        /// Also split a GPU segment's command buffer once the push-descriptor writes it has recorded
        /// reach this many, independent of the node count above. Each dispatch pushes one storage-buffer
        /// descriptor per bound buffer; a fused pointwise/epilogue dispatch binds ~9-11 (plan SSBO + up to
        /// kPwMaxOperands operands) versus ~2-4 for a plain op. A newer-driver device caps the descriptors
        /// one command buffer may hold, and silently corrupts the recording past it, so a long run of
        /// binding-dense fused dispatches must break into more submits than the node count alone implies
        /// (a plain-op graph of the same node count binds far fewer and never trips it). The submit fence
        /// between chunks is a full barrier, so results stay numerically identical. 0 disables this cap.
        /// The default keeps a ~2x margin under the observed corruption point (binding-dense chains only
        /// approach it; a plain-op graph binds far too few to ever split on this).
        int maxSubmitBindings = 1024;

        // Debug.
        bool        timing        = false; ///< print pack/submit/unpack + per-stage timing
        bool        debugSegments = false; ///< trace per-segment + per-CPU-op execution
        std::string disableVkOps;          ///< comma list of op types to force onto CPU
        std::string dumpTensors;           ///< comma list of tensor names to dump to disk

        /// Advanced override of the selective-fp32 set: comma list of tensor-name substrings (leading '-'
        /// excludes) whose activations are kept in fp32 storage even when the segment runs fp16. Empty +
        /// Precision::Normal uses the built-in mixedPrecisionFp32Tensors() preset; a non-empty value replaces
        /// it (and also applies under Precision::Low). The markFp32 pass marks matching flat tensors and
        /// bridges the fp16/fp32 frontier with convert_dtype nodes. Only affects accuracy/runtime.
        std::string fp32Tensors;

        // Profiling / debug.
        bool        profile      = false;
        int         verbosity    = 1; ///< maps to log level
        bool        layerDump    = false;
        std::string layerDumpDir = "/data/local/tmp/vxrt/dump";

        /// Conv kernel selection + GPU-pass knobs, set via setHint(Hint, value) (see the Hint enum):
        /// Hint::Winograd (auto/on/off), the experimental Winograd variant hints, and FlatLayout /
        /// GpuIslandFold (on/off). Forcing Winograd on/off makes the 3x3-conv choice deterministic.
        std::vector<int> hints; ///< indexed by (int)Hint; 0 = production default. Use setHint()/hint().

        /// Set the value of hint @p h, growing the backing vector (zero-filled) if needed.
        void setHint(Hint h, int value) {
            if ((int) h >= (int) hints.size())
            {
                hints.resize((int) h + 1, 0);
            }
            hints[(int) h] = value;
        }
        /// Typed overload so a caller passes the enum directly: setHint(Hint::Winograd, Mode::Off).
        void setHint(Hint h, Mode v) {
            setHint(h, (int) v);
        }
        /// Value of hint @p h, or @p dflt when it has never been set.
        int hint(Hint h, int dflt = 0) const noexcept {
            return (int) h < (int) hints.size() ? hints[(int) h] : dflt;
        }
        /// True when the flat-layout GPU pass is enabled (On by default, the fastest path; --no-flat
        /// sets it Off). Controlled through the hint mechanism.
        bool flatLayout() const noexcept {
            return hint(Hint::FlatLayout, (int) Mode::On) != (int) Mode::Off;
        }
        /// True when tiny-GPU-island folding is enabled (On by default; --no-fold-islands sets it Off).
        /// Controlled through the hint mechanism.
        bool gpuIslandFold() const noexcept {
            return hint(Hint::GpuIslandFold, (int) Mode::On) != (int) Mode::Off;
        }

        static Config fromJsonFile(const std::string &path);
        static Config fromJsonString(const std::string &json);
        std::string   toJson() const;
        void          applyLogLevel() const;

        /// True when comma-separated @p list contains @p name as a whole entry. Entries are trimmed
        /// of surrounding whitespace and compared exactly, so "Conv" matches only Conv — never
        /// ConvTranspose/ConvGemm/ConvertLayout. Matcher for op-name lists such as disableVkOps.
        static bool listContains(const std::string &list, std::string_view name);
    };

} // namespace vknn
