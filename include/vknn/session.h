// Session holds the planned graph, the chosen backends and the caches. Runtime is the thin
// entry point callers actually use.
#pragma once
#include "vknn/backend.h"
#include "vknn/config.h"
#include "vknn/graph.h"
#include "vknn/profiler.h"
#include "vknn/tensor.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vknn {

    /// A named tensor handed in/out of the engine (host side, NCHW canonical, fp32). For zero-copy I/O,
    /// set dmaBufFd >= 0 instead of filling `data`: the engine imports the fd as the boundary GPU buffer
    /// and reads the input from it / writes the output into it directly (no host buffer). The fd's memory
    /// must be the device-native boundary buffer (NCHW at the model's compute precision — see
    /// IOInfo::deviceBytes); the engine never allocates it.
    struct IOTensor {
        std::string          name;
        Shape                shape;
        DType                dtype    = DType::kFloat32;
        int                  dmaBufFd = -1; // >=0 => zero-copy: this fd IS the GPU boundary buffer
        std::vector<uint8_t> data;
        float               *f32() {
            return reinterpret_cast<float *>(data.data());
        }
        const float *f32() const {
            return reinterpret_cast<const float *>(data.data());
        }
    };

    /// Describes a model input or output. Everything the caller would otherwise hand-specify, read
    /// straight from the model (concrete shapes resolved at load time, batch fixed to 1).
    struct IOInfo {
        std::string name;
        Shape       shape;
        DType       dtype = DType::kFloat32;
        int64_t     elems = 0; // product of shape = number of fp32 values expected/produced
        // For zero-copy (IOTensor::dmaBufFd): the size and layout of the device-native boundary buffer
        // the caller must put in the dma-buf. Row-major NCHW at the model's compute precision (fp16/fp32).
        int64_t      deviceBytes  = 0;
        TensorFormat deviceFormat = TensorFormat::kNCHW;
    };

    /// Owns the planned graph, the chosen backend(s), caches, and the tensor pool.
    class Session {
      public:
        ~Session();
        /// Build a session from an ONNX model file.
        static std::unique_ptr<Session> createFromOnnx(const std::string &path, const Config &cfg);
        /// Build from a pre-optimized ".vxm" file (skips ONNX parsing + graph passes).
        static std::unique_ptr<Session> createFromVxm(const std::string &path, const Config &cfg);
        /// Build from an already-imported graph (testing / surgery).
        static std::unique_ptr<Session> create(Graph &&g, const Config &cfg);
        /// Serialize the optimized graph to a ".vxm" file for fast reloads.
        bool saveOptimized(const std::string &path) const;

        /// Write the unified cache file (cfg.cacheFile) if the cache changed. Called automatically from
        /// ~Session(); also callable manually (e.g. before a checkpoint).
        void updateCache();

        /// Run the model. To bind a zero-copy output, pre-fill `outputs` with an entry whose name +
        /// dmaBufFd select that output's caller buffer; that output is written into the fd and returned
        /// with no host data. `outputs` is then (re)filled with all results.
        Status run(const std::vector<IOTensor> &inputs, std::vector<IOTensor> &outputs);

        // --- ergonomic API: names/shapes/dtypes come from the model; the caller passes only data ---
        /// Model inputs/outputs (name, concrete shape, dtype, element count). Use these to size buffers.
        std::vector<IOInfo> inputInfo() const;
        std::vector<IOInfo> outputInfo() const;
        /// Run with raw fp32 data, one buffer per model input in model order. Names/shapes/dtypes are
        /// filled from the model and the element counts are validated. Outputs come back fully described.
        Status run(const std::vector<std::vector<float>> &inputData, std::vector<IOTensor> &outputs);
        /// Single-input / single-output convenience: feed the input values, get the output values back.
        /// Returns empty on error. The shape is whatever the model declares (see inputInfo()).
        std::vector<float> infer(const std::vector<float> &input);

        const Graph &graph() const {
            return graph_;
        }
        const Config &config() const {
            return cfg_;
        }
        Profiler &profiler() {
            return profiler_;
        }
        // Backend assignment per node (for reporting fallbacks).
        std::vector<BackendKind> nodeBackends() const;

        // Per-tensor accessor for layer-dump / debugging (host residency).
        const RtTensor *tensor(const std::string &name) const;

      private:
        Session() = default;
        void plan();               // assign backends, partition into segments, compile
        void foldTinyGpuIslands(); // reassign small CPU-bounded GPU runs to CPU (avoid round trips)
        void reconcileInputs(Segment &seg);

        Graph    graph_;
        Config   cfg_;
        Profiler profiler_;
        // Declaration order matters for teardown: backends_ (owns the VulkanContext) must be
        // destroyed LAST, after segments_ and pool_ release their device buffers. Members are
        // destroyed in reverse declaration order, so backends_ is declared first here.
        std::vector<std::unique_ptr<Backend>> backends_; // active, in priority order
        std::map<BackendKind, Backend *>      byKind_;
        std::vector<int>                      nodeBackendIdx_; // backend index per node
        std::vector<std::unique_ptr<Segment>> segments_;
        std::vector<RtTensor>                 pool_;
        bool                                  planned_        = false;
        bool                                  graphOptimized_ = false; // graph came from .vxm (passes already applied)
    };

    /// Top-level facade users call.
    class Runtime {
      public:
        /// Load a model. `cacheFile` is the unified per-model cache file: existing -> fast warm start;
        /// absent -> populated and written on session teardown. Empty (default) -> "<model>.cache" next
        /// to the model.
        static std::unique_ptr<Session> load(const std::string &path, const Config &cfg = {}, const std::string &cacheFile = "") {
            Config c    = cfg;
            c.cacheFile = cacheFile.empty() ? defaultCacheFile(path) : cacheFile;
            // Dispatch on extension: a pre-optimized ".vxm" skips ONNX parsing + passes; anything else is
            // ONNX.
            bool isVxm = path.size() >= 4 && path.compare(path.size() - 4, 4, ".vxm") == 0;
            return isVxm ? Session::createFromVxm(path, c) : Session::createFromOnnx(path, c);
        }
        /// "<model path without extension>.cache" — e.g. enc.vxm -> enc.cache.
        static std::string defaultCacheFile(const std::string &modelPath) {
            auto slash = modelPath.find_last_of("/\\");
            auto dot   = modelPath.find_last_of('.');
            if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
            {
                return modelPath.substr(0, dot) + ".cache";
            }
            return modelPath + ".cache";
        }
    };

} // namespace vknn
