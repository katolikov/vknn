// Top-level facade users call.
#pragma once
#include "vknn/config.h"
#include "vknn/session_class.h"
#include <memory>
#include <string>

namespace vknn {

    /// Top-level facade the public API is used through: it selects the model loader from the file
    /// extension and resolves the per-model cache path, then delegates to Session.
    class Runtime {
      public:
        /// Load a model and build a ready-to-run Session.
        /// @param path      Model file. A ".vxm" extension selects the pre-optimized loader; any other
        ///                  extension is treated as ONNX.
        /// @param cfg       Runtime configuration; copied so the resolved cache path can be stamped in.
        /// @param cacheFile Unified per-model cache file: an existing file gives a fast warm start;
        ///                  an absent one is populated and written on session teardown. Empty (the
        ///                  default) resolves to "<model>.cache" next to the model via defaultCacheFile().
        /// @returns An owning Session for the loaded model.
        static std::unique_ptr<Session> load(const std::string &path, const Config &cfg = {}, const std::string &cacheFile = "") {
            Config c    = cfg;
            c.cacheFile = cacheFile.empty() ? defaultCacheFile(path) : cacheFile;
            // Dispatch on extension: a pre-optimized ".vxm" skips ONNX parsing + passes; anything else is
            // ONNX.
            constexpr size_t kVxmExtLen = 4; // length of the ".vxm" extension suffix
            bool isVxm = path.size() >= kVxmExtLen && path.compare(path.size() - kVxmExtLen, kVxmExtLen, ".vxm") == 0;
            return isVxm ? Session::createFromVxm(path, c) : Session::createFromOnnx(path, c);
        }
        /// Derive the default cache path from a model path by swapping the final extension for ".cache"
        /// ("<model path without extension>.cache" — e.g. enc.vxm -> enc.cache). A model path with no
        /// extension gets ".cache" appended.
        /// @param modelPath Path to the model file.
        /// @returns The resolved cache-file path.
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
