// Top-level facade users call.
#pragma once
#include "vknn/config.h"
#include "vknn/session_class.h"
#include <memory>
#include <string>

namespace vknn {

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
