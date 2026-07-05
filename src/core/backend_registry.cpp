#include "vknn/backend.h"
#include "vknn/logging.h"

namespace vknn {

    BackendRegistry &BackendRegistry::instance() {
        // Function-local static: constructed on first call (thread-safe under C++11+) so the map is
        // ready before any static-initialization-time BackendRegistrar reaches registerBackend().
        static BackendRegistry r;
        return r;
    }
    void BackendRegistry::registerBackend(BackendKind k, Factory f) {
        factories_[k] = std::move(f);
    }
    bool BackendRegistry::has(BackendKind k) const {
        return factories_.count(k) > 0;
    }
    std::unique_ptr<Backend> BackendRegistry::create(BackendKind k, const Config &cfg) const {
        auto it = factories_.find(k);
        if (it == factories_.end())
        {
            // Unregistered kind (the backend translation unit was not linked in): the caller
            // distinguishes this from a factory that itself returns nullptr on a construction failure.
            return nullptr;
        }
        return it->second(cfg);
    }

} // namespace vknn
