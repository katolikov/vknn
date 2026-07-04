#include "vknn/backend.h"
#include "vknn/logging.h"

namespace vknn {

    BackendRegistry &BackendRegistry::instance() {
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
            return nullptr;
        }
        return it->second(cfg);
    }

} // namespace vknn
