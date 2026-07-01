// Backend registry: name-keyed factories plus the self-registration helper + macro.
#pragma once
#include "vknn/backend_class.h"
#include "vknn/config.h"
#include <functional>
#include <map>
#include <memory>

namespace vknn {

    // --------------------------- Backend registry ---------------------------
    class BackendRegistry {
      public:
        using Factory = std::function<std::unique_ptr<Backend>()>;
        static BackendRegistry  &instance();
        void                     registerBackend(BackendKind k, Factory f);
        bool                     has(BackendKind k) const;
        std::unique_ptr<Backend> create(BackendKind k) const;

      private:
        std::map<BackendKind, Factory> factories_;
    };

    struct BackendRegistrar {
        BackendRegistrar(BackendKind k, BackendRegistry::Factory f) {
            BackendRegistry::instance().registerBackend(k, std::move(f));
        }
    };
#define VKNN_REGISTER_BACKEND(KIND, TYPE)                                                                   \
    static ::vknn::BackendRegistrar _vx_backend_reg_##TYPE(KIND, []() -> std::unique_ptr<::vknn::Backend> { \
        return std::unique_ptr<::vknn::Backend>(new TYPE());                                                \
    })

} // namespace vknn
