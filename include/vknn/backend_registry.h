// Backend registry: BackendKind-keyed factories plus the self-registration helper + macro.
//
// Each backend translation unit registers its factory at static-initialization time via the
// VKNN_REGISTER_BACKEND macro, so linking a backend in is sufficient to make it selectable; the
// core never names concrete backend types. Session creation looks a kind up here and calls the
// factory with the session Config.
#pragma once
#include "vknn/backend_class.h"
#include "vknn/config.h"
#include <functional>
#include <map>
#include <memory>

namespace vknn {

    // --------------------------- Backend registry ---------------------------
    /// Process-wide map from BackendKind to a factory that constructs that backend. Populated at
    /// static-initialization time by BackendRegistrar (typically via VKNN_REGISTER_BACKEND) and read
    /// during session creation to instantiate the selected backend.
    class BackendRegistry {
      public:
        /// Constructs a backend from a session Config. The Config is passed so a backend can honor
        /// creation-time settings (e.g. the Vulkan queue priority, applied at device/queue creation
        /// before configure() runs).
        using Factory = std::function<std::unique_ptr<Backend>(const Config &)>;
        /// The single shared registry (function-local static, constructed on first use).
        static BackendRegistry &instance();
        /// Install (or replace) the factory for kind `k`. A later registration of the same kind wins.
        void registerBackend(BackendKind k, Factory f);
        /// @returns True if a factory is registered for kind `k`.
        bool has(BackendKind k) const;
        /// Instantiate the backend for kind `k`, forwarding `cfg` to its factory.
        /// @returns The new backend, or nullptr if no factory is registered for `k`.
        std::unique_ptr<Backend> create(BackendKind k, const Config &cfg) const;

      private:
        std::map<BackendKind, Factory> factories_;
    };

    /// Self-registration helper: constructing one at namespace scope registers `f` for kind `k`, so a
    /// static instance registers its backend as a side effect of the translation unit being linked in.
    struct BackendRegistrar {
        BackendRegistrar(BackendKind k, BackendRegistry::Factory f) {
            BackendRegistry::instance().registerBackend(k, std::move(f));
        }
    };
/// Define a file-scope static BackendRegistrar that registers TYPE (a Backend subclass) for KIND. The
/// factory constructs `new TYPE(cfg)`, so TYPE must accept the session Config. Place once in a backend
/// source file.
#define VKNN_REGISTER_BACKEND(KIND, TYPE)                                                                                            \
    static ::vknn::BackendRegistrar _vx_backend_reg_##TYPE(KIND, [](const ::vknn::Config &cfg) -> std::unique_ptr<::vknn::Backend> { \
        return std::unique_ptr<::vknn::Backend>(new TYPE(cfg));                                                                      \
    })

} // namespace vknn
