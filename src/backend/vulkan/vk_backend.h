// Vulkan backend: device tensors (NC4HW4), op registry, pre-recorded segments.
#pragma once
#include "vk_buffer.h"
#include "vk_command.h"
#include "vk_context.h"
#include "vk_pipeline.h"
#include "vknn/backend.h"
#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace vknn {

    /// Opaque (to core) device storage = a Vulkan buffer holding an NC4HW4 tensor.
    struct DeviceStorage {
        std::shared_ptr<vk::Buffer> buffer;
    };

    /// In-memory cache of prepacked weights (keyed by op+role+shape) and autotuned workgroup sizes.
    /// Skips the host repacking + per-shape autotune on warm session creation. Serializes to a simple
    /// length-prefixed blob that the backend bundles into the unified per-model cache file.
    class WeightCache {
      public:
        // Populate from a serialized blob (the weights section of the unified cache file). keepWeights
        // retains the (large) prepacked weights for the next save; keepTune retains the (tiny) autotune
        // table. They are independent so a big model can skip caching weights yet still persist its
        // autotune, sparing a re-tune on every warm start.
        void                 loadBytes(const uint8_t *data, size_t n, bool keepWeights, bool keepTune);
        std::vector<uint8_t> serialize() const; // weights + tuning -> blob
        bool                 enabled() const {
            return enabled_;
        }
        bool dirty() const {
            return dirty_;
        }
        bool get(const std::string &key, std::vector<float> &out) const;
        void put(const std::string &key, const std::vector<float> &data);
        // autotune table: op-signature -> chosen local_size_x
        int  tuned(const std::string &sig, int dflt) const;
        void setTuned(const std::string &sig, int val);

      private:
        std::map<std::string, std::vector<float>> weights_;
        std::map<std::string, int>                tune_;
        bool                                      enabled_     = false; // retain prepacked weights (cacheWeights)
        bool                                      tuneEnabled_ = false; // persist the autotune table (cacheTuning)
        mutable bool                              dirty_       = false;
    };

    class VulkanBackend;

    /// Environment passed to Vulkan operators during prepare/record.
    struct VkOpEnv {
        VulkanBackend                        *backend = nullptr;
        vk::VulkanContext                    *ctx     = nullptr;
        vk::PipelineCache                    *cache   = nullptr;
        const Graph                          *graph   = nullptr;
        const Config                         *config  = nullptr;
        std::function<vk::Buffer *(TensorId)> devBuf; // activation buffer for a tensor id
        bool                                  useFp16  = false;
        WeightCache                          *weights  = nullptr; // prepacked-weight + tuning cache (may be null)
        vk::CommandRunner                    *runner   = nullptr; // for on-device autotuning benchmarks
        TuningLevel                           tuning   = TuningLevel::Fast;
        WinogradMode                          winograd = WinogradMode::Auto;
        // Per-model namespace for the weight cache, so reusing one cacheDir across different models can't
        // collide on shared node names (e.g. ResNet + Inception both have a node called "/Conv").
        std::string modelTag;
        // Per-GPU namespace for the autotune table. The fastest kernel is GPU/driver-specific, so a cache
        // tuned on one device must not apply its choices on another; keying the autotune signature by this
        // tag keeps a separate set of tuned entries per device in the same cache file.
        std::string gpuTag;
    };

    /// One operator on the Vulkan backend. Adding an op: subclass + VKNN_REGISTER_VK_OP.
    class VulkanOp {
      public:
        virtual ~VulkanOp() = default;
        /// Create pipeline(s), prepack + upload weights, allocate op-private buffers.
        virtual void prepare(const Node &node, VkOpEnv &env) = 0;
        /// Record dispatch(es) into the command buffer.
        virtual void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) = 0;
    };

    using VkOpFactory = std::function<std::unique_ptr<VulkanOp>()>;

    class VkOpRegistry {
      public:
        static VkOpRegistry &instance();
        void                 reg(OpType t, VkOpFactory f) {
            factories_[t] = std::move(f);
        }
        bool has(OpType t) const {
            return factories_.count(t) > 0;
        }
        std::unique_ptr<VulkanOp> create(OpType t) const {
            auto it = factories_.find(t);
            return it == factories_.end() ? nullptr : it->second();
        }

      private:
        std::map<OpType, VkOpFactory> factories_;
    };

    struct VkOpRegistrar {
        VkOpRegistrar(OpType t, VkOpFactory f) {
            VkOpRegistry::instance().reg(t, std::move(f));
        }
    };
#define VKNN_REGISTER_VK_OP(OPTYPE, CLASS)                           \
    static ::vknn::VkOpRegistrar _vx_vkop_reg_##CLASS(OPTYPE, []() { \
        return std::unique_ptr<::vknn::VulkanOp>(new CLASS());       \
    })

    // NC4HW4 element count for a logical NCHW shape.
    inline int64_t packedElems(const Shape &shape) {
        NCHW x = NCHW::from(shape);
        return x.n * cBlocks(x.c) * 4 * x.h * x.w;
    }

} // namespace vknn
