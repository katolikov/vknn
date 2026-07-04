// Vulkan backend: device tensors (NC4HW4), op registry, pre-recorded segments.
#pragma once
#include "vk_buffer.h"
#include "vk_command.h"
#include "vk_context.h"
#include "vk_pipeline.h"
#include "core/cache_codec.h"
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
    /// Skips the host repacking + per-shape autotune on warm session creation. It maps to/from one
    /// CacheVariant of the multi-variant model cache (see cache_codec.h).
    class WeightCache {
      public:
        // Clear and set whether prepacked weights are retained for saving. `enabled` is true when a
        // persistent cache file is in use; without a file, weights are uploaded and freed (never
        // retained) to avoid ballooning RAM (a 965M model would hold ~3.85GB of prepacked fp32).
        void reset(bool enabled) {
            weights_.clear();
            tune_.clear();
            enabled_ = enabled;
            dirty_   = false;
        }
        // Populate from a cached variant (warm start), then retain for the next save.
        void loadFrom(const CacheVariant &v);
        // Copy the retained weights + autotune table into a variant for serialization.
        void writeInto(CacheVariant &v) const;
        bool enabled() const {
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
        bool                                      enabled_ = false; // retain prepacked weights for saving
        mutable bool                              dirty_   = false;
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
        bool                                  useFp16  = false; // per-node: false for a storeFp32 node so it runs its fp32 kernel
        bool                                  baseFp16 = false; // segment-wide precision (what a non-storeFp32 tensor is stored as)
        WeightCache                          *weights  = nullptr; // prepacked-weight + tuning cache (may be null)
        vk::CommandRunner                    *runner   = nullptr; // for on-device autotuning benchmarks
        Tuning                                tuning   = Tuning::Fast;
        Mode                                  winograd = Mode::Auto;
        // Per-model namespace for the weight cache, so reusing one cacheDir across different models can't
        // collide on shared node names (e.g. ResNet + Inception both have a node called "/Conv").
        std::string modelTag;
        // Per-GPU namespace for the autotune table. The fastest kernel is GPU/driver-specific, so a cache
        // tuned on one device must not apply its choices on another; keying the autotune signature by this
        // tag keeps a separate set of tuned entries per device in the same cache file.
        std::string gpuTag;

        // Session-shared compute pipeline, keyed by (shader, buffer count, push-constant size, spec
        // constants). Nodes with the same kernel configuration share one VkPipeline + shader module
        // instead of each building their own — the driver's per-pipeline host memory and creation time
        // scale with the number of DISTINCT kernels, not the node count.
        std::shared_ptr<vk::ComputePipeline> pipeline(const std::string &shaderName, uint32_t numBuffers, uint32_t pushConstBytes, const std::vector<uint32_t> &specData = {}) const;

        // Content-addressed upload for small parameter blocks (e.g. pw_epilogue plans): identical bytes
        // yield one shared device buffer, so per-node metadata does not multiply vkAllocateMemory count.
        std::shared_ptr<vk::Buffer> uploadPooled(const void *data, size_t bytes) const;
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
