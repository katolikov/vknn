// Vulkan operator surface: op environment, op base class, registry + registration macro.
#pragma once
#include "vk_buffer.h"
#include "vk_command.h"
#include "vk_context.h"
#include "vk_pipeline.h"
#include "vknn/backend.h"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vknn {

    /// Opaque (to core) device storage = a Vulkan buffer holding an NC4HW4 tensor.
    struct DeviceStorage {
        std::shared_ptr<vk::Buffer> buffer;
    };

    class VulkanBackend;
    class WeightCache;

    /// Environment passed to Vulkan operators during prepare/record.
    struct VkOpEnv {
        VulkanBackend     *backend = nullptr;
        vk::VulkanContext *ctx     = nullptr;
        vk::PipelineCache *cache   = nullptr;
        const Graph       *graph   = nullptr;
        const Config      *config  = nullptr;
        /// Workgroup width of the flat/element-parallel kernel family, resolved ONCE at segment
        /// build from exact device caps (min(256, maxWorkGroupInvocations) in whole subgroups -
        /// flatLocalSizeFor). Every family pipeline takes it as its workgroup-size spec constant
        /// and every dispatch derives its group count from the same value, so the two can never
        /// disagree. Caps are exact per device, so this may steer byte-affecting choices (a
        /// workgroup reduction tree); MEASURED probe values (deviceTuneModel) may only ever steer
        /// placement-only choices such as items-per-lane.
        uint32_t                              flatLocalSize = 256;
        std::function<vk::Buffer *(TensorId)> devBuf; // resolves a tensor id to its (possibly pool-aliased) activation buffer
        // Resolves a tensor id to its int8 KV-cache SCALE buffer (Hint::KvCacheQuant): non-null
        // exactly for the cache tensors the owning segment allocated as int8 payload + fp16 scales
        // (src/core/kv_quant.h), so FusedAttention derives its kernel variant from the segment's
        // allocation and the two can never disagree. Null on a segment without the scheme.
        std::function<vk::Buffer *(TensorId)> kvqScale;
        // Physical last-axis extent of a tensor's activation buffer, or 0 when the buffer is packed
        // (the logical last axis). Non-zero exactly for the tensors the owning segment allocated
        // with a virtualized row stride so a consuming tiled MatMul could take the vec4-load
        // kernels (rule in core/matmul_tile.h): the producing kernel stores at this stride and
        // zero-fills the pad, the consumer reads at it. Null on a segment without the scheme, so
        // callers use `rowPad ? rowPad(t) : 0`.
        std::function<int64_t(TensorId)> rowPad;
        // Drops an initializer's HOST bytes the moment its device copy exists, so a large-weight model
        // never holds the host and device copy of the same weight at once (the load-time peak would
        // otherwise be twice the weight set, which exhausts phone RAM on a multi-GB model). Null when
        // the session keeps its weights (Config::freeWeightsAfterUpload off). Only weights uploaded at
        // prepare() time are released; record-time constant operands stay resident.
        std::function<void(TensorId)> releaseInitializer;
        // Memo of the flat device buffer already uploaded for an initializer of THIS graph. A weight may
        // feed several nodes; once its host bytes are released the content digest can no longer be
        // recomputed, so every consumer after the first resolves through this memo instead.
        std::function<std::shared_ptr<vk::Buffer>(TensorId)>       lookupFlatWeight;
        std::function<void(TensorId, std::shared_ptr<vk::Buffer>)> rememberFlatWeight;
        bool                                                       useFp16  = false;   // per-node: false for a storeFp32 node so it runs its fp32 kernel
        bool                                                       baseFp16 = false;   // segment-wide precision (what a non-storeFp32 tensor is stored as)
        WeightCache                                               *weights  = nullptr; // prepacked-weight + tuning cache (may be null)
        vk::CommandRunner                                         *runner   = nullptr; // for on-device autotuning benchmarks
        Tuning                                                     tuning   = Tuning::Fast;
        Mode                                                       winograd = Mode::Auto;
        // Per-model namespace for the weight cache, so reusing one cache directory across different models can't
        // collide on shared node names (e.g. ResNet + Inception both have a node called "/Conv").
        std::string modelTag;
        // Per-GPU namespace for the autotune table. The fastest kernel is GPU/driver-specific, so a cache
        // tuned on one device must not apply its choices on another; keying the autotune signature by this
        // tag keeps a separate set of tuned entries per device in the same cache file.
        std::string gpuTag;

        // Cache-first autotune reuse decision for a pick site. On a cached entry for `sig`, writes its
        // chosen value to `out` and returns true when it may be reused: always under Tuning::None (none
        // runs no new sweep but honors a stored measurement) and otherwise only when the entry was
        // measured at a level >= the requested one (a fast entry is re-swept for a heavy request). A
        // miss, a lower cached level, or a null weight cache returns false and the site sweeps (or, under
        // None, falls back to its default kernel). The site applies any value-specific validity gate.
        bool reuseTuned(const std::string &sig, int &out) const;

        // Session-shared compute pipeline, keyed by (shader, buffer count, push-constant size, spec
        // constants, pinned subgroup size). Nodes with the same kernel configuration share one
        // VkPipeline + shader module instead of each building their own — the driver's per-pipeline
        // host memory and creation time scale with the number of DISTINCT kernels, not the node count.
        // `requiredSubgroupSize` 0 leaves the driver's width choice; non-zero pins it (coopmat kernels)
        // and requires caps().subgroupSizeControl, which the requesting op gates on.
        std::shared_ptr<vk::ComputePipeline> pipeline(const std::string &shaderName, uint32_t numBuffers, uint32_t pushConstBytes, const std::vector<uint32_t> &specData = {}, uint32_t requiredSubgroupSize = 0) const;

        // Content-addressed upload for small parameter blocks (e.g. pw_epilogue plans): identical bytes
        // yield one shared device buffer, so per-node metadata does not multiply vkAllocateMemory count.
        std::shared_ptr<vk::Buffer> uploadPooled(const void *data, size_t bytes) const;

        // Upload a weight payload into device-only memory through the backend's persistent staging
        // buffer (VulkanBackend::stageWeightToDevice): `srcBytes` are copied into a fresh buffer of
        // `bufferBytes` (>= srcBytes; the tail is allocated padding, never read as data). The
        // destination has no host mapping, so weights stay outside the driver's per-process
        // host-mappable memory budget.
        std::shared_ptr<vk::Buffer> uploadWeightDeviceOnly(const void *src, size_t srcBytes, size_t bufferBytes) const;

        // Backend-level device-weight pool. Uploaded weight/bias/transformed-weight buffers are shared
        // across plan buckets keyed by (weight-cache key, precision): the first op instance to acquire a
        // key runs `make` (host-cache consult + prepack + upload); later instances — including a second
        // shape bucket's re-prepare — get the same device buffer instead of a duplicate upload. Weakly
        // held, so a fixed-shape model's single instance keeps today's allocation count exactly.
        std::shared_ptr<vk::Buffer> acquireWeight(const std::string &key, bool fp16, std::function<std::shared_ptr<vk::Buffer>()> make) const;
    };

    /// One operator on the Vulkan backend. Adding an op: subclass + VKNN_REGISTER_VK_OP.
    ///
    /// The two phases run at distinct times: prepare() once at plan time (session build / warm start),
    /// record() once per segment when the command buffer is pre-recorded. Any state an op computes in
    /// prepare() and reads back in record() must be stored on the op instance, since the same instance
    /// serves both calls.
    class VulkanOp {
      public:
        virtual ~VulkanOp() = default;
        /// Create pipeline(s), prepack + upload weights, allocate op-private buffers. Runs once at plan
        /// time; may consult and populate the weight/tune cache in `env`.
        virtual void prepare(const Node &node, VkOpEnv &env) = 0;
        /// Record dispatch(es) into the command buffer. Runs at pre-record time and must not allocate or
        /// upload (all device resources are already created in prepare()); it only binds and dispatches.
        virtual void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) = 0;
    };

    /// Produces a fresh backend op instance for one node (each node gets its own instance so prepare()
    /// state does not alias between nodes of the same OpType).
    using VkOpFactory = std::function<std::unique_ptr<VulkanOp>()>;

    /// Global OpType -> factory table. Populated at static-init time by the VKNN_REGISTER_VK_OP macro;
    /// the planner queries has() to decide GPU-vs-fallback and calls create() to instantiate ops.
    class VkOpRegistry {
      public:
        /// The process-wide singleton (constructed on first use, so registration order is irrelevant).
        static VkOpRegistry &instance();
        /// Register (or replace) the factory for an OpType.
        void reg(OpType t, VkOpFactory f) {
            factories_[t] = std::move(f);
        }
        /// True if a Vulkan implementation is registered for `t` (i.e. the op can run on the GPU).
        bool has(OpType t) const {
            return factories_.count(t) > 0;
        }
        /// Instantiate the op for `t`, or nullptr if none is registered.
        std::unique_ptr<VulkanOp> create(OpType t) const {
            auto it = factories_.find(t);
            return it == factories_.end() ? nullptr : it->second();
        }

      private:
        std::map<OpType, VkOpFactory> factories_;
    };

    /// Static-init helper: constructing one registers `f` for `t`. VKNN_REGISTER_VK_OP declares a file-
    /// scope instance so each op source file self-registers when its translation unit is loaded.
    struct VkOpRegistrar {
        VkOpRegistrar(OpType t, VkOpFactory f) {
            VkOpRegistry::instance().reg(t, std::move(f));
        }
    };
#define VKNN_REGISTER_VK_OP(OPTYPE, CLASS)                           \
    static ::vknn::VkOpRegistrar _vx_vkop_reg_##CLASS(OPTYPE, []() { \
        return std::unique_ptr<::vknn::VulkanOp>(new CLASS());       \
    })

    /// Physically-stored element count for a logical shape in the NC4HW4 device layout. Channels are
    /// grouped into blocks of four (cBlocks rounds the channel count up, so a partial final block still
    /// costs a full four channels of padding), and the block width of four is the `* 4` factor. Matches
    /// formatElems(TensorFormat::NC4HW4, ...) and is what device activation buffers are sized by.
    /// @param shape Logical (NCHW-interpreted) tensor shape.
    /// @returns The padded NC4HW4 element count `N * cBlocks(C) * 4 * H * W`.
    inline int64_t packedElems(const Shape &shape) {
        NCHW x = NCHW::from(shape);
        return x.n * cBlocks(x.c) * 4 * x.h * x.w;
    }

} // namespace vknn
