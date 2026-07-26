// compute pipeline, shader-module cache, on-disk VkPipelineCache.
#pragma once
#include "vk_context.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vknn {
    // Provided by the build-time-generated translation unit (tools/embed_spirv.py).
    const std::map<std::string, std::vector<uint32_t>> &embeddedShaders();
    // md5 hex of all embedded SPIR-V kernels — the model cache's kernel-version guard.
    const char *embeddedShadersHash();

    namespace vk {

        /// Serializable VkPipelineCache keyed by device+driver. Speeds warm session creation. Built from
        /// the pipeline section of the unified per-model cache file; getData() returns the bytes the
        /// backend writes back.
        ///
        /// Owns one VkPipelineCache (RAII); not copyable or movable.
        class PipelineCache {
          public:
            /// @param ctx         The Vulkan context whose device owns the cache (retained by reference).
            /// @param initialData Serialized cache bytes from a prior session, or empty for a cold cache.
            ///                    The driver validates them against its own UUID and ignores a mismatch.
            explicit PipelineCache(VulkanContext &ctx, const std::vector<char> &initialData = {});
            ~PipelineCache();
            PipelineCache(const PipelineCache &)            = delete;
            PipelineCache &operator=(const PipelineCache &) = delete;
            PipelineCache(PipelineCache &&)                 = delete;
            PipelineCache &operator=(PipelineCache &&)      = delete;

            VkPipelineCache handle() const noexcept {
                return cache_;
            }
            std::vector<char> getData() const; ///< Serialize the current cache for the unified model file.
            size_t            diskBytes() const noexcept {
                return diskBytes_;
            }
            /// Size the driver would serialize right now, from the size-only half of the two-call idiom
            /// (no blob copy). Grows as the driver compiles pipelines, so it tells a flush whether any
            /// driver work happened since the last save without paying for the bytes.
            size_t currentBytes() const noexcept;

          private:
            VulkanContext  &ctx_;
            VkPipelineCache cache_     = VK_NULL_HANDLE;
            size_t          diskBytes_ = 0;
        };

        /// A compute pipeline bound to N storage buffers (via push descriptors) with a
        /// push-constant block and optional specialization constants.
        ///
        /// Owns its shader module, descriptor-set layout, pipeline layout, and pipeline (RAII); not
        /// copyable or movable.
        class ComputePipeline {
          public:
            /// @param ctx            The Vulkan context whose device creates and owns the pipeline (retained by reference).
            /// @param shaderName     Key into the embedded SPIR-V table (see embeddedShaders()).
            /// @param numBuffers     Storage buffers bound at descriptor bindings 0..numBuffers-1.
            /// @param pushConstBytes Size of the push-constant block (0 for none).
            /// @param specData       Specialization constants, one uint32 at ids 0..N-1.
            /// @param cache          Pipeline cache to accelerate creation, or VK_NULL_HANDLE.
            /// @param requiredSubgroupSize  Exact subgroup width the pipeline must run at, or 0 for
            ///                       the driver default. Non-zero requires the subgroupSizeControl
            ///                       capability (cooperative-matrix kernels pin their wave width);
            ///                       the caller gates on caps before requesting one.
            /// @throws Error if the shader is unknown or a Vulkan object cannot be created.
            ComputePipeline(VulkanContext &ctx, const std::string &shaderName, uint32_t numBuffers, uint32_t pushConstBytes, const std::vector<uint32_t> &specData = {}, VkPipelineCache cache = VK_NULL_HANDLE, uint32_t requiredSubgroupSize = 0);
            ~ComputePipeline();
            ComputePipeline(const ComputePipeline &)            = delete;
            ComputePipeline &operator=(const ComputePipeline &) = delete;
            ComputePipeline(ComputePipeline &&)                 = delete;
            ComputePipeline &operator=(ComputePipeline &&)      = delete;

            VkPipeline pipeline() const noexcept {
                return pipeline_;
            }
            VkPipelineLayout layout() const noexcept {
                return layout_;
            }
            uint32_t numBuffers() const noexcept {
                return numBuffers_;
            }

            /// Record bind + push-descriptors + push-constants + dispatch into `cmd`. A 1D dispatch whose
            /// group count exceeds the device limit is spilled into the y dimension (see the .cpp).
            void dispatch(VkCommandBuffer cmd, const std::vector<VkBuffer> &buffers, const void *pushConst, uint32_t pcBytes, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1);

          private:
            void destroy() noexcept; ///< Release every owned handle; safe from the destructor and a failing constructor.

            VulkanContext        &ctx_;
            VkShaderModule        module_     = VK_NULL_HANDLE;
            VkDescriptorSetLayout setLayout_  = VK_NULL_HANDLE;
            VkPipelineLayout      layout_     = VK_NULL_HANDLE;
            VkPipeline            pipeline_   = VK_NULL_HANDLE;
            uint32_t              numBuffers_ = 0;
            std::string           name_;
        };

    } // namespace vk
} // namespace vknn
