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

    namespace vk {

        /// Serializable VkPipelineCache keyed by device+driver. Speeds warm session creation.
        class PipelineCache {
          public:
            PipelineCache(VulkanContext &ctx, std::string path);
            ~PipelineCache();
            VkPipelineCache handle() const {
                return cache_;
            }
            void   save();
            size_t diskBytes() const {
                return diskBytes_;
            }

          private:
            VulkanContext  &ctx_;
            std::string     path_;
            VkPipelineCache cache_     = VK_NULL_HANDLE;
            size_t          diskBytes_ = 0;
        };

        /// A compute pipeline bound to N storage buffers (via push descriptors) with a
        /// push-constant block and optional specialization constants.
        class ComputePipeline {
          public:
            ComputePipeline(VulkanContext &ctx, const std::string &shaderName, uint32_t numBuffers, uint32_t pushConstBytes, const std::vector<uint32_t> &specData = {}, VkPipelineCache cache = VK_NULL_HANDLE);
            ~ComputePipeline();

            VkPipeline pipeline() const {
                return pipeline_;
            }
            VkPipelineLayout layout() const {
                return layout_;
            }
            uint32_t numBuffers() const {
                return numBuffers_;
            }

            // Records bind + push-descriptors + push-constants + dispatch into `cmd`.
            void dispatch(VkCommandBuffer cmd, const std::vector<VkBuffer> &buffers, const void *pushConst, uint32_t pcBytes, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1);

          private:
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
