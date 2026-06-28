// GPU conversion between a caller's declared dma-buf layout/dtype and the device-native boundary
// buffer, for declared-format zero-copy I/O. One shader source compiled to four cross-dtype SPIR-V
// (f32_f32, f32_f16, f16_f32, f16_f16); the layout pair is a push constant. The segment owns one
// instance and records a dispatch per converted boundary tensor (input: imported -> pooled boundary;
// output: pooled boundary -> imported).
#pragma once
#include "backend/vulkan/vk_buffer.h"
#include "backend/vulkan/vk_pipeline.h"
#include "vknn/dtype.h"
#include "vknn/tensor_format.h"
#include <memory>

namespace vknn {

    class BoundaryConvert {
      public:
        // Record a dispatch reading `src` (srcFmt/srcDt) and writing `dst` (dstFmt/dstDt) for the logical
        // NCHW `shape`. Pipelines are built lazily and cached across runs.
        void record(VkCommandBuffer cmd, vk::VulkanContext &ctx, vk::PipelineCache *cache, vk::Buffer *src, vk::Buffer *dst, const NCHW &shape, TensorFormat srcFmt, DType srcDt, TensorFormat dstFmt, DType dstDt);

      private:
        std::unique_ptr<vk::ComputePipeline> pipes_[4]; // index = (srcFp16 << 1) | dstFp16
    };

} // namespace vknn
