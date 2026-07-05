// GPU conversion between a caller's declared layout/dtype and the device-native boundary buffer, for
// both declared-format dma-buf zero-copy and the default host-buffer I/O path. One shader source
// compiled to cross-dtype SPIR-V (the fp32/fp16 pairs plus uint8 source/destination variants); the
// layout pair is a push constant. The segment owns one instance and records a dispatch per converted
// boundary tensor (input: staging/imported -> pooled boundary; output: pooled boundary -> staging/imported).
#pragma once
#include "backend/vulkan/vk_buffer.h"
#include "backend/vulkan/vk_pipeline.h"
#include "vknn/dtype.h"
#include "vknn/tensor_format.h"
#include <map>
#include <memory>
#include <utility>

namespace vknn {

    class BoundaryConvert {
      public:
        // Record a dispatch reading `src` (srcFmt/srcDt) and writing `dst` (dstFmt/dstDt) for the logical
        // NCHW `shape`. Pipelines are built lazily and cached across runs, keyed by the (src,dst) dtype pair.
        void record(VkCommandBuffer cmd, vk::VulkanContext &ctx, vk::PipelineCache *cache, vk::Buffer *src, vk::Buffer *dst, const NCHW &shape, TensorFormat srcFmt, DType srcDt, TensorFormat dstFmt, DType dstDt);

      private:
        // Lazily built pipelines, keyed only by the (src,dst) dtype pair. The layout pair is not part of
        // the key: it is a push constant, so one pipeline per dtype pair serves every layout combination.
        std::map<std::pair<DType, DType>, std::unique_ptr<vk::ComputePipeline>> pipes_;
    };

} // namespace vknn
