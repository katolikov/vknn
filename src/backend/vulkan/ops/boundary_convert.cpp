// See boundary_convert.h. The dispatch covers the destination element count and converts layout +
// dtype in one pass; index math matches VulkanBackend::packToBuffer / unpackFromBuffer.
#include "boundary_convert.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct BoundaryPC {
            int      N, C, H, W, srcFmt, dstFmt;
            uint32_t count; // unsigned: the shader's bounds check stays correct up to 2^32 elements
        };

        int fmtCode(TensorFormat f) {
            return f == TensorFormat::NHWC ? 1 : f == TensorFormat::NC4HW4 ? 2 : 0;
        }

        const char *variantName(DType srcDt, DType dstDt) {
            bool s = srcDt == DType::Float16, d = dstDt == DType::Float16;
            return s ? (d ? "boundary_convert_f16_f16" : "boundary_convert_f16_f32") : (d ? "boundary_convert_f32_f16" : "boundary_convert_f32_f32");
        }

    } // namespace

    void BoundaryConvert::record(VkCommandBuffer cmd, vk::VulkanContext &ctx, vk::PipelineCache *cache, vk::Buffer *src, vk::Buffer *dst, const NCHW &shape, TensorFormat srcFmt, DType srcDt, TensorFormat dstFmt, DType dstDt) {
        int idx = (srcDt == DType::Float16 ? 2 : 0) + (dstDt == DType::Float16 ? 1 : 0);
        if (!pipes_[idx])
        {
            pipes_[idx] = std::make_unique<vk::ComputePipeline>(ctx, variantName(srcDt, dstDt), 2, sizeof(BoundaryPC), std::vector<uint32_t> {}, cache ? cache->handle() : VK_NULL_HANDLE);
        }
        int64_t    count = formatElems(dstFmt, shape);
        BoundaryPC pc {(int) shape.n, (int) shape.c, (int) shape.h, (int) shape.w, fmtCode(srcFmt), fmtCode(dstFmt), (uint32_t) count};
        pipes_[idx]->dispatch(cmd, {src->handle(), dst->handle()}, &pc, sizeof(pc), groups(count, 256));
    }

} // namespace vknn
