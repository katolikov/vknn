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

        // Storage-type tag for a boundary dtype: uint8 rides the 8-bit variants, fp16 the 16-bit variants,
        // everything else the fp32 variant. (The device boundary is only ever fp16/fp32; declared I/O adds
        // uint8. Int8/Int32/Int64 have no boundary_convert variant — they never reach this path.)
        const char *dtTag(DType d) {
            return d == DType::UInt8 ? "u8" : d == DType::Float16 ? "f16" : "f32";
        }
        std::string variantName(DType srcDt, DType dstDt) {
            return std::string("boundary_convert_") + dtTag(srcDt) + "_" + dtTag(dstDt);
        }

    } // namespace

    void BoundaryConvert::record(VkCommandBuffer cmd, vk::VulkanContext &ctx, vk::PipelineCache *cache, vk::Buffer *src, vk::Buffer *dst, const NCHW &shape, TensorFormat srcFmt, DType srcDt, TensorFormat dstFmt, DType dstDt) {
        auto  key = std::make_pair(srcDt, dstDt);
        auto &pipe = pipes_[key];
        if (!pipe)
        {
            pipe = std::make_unique<vk::ComputePipeline>(ctx, variantName(srcDt, dstDt), 2, sizeof(BoundaryPC), std::vector<uint32_t> {}, cache ? cache->handle() : VK_NULL_HANDLE);
        }
        int64_t    count = formatElems(dstFmt, shape);
        BoundaryPC pc {(int) shape.n, (int) shape.c, (int) shape.h, (int) shape.w, fmtCode(srcFmt), fmtCode(dstFmt), (uint32_t) count};
        pipe->dispatch(cmd, {src->handle(), dst->handle()}, &pc, sizeof(pc), groups(count, 256));
    }

} // namespace vknn
