// ConvertDtype: fp16 <-> fp32 storage conversion at a selective-fp32 region frontier (inserted by the
// markFp32 pass). Logical shape + flat layout are identical on both sides; only the buffer element type
// differs. Reuses BoundaryConvert (one thread per element, four cross-dtype SPIR-V) as a flat identity
// map: the tensor is presented as NCHW {1, numElements, 1, 1} so the index math is a straight copy with
// a dtype cast. A non-storeFp32 tensor is stored at the segment's base precision (env.baseFp16).
#include "backend/vulkan/ops/boundary_convert.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct ConvertDtypeOp: VulkanOp {
            BoundaryConvert conv;
            NCHW            sh {};
            DType           srcDt = DType::Float32, dstDt = DType::Float32;

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g = *env.graph;
                int64_t      n = numElements(g.desc(node.outputs[0]).shape);
                sh.n           = 1;
                sh.c           = n;
                sh.h           = 1;
                sh.w           = 1;
                auto dt        = [&](TensorId t) {
                    return g.desc(t).storeFp32 ? DType::Float32 : (env.baseFp16 ? DType::Float16 : DType::Float32);
                };
                srcDt = dt(node.inputs[0]);
                dstDt = dt(node.outputs[0]);
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *s = env.devBuf(node.inputs[0]);
                vk::Buffer *d = env.devBuf(node.outputs[0]);
                conv.record(cmd, *env.ctx, env.cache, s, d, sh, TensorFormat::NCHW, srcDt, TensorFormat::NCHW, dstDt);
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::ConvertDtype, ConvertDtypeOp);

} // namespace vknn
