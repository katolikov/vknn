// FusedAttention on the flat GPU path: the whole M == 1 decode-attention core — q.k^T scores with
// scale and additive mask, online softmax, p.V context — in ONE dispatch of
// shaders/fused_attention.comp (one workgroup per attention row; contract in
// core/fused_attention.h). Operands are read through the per-axis strides the fuseDecodeAttention
// pass composed from the folded MatMul operand views, so the GQA KV cache is read in place. The
// kernel accumulates in fp32 and regroups the same sums across subgroup lanes, so it matches the
// CPU oracle to fp32 rounding (cosine), not byte-for-byte.
#include "core/fused_attention.h"
#include "flat_ops.h"
#include "vk_op_common.h"
#include "vknn/op.h"
#include <vector>

namespace vknn {
    namespace {

        // Field order/types mirror fused_attention.comp's push_constant block.
        struct FaPC {
            int   rank, rows, C, hd;
            int   qK, kN, kK, vN, vK, mN, hasMask;
            float scale, maskScale;
        };

        struct FusedAttentionOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            FaPC                                 pc {};
            std::shared_ptr<vk::Buffer>          geom;      // dims + q/k/v/mask strides, deduped SSBO
            std::shared_ptr<vk::Buffer>          hold[4];   // per-operand, set when that operand is a constant initializer
            std::shared_ptr<vk::Buffer>          maskDummy; // bound in the mask slot when the node has no mask

            void prepare(const Node &node, VkOpEnv &env) override {
                const std::vector<int64_t> &dims    = node.attr.getints(kFaDims);
                const std::vector<int64_t> &qStride = node.attr.getints(kFaQStride);
                const std::vector<int64_t> &kStride = node.attr.getints(kFaKStride);
                const std::vector<int64_t> &vStride = node.attr.getints(kFaVStride);
                const std::vector<int64_t> &mStride = node.attr.getints(kFaMStride);
                const int                   rank    = (int) dims.size();

                pc.rank      = rank;
                pc.C         = (int) node.attr.geti(kFaC);
                pc.hd        = (int) node.attr.geti(kFaHd);
                pc.qK        = (int) node.attr.geti(kFaQK);
                pc.kN        = (int) node.attr.geti(kFaKN);
                pc.kK        = (int) node.attr.geti(kFaKK);
                pc.vN        = (int) node.attr.geti(kFaVN);
                pc.vK        = (int) node.attr.geti(kFaVK);
                pc.mN        = (int) node.attr.geti(kFaMN);
                pc.hasMask   = node.inputs.size() > 3 && node.inputs[3] != kNoTensor ? 1 : 0;
                pc.scale     = node.attr.getf(kFaScale, 1.f);
                pc.maskScale = node.attr.getf(kFaMaskScale, 1.f);
                int64_t rows = 1;
                for (int64_t d: dims)
                {
                    rows *= d;
                }
                pc.rows = (int) rows;

                // Geometry SSBO: dims, then the four stride arrays (zeros for a maskless node so
                // the kernel's base decode reads a well-defined array).
                std::vector<int32_t> dimsI(rank), qs(rank), ks(rank), vs(rank), ms(rank, 0);
                for (int i = 0; i < rank; ++i)
                {
                    dimsI[i] = (int32_t) dims[i];
                    qs[i]    = (int32_t) qStride[i];
                    ks[i]    = (int32_t) kStride[i];
                    vs[i]    = (int32_t) vStride[i];
                    if (pc.hasMask && i < (int) mStride.size())
                    {
                        ms[i] = (int32_t) mStride[i];
                    }
                }
                geom = flat::uploadFlatGeom(env, {dimsI, qs, ks, vs, ms});

                if (!pc.hasMask)
                {
                    // Descriptor sets bind every declared buffer; a maskless node binds a shared
                    // dummy word the kernel's hasMask == 0 path never reads.
                    maskDummy = env.acquireWeight("fattn#dummy", env.useFp16, [&] {
                        const uint32_t zero[4] = {0, 0, 0, 0};
                        return env.uploadWeightDeviceOnly(zero, sizeof zero, sizeof zero);
                    });
                }

                pipe = env.pipeline(shader("fused_attention", env.useFp16), 6, sizeof(FaPC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *q    = operandBuf(env, node.inputs[0], hold[0]);
                vk::Buffer *k    = operandBuf(env, node.inputs[1], hold[1]);
                vk::Buffer *v    = operandBuf(env, node.inputs[2], hold[2]);
                vk::Buffer *mask = pc.hasMask ? operandBuf(env, node.inputs[3], hold[3]) : maskDummy.get();
                vk::Buffer *dst  = env.devBuf(node.outputs[0]);
                // One workgroup per row; ComputePipeline::dispatch spills a 1-D overflow into y,
                // which the kernel folds back through gl_WorkGroupID.y.
                pipe->dispatch(cmd, {q->handle(), k->handle(), v->handle(), dst->handle(), mask->handle(), geom->handle()}, &pc, sizeof(pc), (uint32_t) pc.rows);
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::FusedAttention, FusedAttentionOp);
} // namespace vknn
