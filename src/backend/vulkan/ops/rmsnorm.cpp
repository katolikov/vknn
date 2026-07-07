// RMSNormalization on the GPU via the FLAT (row-major) path. One WORKGROUP per OUTER row (outer =
// product of every dim before the last); the workgroup's threads cooperatively LDS-reduce the row's
// sum of squares in fp32, then write x * rsqrt(mean(x^2) + eps) * gamma. The fp32 accumulate is free
// (every VKNN kernel accumulates fp32) and is what keeps the norm correct at fp16 storage, where the
// decomposed Pow/ReduceMean chain overflowed/lost precision. gamma (Scale) is a 1-D [norm] tensor: a
// constant uploads flat in prepare(); a runtime (computed) scale binds its activation buffer at
// dispatch -- the shader reads gamma[j] the same way. No bias term (unlike LayerNorm). Output shape ==
// input shape. A fused pointwise chain rides the store via the shared _epi variant.
#include "flat_ops.h"
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {

        struct RmsPC {
            int   outer, norm;
            float eps;
        };

        struct RMSNormOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            RmsPC                                pc {};
            std::shared_ptr<vk::Buffer>          gammaBuf; // holds the CONST buffer; null for a runtime scale
            bool                                 rtGamma = false;
            PwEpi                                epi;

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g    = *env.graph;
                Shape        s    = g.desc(node.inputs[0]).shape;
                int          rank = (int) s.size();
                // norm = the trailing dim (the block each row normalizes over); outer = every dim
                // before it, i.e. the number of independent rows.
                int64_t norm = rank > 0 ? s[rank - 1] : 1;
                if (norm < 1)
                {
                    norm = 1;
                }
                int64_t outer = numElements(s) / norm;
                pc            = {(int) outer, (int) norm, node.attr.getf("epsilon", 1e-6f)};

                // gamma (Scale) is a 1-D [norm] tensor. A constant uploads flat here; a runtime scale
                // binds its activation buffer at dispatch (record()).
                rtGamma = !g.isInitializer(node.inputs[1]);
                if (!rtGamma)
                {
                    std::vector<float> gv = initFloats(g, node.inputs[1]);
                    gv.resize(norm);
                    gammaBuf = upload(*env.ctx, gv, env.useFp16);
                }
                epi.prepare(node, env, /*flat=*/true, g.desc(node.outputs[0]).shape);
                // 3 fixed bindings (input, gamma, output) plus any extra buffers the fused pointwise
                // epilogue binds after them; suffix() picks the matching PW_EPI shader variant.
                pipe = env.pipeline(shader((std::string("flat_rmsnorm") + epi.suffix()).c_str(), env.useFp16), 3 + epi.extraBufs(), sizeof(RmsPC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                // Bind order must match the shader: input, gamma, output, then epilogue buffers. A const
                // scale uses its uploaded buffer; a runtime one binds its activation buffer.
                VkBuffer              dst   = env.devBuf(node.outputs[0])->handle();
                VkBuffer              gamma = rtGamma ? env.devBuf(node.inputs[1])->handle() : gammaBuf->handle();
                std::vector<VkBuffer> bufs  = {env.devBuf(node.inputs[0])->handle(), gamma, dst};
                epi.append(bufs, node, env, dst);
                // Group count = outer (one workgroup per row; flat_rmsnorm reduces across the workgroup
                // via LDS). dispatch() spills an outer that exceeds the per-dim group limit into the y
                // dimension, which the shader unfolds back to a linear row index.
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), (uint32_t) pc.outer);
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::RMSNorm, RMSNormOp);
} // namespace vknn
