// LayerNormalization on the GPU via the FLAT (row-major) path. One WORKGROUP per OUTER row (outer =
// product of dims before `axis`); the workgroup's threads cooperatively LDS-reduce the row's
// normalized block (norm = product of dims from `axis` to the end) to mean and variance, then write
// (x-mean)/sqrt(var+eps)*gamma + beta. gamma (Scale) and beta (B, optional) are 1-D [norm] tensors.
// A constant scale/bias uploads as a flat buffer in prepare(); a RUNTIME (computed) scale/bias binds
// its activation buffer directly at dispatch — the shader reads gamma[j]/beta[j] the same way in both
// cases, so only the buffer source differs. Output shape == input shape.
#include "flat_ops.h"
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {

        struct LnPC {
            int   outer, norm, hasBeta;
            float eps;
        };

        struct LayerNormOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            LnPC                                 pc {};
            std::shared_ptr<vk::Buffer>          gammaBuf, betaBuf; // hold the CONST/zero buffers; null for a runtime operand
            bool                                 rtGamma = false, rtBeta = false;
            PwEpi                                epi;

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g    = *env.graph;
                Shape        s    = g.desc(node.inputs[0]).shape;
                int          rank = (int) s.size();
                // ONNX axis defaults to -1 and may be negative (counted from the end). Wrap once, then
                // floor at 0 so a still-negative or out-of-range axis normalizes over the whole tensor
                // rather than indexing s[] out of bounds below.
                int64_t axis = node.attr.geti("axis", -1);
                if (axis < 0)
                {
                    axis += rank;
                }
                if (axis < 0)
                {
                    axis = 0;
                }
                // norm = product of dims from axis to the end (the block each row normalizes over);
                // outer = every dim before axis, i.e. the number of independent rows.
                int64_t norm = 1;
                for (int k = (int) axis; k < rank; ++k)
                {
                    norm *= s[k];
                }
                if (norm < 1)
                {
                    norm = 1;
                }
                int64_t outer   = numElements(s) / norm;
                bool    hasBeta = pwCoreInputs(node) > 2 && node.inputs[2] != kNoTensor;
                pc              = {(int) outer, (int) norm, hasBeta ? 1 : 0, node.attr.getf("epsilon", 1e-5f)};

                // gamma (Scale) is a 1-D [norm] tensor. A constant uploads flat here; a runtime scale
                // binds its activation buffer at dispatch (record()).
                rtGamma = !g.isInitializer(node.inputs[1]);
                if (!rtGamma)
                {
                    std::vector<float> gv = initFloats(g, node.inputs[1]);
                    gv.resize(norm);
                    gammaBuf = upload(*env.ctx, gv, env.useFp16);
                }
                // beta (optional): a constant real bias uploads here, a runtime bias binds at dispatch,
                // and an absent bias becomes a zero buffer so binding 2 is always valid.
                rtBeta = hasBeta && !g.isInitializer(node.inputs[2]);
                if (hasBeta && !rtBeta)
                {
                    std::vector<float> bv = initFloats(g, node.inputs[2]);
                    bv.resize(norm);
                    betaBuf = upload(*env.ctx, bv, env.useFp16);
                } else if (!hasBeta)
                { betaBuf = upload(*env.ctx, std::vector<float>((size_t) norm, 0.0f), env.useFp16); }
                epi.prepare(node, env, /*flat=*/true, g.desc(node.outputs[0]).shape);
                // 4 fixed bindings (input, gamma, beta, output) plus any extra buffers the fused
                // pointwise epilogue binds after them; suffix() picks the matching PW_EPI shader variant.
                pipe = env.pipeline(shader((std::string("flat_layernorm") + epi.suffix()).c_str(), env.useFp16), 4 + epi.extraBufs(), sizeof(LnPC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                // Bind order must match the shader: input, gamma, beta, output, then epilogue buffers.
                // A const scale/bias uses its uploaded buffer; a runtime one binds its activation buffer.
                VkBuffer              dst   = env.devBuf(node.outputs[0])->handle();
                VkBuffer              gamma = rtGamma ? env.devBuf(node.inputs[1])->handle() : gammaBuf->handle();
                VkBuffer              beta  = rtBeta ? env.devBuf(node.inputs[2])->handle() : betaBuf->handle();
                std::vector<VkBuffer> bufs  = {env.devBuf(node.inputs[0])->handle(), gamma, beta, dst};
                epi.append(bufs, node, env, dst);
                // Group count = outer (one workgroup per row; flat_layernorm reduces across the
                // workgroup via LDS). dispatch() spills an outer that exceeds the per-dim group limit
                // into the y dimension, which the shader unfolds back to a linear row index.
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), (uint32_t) pc.outer);
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::LayerNorm, LayerNormOp);
} // namespace vknn
