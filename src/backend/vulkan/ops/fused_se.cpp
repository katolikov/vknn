// Fused Squeeze-Excite scale on the GPU. One dispatch (one workgroup per image) replaces the
// GAP->FC->relu->FC->hardsigmoid chain. Weights uploaded once (fp16). The following Mul is
// unchanged.
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        // Field order/types mirror fused_se.comp's push_constant block. Cr is the reduced (squeeze)
        // channel count; alpha/beta are the hardsigmoid slope/offset applied to the FC2 output
        // (scale = clamp(alpha*acc + beta, 0, 1)), carried through from the fused HardSigmoid node.
        struct SePC {
            int   N, C, Cr;
            float alpha, beta;
        };
        struct FusedSeOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          w1, b1, w2, b2;
            SePC                                 pc {};
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g = *env.graph;
                // inputs: [0] pooled avg [N,C,1,1], [1] W1, [2] b1, [3] W2, [4] b2. C is the full
                // channel count (FC2 output width); Cr is the reduced squeeze width. W1 is [Cr][C] and
                // W2 is [C][Cr], so the FC1 output count Cr is W1's row count (shape[0] of inputs[1]).
                NCHW         x = NCHW::from(g.desc(node.inputs[0]).shape);
                int64_t      C = x.c, Cr = g.desc(node.inputs[3]).shape[0];
                Cr = g.desc(node.inputs[1]).shape[0]; // W1 is [Cr][C], so Cr is the row count of W1
                pc = {(int) x.n, (int) C, (int) Cr, node.actLo, node.actHi};
                w1 = uploadCached(env, node.name + "#w1", [&] {
                    return initFloats(g, node.inputs[1]);
                });
                w2 = uploadCached(env, node.name + "#w2", [&] {
                    return initFloats(g, node.inputs[3]);
                });
                // Biases default to zeros (an absent Conv bias is kNoTensor) and are copied in at their
                // exact FC width: b1 has Cr entries (FC1), b2 has C entries (FC2). The clamp on the copy
                // guards a bias tensor that is shorter than the weight-derived width.
                b1 = uploadCached(env, node.name + "#b1", [&] {
                    std::vector<float> v(Cr, 0.f);
                    if (node.inputs[2] != kNoTensor)
                    {
                        std::vector<float> t = initFloats(g, node.inputs[2]);
                        for (int64_t i = 0; i < Cr && i < (int64_t) t.size(); ++i)
                        {
                            v[i] = t[i];
                        }
                    }
                    return v;
                });
                b2 = uploadCached(env, node.name + "#b2", [&] {
                    std::vector<float> v(C, 0.f);
                    if (node.inputs[4] != kNoTensor)
                    {
                        std::vector<float> t = initFloats(g, node.inputs[4]);
                        for (int64_t i = 0; i < C && i < (int64_t) t.size(); ++i)
                        {
                            v[i] = t[i];
                        }
                    }
                    return v;
                });
                // 6 buffer bindings: avg, w1, b1, w2, b2, scale (binding order 0..5 in fused_se.comp).
                pipe = env.pipeline(shader("fused_se", env.useFp16), 6, sizeof(SePC), std::vector<uint32_t> {});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *f = env.devBuf(node.inputs[0]); // pooled avg in
                vk::Buffer *s = env.devBuf(node.outputs[0]); // channel scale out
                // One workgroup per image (pc.N groups): each group does the whole FC1->relu->FC2->
                // hardsigmoid for its image, so the FC accumulations stay in shared memory. Buffer order
                // matches the pipeline's binding layout (avg, w1, b1, w2, b2, scale).
                pipe->dispatch(cmd, {f->handle(), w1->handle(), b1->handle(), w2->handle(), b2->handle(), s->handle()}, &pc, sizeof(pc), (uint32_t) pc.N);
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::FusedSE, FusedSeOp);
} // namespace vknn
