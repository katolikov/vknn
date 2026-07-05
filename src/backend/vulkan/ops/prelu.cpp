// PRelu on the GPU (NC4HW4). Per-channel slope packed to [Cb][4] and uploaded once.
#include "vk_op_common.h"

namespace vknn {
    namespace {
        // Mirrors prelu.comp's push_constant block. count is the NC4HW4 packed lane count
        // (one vec4 per lane); the shader recovers each lane's channel block as (i/HW)%Cb to
        // index the per-block slope. HW = h*w spatial stride, Cb = number of channel blocks.
        struct PReluPC {
            int count, HW, Cb;
        };
        struct PReluOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          slope;
            PReluPC                              pc {};
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g        = *env.graph;
                NCHW         x        = NCHW::from(g.desc(node.outputs[0]).shape);
                int64_t      Cb       = cBlocks(x.c);
                // count spans every NC4HW4 packed lane of the output (n * Cb * h * w vec4s).
                pc                    = {(int) ((int64_t) x.n * Cb * x.h * x.w), (int) (x.h * x.w), (int) Cb};
                std::vector<float> sv = initFloats(g, node.inputs[1]);
                int64_t            ns = numElements(g.desc(node.inputs[1]).shape); // slope element count (dtype-agnostic)
                slope                 = uploadCached(env, node.name + "#slope", [&] {
                    // Slope packs to Cb channel blocks of 4 lanes each (NC4HW4 width). Real channels
                    // fill [0, x.c); the padding lanes up to Cb*4 stay 0 (a 0 slope leaves the padded
                    // activations untouched). ns==1 is the scalar-slope broadcast (single shared value).
                    std::vector<float> sp(Cb * 4, 0.f);
                    const float       *s = sv.data();
                    for (int64_t c = 0; c < x.c; ++c)
                    {
                        sp[c] = ns == 1 ? s[0] : s[c];
                    }
                    return sp;
                });
                pipe = env.pipeline(shader("prelu", env.useFp16), 3, sizeof(PReluPC), std::vector<uint32_t> {});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *s = env.devBuf(node.inputs[0]);
                vk::Buffer *d = env.devBuf(node.outputs[0]);
                // Binding order src/slope/dst matches prelu.comp bindings 0/1/2. One flat 1D grid of
                // 256-wide workgroups (the shader's local_size_x) covers every packed lane in pc.count.
                pipe->dispatch(cmd, {s->handle(), slope->handle(), d->handle()}, &pc, sizeof(pc), groups(pc.count, 256));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::PRelu, PReluOp);
} // namespace vknn
