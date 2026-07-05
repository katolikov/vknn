// Elementwise binary family on the GPU (Mul/Sub/Div/Max/Min/Pow). Handles same-shape inputs and
// the channel-broadcast case (second operand [N,C,1,1], the Squeeze-Excite scale). Other broadcast
// patterns fall back to the CPU binary op (gated in the backend's supportsNode()).
#include "flat_ops.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct BinaryPC {
            int count, HW, op;
        };

        struct BinaryOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            BinaryPC                             pc {};
            flat::Binary                         flatImpl;
            bool                                 flat = false;

            void prepare(const Node &node, VkOpEnv &env) override {
                if (opIsFlat(node, env))
                {
                    flat = true;
                    flatImpl.prepare(node, env);
                    return;
                }
                NCHW y  = NCHW::from(env.graph->desc(node.outputs[0]).shape);
                NCHW a  = NCHW::from(env.graph->desc(node.inputs[0]).shape);
                NCHW b  = NCHW::from(env.graph->desc(node.inputs[1]).shape);
                // Spatial extent per channel block. The shader divides the linear vec4 index by HW to
                // collapse a broadcast [N,C,1,1] operand to its matching per-channel-block element.
                int  HW = (int) (y.h * y.w);
                // 0 = same shape, 1 = A is the [N,C,1,1] broadcast operand, 2 = B is.
                uint32_t bcast = 0;
                if (y.h * y.w != 1)
                {
                    if (a.h * a.w == 1)
                    {
                        bcast = 1;
                    } else if (b.h * b.w == 1)
                    { bcast = 2; }
                }
                // One GPU thread per NC4HW4 vec4 slot: cBlocks(y.c) packs four channels into each vec4,
                // so the thread count is y.n * ceil(C/4) * HW. The int64 product guards the multiply from
                // overflow before it is truncated to the shader's int `count`.
                pc = {(int) ((int64_t) y.n * cBlocks(y.c) * HW), HW, node.subOp};
                pipe =
                    env.pipeline(shader("binary", env.useFp16), 3, sizeof(BinaryPC), std::vector<uint32_t> {bcast});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                if (flat)
                {
                    flatImpl.record(cmd, node, env);
                    return;
                }
                vk::Buffer *a = env.devBuf(node.inputs[0]);
                vk::Buffer *b = env.devBuf(node.inputs[1]);
                vk::Buffer *c = env.devBuf(node.outputs[0]);
                // 256 = the shader's local_size_x; groups() rounds pc.count up to whole workgroups.
                pipe->dispatch(cmd, {a->handle(), b->handle(), c->handle()}, &pc, sizeof(pc), groups(pc.count, 256));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Binary, BinaryOp);
} // namespace vknn
