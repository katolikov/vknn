// ConvertLayout: NC4HW4 <-> flat row-major on the GPU. Inserted by the format pass at boundaries
// between NC4HW4-native ops (conv/pool) and the generic flat head ops. node.subOp = direction
// (0: NC4HW4 -> flat, 1: flat -> NC4HW4). Logical NCHW shape is identical on both sides.
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct ConvertPC {
            int N, C, H, W, dir;
        };

        struct ConvertLayoutOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            ConvertPC                            pc {};
            uint32_t                             count = 0;

            void prepare(const Node &node, VkOpEnv &env) override {
                NCHW x     = NCHW::from(env.graph->desc(node.outputs[0]).shape);
                int  dir   = node.subOp; // 0: NC4HW4->flat, 1: flat->NC4HW4
                pc         = {(int) x.n, (int) x.c, (int) x.h, (int) x.w, dir};
                int64_t Cb = cBlocks(x.c), HW = x.h * x.w;
                count = (uint32_t) (x.n * Cb * HW); // one thread per NC4 lane-quad, both directions
                // Binding count is 4: bindings 0/1 (scalar) and 2/3 (vec4) alias the same src/dst
                // buffers, so each direction reads scalars on the flat side and one STORE4 on the NC4 side.
                pipe = env.pipeline(shader("convert_layout", env.useFp16), 4, sizeof(ConvertPC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                // bindings 0/1 = scalar views, 2/3 = vec4 views of the same src/dst buffers
                vk::Buffer *s = env.devBuf(node.inputs[0]);
                vk::Buffer *d = env.devBuf(node.outputs[0]);
                pipe->dispatch(cmd, {s->handle(), d->handle(), s->handle(), d->handle()}, &pc, sizeof(pc), groups(count, 256));
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::ConvertLayout, ConvertLayoutOp);

} // namespace vknn
