// MaxPool2D on the GPU. Runs in the NC4HW4 world: one thread per (n, channel-block, out-y, out-x)
// takes the max over its pooling window as a vec4, so the four packed channels of a block reduce
// together in lockstep (see shaders/maxpool.comp).
#include "core/conv_geom.h"
#include "pw_plan.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct MaxPoolOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            MaxPC                                pc {};
            PwEpi                                epi;
            int64_t                              total = 0;

            void prepare(const Node &node, VkOpEnv &env) override {
                NCHW x    = NCHW::from(env.graph->desc(node.inputs[0]).shape);
                NCHW y    = NCHW::from(env.graph->desc(node.outputs[0]).shape);
                auto ints = [&](const char *k, std::vector<int64_t> d) {
                    const auto &v = node.attr.getints(k);
                    return v.empty() ? d : v;
                };
                auto ks  = ints("kernel_shape", {1, 1});
                auto st  = ints("strides", {1, 1});
                // Shared pool geometry (core/conv_geom.h): resolves auto_pad. Only the top/left origin
                // offsets (pad[0], pad[1]) reach the shader, which reads the trailing pad implicitly by
                // clamping to H/W; the end pads are already folded into the output extent (y).
                auto pad = poolGeom(x.h, x.w, node.attr).pads();
                // Field order/types mirror maxpool.comp's push_constant block (N, C, H, W, OH, OW,
                // KH, KW, SH, SW, PT, PL). C is the true channel count; the shader derives the block
                // count as ceil(C/4) itself.
                pc       = {(int) x.n,   (int) x.c,   (int) x.h,   (int) x.w,   (int) y.h,    (int) y.w,
                            (int) ks[0], (int) ks[1], (int) st[0], (int) st[1], (int) pad[0], (int) pad[1]};
                // One thread per output vec4 lane: n * channel-blocks * out-H * out-W. No factor of 4
                // (unlike packedElems), because each thread already handles a full 4-channel block.
                total    = x.n * cBlocks(x.c) * y.h * y.w;
                // flat=false: the epilogue indexes in NC4HW4 (channel-block) space to match the max
                // reduction above, not a flat elementwise layout.
                epi.prepare(node, env, /*flat=*/false, env.graph->desc(node.outputs[0]).shape);
                pipe = env.pipeline(shader((std::string("maxpool") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(MaxPC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer           *src  = env.devBuf(node.inputs[0]);
                vk::Buffer           *dst  = env.devBuf(node.outputs[0]);
                std::vector<VkBuffer> bufs = {src->handle(), dst->handle()};
                epi.append(bufs, node, env, dst->handle());
                // groups(total, 64) matches maxpool.comp's local_size_x = 64.
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(total, 64));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::MaxPool, MaxPoolOp);
} // namespace vknn
