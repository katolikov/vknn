// GlobalAveragePool: average each channel over H*W. One workgroup per (n, channel-block); its
// threads cooperatively reduce the spatial dimension (see shaders/avgpool.comp).
#include "pw_plan.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct GlobalAvgPoolOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            PoolPC                               pc {};
            PwEpi                                epi;
            int64_t                              total = 0;

            void prepare(const Node &node, VkOpEnv &env) override {
                NCHW x = NCHW::from(env.graph->desc(node.inputs[0]).shape);
                // PoolPC is byte-matched to shaders/avgpool.comp's push constant {N, C, H, W}.
                pc = {(int) x.n, (int) x.c, (int) x.h, (int) x.w};
                // One workgroup per (n, channel-block): total is the dispatch group count, not an
                // element count. cBlocks(c) = ceil(c/4) is the NC4HW4 channel-block count (four
                // channels per vec4 slot); the shader derives (n, cb) from gl_WorkGroupID.x.
                total = x.n * cBlocks(x.c);
                epi.prepare(node, env, /*flat=*/false, env.graph->desc(node.outputs[0]).shape);
                // Base binding count is 2 (src, dst); a fused pointwise epilogue appends its own
                // operand buffers and swaps in the _epi shader variant via suffix().
                pipe = env.pipeline(shader((std::string("avgpool") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(PoolPC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer           *src  = env.devBuf(node.inputs[0]);
                vk::Buffer           *dst  = env.devBuf(node.outputs[0]);
                // Binding order must match the shader: 0 = src, 1 = dst, then any epilogue operands.
                std::vector<VkBuffer> bufs = {src->handle(), dst->handle()};
                epi.append(bufs, node, env, dst->handle());
                // total is passed as the x group count: one workgroup per (n, channel-block). The
                // pipeline spills the count into y if it exceeds the device's max workgroup count.
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), (uint32_t) total);
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::GlobalAvgPool, GlobalAvgPoolOp);

} // namespace vknn
