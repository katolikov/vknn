// DepthToSpace on the GPU: [N,C,H,W] -> [N,C/b^2,H*b,W*b]. Two paths, chosen by depthToSpaceIsNc4.
//
// Packed (both channel counts 4-aligned): one thread per output NC4HW4 block-pixel assembles the
// block's four source channels and stores one quad. Nothing converts around it.
//
// Flat row-major (anything else): one thread per output element does the DCR/CRD index remap, and
// the layout pass converts at each NC4HW4 boundary.
#include "flat_ops.h"
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        struct D2sPC {
            int total, N, C, H, W, C2, OH, OW, b, mode;
        };
        // Byte-matched to shaders/depth_to_space_nc4.comp's push_constant block; the packed kernel
        // derives its own block counts, so it needs no thread total in the block.
        struct D2sNc4PC {
            int N, C, H, W, C2, OH, OW, b, mode;
        };

        struct DepthToSpaceOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            D2sPC                                pc {};
            D2sNc4PC                             nc4Pc {};
            uint32_t                             nc4Count = 0; // non-zero on the packed path
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                NCHW x = NCHW::from(env.graph->desc(node.inputs[0]).shape);
                int  b = (int) node.attr.geti("blocksize", 1);
                if (b < 1)
                {
                    b = 1;
                }
                // Output channels shrink by b^2 while each spatial dim grows by b, so total element
                // count is preserved; the shader reads mode to pick the DCR vs CRD channel unpacking.
                int C2 = (int) x.c / (b * b), OH = (int) x.h * b, OW = (int) x.w * b;
                int mode = node.attr.gets("mode", "DCR") == "CRD" ? 1 : 0;
                // total (thread count) is computed in int64 to avoid overflowing the N*C2*OH*OW
                // product before it is narrowed into the int push-constant field.
                pc = {(int) ((int64_t) x.n * C2 * OH * OW), (int) x.n, (int) x.c, (int) x.h, (int) x.w, C2, OH, OW, b, mode};
                if (depthToSpaceIsNc4(*env.graph, node))
                {
                    nc4Pc    = {(int) x.n, (int) x.c, (int) x.h, (int) x.w, C2, OH, OW, b, mode};
                    nc4Count = (uint32_t) ((int64_t) x.n * cBlocks(C2) * OH * OW); // one lane per output block-pixel
                    pipe = env.pipeline(shader("depth_to_space_nc4", env.useFp16), 2, sizeof(D2sNc4PC), std::vector<uint32_t> {env.convLocalSize});
                    return;
                }
                pipe = env.pipeline(shader("flat_depth_to_space", env.useFp16), 2, sizeof(D2sPC), std::vector<uint32_t> {env.flatLocalSize});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                std::vector<VkBuffer> bufs = {env.devBuf(node.inputs[0])->handle(), env.devBuf(node.outputs[0])->handle()};
                if (nc4Count)
                {
                    pipe->dispatch(cmd, bufs, &nc4Pc, sizeof(nc4Pc), groups(nc4Count, env.convLocalSize));
                    return;
                }
                // One thread per output element; flat_depth_to_space.comp is local_size_x=256 ==
                // flat::kFlatLocalSize. groups() rounds pc.total up to whole workgroups.
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(pc.total, env.flatLocalSize));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::DepthToSpace, DepthToSpaceOp);
} // namespace vknn
