// ChannelShuffle on the GPU: out[n][c] = in[n][(c % g) * (C/g) + c / g] — the group-interleave
// channel permutation fuseChannelShuffle folds from the ShuffleNetV2 Reshape/Transpose/Reshape
// chain. ONE dispatch in whichever layout the layout pass assigned (the op is layout-agnostic, so
// it adopts its input's layout and never forces a ConvertLayout): channel_shuffle_flat runs one
// thread per output element on the flat row-major buffer; channel_shuffle_nc4 runs one thread per
// output vec4 lane-quad, gathering each lane's source scalar from its NC4HW4 block. Both kernels
// are pure index remaps with STORE-to-STORE copies, so the output is bit-identical to the chain in
// either precision.
#include "flat_ops.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {
        struct ChannelShufflePC {
            int total, C, HW, groups;
        };
        struct ChannelShuffleOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          hold0; // when the input is a constant initializer
            ChannelShufflePC                     pc {};
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                const Shape &in         = env.graph->desc(node.inputs[0]).shape;
                int          groupCount = (int) node.attr.geti("groups", 1);
                int64_t      batch      = in.size() > 0 ? in[0] : 1;
                int64_t      channels   = in.size() > 1 ? in[1] : 1;
                int64_t      spatial    = 1; // product of every dim after the channel axis
                for (size_t k = 2; k < in.size(); ++k)
                {
                    spatial *= in[k];
                }
                if (opIsFlat(node, env))
                {
                    // total is computed in int64 before narrowing into the int push-constant field.
                    pc = {(int) (batch * channels * spatial), (int) channels, (int) spatial, groupCount};
                    pipe = env.pipeline(shader("channel_shuffle_flat", env.useFp16), 2, sizeof(ChannelShufflePC), std::vector<uint32_t> {env.flatLocalSize});
                } else
                {
                    int64_t channelBlocks = (channels + 3) / 4;
                    pc = {(int) (batch * channelBlocks * spatial), (int) channels, (int) spatial, groupCount};
                    pipe = env.pipeline(shader("channel_shuffle_nc4", env.useFp16), 2, sizeof(ChannelShufflePC), std::vector<uint32_t> {env.flatLocalSize});
                }
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                // Both kernels are local_size_x=256 == flat::kFlatLocalSize; groups() rounds pc.total
                // up to whole workgroups and dispatch() 2D-splits past the device's x-limit.
                pipe->dispatch(cmd, {operandBuf(env, node.inputs[0], hold0)->handle(), env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.total, env.flatLocalSize));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::ChannelShuffle, ChannelShuffleOp);
} // namespace vknn
