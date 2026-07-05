// Channel-axis Concat on the GPU (NC4HW4). Each input is copied into the output at its channel-
// block offset. Only valid when every input's channel count is a multiple of 4 (block alignment);
// the backend's supportsNode() enforces that, otherwise it falls back to the CPU concat.
#include "flat_ops.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        // Mirror of the concat shader's push_constant block. Cib/Cob are input/output channel-block
        // counts (channels/4 in NC4HW4), cbOff is this input's starting channel block in the output,
        // and HW is the flattened spatial extent. One invocation copies one [n][cb][hw] vec4 element.
        struct ConcatPC {
            int N, Cib, Cob, cbOff, HW;
        };

        struct ConcatOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::vector<ConcatPC>                parts; // one per input
            std::vector<int64_t>                 partGroups;
            flat::Concat                         flatImpl;
            bool                                 flat = false;

            void prepare(const Node &node, VkOpEnv &env) override {
                if (opIsFlat(node, env))
                {
                    flat = true;
                    flatImpl.prepare(node, env);
                    return;
                }
                NCHW y   = NCHW::from(env.graph->desc(node.outputs[0]).shape);
                int  Cob = (int) cBlocks(y.c), HW = (int) (y.h * y.w);
                int  cbOff = 0;
                for (TensorId in: node.inputs)
                {
                    NCHW xi  = NCHW::from(env.graph->desc(in).shape);
                    int  Cib = (int) cBlocks(xi.c);
                    parts.push_back({(int) y.n, Cib, Cob, cbOff, HW});
                    // One invocation per [n][cb][hw] vec4; 64 matches the shader's local_size_x.
                    partGroups.push_back(groups((int64_t) y.n * Cib * HW, 64));
                    // Advance the output channel-block cursor so the next input lands after this one.
                    cbOff += Cib;
                }
                pipe = env.pipeline(shader("concat", env.useFp16), 2, sizeof(ConcatPC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                if (flat)
                {
                    flatImpl.record(cmd, node, env);
                    return;
                }
                vk::Buffer *dst = env.devBuf(node.outputs[0]);
                // Each input writes a disjoint channel-block range of the output, so no barriers between them.
                for (size_t i = 0; i < node.inputs.size(); ++i)
                {
                    vk::Buffer *src = env.devBuf(node.inputs[i]);
                    pipe->dispatch(cmd, {src->handle(), dst->handle()}, &parts[i], sizeof(ConcatPC), (uint32_t) partGroups[i]);
                }
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Concat, ConcatOp);
} // namespace vknn
