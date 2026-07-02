// Channel-axis Softmax on the GPU (NC4HW4, HW==1). supportsNode gates to the channel-softmax case
// (logits [N,C] / [N,C,1,1]); other axes fall back to the CPU softmax.
#include "flat_ops.h"
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        struct SmPC {
            int N, C;
        };
        struct SoftmaxOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            SmPC                                 pc {};
            PwEpi                                epi;
            flat::Softmax                        flatImpl;
            bool                                 flat = false;
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                if (opIsFlat(node, env))
                {
                    flat = true;
                    flatImpl.prepare(node, env);
                    return;
                }
                NCHW x = NCHW::from(env.graph->desc(node.inputs[0]).shape);
                pc     = {(int) x.n, (int) x.c};
                epi.prepare(node, env, /*flat=*/false, env.graph->desc(node.outputs[0]).shape);
                pipe = env.pipeline(shader((std::string("softmax") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(SmPC), std::vector<uint32_t> {});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                if (flat)
                {
                    flatImpl.record(cmd, node, env);
                    return;
                }
                vk::Buffer           *s    = env.devBuf(node.inputs[0]);
                vk::Buffer           *d    = env.devBuf(node.outputs[0]);
                std::vector<VkBuffer> bufs = {s->handle(), d->handle()};
                epi.append(bufs, node, env, d->handle());
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), (uint32_t) pc.N);
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Softmax, SoftmaxOp);
} // namespace vknn
