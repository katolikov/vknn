// Channel-axis Softmax on the GPU (NC4HW4, HW==1). supportsNode gates to the channel-softmax case
// (logits [N,C] / [N,C,1,1]); other axes fall back to the CPU softmax.
#include "flat_ops.h"
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        // Field order/types mirror softmax.comp's push_constant block. N = number of independent
        // images (one workgroup each); C = logit count reduced over (channel axis).
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
                // A flat (non-NC4HW4) layout routes to the shared flat softmax kernel; otherwise this
                // op reduces over the packed channel axis via the NC4HW4 softmax.comp below.
                if (opIsFlat(node, env))
                {
                    flat = true;
                    flatImpl.prepare(node, env);
                    return;
                }
                // supportsNode has gated to channel-axis softmax with HW==1, so NCHW::from folds the
                // logits to [N,C]: n is the image count, c the channel count reduced over.
                NCHW x = NCHW::from(env.graph->desc(node.inputs[0]).shape);
                pc     = {(int) x.n, (int) x.c};
                epi.prepare(node, env, /*flat=*/false, env.graph->desc(node.outputs[0]).shape);
                // 2 fixed bindings (source, dest) plus any extra buffers the fused pointwise epilogue
                // binds after them at PW_EPI_BASE=2; suffix() picks the matching PW_EPI shader variant.
                pipe = env.pipeline(shader((std::string("softmax") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(SmPC), std::vector<uint32_t> {});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                if (flat)
                {
                    flatImpl.record(cmd, node, env);
                    return;
                }
                // Bind order must match the shader: source, dest, then epilogue buffers.
                vk::Buffer           *s    = env.devBuf(node.inputs[0]);
                vk::Buffer           *d    = env.devBuf(node.outputs[0]);
                std::vector<VkBuffer> bufs = {s->handle(), d->handle()};
                epi.append(bufs, node, env, d->handle());
                // Group count = N (one workgroup per image; softmax.comp max/sum-reduces the image's
                // channels across the workgroup via LDS, then normalizes in place).
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), (uint32_t) pc.N);
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Softmax, SoftmaxOp);
} // namespace vknn
