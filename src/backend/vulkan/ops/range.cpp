// Flat Range on the GPU: y[i] = start + i * delta over the plan-time output size. start/delta are
// scalar operands — constants upload once, runtime scalars bind from their flat buffers (the
// static plan fixes the element count; the values may still be computed on-GPU). int64 ranges
// (index vectors) const-fold or stay on the exact CPU op, as does a Range whose size cannot
// resolve at plan time.
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {

        struct RangeVk: VulkanOp {
            struct PC {
                int total;
            } pc {};
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          startHold, deltaHold;

            void prepare(const Node &node, VkOpEnv &env) override {
                pc.total = (int) numElements(env.graph->desc(node.outputs[0]).shape);
                pipe     = env.pipeline(shader("range", env.useFp16), 3, sizeof(PC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *s = operandBuf(env, node.inputs[0], startHold);
                vk::Buffer *d = operandBuf(env, node.inputs[2], deltaHold);
                pipe->dispatch(cmd, {s->handle(), d->handle(), env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.total, 256));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Range, RangeVk);
} // namespace vknn
