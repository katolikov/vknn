// Elementwise unary family on the GPU (Sigmoid/Tanh/HardSwish/HardSigmoid/LeakyRelu/Elu/...).
// Operates on the packed buffer, so it's correct for any layout.
#include "vk_op_common.h"

namespace vknn {
    namespace {

        // Field order/types mirror unary.comp's push_constant block byte-for-byte.
        // op selects the activation function; a/b carry its scalar parameters (e.g. LeakyRelu
        // slope, Elu alpha, HardSigmoid alpha/beta) via the node's actLo/actHi.
        struct UnaryPC {
            uint32_t count;
            int      op;
            float    a, b;
        };

        struct UnaryOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            UnaryPC                              pc {};
            std::shared_ptr<vk::Buffer>          hold; // when the input is a constant initializer
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                const auto &od = env.graph->desc(node.outputs[0]);
                // flat (B,N,C) buffers are numElements-sized; NC4HW4 buffers are packedElems-sized.
                uint32_t count = (uint32_t) (od.gpuFlat ? numElements(od.shape) : packedElems(od.shape));
                pc             = {count, node.subOp, node.actLo, node.actHi};
                // 2 SSBOs: source (binding 0) and destination (binding 1); the kernel is a pure
                // elementwise map so no packed-layout math is needed here.
                pipe = env.pipeline(shader("unary", env.useFp16), 2, sizeof(UnaryPC), std::vector<uint32_t> {});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *s = operandBuf(env, node.inputs[0], hold);
                vk::Buffer *d = env.devBuf(node.outputs[0]);
                // One flat 1D grid of 256-wide workgroups (matches unary.comp's local_size_x = 256)
                // covers every element/packed lane; the kernel's bounds check drops the tail past count.
                pipe->dispatch(cmd, {s->handle(), d->handle()}, &pc, sizeof(pc), groups(pc.count, 256));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Unary, UnaryOp);
} // namespace vknn
