// Standalone ReLU on the GPU (elementwise over the packed buffer). Used for ReLUs that aren't
// fused into a Conv/Gemm (e.g. the ReLU after a residual Add in ResNet).
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct ReluOp: VulkanOp {
            std::unique_ptr<vk::ComputePipeline> pipe;
            uint32_t                             count = 0;

            void prepare(const Node &node, VkOpEnv &env) override {
                count = (uint32_t) packedElems(env.graph->desc(node.outputs[0]).shape);
                pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("relu", env.useFp16), 2, sizeof(uint32_t), std::vector<uint32_t> {}, env.cache->handle());
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *src = env.devBuf(node.inputs[0]);
                vk::Buffer *dst = env.devBuf(node.outputs[0]);
                pipe->dispatch(cmd, {src->handle(), dst->handle()}, &count, sizeof(count), groups(count, 256));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Relu, ReluOp);
} // namespace vknn
