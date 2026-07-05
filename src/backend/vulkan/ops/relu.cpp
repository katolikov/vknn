// Standalone ReLU on the GPU (elementwise over the packed buffer). Used for ReLUs that aren't
// fused into a Conv/Gemm (e.g. the ReLU after a residual Add in ResNet).
#include "vk_op_common.h"

namespace vknn {
    namespace {

        // Local workgroup size along x; matches layout(local_size_x = ...) in shaders/relu.comp.
        constexpr uint32_t kReluLocalSize = 256;

        struct ReluOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            uint32_t                             count = 0;

            void prepare(const Node &node, VkOpEnv &env) override {
                // NC4HW4 packed element count: ReLU is a per-element monotonic map, so running it over the
                // padded channel-block lanes (c rounded up to a multiple of 4) is harmless and keeps the
                // dispatch a single flat 1D range. Two SSBO bindings (src, dst); one uint32 push constant.
                count = (uint32_t) packedElems(env.graph->desc(node.outputs[0]).shape);
                pipe = env.pipeline(shader("relu", env.useFp16), 2, sizeof(uint32_t), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *src = env.devBuf(node.inputs[0]);
                vk::Buffer *dst = env.devBuf(node.outputs[0]);
                // One thread per packed element; ceil-div the count into local-size-x workgroups.
                pipe->dispatch(cmd, {src->handle(), dst->handle()}, &count, sizeof(count),
                               groups(count, kReluLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Relu, ReluOp);
} // namespace vknn
