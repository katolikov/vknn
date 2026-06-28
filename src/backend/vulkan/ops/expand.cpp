// Expand on the GPU — flat row-major only (the layout pass routes it flat). Broadcast-style gather
// over flat::Broadcast (size-1 dims broadcast to the target shape).
#include "flat_ops.h"

namespace vknn {
    namespace {

        struct ExpandOp: VulkanOp {
            flat::Broadcast impl;
            void            prepare(const Node &node, VkOpEnv &env) override {
                impl.prepare(node, env);
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                impl.record(cmd, node, env);
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Expand, ExpandOp);
} // namespace vknn
