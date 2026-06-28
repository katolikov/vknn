// Transpose and Slice on the GPU — flat row-major only (the layout pass guarantees flat I/O).
#include "flat_ops.h"

namespace vknn {
    namespace {

        struct TransposeOp: VulkanOp {
            flat::Gather impl;
            void         prepare(const Node &node, VkOpEnv &env) override {
                impl.prepare(node, env);
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                impl.record(cmd, node, env);
            }
        };
        struct SliceOp: VulkanOp {
            flat::Gather impl;
            void         prepare(const Node &node, VkOpEnv &env) override {
                impl.prepare(node, env);
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                impl.record(cmd, node, env);
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::Transpose, TransposeOp);
    VKNN_REGISTER_VK_OP(OpType::Slice, SliceOp);

} // namespace vknn
