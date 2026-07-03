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
                // A full-range unit-step Slice is aliased onto its input buffer by the segment planner
                // (geometry-as-metadata); input and output then resolve to the same buffer, so the gather
                // is a no-op — skip it. An initializer data operand is never aliased.
                if (!node.attr.has("pw_steps") && !node.inputs.empty() && node.inputs[0] != kNoTensor && !env.graph->isInitializer(node.inputs[0]))
                {
                    vk::Buffer *src = env.devBuf(node.inputs[0]);
                    vk::Buffer *dst = env.devBuf(node.outputs[0]);
                    if (src && dst && src->handle() == dst->handle())
                    {
                        return; // identity slice with no epilogue: aliased onto its input, dispatch is a no-op
                    }
                }
                impl.record(cmd, node, env);
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::Transpose, TransposeOp);
    VKNN_REGISTER_VK_OP(OpType::Slice, SliceOp);

} // namespace vknn
