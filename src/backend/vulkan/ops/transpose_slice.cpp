// Transpose and Slice on the GPU — flat row-major only (the layout pass guarantees flat I/O).
#include "flat_ops.h"

namespace vknn {
    namespace {

        // Both Transpose and Slice are pure index remaps with no arithmetic, so they share one gather
        // kernel: out[i] = in[base + sum_k outCoord_k * inStride_k]. Transpose permutes the per-axis
        // strides (base 0); Slice folds start offsets into base and the step into inStride. flat::Gather
        // builds the matching push constants per op type in its prepare().
        struct TransposeOp: VulkanOp {
            flat::Gather impl;
            void         prepare(const Node &node, VkOpEnv &env) override {
                impl.prepare(node, env);
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                // An identity-perm Transpose is aliased onto its input by the segment planner (the
                // same geometry-as-metadata rule as Reshape); the gather is then a no-op. Same
                // guards as Slice: a fused epilogue must run, and an initializer operand is never
                // aliased.
                if (!node.attr.has("pw_steps") && !node.inputs.empty() && node.inputs[0] != kNoTensor && !env.graph->isInitializer(node.inputs[0]))
                {
                    vk::Buffer *src = env.devBuf(node.inputs[0]);
                    vk::Buffer *dst = env.devBuf(node.outputs[0]);
                    if (src && dst && src->handle() == dst->handle())
                    {
                        return;
                    }
                }
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
                // is a no-op — skip it. Guards, in order: a folded pointwise epilogue ("pw_steps") still
                // has to run so it cannot be skipped; the data operand must exist; and an initializer
                // operand is never aliased (it lives in its own constant buffer, not the input's).
                if (!node.attr.has("pw_steps") && !node.inputs.empty() && node.inputs[0] != kNoTensor && !env.graph->isInitializer(node.inputs[0]))
                {
                    vk::Buffer *src = env.devBuf(node.inputs[0]);
                    vk::Buffer *dst = env.devBuf(node.outputs[0]);
                    if (src && dst && src->handle() == dst->handle())
                    {
                        return; // identity slice with no epilogue: aliased onto its input, dispatch is a no-op
                    }
                    // Zero-copy: a contiguous unit-step slice whose output the planner made a
                    // sub-buffer view of the input at exactly pc.base elements — bytes already in place.
                    const size_t elemBytes = env.useFp16 ? 2 : 4;
                    if (src && dst && impl.contiguousSlice && dst->hazardRoot() == src->hazardRoot() && dst->rootOffset() == src->rootOffset() + (size_t) impl.pc.base * elemBytes)
                    {
                        return;
                    }
                }
                impl.record(cmd, node, env);
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::Transpose, TransposeOp);
    VKNN_REGISTER_VK_OP(OpType::Slice, SliceOp);

} // namespace vknn
