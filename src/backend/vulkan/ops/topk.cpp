// TopK on the GPU (FLAT row-major path): the k largest (largest=1, default) or smallest (largest=0)
// elements of each axis slice, with values (input dtype) and int64 source indices. One shader
// invocation owns one output slice and runs k selection passes over the axis (shaders/topk.comp),
// breaking value ties on the ascending source index — the ONNX rule, matching the CPU op exactly.
// k is the `k` attribute (opset < 10) or the constant int64 input[1] (opset 10+), clamped to the axis
// length; sorted is always emitted (the order is deterministic). Indices are carried as
// compute-precision float like every flat integer tensor (exact for axis lengths within the compute
// float's integer range — the whole span in fp32, up to 2048 in fp16); the graph boundary repacks
// them to int64.
#include "flat_ops.h"
#include "vk_op_common.h"
#include "vknn/op.h"
#include <algorithm>

namespace vknn {
    namespace {
        // Field order/types mirror shaders/topk.comp's push_constant block.
        struct TopKPC {
            int outer, dim, inner, k, largest, writeIdx;
        };
        struct TopKOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            TopKPC                               pc {};
            bool                                 hasIdx = false;
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g    = *env.graph;
                Shape        in   = g.desc(node.inputs[0]).shape;
                int          rank = (int) in.size();
                int64_t      axis = node.attr.geti("axis", -1);
                if (axis < 0)
                {
                    axis += rank;
                }
                axis = std::max<int64_t>(0, std::min<int64_t>(axis, rank - 1));
                // k: attribute form (opset < 10) or the constant int64 input[1] (opset 10+), clamped to
                // the axis length exactly as the CPU op and infer_shapes do.
                int64_t k = -1;
                if (node.attr.has("k"))
                {
                    k = node.attr.geti("k", -1);
                } else
                {
                    std::vector<int64_t> kv = readI64Param(g, node, "k", 1);
                    if (!kv.empty())
                    {
                        k = kv[0];
                    }
                }
                int64_t dim = rank > 0 ? in[axis] : 0;
                k           = std::max<int64_t>(0, std::min<int64_t>(k, dim));
                int64_t outer = 1, inner = 1;
                for (int64_t i = 0; i < axis; ++i)
                {
                    outer *= in[i];
                }
                for (int64_t i = axis + 1; i < rank; ++i)
                {
                    inner *= in[i];
                }
                hasIdx      = node.outputs.size() > 1 && node.outputs[1] != kNoTensor;
                pc          = {(int) outer, (int) dim, (int) inner, (int) k, node.attr.geti("largest", 1) != 0 ? 1 : 0, hasIdx ? 1 : 0};
                // 3 SSBOs (input, values, indices); the indices binding is bound to the values buffer when
                // the node omits the indices output (the shader's writeIdx flag then skips the store).
                pipe = env.pipeline(shader("topk", env.useFp16), 3, sizeof(TopKPC), std::vector<uint32_t> {});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *src  = env.devBuf(node.inputs[0]);
                vk::Buffer *vals = env.devBuf(node.outputs[0]);
                vk::Buffer *idx  = hasIdx ? env.devBuf(node.outputs[1]) : vals;
                // One invocation per output slice (outer * inner); the shader loops the axis internally.
                // topk.comp is local_size_x=256 == flat::kFlatLocalSize.
                int64_t slices = (int64_t) pc.outer * pc.inner;
                pipe->dispatch(cmd, {src->handle(), vals->handle(), idx->handle()}, &pc, sizeof(pc), groups(slices, flat::kFlatLocalSize));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::TopK, TopKOp);
} // namespace vknn
