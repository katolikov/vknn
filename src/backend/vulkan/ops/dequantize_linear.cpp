// DequantizeLinear on the GPU: y = (x - zero_point) * scale, the ONNX affine dequant. The data input
// x is an integer tensor carried as integer-valued fp32 (the importer widens int8/uint8/int32 payloads
// to fp32 host storage), so the flat kernel reads it at compute precision; the output is real-valued
// fp32. scale (and zero_point, when present) are constant initializers uploaded flat as fp32 buffers:
// a rank-0 scalar is the per-tensor form (one scale/zp spans the tensor), a 1-D vector of length
// dims[axis] the per-axis form (channel selected by the flat index / inner stride, matching the CPU
// op). An absent zero_point defaults to 0. Runs on the flat row-major path (embedding tables are 2-D,
// not NC4HW4-packable); the CPU op in backend/cpu/ops/dequantize_linear.cpp is the reference/oracle.
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {

        // Local workgroup size along x; matches local_size_x in shaders/dequantize_linear.comp.
        constexpr uint32_t kDequantizeLocalSize = 256;

        // Field order/types mirror dequantize_linear.comp's push_constant block byte-for-byte.
        // total is the flat element count; sCount/zCount the scale/zero_point element counts (sCount 1 ==
        // per-tensor, zCount 0 == absent zero_point); inner the per-axis stride (elements between channel
        // steps; total for the per-tensor form so the channel index collapses to 0).
        struct DqPC {
            int total, sCount, zCount, inner;
        };

        // Per-axis stride matching the CPU op: elements between consecutive channel steps for a scale of
        // more than one element. inner stays `total` for the per-tensor form (one scale spans the tensor).
        int perAxisInner(const Node &node, const Shape &xShape, int64_t total, int64_t sCount) {
            if (sCount <= 1)
            {
                return (int) total;
            }
            int64_t rank = (int64_t) xShape.size();
            int64_t axis = node.attr.geti("axis", 1);
            if (axis < 0)
            {
                axis += rank;
            }
            int64_t inner = 1;
            for (int64_t d = axis + 1; d < rank; ++d)
            {
                inner *= xShape[d];
            }
            return (int) (inner < 1 ? 1 : inner);
        }

        struct DequantizeLinearOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          scaleBuf, zpBuf, dataHold;
            DqPC                                 pc {};

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g = *env.graph;
                // A rank-0 scalar scale/zp counts as ONE element (per-tensor): elemCount() clamps the
                // empty-shape count to 1, so the numElements([]) == 0 bug never reintroduces a zero count.
                auto elemCount = [](const Shape &s) {
                    return s.empty() ? 1 : numElements(s);
                };
                int64_t total  = numElements(g.desc(node.outputs[0]).shape);
                int64_t sCount = elemCount(g.desc(node.inputs[1]).shape);
                int64_t zCount = 0;
                // scale (and zero_point, when present) upload flat as fp32 — the shader binds them as
                // float, not STORE, so they stay full precision regardless of the activation precision.
                scaleBuf = upload(*env.ctx, initFloats(g, node.inputs[1]), false);
                if (node.inputs.size() > 2 && node.inputs[2] != kNoTensor)
                {
                    zCount = elemCount(g.desc(node.inputs[2]).shape);
                    zpBuf  = upload(*env.ctx, initFloats(g, node.inputs[2]), false);
                } else
                {
                    // The kernel still binds 4 buffers; a 1-element placeholder keeps zCount == 0 (zp = 0)
                    // without a null binding.
                    zpBuf = upload(*env.ctx, std::vector<float> {0.0f}, false);
                }
                pc   = {(int) total, (int) sCount, (int) zCount, perAxisInner(node, g.desc(node.inputs[0]).shape, total, sCount)};
                pipe = env.pipeline(shader("dequantize_linear", env.useFp16), 4, sizeof(DqPC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                // operandBuf so a constant-initializer data input (rare) uploads flat instead of null-
                // crashing; the common case is a runtime activation (a quantized Gather output).
                vk::Buffer *x = operandBuf(env, node.inputs[0], dataHold);
                pipe->dispatch(cmd, {x->handle(), scaleBuf->handle(), zpBuf->handle(), env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.total, kDequantizeLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::DequantizeLinear, DequantizeLinearOp);
} // namespace vknn
