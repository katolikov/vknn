// QuantizeLinear on the GPU: y = saturate(roundEven(x / scale) + zero_point), the ONNX affine
// quantize. The rounding is round-half-to-even (ONNX-specified; the shader's roundEven matches the CPU
// op's std::nearbyint under the default round-to-nearest-even mode), and the result saturates into the
// output integer type's range (int8 [-128,127], uint8 [0,255], int32 effectively unbounded). vknn
// computes in fp32 and writes the integer result as integer-valued fp32 (the output tensor's dtype
// label still marks it int8/uint8). scale (and zero_point, when present) are constant initializers
// uploaded flat as fp32 buffers: a rank-0 scalar is the per-tensor form, a 1-D vector of length
// dims[axis] the per-axis form (channel by the flat index / inner stride). An absent zero_point is 0.
// Runs on the flat row-major path; the CPU op in backend/cpu/ops/quantize_linear.cpp is the oracle.
#include "vk_op_common.h"
#include "vknn/op.h"
#include <cmath>

namespace vknn {
    namespace {

        // Local workgroup size along x; matches local_size_x in shaders/quantize_linear.comp.
        constexpr uint32_t kQuantizeLocalSize = 256;

        // Field order/types mirror quantize_linear.comp's push_constant block byte-for-byte. total is the
        // flat element count; sCount/zCount the scale/zero_point element counts (sCount 1 == per-tensor,
        // zCount 0 == absent); inner the per-axis stride; qLo/qHi the output-dtype saturation range.
        struct QPC {
            int   total, sCount, zCount, inner;
            float qLo, qHi;
        };

        // Range the output integer dtype saturates into, matching the CPU op's quantRange. int32 (a
        // bias-quant target) is effectively unbounded; anything but int8 falls back to the uint8 range
        // (the ONNX default when no zero_point pins the type).
        void quantRange(DType dt, float &qLo, float &qHi) {
            if (dt == DType::Int8)
            {
                qLo = -128.0f;
                qHi = 127.0f;
            } else if (dt == DType::Int32)
            {
                qLo = -2147483648.0f;
                qHi = 2147483647.0f;
            } else
            {
                qLo = 0.0f; // UInt8 (and the default)
                qHi = 255.0f;
            }
        }

        // Per-axis stride matching the CPU op (see dequantize_linear.cpp); inner is `total` per-tensor.
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

        struct QuantizeLinearOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          scaleBuf, zpBuf, dataHold;
            QPC                                  pc {};

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g = *env.graph;
                // A rank-0 scalar counts as ONE element (per-tensor): the empty-shape count clamps to 1,
                // so the numElements([]) == 0 bug never reintroduces a zero count.
                auto elemCount = [](const Shape &s) {
                    return s.empty() ? 1 : numElements(s);
                };
                int64_t total  = numElements(g.desc(node.outputs[0]).shape);
                int64_t sCount = elemCount(g.desc(node.inputs[1]).shape);
                int64_t zCount = 0;
                scaleBuf       = upload(*env.ctx, initFloats(g, node.inputs[1]), false);
                if (node.inputs.size() > 2 && node.inputs[2] != kNoTensor)
                {
                    zCount = elemCount(g.desc(node.inputs[2]).shape);
                    zpBuf  = upload(*env.ctx, initFloats(g, node.inputs[2]), false);
                } else
                {
                    zpBuf = upload(*env.ctx, std::vector<float> {0.0f}, false);
                }
                pc = {(int) total, (int) sCount, (int) zCount, perAxisInner(node, g.desc(node.inputs[0]).shape, total, sCount), 0.0f, 0.0f};
                quantRange(g.desc(node.outputs[0]).dtype, pc.qLo, pc.qHi);
                pipe = env.pipeline(shader("quantize_linear", env.useFp16), 4, sizeof(QPC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *x = operandBuf(env, node.inputs[0], dataHold);
                pipe->dispatch(cmd, {x->handle(), scaleBuf->handle(), zpBuf->handle(), env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.total, kQuantizeLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::QuantizeLinear, QuantizeLinearOp);
} // namespace vknn
