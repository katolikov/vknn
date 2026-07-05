// ONNX Concat: join N same-rank inputs along `axis`, producing an output whose shape equals the
// first input's shape except that the axis dimension is the sum of the inputs' axis dimensions;
// all other dimensions must match across inputs. The layout is row-major, so everything from `axis`
// onward forms one contiguous block per "outer" position (the product of the dims left of `axis`).
// Each input therefore contributes a run of `blockElems` contiguous elements per outer position, and
// successive inputs are laid down side by side along the axis by advancing a per-outer write offset —
// which is the memcpy interleave below. Int64 and float share this structure and differ only in
// element size.
#include "backend/cpu/cpu_backend.h"
#include <cstring>

namespace vknn {
    namespace {

        struct ConcatCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                RtTensor       &Y     = ctx.t(node.outputs[0]);
                int64_t         axis  = node.attr.geti("axis", 0);
                const RtTensor &first = ctx.t(node.inputs[0]);
                int64_t         rank  = (int64_t) first.shape.size();
                // ONNX allows a negative axis counting from the end of the rank.
                if (axis < 0)
                {
                    axis += rank;
                }
                // Inputs from pwCoreInputs on are fused-epilogue operands, not concatenated parts.
                std::vector<TensorId> parts(node.inputs.begin(), node.inputs.begin() + (long) pwCoreInputs(node));
                // Output shape = first input's shape with the axis dimension summed over all parts.
                Shape   out   = first.shape;
                int64_t total = 0;
                for (TensorId in: parts)
                {
                    total += ctx.t(in).shape[axis];
                }
                out[axis]       = total;
                // Elements in one contiguous block: the product of the dims from `axis` to the end.
                // For an input this is its per-outer run length; for `out` it is the destination stride.
                auto blockElems = [&](const Shape &s) {
                    int64_t b = 1;
                    for (int64_t i = axis; i < (int64_t) s.size(); ++i)
                    {
                        b *= s[i];
                    }
                    return b;
                };
                // Outer positions: the product of the dims left of `axis`, shared by every input and
                // by the output. Each outer position is copied independently.
                int64_t outer = 1;
                for (int64_t i = 0; i < axis; ++i)
                {
                    outer *= first.shape[i];
                }
                bool isI64 = first.dtype == DType::Int64;
                if (isI64)
                {
                    int64_t *y        = cpu::allocOutI64(Y, out);
                    // `off` is the running write position along the axis within each output block; it
                    // advances by every input's block length so inputs sit side by side, not overlapped.
                    int64_t  outBlock = blockElems(out), off = 0;
                    for (TensorId in: parts)
                    {
                        const RtTensor &T  = ctx.t(in);
                        int64_t         bk = blockElems(T.shape);
                        for (int64_t o = 0; o < outer; ++o)
                        {
                            std::memcpy(y + o * outBlock + off, T.host.i64() + o * bk, bk * sizeof(int64_t));
                        }
                        off += bk;
                    }
                } else
                {
                    // Float path: identical block interleave as the int64 branch above, 4-byte elements.
                    float  *y        = cpu::allocOut(Y, out);
                    int64_t outBlock = blockElems(out), off = 0;
                    for (TensorId in: parts)
                    {
                        const RtTensor &T  = ctx.t(in);
                        int64_t         bk = blockElems(T.shape);
                        for (int64_t o = 0; o < outer; ++o)
                        {
                            std::memcpy(y + o * outBlock + off, T.host.f32() + o * bk, bk * sizeof(float));
                        }
                        off += bk;
                    }
                }
            }
            bool supportsDType(DType) const override {
                return true;
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Concat, ConcatCpu);
} // namespace vknn
