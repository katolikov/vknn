// ConstantOfShape: input is an int64 1-D shape tensor; output is a tensor of that shape filled with
// the `value` attribute (default float 0). Usually const-folded, but registered so the parser
// accepts it and so a runtime shape input still works. Emits float (the canonical compute dtype);
// integer `value` attrs are emitted as int64.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
    namespace {

        struct ConstantOfShapeCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &S = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                // The shape operand is a 1-D vector whose `elems()` entries are the output extents,
                // one per output dimension. ONNX specifies it as int64, but a runtime float shape is
                // tolerated by truncating each entry to an integer extent.
                int64_t r = S.elems();
                Shape   out;
                if (S.dtype == DType::Int64)
                {
                    const int64_t *s = S.host.i64();
                    for (int64_t i = 0; i < r; ++i)
                    {
                        out.push_back(s[i]);
                    }
                } else
                {
                    const float *s = S.host.f32();
                    for (int64_t i = 0; i < r; ++i)
                    {
                        out.push_back((int64_t) s[i]);
                    }
                }
                if (out.empty())
                {
                    // An empty shape vector denotes a rank-0 (scalar) output in ONNX; represent it as
                    // a single-element 1-D tensor so the fill loop still writes exactly one value.
                    out = {1};
                }

                // ONNX supplies the fill via a one-element `value` tensor whose element type also
                // dictates the output dtype. The importer records an integer `value` as an Ints
                // attribute and a float `value` as Floats, so the attribute kind is the dtype switch:
                // Ints -> int64 output, anything else (including a missing attribute) -> float output.
                auto    it     = node.attr.map.find("value");
                bool    intVal = it != node.attr.map.end() && it->second.kind == Attr::Ints;
                int64_t n      = numElements(out);
                if (intVal)
                {
                    // The fill is the tensor's single element; an empty list defaults to 0 per ONNX.
                    int64_t  v = it->second.ints.empty() ? 0 : it->second.ints[0];
                    int64_t *y = cpu::allocOutI64(Y, out);
                    for (int64_t i = 0; i < n; ++i)
                    {
                        y[i] = v;
                    }
                } else
                {
                    // Absent `value`, or a float `value` tensor: fill with its single element, or the
                    // ONNX default 0.0 when the attribute is missing or carries no element.
                    float  v = (it != node.attr.map.end() && !it->second.floats.empty()) ? it->second.floats[0] : 0.f;
                    float *y = cpu::allocOut(Y, out);
                    for (int64_t i = 0; i < n; ++i)
                    {
                        y[i] = v;
                    }
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::ConstantOfShape, ConstantOfShapeCpu);
} // namespace vknn
