// QuantizeLinear: y = saturate(round_half_even(x / y_scale) + y_zero_point), the ONNX affine
// quantize. The rounding is round-half-to-even (banker's rounding, ONNX-specified), and the result
// saturates into the range of the output integer type: int8 -> [-128,127], uint8 -> [0,255]. The
// output type is the graph tensor's stamped dtype (the zero_point's dtype at import), defaulting to
// uint8 when no zero_point is supplied.
//
// vknn computes in fp32, so the integer result is written into fp32 host storage as an exact small
// integer value (the tensor's dtype label still marks it int8/uint8). Per-tensor and per-axis forms
// both apply: a per-axis scale/zero_point (1-D of length dims[axis], `axis` attribute, negatives
// from the back) selects each element's channel; a scalar scale spans the whole tensor. This is the
// graph-boundary quantize kernel the import-time decomposition keeps as a node when an int8-declared
// graph output must be re-quantized on the way out; interior activation quantize hops collapse to a
// saturation Clip at import instead.
#include "backend/cpu/cpu_backend.h"
#include <cmath>

namespace vknn {
    namespace {

        // Range the output integer dtype saturates into. int32 (a bias-quant target) is effectively
        // unbounded here; anything else falls back to the uint8 range (the ONNX default when no
        // zero_point pins the type).
        void quantRange(DType dt, double &qmin, double &qmax) {
            if (dt == DType::Int8)
            {
                qmin = -128.0;
                qmax = 127.0;
            } else if (dt == DType::Int32)
            {
                qmin = -2147483648.0;
                qmax = 2147483647.0;
            } else
            {
                qmin = 0.0; // UInt8 (and the default)
                qmax = 255.0;
            }
        }

        struct QuantizeLinearCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X      = ctx.t(node.inputs[0]);
                const RtTensor &S      = ctx.t(node.inputs[1]);
                RtTensor       &Y      = ctx.t(node.outputs[0]);
                int64_t         n      = cpu::elemCount(X.shape);
                const float    *x      = X.host.f32();
                const float    *s      = S.host.f32();
                int64_t         sCount = cpu::elemCount(S.shape); // rank-0 scalar counts as 1 (per-tensor)
                const float    *z      = nullptr;
                int64_t         zCount = 0;
                if (node.inputs.size() > 2 && node.inputs[2] != kNoTensor)
                {
                    const RtTensor &Z = ctx.t(node.inputs[2]);
                    z                 = Z.host.f32();
                    zCount            = cpu::elemCount(Z.shape);
                }
                double qmin, qmax;
                quantRange(ctx.graph->desc(node.outputs[0]).dtype, qmin, qmax);
                int64_t inner = n;
                if (sCount > 1)
                {
                    int64_t rank = (int64_t) X.shape.size();
                    int64_t axis = node.attr.geti("axis", 1);
                    if (axis < 0)
                    {
                        axis += rank;
                    }
                    inner = 1;
                    for (int64_t d = axis + 1; d < rank; ++d)
                    {
                        inner *= X.shape[d];
                    }
                    if (inner < 1)
                    {
                        inner = 1;
                    }
                }
                float *y = cpu::allocOut(Y, X.shape);
                Y.dtype  = ctx.graph->desc(node.outputs[0]).dtype; // keep the int8/uint8 label; storage stays fp32
                for (int64_t i = 0; i < n; ++i)
                {
                    int64_t c  = sCount == 1 ? 0 : (i / inner) % sCount;
                    float   zp = z == nullptr ? 0.0f : z[zCount == 1 ? 0 : c];
                    // std::nearbyint honors the current rounding mode, which is round-to-nearest-even
                    // by default (never changed in this process) -- exactly ONNX's round-half-even.
                    double q = std::nearbyint((double) x[i] / (double) s[c]) + (double) zp;
                    q        = q < qmin ? qmin : (q > qmax ? qmax : q);
                    y[i]     = (float) q;
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::QuantizeLinear, QuantizeLinearCpu);
} // namespace vknn
