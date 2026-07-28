// DequantizeLinear: y = (x - x_zero_point) * x_scale, the ONNX affine dequant. Input x is an
// integer tensor (int8/uint8/int32), carried here in the fp32 host storage the importer widens
// every integer payload into, so x is read as fp32 and its values are exact small integers. The
// output is real-valued fp32.
//
// Per-tensor form: scale/zero_point are scalars, applied to every element. Per-axis form: scale
// (and matching zero_point) is a 1-D vector of length dims[axis], selected by the `axis` attribute
// (negative counted from the back); every element on a given axis index shares that channel's
// scale/zp. An absent zero_point input defaults to 0. This is the graph-boundary activation-dequant
// kernel the import-time decomposition keeps as a node when a quantized tensor enters a decomposed
// op from outside (a genuine int8 graph input, or a QuantizeLinear the decomposition could not
// collapse); weight-side dequant of an initializer is folded to a constant at import instead.
#include "backend/cpu/cpu_backend.h"

namespace vknn {
    namespace {

        struct DequantizeLinearCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X      = ctx.t(node.inputs[0]);
                const RtTensor &S      = ctx.t(node.inputs[1]);
                RtTensor       &Y      = ctx.t(node.outputs[0]);
                int64_t         n      = cpu::elemCount(X.shape);
                float          *y      = cpu::allocOut(Y, X.shape);
                const float    *x      = X.host.f32();
                const float    *s      = S.host.f32();
                int64_t         sCount = cpu::elemCount(S.shape); // rank-0 scalar counts as 1 (per-tensor)
                // zero_point is optional; an absent input leaves the default of 0.
                const float *z      = nullptr;
                int64_t      zCount = 0;
                if (node.inputs.size() > 2 && node.inputs[2] != kNoTensor)
                {
                    const RtTensor &Z = ctx.t(node.inputs[2]);
                    z                 = Z.host.f32();
                    zCount            = cpu::elemCount(Z.shape);
                }
                // Per-axis stride: elements between consecutive channel steps. inner==n for the
                // per-tensor form (a single scale spans the whole tensor), so the channel index below
                // collapses to 0.
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
                for (int64_t i = 0; i < n; ++i)
                {
                    int64_t c  = sCount == 1 ? 0 : (i / inner) % sCount;
                    float   zp = z == nullptr ? 0.0f : z[zCount == 1 ? 0 : c];
                    y[i]       = (x[i] - zp) * s[c];
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::DequantizeLinear, DequantizeLinearCpu);
} // namespace vknn
