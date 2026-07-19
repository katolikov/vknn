// Cast: change dtype. vknn computes in fp32, carries int64 for shape paths, and carries real fp64 on
// the double-precision (SVD / camera-head) path, so we support float<->int64<->double conversions
// (other integer widths map onto int64). Shape unchanged. Cast is the fp32<->fp64 bridge: a Cast to
// DOUBLE widens to real fp64, a Cast off a fp64 input reads every bit before narrowing.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        struct CastCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X     = ctx.t(node.inputs[0]);
                RtTensor       &Y     = ctx.t(node.outputs[0]);
                int64_t         to    = node.attr.geti("to", 1); // ONNX TensorProto: 1=FLOAT, 11=DOUBLE, 7=INT64, 6=INT32
                int64_t         n     = cpu::elemCount(X.shape); // a rank-0 scalar carries its one element
                bool            inI64 = X.dtype == DType::Int64;
                bool            inF64 = X.dtype == DType::Float64;
                // Read source element i as a double, exact for every storage the engine carries: int64
                // lanes, native fp64 lanes, or the fp32 compute storage.
                auto src = [&](int64_t i) -> double {
                    if (inI64)
                    {
                        return (double) X.host.i64()[i];
                    }
                    if (inF64)
                    {
                        return X.host.f64()[i];
                    }
                    return (double) X.host.f32()[i];
                };
                // Integer targets are carried as int64 storage (truncate toward zero, ONNX Cast semantics):
                // 2=UINT8 3=INT8 4=UINT16 5=INT16 6=INT32 7=INT64 9=BOOL 12=UINT32 13=UINT64.
                bool outI64 = (to == 2 || to == 3 || to == 4 || to == 5 || to == 6 || to == 7 || to == 9 || to == 12 || to == 13);
                bool outF64 = (to == 11); // DOUBLE -> real fp64
                if (outI64)
                {
                    int64_t *y = cpu::allocOutI64(Y, X.shape);
                    for (int64_t i = 0; i < n; ++i)
                    {
                        // C double-to-integer conversion truncates toward zero, matching ONNX Cast to an
                        // integer type. Narrower int targets round-trip through this int64 store.
                        y[i] = (int64_t) src(i);
                    }
                } else if (outF64)
                {
                    double *y = cpu::allocOutF64(Y, X.shape);
                    for (int64_t i = 0; i < n; ++i)
                    {
                        y[i] = src(i); // widen to real fp64 (or copy a fp64 source verbatim)
                    }
                } else
                {
                    float *y = cpu::allocOut(Y, X.shape);
                    for (int64_t i = 0; i < n; ++i)
                    {
                        y[i] = (float) src(i); // narrow to the fp32 compute storage
                    }
                }
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Cast, CastCpu);
} // namespace vknn
