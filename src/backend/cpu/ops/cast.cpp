// Cast: change dtype. vknn computes in fp32 and carries int64 for shape paths, so we support
// float<->int64 conversions (other integer widths map onto these). Shape unchanged.
#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/int64_arithmetic.h"
#include "import/onnx/onnx_types.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        // ONNX TensorProto.DataType code of BOOL, the one integer target that normalizes instead of
        // truncating.
        constexpr int64_t kOnnxBool = (int64_t) onnx::OnnxType::Bool;

        struct CastCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X     = ctx.t(node.inputs[0]);
                RtTensor       &Y     = ctx.t(node.outputs[0]);
                int64_t         to    = node.attr.geti("to", (int64_t) onnx::OnnxType::Float); // ONNX TensorProto.DataType
                int64_t         n     = cpu::elemCount(X.shape);                               // a rank-0 scalar carries its one element
                bool            inI64 = X.dtype == DType::Int64;
                // Integer targets (UINT8, INT8, UINT16, INT16, INT32, INT64, UINT32, UINT64) are carried as
                // int64 storage (truncate toward zero, ONNX Cast semantics). BOOL (kOnnxBool) is also stored as
                // int64 but has its own truth-test branch below.
                bool outI64 = onnx::isIntegerElementType(to);
                // Two element loops per branch, selected by the (int64 storage, float storage)
                // product of input and output kind; BOOL has its own branch. Casts are elementwise, so
                // index i maps 1:1 and the output keeps X.shape.
                if (to == kOnnxBool)
                {
                    // BOOL is a truth test, not a truncation: any nonzero value (negative, fractional,
                    // infinite or NaN) is 1 and only +0 / -0 are 0, stored as int64 0/1 like the other
                    // integer targets. NaN != 0 compares true under IEEE, so a NaN is 1.
                    int64_t *y = cpu::allocOutI64(Y, X.shape);
                    if (inI64)
                    {
                        const int64_t *x = X.host.i64();
                        for (int64_t i = 0; i < n; ++i)
                        {
                            y[i] = x[i] != 0 ? 1 : 0;
                        }
                    } else
                    {
                        const float *x = X.host.f32();
                        for (int64_t i = 0; i < n; ++i)
                        {
                            y[i] = x[i] != 0.0f ? 1 : 0;
                        }
                    }
                } else if (outI64)
                {
                    int64_t *y = cpu::allocOutI64(Y, X.shape);
                    if (inI64)
                    {
                        const int64_t *x = X.host.i64();
                        for (int64_t i = 0; i < n; ++i)
                        {
                            y[i] = x[i];
                        }
                    } else
                    {
                        const float *x = X.host.f32();
                        for (int64_t i = 0; i < n; ++i)
                        {
                            // Truncates toward zero, matching ONNX Cast to an integer type; a NaN reads 0
                            // and a value outside the int64 range (infinities included) saturates, so the
                            // conversion is defined for every input and the result is the same on every
                            // platform. Narrower int targets round-trip through this int64 store (their
                            // reduced range is enforced downstream, not here).
                            y[i] = cpu::int64FromFp32Operand(x[i]);
                        }
                    }
                } else
                {
                    float *y = cpu::allocOut(Y, X.shape);
                    if (inI64)
                    {
                        const int64_t *x = X.host.i64();
                        for (int64_t i = 0; i < n; ++i)
                        {
                            y[i] = (float) x[i];
                        }
                    } else
                    {
                        const float *x = X.host.f32();
                        for (int64_t i = 0; i < n; ++i)
                        {
                            y[i] = x[i];
                        }
                    }
                }
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Cast, CastCpu);
} // namespace vknn
