// Range: arange(start, limit, delta) from three scalar inputs; output is the 1-D vector
// [start, start+delta, ...) with n = max(ceil((limit-start)/delta), 0) elements. Usually
// const-folded (index/position vectors must be exact), but registered so a runtime
// start/limit/delta still works. Emits int64 when every input is int64, else float.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <cmath>

namespace vknn {
    namespace {

        struct RangeCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &S = ctx.t(node.inputs[0]);
                const RtTensor &L = ctx.t(node.inputs[1]);
                const RtTensor &D = ctx.t(node.inputs[2]);
                RtTensor       &Y = ctx.t(node.outputs[0]);

                // Read a single-element scalar input as double, widening whichever dtype it carries
                // (int64 / fp16 / f32) to a common type so the element count `n` is derived uniformly.
                auto scalar = [](const RtTensor &t) {
                    if (t.dtype == DType::Int64)
                    {
                        return (double) t.host.i64()[0];
                    }
                    if (t.dtype == DType::Float16)
                    {
                        return (double) halfToFloat(reinterpret_cast<const fp16_t *>(t.host.bytes.data())[0]);
                    }
                    return (double) t.host.f32()[0];
                };
                double  start = scalar(S), limit = scalar(L), delta = scalar(D);
                // Element count per the ONNX Range rule: n = max(ceil((limit-start)/delta), 0), which
                // also yields 0 for a range whose sign disagrees with `delta`. A zero `delta` is treated
                // as an empty range rather than dividing by zero.
                int64_t n = 0;
                if (delta != 0.0)
                {
                    n = std::max<int64_t>((int64_t) std::ceil((limit - start) / delta), 0);
                }

                // Output dtype follows the inputs: an all-int64 (start, limit, delta) triple yields int64,
                // any other combination yields float.
                bool i64 = S.dtype == DType::Int64 && L.dtype == DType::Int64 && D.dtype == DType::Int64;
                if (i64)
                {
                    // Integer path uses exact int64 arithmetic (s + i*d) from the raw inputs, not the
                    // double-widened `start`/`delta`, so large magnitudes stay bit-exact.
                    int64_t *y = cpu::allocOutI64(Y, {n});
                    int64_t  s = S.host.i64()[0], d = D.host.i64()[0];
                    for (int64_t i = 0; i < n; ++i)
                    {
                        y[i] = s + i * d;
                    }
                } else
                {
                    // Float path accumulates in double then narrows to float, matching typical index/
                    // position-vector generation.
                    float *y = cpu::allocOut(Y, {n});
                    for (int64_t i = 0; i < n; ++i)
                    {
                        y[i] = (float) (start + (double) i * delta);
                    }
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Range, RangeCpu);
} // namespace vknn
