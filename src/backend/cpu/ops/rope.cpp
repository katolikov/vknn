// Rope (fused rotate-half rotary embedding). Inputs: X [.., H, head], positions [..] (one per
// leading index, int64 or float — same convention as the Gather index), cos/sin tables [P, half]
// with half = head / 2. Per head row at position p:
//   y[j]        = x[j] * cos[p, j] - x[j + half] * sin[p, j]
//   y[j + half] = x[j] * sin[p, j] + x[j + half] * cos[p, j]        for j in [0, half)
// Output shape == input shape. All math in fp32 — this is the numeric oracle the Vulkan rope
// kernel is diffed against. The graph creates Rope only via fuseRope at session load (never parsed
// from ONNX, never serialized); a negative position wraps by the table height like GatherCpu.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
    namespace {

        struct RopeCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X   = ctx.t(node.inputs[0]);
                const RtTensor &P   = ctx.t(node.inputs[1]);
                const RtTensor &Cos = ctx.t(node.inputs[2]);
                const RtTensor &Sin = ctx.t(node.inputs[3]);
                RtTensor       &Y   = ctx.t(node.outputs[0]);

                const int64_t half = node.attr.geti("half", 0);
                const int64_t head = half * 2;
                const int     rank = (int) X.shape.size();
                // headsPerPos = the H axis (rank-2): consecutive head rows sharing one position.
                const int64_t headsPerPos = rank >= 2 ? X.shape[rank - 2] : 1;
                const int64_t rows        = head > 0 ? X.elems() / head : 0;
                // Table height, for the negative-position wrap (mirrors GatherCpu's index wrap).
                const int64_t tableRows = Cos.shape.empty() ? 0 : Cos.shape[0];

                float       *y   = cpu::allocOut(Y, X.shape);
                const float *x   = X.host.f32();
                const float *cos = Cos.host.f32();
                const float *sin = Sin.host.f32();
                for (int64_t r = 0; r < rows; ++r)
                {
                    const int64_t posIdx = r / headsPerPos;
                    int64_t       p      = P.dtype == DType::Int64 ? P.host.i64()[posIdx] : (int64_t) P.host.f32()[posIdx];
                    if (p < 0)
                    {
                        p += tableRows;
                    }
                    const float *cr = cos + p * half;
                    const float *sr = sin + p * half;
                    const float *xr = x + r * head;
                    float       *yr = y + r * head;
                    for (int64_t j = 0; j < half; ++j)
                    {
                        // Strict IEEE mul then sub/add, contraction pinned off: an fma here would
                        // skip the product rounding and drift from the decomposed chain's per-op
                        // rounding (each Mul/Sub/Add of the unfused graph rounds separately),
                        // breaking the fused == decomposed equality this oracle is diffed by.
#pragma clang fp contract(off)
                        const float x1 = xr[j], x2 = xr[j + half];
                        yr[j]        = x1 * cr[j] - x2 * sr[j];
                        yr[j + half] = x1 * sr[j] + x2 * cr[j];
                    }
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Rope, RopeCpu);
} // namespace vknn
