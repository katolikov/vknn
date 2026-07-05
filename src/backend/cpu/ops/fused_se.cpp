/// @file
/// CPU reference for the fused Squeeze-Excite middle (OpType::FusedSE).
///
/// Collapses the Squeeze-Excite scale chain GlobalAvgPool -> Conv1x1(+relu) -> Conv1x1 ->
/// HardSigmoid into a single kernel that reads the already-pooled channel descriptor and emits the
/// per-channel scale. The 1x1 convolutions over a [N,C,1,1] tensor are exactly two fully-connected
/// layers (matrix-vector products), so the kernel is FC1 -> ReLU -> FC2 -> HardSigmoid. The
/// GlobalAvgPool and the downstream channel-broadcast Mul are left as separate nodes; this op
/// produces only the scale that the Mul consumes.
///
/// This is the numerically-exact oracle: FC accumulations run in double so the reference does not
/// itself introduce fp32 rounding.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>

namespace vknn {
    namespace {
        struct FusedSeCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                // Operand layout (matches the fusion pass and the Vulkan sibling):
                //   [0] avg  pooled channel descriptor [N,C,1,1]
                //   [1] W1   FC1 weights, row-major [Cr][C]   [2] b1  FC1 bias [Cr] (optional)
                //   [3] W2   FC2 weights, row-major [C][Cr]   [4] b2  FC2 bias [C]  (optional)
                // An absent Conv bias is kNoTensor, decoded here to a null pointer treated as zero.
                const RtTensor    &A  = ctx.t(node.inputs[0]); // pooled avg [N,C,1,1]
                const RtTensor    &W1 = ctx.t(node.inputs[1]);
                const RtTensor    &W2 = ctx.t(node.inputs[3]);
                const float       *b1 = node.inputs[2] != kNoTensor ? ctx.t(node.inputs[2]).host.f32() : nullptr;
                const float       *b2 = node.inputs[4] != kNoTensor ? ctx.t(node.inputs[4]).host.f32() : nullptr;
                RtTensor          &Y  = ctx.t(node.outputs[0]);
                NCHW               x  = NCHW::from(A.shape);
                // C is the full channel width (FC2 output); Cr is the reduced "squeeze" width, taken as
                // W1's row count since W1 is [Cr][C].
                int64_t            N = x.n, C = x.c, Cr = W1.shape[0];
                const float       *avg = A.host.f32();
                const float       *w1  = W1.host.f32();
                const float       *w2  = W2.host.f32();
                // HardSigmoid slope/offset carried through from the fused node: scale = clamp(a*z + b, 0, 1).
                float              a = node.actLo, b = node.actHi;
                float             *y = cpu::allocOut(Y, {N, C, 1, 1});
                std::vector<float> s1(Cr); // FC1 activations for the current image, reused across n
                for (int64_t n = 0; n < N; ++n)
                {
                    // FC1: s1[j] = ReLU( b1[j] + sum_c W1[j][c] * avg[n][c] ), squeeze to Cr channels.
                    for (int64_t j = 0; j < Cr; ++j)
                    {
                        double s = b1 ? b1[j] : 0.0;
                        for (int64_t c = 0; c < C; ++c)
                        {
                            s += (double) w1[j * C + c] * avg[n * C + c];
                        }
                        s1[j] = s > 0 ? (float) s : 0.f;
                    }
                    // FC2 + HardSigmoid: expand back to C channels and clamp to the [0,1] scale.
                    for (int64_t k = 0; k < C; ++k)
                    {
                        double s = b2 ? b2[k] : 0.0;
                        for (int64_t j = 0; j < Cr; ++j)
                        {
                            s += (double) w2[k * Cr + j] * s1[j];
                        }
                        y[n * C + k] = std::min(std::max(a * (float) s + b, 0.f), 1.f);
                    }
                }
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::FusedSE, FusedSeCpu);
} // namespace vknn
