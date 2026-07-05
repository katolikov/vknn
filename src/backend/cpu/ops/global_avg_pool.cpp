// GlobalAveragePool: collapse each channel's HxW plane to its mean.
// ONNX semantics: a full-plane average pool that reduces every spatial dimension, so an [N,C,H,W]
// input yields an [N,C,1,1] output whose (n,c) entry is the mean over that channel's H*W cells.
#include "backend/cpu/cpu_backend.h"

namespace vknn {
    namespace {

        struct GlobalAvgPoolCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X  = ctx.t(node.inputs[0]);
                RtTensor       &Y  = ctx.t(node.outputs[0]);
                NCHW            x  = NCHW::from(X.shape);
                // Output keeps N and C but collapses the spatial extent to a single 1x1 cell, so the
                // dense output is one scalar per (n, c) laid out with stride 1 over C within each n.
                float          *y  = cpu::allocOut(Y, {x.n, x.c, 1, 1});
                const float    *xd = X.host.f32();
                int64_t         hw = x.h * x.w;
                for (int64_t n = 0; n < x.n; ++n)
                {
                    for (int64_t c = 0; c < x.c; ++c)
                    {
                        // Dense NCHW: channel (n,c)'s plane starts at flat offset (n*C + c)*H*W, and its
                        // H*W elements are contiguous, so a linear walk covers the whole plane.
                        const float *p = xd + (n * x.c + c) * hw;
                        // Accumulate in double to keep the running sum precise over large planes before
                        // narrowing the divided mean back to fp32; the div-by-hw is the plane area H*W.
                        double       s = 0;
                        for (int64_t i = 0; i < hw; ++i)
                        {
                            s += p[i];
                        }
                        y[n * x.c + c] = (float) (s / hw);
                    }
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::GlobalAvgPool, GlobalAvgPoolCpu);
} // namespace vknn
