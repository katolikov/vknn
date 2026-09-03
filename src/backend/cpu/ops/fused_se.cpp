// CPU oracle of the fused Squeeze-Excite scale (core/squeeze_excite.h): pools the feature map
// [N,C,H,W] to one value per channel, runs FC1 -> activation -> FC2 -> gate and writes the channel
// scale [N,C,1,1]. Sums accumulate in double, so the result is the reference the GPU kernel's
// fp32 arithmetic is gated against.
#include "backend/cpu/cpu_backend.h"
#include "core/squeeze_excite.h"
#include "vknn/op.h"
#include <algorithm>
#include <cmath>

namespace vknn {
    namespace {
        float seActivate(ActType act, float x) {
            switch (act)
            {
                case ActType::SiLU:
                    return x / (1.0f + std::exp(-x));
                case ActType::HardSwish:
                    return x * std::min(std::max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
                default:
                    return std::max(x, 0.0f); // Relu
            }
        }
        float seGateValue(UnaryType gate, float x, float alpha, float beta) {
            if (gate == UnaryType::Sigmoid)
            {
                return 1.0f / (1.0f + std::exp(-x));
            }
            return std::min(std::max(alpha * x + beta, 0.0f), 1.0f); // HardSigmoid
        }

        struct FusedSeCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X  = ctx.t(node.inputs[0]); // feature map [N,C,H,W]
                const RtTensor &W1 = ctx.t(node.inputs[1]); // [Cr][C][1][1]
                const RtTensor &W2 = ctx.t(node.inputs[3]); // [C][Cr][1][1]
                const float    *b1 = node.inputs[2] != kNoTensor ? ctx.t(node.inputs[2]).host.f32() : nullptr;
                const float    *b2 = node.inputs[4] != kNoTensor ? ctx.t(node.inputs[4]).host.f32() : nullptr;
                RtTensor       &Y  = ctx.t(node.outputs[0]);
                const NCHW      x  = NCHW::from(X.shape);
                const int64_t   N = x.n, C = x.c, Cr = W1.shape[0], hw = x.h * x.w;
                const float    *xd = X.host.f32();
                const float    *w1 = W1.host.f32();
                const float    *w2 = W2.host.f32();
                const ActType   act   = node.fusedAct;
                const UnaryType gate  = (UnaryType) node.subOp;
                const float     alpha = node.actLo, beta = node.actHi;
                float          *y     = cpu::allocOut(Y, {N, C, 1, 1});
                std::vector<float> avg((size_t) C), s1((size_t) Cr); // per image, reused across n
                for (int64_t n = 0; n < N; ++n)
                {
                    for (int64_t c = 0; c < C; ++c)
                    {
                        const float *p = xd + (n * C + c) * hw;
                        double       s = 0;
                        for (int64_t i = 0; i < hw; ++i)
                        {
                            s += p[i];
                        }
                        avg[(size_t) c] = hw > 0 ? (float) (s / (double) hw) : 0.f;
                    }
                    for (int64_t j = 0; j < Cr; ++j)
                    {
                        double s = b1 ? b1[j] : 0.0;
                        for (int64_t c = 0; c < C; ++c)
                        {
                            s += (double) w1[j * C + c] * avg[(size_t) c];
                        }
                        s1[(size_t) j] = seActivate(act, (float) s);
                    }
                    for (int64_t k = 0; k < C; ++k)
                    {
                        double s = b2 ? b2[k] : 0.0;
                        for (int64_t j = 0; j < Cr; ++j)
                        {
                            s += (double) w2[k * Cr + j] * s1[(size_t) j];
                        }
                        y[n * C + k] = seGateValue(gate, (float) s, alpha, beta);
                    }
                }
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::FusedSE, FusedSeCpu);
} // namespace vknn
