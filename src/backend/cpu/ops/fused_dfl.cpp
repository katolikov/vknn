// CPU oracle of the distribution-focal decode (core/dfl.h): per (image, side, anchor) a softmax
// over the B bins and its expectation against the bin weights, sums in double.
#include "backend/cpu/cpu_backend.h"
#include "core/dfl.h"
#include "vknn/op.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace vknn {
    namespace {
        struct FusedDflCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]); // [N, S*B, L]
                const RtTensor &W = ctx.t(node.inputs[1]); // [1, B, 1, 1]
                RtTensor       &Y = ctx.t(node.outputs[0]);
                const int64_t   B = node.attr.geti("bins", 0);
                const int64_t   N = X.shape[0], C = X.shape[1], L = X.shape[2];
                const int64_t   S = B > 0 ? C / B : 0;
                const float    *x = X.host.f32();
                const float    *w = W.host.f32();
                float          *y = cpu::allocOut(Y, {N, S, L});
                std::vector<double> e((size_t) std::max<int64_t>(B, 1));
                for (int64_t n = 0; n < N; ++n)
                {
                    for (int64_t s = 0; s < S; ++s)
                    {
                        const float *row = x + ((n * C) + s * B) * L; // bin b of anchor l at row[b * L + l]
                        for (int64_t l = 0; l < L; ++l)
                        {
                            float m = row[l];
                            for (int64_t b = 1; b < B; ++b)
                            {
                                m = std::max(m, row[b * L + l]);
                            }
                            double sum = 0, acc = 0;
                            for (int64_t b = 0; b < B; ++b)
                            {
                                e[(size_t) b] = std::exp((double) (row[b * L + l] - m));
                                sum += e[(size_t) b];
                                acc += e[(size_t) b] * (double) w[b];
                            }
                            y[(n * S + s) * L + l] = sum > 0 ? (float) (acc / sum) : 0.f;
                        }
                    }
                }
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::FusedDfl, FusedDflCpu);
} // namespace vknn
