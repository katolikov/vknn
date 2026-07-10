// FusedAttention: single-query decode attention core in one op —
//   out[row][n] = sum_s softmax_s(q.k^T * scale + mask * maskScale)[s] * v[s][n]
// with q/k/v/mask read through the per-axis strides the fuseDecodeAttention pass composed from the
// folded MatMul operand views (contract in core/fused_attention.h). CPU reference: per attention
// row, fp32 scores over ascending s, max-subtract softmax, fp32 p.V accumulation over ascending s —
// the numeric oracle the GPU kernel's fp32 flash accumulation is validated against (cosine, not
// bytes: the GPU regroups the same fp32 sums across lanes).
#include "core/fused_attention.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/parallel.h"
#include "vknn/op.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace vknn {
    namespace {

        // One row's score vector: scores[s] = dot(q, k_s) * scale + mask[s] * maskScale, fp32 over
        // ascending k. Contraction pinned off for the same reason as matmul.cpp's matmulRow: only
        // the strict IEEE mul+add chain is bit-stable across compiler specializations.
        __attribute__((noinline)) void attentionScores(const float *q, const float *k, const float *mask, float *scores, int64_t C, int64_t hd, int64_t qK, int64_t kN, int64_t kK, int64_t mN, float scale, float maskScale) {
#pragma clang fp contract(off)
            for (int64_t s = 0; s < C; ++s)
            {
                float acc = 0.f;
                for (int64_t d = 0; d < hd; ++d)
                {
                    acc += q[d * qK] * k[s * kN + d * kK];
                }
                scores[s] = mask ? acc * scale + mask[s * mN] * maskScale : acc * scale;
            }
        }

        // One row's context vector: out[n] = sum_s p[s] * v[s][n], fp32 over ascending s.
        __attribute__((noinline)) void attentionContext(const float *p, const float *v, float *out, int64_t C, int64_t hd, int64_t vK, int64_t vN) {
#pragma clang fp contract(off)
            for (int64_t n = 0; n < hd; ++n)
            {
                float acc = 0.f;
                for (int64_t s = 0; s < C; ++s)
                {
                    acc += p[s] * v[s * vK + n * vN];
                }
                out[n] = acc;
            }
        }

        struct FusedAttentionCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const Graph &g = *ctx.graph;

                const std::vector<int64_t> &dims    = node.attr.getints(kFaDims);
                const std::vector<int64_t> &qStride = node.attr.getints(kFaQStride);
                const std::vector<int64_t> &kStride = node.attr.getints(kFaKStride);
                const std::vector<int64_t> &vStride = node.attr.getints(kFaVStride);
                const std::vector<int64_t> &mStride = node.attr.getints(kFaMStride);
                const int64_t               rank    = (int64_t) dims.size();
                const int64_t               C = node.attr.geti(kFaC), hd = node.attr.geti(kFaHd);
                const int64_t               qK = node.attr.geti(kFaQK);
                const int64_t               kN = node.attr.geti(kFaKN), kK = node.attr.geti(kFaKK);
                const int64_t               vN = node.attr.geti(kFaVN), vK = node.attr.geti(kFaVK);
                const int64_t               mN        = node.attr.geti(kFaMN);
                const float                 scale     = node.attr.getf(kFaScale, 1.f);
                const float                 maskScale = node.attr.getf(kFaMaskScale, 1.f);
                const bool                  hasMask   = node.inputs.size() > 3 && node.inputs[3] != kNoTensor;

                const float *q    = ctx.t(node.inputs[0]).host.f32();
                const float *k    = ctx.t(node.inputs[1]).host.f32();
                const float *v    = ctx.t(node.inputs[2]).host.f32();
                const float *mask = hasMask ? ctx.t(node.inputs[3]).host.f32() : nullptr;
                float       *out  = cpu::allocOut(ctx.t(node.outputs[0]), g.desc(node.outputs[0]).shape);

                int64_t rows = 1;
                for (int64_t d: dims)
                {
                    rows *= d;
                }

                // Rows are independent (each writes its own out[row*hd, (row+1)*hd) span), so they
                // partition across threads bit-identically to a serial run.
                cpu::parallelFor(cpu::threadCount(ctx.config), 0, rows, cpu::minChunkForWork(C * hd), [&](int64_t rowBegin, int64_t rowEnd) {
                    std::vector<float> scores((size_t) C);
                    for (int64_t row = rowBegin; row < rowEnd; ++row)
                    {
                        int64_t qBase = 0, kBase = 0, vBase = 0, mBase = 0, rem = row;
                        for (int64_t i = rank - 1; i >= 0; --i)
                        {
                            const int64_t c = rem % dims[i];
                            rem /= dims[i];
                            qBase += c * qStride[i];
                            kBase += c * kStride[i];
                            vBase += c * vStride[i];
                            if (hasMask)
                            {
                                mBase += c * mStride[i];
                            }
                        }
                        attentionScores(q + qBase, k + kBase, mask ? mask + mBase : nullptr, scores.data(), C, hd, qK, kN, kK, mN, scale, maskScale);
                        // Max-subtract softmax in place: fp32 exponentials, fp32 sum over ascending s.
                        float mx = scores[0];
                        for (int64_t s = 1; s < C; ++s)
                        {
                            mx = std::max(mx, scores[(size_t) s]);
                        }
                        float sum = 0.f;
                        for (int64_t s = 0; s < C; ++s)
                        {
                            scores[(size_t) s] = std::exp(scores[(size_t) s] - mx);
                            sum += scores[(size_t) s];
                        }
                        const float inv = 1.f / sum;
                        for (int64_t s = 0; s < C; ++s)
                        {
                            scores[(size_t) s] *= inv;
                        }
                        attentionContext(scores.data(), v + vBase, out + row * hd, C, hd, vK, vN);
                    }
                });
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::FusedAttention, FusedAttentionCpu);
} // namespace vknn
