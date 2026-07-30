// Det: determinant of (batched) square matrices, [..., n, n] -> [...] (rank-2 input -> {1}).
// n <= kDetMaxAnalyticN uses fixed-order cofactor expansion — the SAME expressions, in the SAME
// operation order, as the det_flat GPU kernel, so CPU and GPU agree bitwise in fp32. Larger n
// (gated off the GPU by vkNodeGate) uses partial-pivot LU, the numerically standard general form.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <cmath>
#include <vector>

namespace vknn {
    namespace {

        // Fixed-order cofactor expansions shared (by transcription) with shaders/det_flat.comp.
        // Any change here must be mirrored there, or CPU-vs-GPU byte parity breaks.
        float det2(const float *m) {
            return m[0] * m[3] - m[1] * m[2];
        }
        float det3(const float *m) {
            return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
        }
        float det4(const float *m) {
            // Expansion along the first row with explicit 3x3 minors, first-row order.
            const float s0 = m[10] * m[15] - m[11] * m[14];
            const float s1 = m[9] * m[15] - m[11] * m[13];
            const float s2 = m[9] * m[14] - m[10] * m[13];
            const float s3 = m[8] * m[15] - m[11] * m[12];
            const float s4 = m[8] * m[14] - m[10] * m[12];
            const float s5 = m[8] * m[13] - m[9] * m[12];
            const float c0 = m[5] * s0 - m[6] * s1 + m[7] * s2;
            const float c1 = m[4] * s0 - m[6] * s3 + m[7] * s4;
            const float c2 = m[4] * s1 - m[5] * s3 + m[7] * s5;
            const float c3 = m[4] * s2 - m[5] * s4 + m[6] * s5;
            return m[0] * c0 - m[1] * c1 + m[2] * c2 - m[3] * c3;
        }

        // General n: LU with partial pivoting; the determinant is the pivot product with the
        // permutation sign. CPU-only (the GPU gate refuses n > kDetMaxAnalyticN by name).
        float detLu(const float *src, int64_t n) {
            std::vector<double> a(src, src + n * n);
            double              det = 1.0;
            for (int64_t col = 0; col < n; ++col)
            {
                int64_t pivot = col;
                for (int64_t r = col + 1; r < n; ++r)
                {
                    if (std::fabs(a[(size_t) (r * n + col)]) > std::fabs(a[(size_t) (pivot * n + col)]))
                    {
                        pivot = r;
                    }
                }
                if (a[(size_t) (pivot * n + col)] == 0.0)
                {
                    return 0.0f; // singular
                }
                if (pivot != col)
                {
                    for (int64_t k = 0; k < n; ++k)
                    {
                        std::swap(a[(size_t) (pivot * n + k)], a[(size_t) (col * n + k)]);
                    }
                    det = -det;
                }
                det *= a[(size_t) (col * n + col)];
                for (int64_t r = col + 1; r < n; ++r)
                {
                    const double f = a[(size_t) (r * n + col)] / a[(size_t) (col * n + col)];
                    for (int64_t k = col + 1; k < n; ++k)
                    {
                        a[(size_t) (r * n + k)] -= f * a[(size_t) (col * n + k)];
                    }
                }
            }
            return (float) det;
        }

        struct DetCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X    = ctx.t(node.inputs[0]);
                RtTensor       &Y    = ctx.t(node.outputs[0]);
                const int       rank = (int) X.shape.size();
                const int64_t   n    = rank >= 2 ? X.shape[(size_t) rank - 1] : 0;
                Shape           out(X.shape.begin(), X.shape.end() - 2);
                if (out.empty())
                {
                    out.push_back(1); // the IR has no rank-0 activations
                }
                const int64_t batches = numElements(out);
                const float  *x       = X.host.f32();
                float        *y       = cpu::allocOut(Y, out);
                for (int64_t b = 0; b < batches; ++b)
                {
                    const float *m = x + b * n * n;
                    switch (n)
                    {
                        case 1:
                            y[b] = m[0];
                            break;
                        case 2:
                            y[b] = det2(m);
                            break;
                        case 3:
                            y[b] = det3(m);
                            break;
                        case 4:
                            y[b] = det4(m);
                            break;
                        default:
                            y[b] = detLu(m, n);
                            break;
                    }
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Det, DetCpu);
} // namespace vknn
