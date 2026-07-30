// Winograd F(m,3) transform construction from interpolation points (the Lavin/wincnn Cook-Toom
// form: Lagrange interpolation with the point at infinity, scale factors gathered into G), plus
// the host-side scoring machinery for candidate point sets: transform condition numbers and an
// fp16 pipeline simulation that mirrors the GPU Winograd path stage by stage.
//
// Consumers are offline/host-only: tools/wino_f63_points.cpp (the point-selection search) and
// tests/test_wino_f63.cpp (transform correctness and point-set pinning). The engine itself reads
// only the baked tables in core/wino_f63.h; nothing here runs at inference time.
//
// Construction, for F(m, 3) with alpha = m + 2 transform points (alpha - 1 finite points p_i plus
// the point at infinity), with M(x) = prod_i (x - p_i) and c_i = prod_{j != i} (p_i - p_j):
//   G  (alpha x 3):     row i = [1, p_i, p_i^2] / c_i;             infinity row = [0, 0, 1]
//   B^T (alpha x alpha): row i = coefficients of M(x) / (x - p_i); infinity row = coefficients of M(x)
//   A^T (m x alpha):     column i = [1, p_i, ..., p_i^(m-1)];      infinity column = e_{m-1}
// The construction reproduces the engine's F(2,3) matrices for points {0, 1, -1} and its F(4,3)
// matrices for points {0, 1, -1, 2, -2} exactly (pinned by tests/test_wino_f63.cpp).
#pragma once
#include "vknn/dtype.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace vknn {

    // Kernel taps of the 3x3 separable Winograd family this header constructs (F(m,3) only).
    constexpr int kWinoKernelTaps = 3;

    struct WinoMatrices {
        int                 outTile = 0; // m: output tile edge
        int                 alpha   = 0; // m + kWinoKernelTaps - 1: transform edge / point count
        std::vector<double> filterG;     // G,   alpha x kWinoKernelTaps, row-major
        std::vector<double> inputBt;     // B^T, alpha x alpha, row-major
        std::vector<double> outputAt;    // A^T, outTile x alpha, row-major
    };

    // Coefficients (ascending powers) of prod_r (x - roots[r]).
    inline std::vector<double> winoPolyFromRoots(const std::vector<double> &roots) {
        std::vector<double> coeffs {1.0};
        for (double root: roots)
        {
            std::vector<double> next(coeffs.size() + 1, 0.0);
            for (size_t j = 0; j < coeffs.size(); ++j)
            {
                next[j] -= root * coeffs[j];
                next[j + 1] += coeffs[j];
            }
            coeffs = std::move(next);
        }
        return coeffs;
    }

    // Build G / B^T / A^T for F(outTile, 3) from the alpha - 1 finite interpolation points.
    inline WinoMatrices buildWinoMatrices(int outTile, const std::vector<double> &finitePoints) {
        WinoMatrices w;
        w.outTile       = outTile;
        w.alpha         = outTile + kWinoKernelTaps - 1;
        const int alpha = w.alpha;
        w.filterG.assign((size_t) alpha * kWinoKernelTaps, 0.0);
        w.inputBt.assign((size_t) alpha * alpha, 0.0);
        w.outputAt.assign((size_t) outTile * alpha, 0.0);
        if ((int) finitePoints.size() != alpha - 1)
        {
            w.outTile = 0; // caller passed the wrong point count; an empty tile marks the failure
            return w;
        }
        for (int i = 0; i < alpha - 1; ++i)
        {
            const double p = finitePoints[i];
            // c_i and the reduced node polynomial M(x) / (x - p_i).
            double              scale = 1.0;
            std::vector<double> others;
            others.reserve(finitePoints.size() - 1);
            for (int j = 0; j < alpha - 1; ++j)
            {
                if (j != i)
                {
                    scale *= (p - finitePoints[j]);
                    others.push_back(finitePoints[j]);
                }
            }
            double power = 1.0;
            for (int k = 0; k < kWinoKernelTaps; ++k)
            {
                w.filterG[(size_t) i * kWinoKernelTaps + k] = power / scale;
                power *= p;
            }
            const std::vector<double> reduced = winoPolyFromRoots(others); // degree alpha - 2
            for (int j = 0; j < alpha - 1; ++j)
            {
                w.inputBt[(size_t) i * alpha + j] = reduced[(size_t) j];
            }
            power = 1.0;
            for (int t = 0; t < outTile; ++t)
            {
                w.outputAt[(size_t) t * alpha + i] = power;
                power *= p;
            }
        }
        // The point at infinity: leading filter coefficient, the full node polynomial M(x), and the
        // last output row.
        w.filterG[(size_t) (alpha - 1) * kWinoKernelTaps + (kWinoKernelTaps - 1)] = 1.0;
        const std::vector<double> node                                            = winoPolyFromRoots(finitePoints); // degree alpha - 1
        for (int j = 0; j < alpha; ++j)
        {
            w.inputBt[(size_t) (alpha - 1) * alpha + j] = node[(size_t) j];
        }
        w.outputAt[(size_t) (outTile - 1) * alpha + (alpha - 1)] = 1.0;
        // wincnn's first-row sign normalization: a paired flip of (G row 0, B^T row 0) leaves the
        // algorithm identical — the per-point product (G_0 g)(B^T_0 d) is unchanged — and makes the
        // leading filter coefficient positive. With it, points {0, 1, -1} reproduce the engine's
        // F(2,3) matrices exactly (F(4,3)'s c_0 is already positive and needs no flip).
        if (w.filterG[0] < 0.0)
        {
            const auto flip = [](double v) {
                return v == 0.0 ? 0.0 : -v;
            }; // keep zeros +0.0
            for (int k = 0; k < kWinoKernelTaps; ++k)
            {
                w.filterG[(size_t) k] = flip(w.filterG[(size_t) k]);
            }
            for (int j = 0; j < alpha; ++j)
            {
                w.inputBt[(size_t) j] = flip(w.inputBt[(size_t) j]);
            }
        }
        return w;
    }

    // Eigenvalues of a symmetric n x n matrix by cyclic Jacobi rotations (n <= 8 here; plenty for
    // the transform Gram matrices). Returns the diagonal after convergence, unsorted.
    inline std::vector<double> winoSymmetricEigenvalues(std::vector<double> a, int n) {
        constexpr int    kMaxSweeps       = 64;
        constexpr double kOffDiagonalStop = 1e-26;
        for (int sweep = 0; sweep < kMaxSweeps; ++sweep)
        {
            double off = 0.0;
            for (int p = 0; p < n; ++p)
            {
                for (int q = p + 1; q < n; ++q)
                {
                    off += a[(size_t) p * n + q] * a[(size_t) p * n + q];
                }
            }
            if (off < kOffDiagonalStop)
            {
                break;
            }
            for (int p = 0; p < n; ++p)
            {
                for (int q = p + 1; q < n; ++q)
                {
                    const double apq = a[(size_t) p * n + q];
                    if (apq == 0.0)
                    {
                        continue;
                    }
                    const double theta = 0.5 * std::atan2(2.0 * apq, a[(size_t) q * n + q] - a[(size_t) p * n + p]);
                    const double c = std::cos(theta), s = std::sin(theta);
                    for (int k = 0; k < n; ++k)
                    {
                        const double akp      = a[(size_t) k * n + p];
                        const double akq      = a[(size_t) k * n + q];
                        a[(size_t) k * n + p] = c * akp - s * akq;
                        a[(size_t) k * n + q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < n; ++k)
                    {
                        const double apk      = a[(size_t) p * n + k];
                        const double aqk      = a[(size_t) q * n + k];
                        a[(size_t) p * n + k] = c * apk - s * aqk;
                        a[(size_t) q * n + k] = s * apk + c * aqk;
                    }
                }
            }
        }
        std::vector<double> eig((size_t) n);
        for (int k = 0; k < n; ++k)
        {
            eig[(size_t) k] = a[(size_t) k * n + k];
        }
        return eig;
    }

    // Spectral (2-norm) condition number of a general rows x cols matrix: sqrt of the eigenvalue
    // ratio of the smaller-side Gram matrix. Infinity when the matrix is rank-deficient.
    inline double winoCondition2(const std::vector<double> &m, int rows, int cols) {
        const int           n = std::min(rows, cols);
        std::vector<double> gram((size_t) n * n, 0.0);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                double dot = 0.0;
                for (int k = 0; k < std::max(rows, cols); ++k)
                {
                    const double a = (rows <= cols) ? m[(size_t) i * cols + k] : m[(size_t) k * cols + i];
                    const double b = (rows <= cols) ? m[(size_t) j * cols + k] : m[(size_t) k * cols + j];
                    dot += a * b;
                }
                gram[(size_t) i * n + j] = dot;
            }
        }
        const std::vector<double> eig = winoSymmetricEigenvalues(gram, n);
        double                    lo = std::numeric_limits<double>::infinity(), hi = 0.0;
        for (double e: eig)
        {
            lo = std::min(lo, e);
            hi = std::max(hi, e);
        }
        if (lo <= 0.0)
        {
            return std::numeric_limits<double>::infinity();
        }
        return std::sqrt(hi / lo);
    }

    // fp32 -> fp16 bit narrowing with IEEE round-to-nearest-even, no saturation: the exact bits an
    // RTE FConvert produces, mirroring shaders/f16bits.glsl vknnF32ToF16RteBits (which is what the
    // GPU's f16vec4() narrowing rounds like under the store16.glsl RoundingModeRTE execution mode).
    // Overflow goes to infinity, exactly like the shader helper.
    inline uint16_t floatToHalfRteBits(float x) {
        uint32_t f;
        std::memcpy(&f, &x, 4);
        const uint32_t sign = (f >> 16) & 0x8000u;
        const uint32_t exp  = (f >> 23) & 0xFFu;
        uint32_t       mant = f & 0x7FFFFFu;
        if (exp == 255u)
        {
            return (uint16_t) (sign | 0x7C00u | (mant != 0u ? 0x200u : 0u)); // inf / quieted nan
        }
        const int rebased = (int) exp - 112; // rebias 127 -> 15
        if (rebased >= 31)
        {
            return (uint16_t) (sign | 0x7C00u); // overflow -> inf
        }
        if (rebased <= 0)
        {
            if (rebased < -10)
            {
                return (uint16_t) sign;
            }
            mant |= 0x800000u;
            const uint32_t shift   = (uint32_t) (14 - rebased);
            uint32_t       half    = mant >> shift;
            const uint32_t rem     = mant & ((1u << shift) - 1u);
            const uint32_t halfway = 1u << (shift - 1u);
            if (rem > halfway || (rem == halfway && (half & 1u) == 1u))
            {
                half += 1u;
            }
            return (uint16_t) (sign | half);
        }
        uint32_t       half = ((uint32_t) rebased << 10) | (mant >> 13);
        const uint32_t rem  = mant & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (half & 1u) == 1u))
        {
            half += 1u; // a carry rolls into the exponent correctly
        }
        return (uint16_t) (sign | half);
    }

    // The value a GPU fp16 intermediate store round-trips to: RTE narrowing without saturation
    // (wino_input / wino_gemm store through plain f16vec4() under the RTE execution mode).
    inline float roundFp16GpuStore(float x) {
        return halfToFloat(floatToHalfRteBits(x));
    }

    // The value a GPU fp16 ACTIVATION store round-trips to: TO_STORE = RTE narrowing of the
    // fp16-range-saturated value (store16.glsl).
    inline float roundFp16GpuActStore(float x) {
        return halfToFloat(floatToHalfRteBits(saturateToFp16Range(x)));
    }

    // The value a HOST-uploaded fp16 weight round-trips to: floatToHalfSat (saturating, ties away
    // from zero) — the uploadWeight() path prepareWinograd's U takes.
    inline float roundFp16HostUpload(float x) {
        return halfToFloat(floatToHalfSat(x));
    }

    // Deterministic standard-normal stream: Box-Muller over raw mt19937 words, so every platform
    // and standard library draws the identical sequence (std::normal_distribution is
    // implementation-defined and would unpin the tests).
    struct WinoNormalStream {
        std::mt19937 rng;
        bool         hasSpare = false;
        double       spare    = 0.0;
        explicit WinoNormalStream(uint32_t seed): rng(seed) {
        }
        double next() {
            if (hasSpare)
            {
                hasSpare = false;
                return spare;
            }
            constexpr double kTwoPi = 6.283185307179586;
            const double     u1     = ((double) rng() + 1.0) * (1.0 / 4294967296.0); // (0, 1]
            const double     u2     = ((double) rng() + 1.0) * (1.0 / 4294967296.0);
            const double     radius = std::sqrt(-2.0 * std::log(u1));
            const double     angle  = kTwoPi * u2;
            spare                   = radius * std::sin(angle);
            hasSpare                = true;
            return radius * std::cos(angle);
        }
    };

    // One conv layer the fp16 pipeline is scored on. inputScale/inputBias shape the activation
    // range (1/0 = unit normal; wider scale + bias models post-BatchNorm activations).
    struct WinoLayerConfig {
        int      cin        = 32;
        int      cout       = 32;
        int      height     = 24;
        int      width      = 24;
        float    inputScale = 1.0f;
        float    inputBias  = 0.0f;
        uint32_t seed       = 1;
    };

    struct WinoFp16Error {
        double maxAbs = 0.0; // max |winograd fp16 pipeline - fp64 direct| over the output map
        double snrDb  = 0.0; // 10 log10(sum ref^2 / sum err^2)
    };

    // The conv layers candidate point sets are scored on, shared by the selection tool
    // (tools/wino_f63_points.cpp) and the pinning test (tests/test_wino_f63.cpp) so both rank
    // candidates by the identical deterministic evaluation: unit-normal activations at two channel
    // depths, a post-BatchNorm-range regime (wider scale, shifted mean), and a larger map for tile
    // coverage.
    inline const std::vector<WinoLayerConfig> &winoScoreConfigs() {
        static const std::vector<WinoLayerConfig> configs = {
            {32, 32, 24, 24, 1.0f, 0.0f, 11},
            {64, 64, 24, 24, 1.0f, 0.0f, 22},
            {32, 32, 24, 24, 2.5f, 0.5f, 33},
            {16, 16, 36, 36, 1.0f, 0.0f, 44},
        };
        return configs;
    }

    namespace winodetail {

        // 3x3 pad-1 direct convolution in double on already-quantized inputs: the reference the
        // fp16 Winograd pipeline is scored against.
        inline std::vector<double> directConv3x3Fp64(const std::vector<float> &input, int cin, int height, int width, const std::vector<float> &weights, int cout) {
            std::vector<double> out((size_t) cout * height * width, 0.0);
            for (int oc = 0; oc < cout; ++oc)
            {
                for (int oy = 0; oy < height; ++oy)
                {
                    for (int ox = 0; ox < width; ++ox)
                    {
                        double acc = 0.0;
                        for (int ic = 0; ic < cin; ++ic)
                        {
                            for (int ky = 0; ky < kWinoKernelTaps; ++ky)
                            {
                                for (int kx = 0; kx < kWinoKernelTaps; ++kx)
                                {
                                    const int iy = oy - 1 + ky, ix = ox - 1 + kx;
                                    if (iy < 0 || iy >= height || ix < 0 || ix >= width)
                                    {
                                        continue;
                                    }
                                    acc += (double) input[((size_t) ic * height + iy) * width + ix] * (double) weights[(((size_t) oc * cin + ic) * kWinoKernelTaps + ky) * kWinoKernelTaps + kx];
                                }
                            }
                        }
                        out[((size_t) oc * height + oy) * width + ox] = acc;
                    }
                }
            }
            return out;
        }

    } // namespace winodetail

    // Full-precision Winograd 3x3 pad-1 convolution: double math throughout, no fp16 stores.
    // Validates the transform algebra of a point set (winograd == direct to fp64 tolerance).
    inline std::vector<double> winoConv2dFp64(const WinoMatrices &w, const std::vector<double> &input, int cin, int height, int width, const std::vector<double> &weights, int cout) {
        const int           m = w.outTile, alpha = w.alpha;
        const int           tilesY = (height + m - 1) / m, tilesX = (width + m - 1) / m;
        std::vector<double> out((size_t) cout * height * width, 0.0);
        // U = G g G^T per (oc, ic).
        std::vector<double> u((size_t) alpha * alpha * cout * cin, 0.0);
        for (int oc = 0; oc < cout; ++oc)
        {
            for (int ic = 0; ic < cin; ++ic)
            {
                const double *kernel = &weights[(((size_t) oc * cin + ic) * kWinoKernelTaps) * kWinoKernelTaps];
                double        gg[8][kWinoKernelTaps]; // alpha <= 8 across the F(m,3) family
                for (int i = 0; i < alpha; ++i)
                {
                    for (int j = 0; j < kWinoKernelTaps; ++j)
                    {
                        double acc = 0.0;
                        for (int k = 0; k < kWinoKernelTaps; ++k)
                        {
                            acc += w.filterG[(size_t) i * kWinoKernelTaps + k] * kernel[(size_t) k * kWinoKernelTaps + j];
                        }
                        gg[i][j] = acc;
                    }
                }
                for (int i = 0; i < alpha; ++i)
                {
                    for (int j = 0; j < alpha; ++j)
                    {
                        double acc = 0.0;
                        for (int k = 0; k < kWinoKernelTaps; ++k)
                        {
                            acc += gg[i][k] * w.filterG[(size_t) j * kWinoKernelTaps + k];
                        }
                        u[(((size_t) i * alpha + j) * cout + oc) * cin + ic] = acc;
                    }
                }
            }
        }
        for (int ty = 0; ty < tilesY; ++ty)
        {
            for (int tx = 0; tx < tilesX; ++tx)
            {
                // V = B^T d B per input channel, M = sum_ic U .* V per output channel, Y = A^T M A.
                std::vector<double> mAcc((size_t) cout * alpha * alpha, 0.0);
                for (int ic = 0; ic < cin; ++ic)
                {
                    double d[8][8], t[8][8], v[8][8];
                    for (int r = 0; r < alpha; ++r)
                    {
                        const int iy = ty * m - 1 + r;
                        for (int c = 0; c < alpha; ++c)
                        {
                            const int ix = tx * m - 1 + c;
                            d[r][c]      = (iy >= 0 && iy < height && ix >= 0 && ix < width) ? input[((size_t) ic * height + iy) * width + ix] : 0.0;
                        }
                    }
                    for (int i = 0; i < alpha; ++i)
                    {
                        for (int c = 0; c < alpha; ++c)
                        {
                            double acc = 0.0;
                            for (int r = 0; r < alpha; ++r)
                            {
                                acc += w.inputBt[(size_t) i * alpha + r] * d[r][c];
                            }
                            t[i][c] = acc;
                        }
                    }
                    for (int i = 0; i < alpha; ++i)
                    {
                        for (int j = 0; j < alpha; ++j)
                        {
                            double acc = 0.0;
                            for (int c = 0; c < alpha; ++c)
                            {
                                acc += t[i][c] * w.inputBt[(size_t) j * alpha + c];
                            }
                            v[i][j] = acc;
                        }
                    }
                    for (int oc = 0; oc < cout; ++oc)
                    {
                        for (int i = 0; i < alpha; ++i)
                        {
                            for (int j = 0; j < alpha; ++j)
                            {
                                mAcc[((size_t) oc * alpha + i) * alpha + j] += v[i][j] * u[(((size_t) i * alpha + j) * cout + oc) * cin + ic];
                            }
                        }
                    }
                }
                for (int oc = 0; oc < cout; ++oc)
                {
                    double t2[8][8], y[8][8];
                    for (int k = 0; k < m; ++k)
                    {
                        for (int c = 0; c < alpha; ++c)
                        {
                            double acc = 0.0;
                            for (int i = 0; i < alpha; ++i)
                            {
                                acc += w.outputAt[(size_t) k * alpha + i] * mAcc[((size_t) oc * alpha + i) * alpha + c];
                            }
                            t2[k][c] = acc;
                        }
                    }
                    for (int k = 0; k < m; ++k)
                    {
                        for (int j = 0; j < m; ++j)
                        {
                            double acc = 0.0;
                            for (int c = 0; c < alpha; ++c)
                            {
                                acc += t2[k][c] * w.outputAt[(size_t) j * alpha + c];
                            }
                            y[k][j] = acc;
                        }
                    }
                    for (int k = 0; k < m; ++k)
                    {
                        const int oy = ty * m + k;
                        if (oy >= height)
                        {
                            continue;
                        }
                        for (int j = 0; j < m; ++j)
                        {
                            const int ox = tx * m + j;
                            if (ox >= width)
                            {
                                continue;
                            }
                            out[((size_t) oc * height + oy) * width + ox] = y[k][j];
                        }
                    }
                }
            }
        }
        return out;
    }

    // fp16 Winograd pipeline simulation, mirroring the GPU path stage by stage:
    //   input activations   fp16 storage (RTE; a previous op's store16 store)
    //   U = G g G^T         fp32 math (float tables), fp16 storage via floatToHalfSat (host upload)
    //   V = B^T d B         fp32 math in the shader's separable two-stage order, fp16 RTE storage
    //   M = sum_ic U .* V   fp32 accumulation over ascending ic, fp16 RTE storage
    //   Y = A^T M A         fp32 math, saturating RTE fp16 store (TO_STORE)
    // Scored against the fp64 direct conv on the SAME quantized inputs and fp32 weights, so the
    // reported error is the Winograd pipeline's own (transform conditioning + intermediate fp16
    // storage), not input quantization noise.
    inline WinoFp16Error simulateWinoFp16Pipeline(const WinoMatrices &w, const WinoLayerConfig &cfg) {
        const int m = w.outTile, alpha = w.alpha;
        const int cin = cfg.cin, cout = cfg.cout, height = cfg.height, width = cfg.width;
        const int tilesY = (height + m - 1) / m, tilesX = (width + m - 1) / m;
        // float copies of the transform tables: the engine bakes them as float (fp32 GPU math).
        std::vector<float> gF(w.filterG.begin(), w.filterG.end());
        std::vector<float> btF(w.inputBt.begin(), w.inputBt.end());
        std::vector<float> atF(w.outputAt.begin(), w.outputAt.end());

        WinoNormalStream   noise(cfg.seed);
        std::vector<float> input((size_t) cin * height * width);
        for (float &x: input)
        {
            x = roundFp16GpuStore((float) noise.next() * cfg.inputScale + cfg.inputBias);
        }
        // Kaiming-style scale keeps per-output magnitudes realistic across cin.
        const float        weightScale = std::sqrt(2.0f / (float) (kWinoKernelTaps * kWinoKernelTaps * cin));
        std::vector<float> weights((size_t) cout * cin * kWinoKernelTaps * kWinoKernelTaps);
        for (float &x: weights)
        {
            x = (float) noise.next() * weightScale;
        }

        // U, fp32 transform math exactly like prepareWinograd, then the host fp16 upload rounding.
        std::vector<float> u((size_t) alpha * alpha * cout * cin);
        for (int oc = 0; oc < cout; ++oc)
        {
            for (int ic = 0; ic < cin; ++ic)
            {
                const float *kernel = &weights[(((size_t) oc * cin + ic) * kWinoKernelTaps) * kWinoKernelTaps];
                float        gg[8][kWinoKernelTaps];
                for (int i = 0; i < alpha; ++i)
                {
                    for (int j = 0; j < kWinoKernelTaps; ++j)
                    {
                        gg[i][j] = gF[(size_t) i * kWinoKernelTaps + 0] * kernel[j] + gF[(size_t) i * kWinoKernelTaps + 1] * kernel[kWinoKernelTaps + j] + gF[(size_t) i * kWinoKernelTaps + 2] * kernel[2 * kWinoKernelTaps + j];
                    }
                }
                for (int i = 0; i < alpha; ++i)
                {
                    for (int j = 0; j < alpha; ++j)
                    {
                        const float val = gg[i][0] * gF[(size_t) j * kWinoKernelTaps + 0] + gg[i][1] * gF[(size_t) j * kWinoKernelTaps + 1] + gg[i][2] * gF[(size_t) j * kWinoKernelTaps + 2];
                        u[(((size_t) i * alpha + j) * cout + oc) * cin + ic] = roundFp16HostUpload(val);
                    }
                }
            }
        }

        std::vector<float> output((size_t) cout * height * width, 0.0f);
        for (int ty = 0; ty < tilesY; ++ty)
        {
            for (int tx = 0; tx < tilesX; ++tx)
            {
                // V per input channel (fp16-stored), then the fp32-accumulated transform-domain GEMM.
                std::vector<float> v16((size_t) cin * alpha * alpha);
                for (int ic = 0; ic < cin; ++ic)
                {
                    float d[8][8], t[8][8];
                    for (int r = 0; r < alpha; ++r)
                    {
                        const int iy = ty * m - 1 + r;
                        for (int c = 0; c < alpha; ++c)
                        {
                            const int ix = tx * m - 1 + c;
                            d[r][c]      = (iy >= 0 && iy < height && ix >= 0 && ix < width) ? input[((size_t) ic * height + iy) * width + ix] : 0.0f;
                        }
                    }
                    for (int i = 0; i < alpha; ++i)
                    {
                        for (int c = 0; c < alpha; ++c)
                        {
                            float acc = 0.0f;
                            for (int r = 0; r < alpha; ++r)
                            {
                                acc += btF[(size_t) i * alpha + r] * d[r][c];
                            }
                            t[i][c] = acc;
                        }
                    }
                    for (int i = 0; i < alpha; ++i)
                    {
                        for (int j = 0; j < alpha; ++j)
                        {
                            float acc = 0.0f;
                            for (int c = 0; c < alpha; ++c)
                            {
                                acc += t[i][c] * btF[(size_t) j * alpha + c];
                            }
                            v16[((size_t) ic * alpha + i) * alpha + j] = roundFp16GpuStore(acc);
                        }
                    }
                }
                for (int oc = 0; oc < cout; ++oc)
                {
                    float m16[8][8];
                    for (int i = 0; i < alpha; ++i)
                    {
                        for (int j = 0; j < alpha; ++j)
                        {
                            float acc = 0.0f;
                            for (int ic = 0; ic < cin; ++ic)
                            {
                                acc += v16[((size_t) ic * alpha + i) * alpha + j] * u[(((size_t) i * alpha + j) * cout + oc) * cin + ic];
                            }
                            m16[i][j] = roundFp16GpuStore(acc);
                        }
                    }
                    float t2[8][8];
                    for (int k = 0; k < m; ++k)
                    {
                        for (int c = 0; c < alpha; ++c)
                        {
                            float acc = 0.0f;
                            for (int i = 0; i < alpha; ++i)
                            {
                                acc += atF[(size_t) k * alpha + i] * m16[i][c];
                            }
                            t2[k][c] = acc;
                        }
                    }
                    for (int k = 0; k < m; ++k)
                    {
                        const int oy = ty * m + k;
                        if (oy >= height)
                        {
                            continue;
                        }
                        for (int j = 0; j < m; ++j)
                        {
                            const int ox = tx * m + j;
                            if (ox >= width)
                            {
                                continue;
                            }
                            float acc = 0.0f;
                            for (int c = 0; c < alpha; ++c)
                            {
                                acc += t2[k][c] * atF[(size_t) j * alpha + c];
                            }
                            output[((size_t) oc * height + oy) * width + ox] = roundFp16GpuActStore(acc);
                        }
                    }
                }
            }
        }

        const std::vector<double> ref = winodetail::directConv3x3Fp64(input, cin, height, width, weights, cout);
        WinoFp16Error             err;
        double                    refEnergy = 0.0, errEnergy = 0.0;
        for (size_t i = 0; i < ref.size(); ++i)
        {
            const double diff = (double) output[i] - ref[i];
            if (!std::isfinite(diff))
            {
                // A transform overflowed fp16 (inf, then possibly NaN): the point set is
                // fp16-unsafe outright. Score it as the worst possible, not as zero error.
                err.maxAbs = std::numeric_limits<double>::infinity();
                errEnergy  = std::numeric_limits<double>::infinity();
                continue;
            }
            err.maxAbs = std::max(err.maxAbs, std::fabs(diff));
            refEnergy += ref[i] * ref[i];
            errEnergy += diff * diff;
        }
        err.snrDb = (errEnergy > 0.0) ? 10.0 * std::log10(refEnergy / errEnergy) : std::numeric_limits<double>::infinity();
        return err;
    }

} // namespace vknn
