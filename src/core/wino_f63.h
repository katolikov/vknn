// F(6,3) Winograd: the interpolation point set, its derived transform tables, and the
// deterministic Winograd output-tile (F-unit) selection rule shared by the Vulkan conv op
// (src/backend/vulkan/ops/conv.cpp) and the host gating test (tests/test_wino_f63.cpp).
//
// Point set: {0, +1/2, -1/2, +1, -1, +2, -2} plus the point at infinity — the classic F(6,3)
// points, retained after a NOVA-style search (arXiv 2512.18453: condition-number-driven point
// selection, fractional points allowed). tools/wino_f63_points.cpp enumerates all 2024 symmetric
// rational sets {0, +-a, +-b, +-c} with denominators <= 4 and magnitudes <= 4, scores each by
// transform condition number AND by empirical fp16 pipeline error (the exact GPU discipline:
// fp32 transforms, fp16 U/V/M storage, fp32 GEMM accumulation, saturating RTE output store; vs
// the fp64 direct conv), and the classic set wins the empirical metric outright:
//
//   points               cond(Bt)  cond(At)  worst SNR    worst max-abs
//   {0,+-1/2,+-1,+-2}      30.95    141.81    42.38 dB    2.29e-01   <- chosen (best on both)
//   {0,+-1/2,+-1,+-7/4}    25.80     73.27    41.89 dB    2.95e-01   (empirical runner-up)
//   {0,+-1/2,+-1,+-4/3}    28.88     20.20    36.54 dB    4.37e-01   (condition-number best)
//
// Condition number alone is a poor predictor for THIS pipeline (the classic set ranks 81st of
// 2024 on it): with fp32 accumulation everywhere, the dominant error source is fp16 storage of
// the V/M intermediates, which the condition-minimizing fractional sets make worse, not better.
// The derived matrices follow the Lavin/wincnn Cook-Toom construction (core/wino_construct.h)
// with scale factors gathered into G and the wincnn first-row sign normalization; the same
// construction reproduces the engine's F(2,3)/F(4,3) matrices from their points.
#pragma once
#include <cstdint>

namespace vknn {

    constexpr int kWinoF63OutTile = 6;                             // output tile edge (m)
    constexpr int kWinoF63Alpha   = 8;                             // transform edge (m + 2) and point count
    constexpr int kWinoF63NumPos  = kWinoF63Alpha * kWinoF63Alpha; // transform-domain positions (64)
    // Threads cooperating on one (channel-block, tile) unit in the separable two-stage
    // wino_input6/wino_out6 kernels: one thread per transform-tile column/row.
    constexpr int kWinoF63TransformLanes = 8;

    // The finite interpolation points (the 8th point is at infinity).
    constexpr double kWinoF63Points[kWinoF63Alpha - 1] = {0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0};

    // Filter transform G (8x3): U = G g G^T, applied on the host in fp32 (prepareWinograd).
    constexpr float kWinoF63G[kWinoF63Alpha][3] = {
        {1.0f, 0.0f, 0.0f},
        {0.711111128f, 0.355555564f, 0.177777782f},
        {0.711111128f, -0.355555564f, 0.177777782f},
        {-0.222222224f, -0.222222224f, -0.222222224f},
        {-0.222222224f, 0.222222224f, -0.222222224f},
        {0.0111111114f, 0.0222222228f, 0.0444444455f},
        {0.0111111114f, -0.0222222228f, 0.0444444455f},
        {0.0f, 0.0f, 1.0f},
    };

    // Input transform B^T (8x8): V = B^T d B (shaders/wino_input6_fp16.comp keeps these rows in
    // lockstep as its kBt constant).
    constexpr float kWinoF63Bt[kWinoF63Alpha][kWinoF63Alpha] = {
        {1.0f, 0.0f, -5.25f, 0.0f, 5.25f, 0.0f, -1.0f, 0.0f},
        {0.0f, 2.0f, 4.0f, -2.5f, -5.0f, 0.5f, 1.0f, 0.0f},
        {0.0f, -2.0f, 4.0f, 2.5f, -5.0f, -0.5f, 1.0f, 0.0f},
        {0.0f, 1.0f, 1.0f, -4.25f, -4.25f, 1.0f, 1.0f, 0.0f},
        {0.0f, -1.0f, 1.0f, 4.25f, -4.25f, -1.0f, 1.0f, 0.0f},
        {0.0f, 0.5f, 0.25f, -2.5f, -1.25f, 2.0f, 1.0f, 0.0f},
        {0.0f, -0.5f, 0.25f, 2.5f, -1.25f, -2.0f, 1.0f, 0.0f},
        {0.0f, -1.0f, 0.0f, 5.25f, 0.0f, -5.25f, 0.0f, 1.0f},
    };

    // Output transform A^T (6x8): Y = A^T M A (shaders/wino_out6_fp16.comp keeps these rows in
    // lockstep as its kAt constant).
    constexpr float kWinoF63At[kWinoF63OutTile][kWinoF63Alpha] = {
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f},
        {0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
        {0.0f, 0.25f, 0.25f, 1.0f, 1.0f, 4.0f, 4.0f, 0.0f},
        {0.0f, 0.125f, -0.125f, 1.0f, -1.0f, 8.0f, -8.0f, 0.0f},
        {0.0f, 0.0625f, 0.0625f, 1.0f, 1.0f, 16.0f, 16.0f, 0.0f},
        {0.0f, 0.03125f, -0.03125f, 1.0f, -1.0f, 32.0f, -32.0f, 1.0f},
    };

    // Research cost model C(n) = 2i(n+k-1) + io(n+k-1) + n(n+k-1)(2n+k-1), k = 3, normalized per
    // output tile (n*n). Defined for n in {2, 4, 6}; the automatic rule below consumes only 2 and 4.
    inline double winoCostPerOutput(int n, int64_t cin, int64_t cout) {
        double i = (double) cin, o = (double) cout, e = n + 2; // n + k - 1, k = 3
        return (2.0 * i * e + i * o * e + (double) n * e * (2.0 * n + 2.0)) / (double) (n * n);
    }

    // The automatic Winograd output-tile pick: a DETERMINISTIC shape rule (ADR-0009 — the units
    // round fp16 differently, so a timing race would break run-to-run bit-exactness). F(4,3)'s
    // 4x FLOP / 0.56x traffic saving wins on deep channels; F(2,3)'s smaller transform wins on
    // shallow. F(6,3) is NOT selectable here: it becomes an automatic candidate only after device
    // measurement establishes real thresholds; until then it is reachable solely through the
    // explicit setHint(Hint::WinogradUnit, 6). tests/test_wino_f63.cpp pins this rule's choices
    // over a shape sweep.
    inline int winoAutoUnit(int64_t cin, int64_t cout) {
        return (winoCostPerOutput(4, cin, cout) < winoCostPerOutput(2, cin, cout)) ? 4 : 2;
    }

} // namespace vknn
