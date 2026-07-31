// GridSample host-side rules, mirrored by the sampler shaders (shaders/gridsample.comp,
// gridsample_fp16.comp, gridsample_warp.comp, gridsample_warp_fp16.comp and the shared
// shaders/gridsample_taps.glsl). Three contracts live here:
//
//   - the MODE / PADMODE specialization codes src/backend/vulkan/ops/gridsample.cpp bakes into the
//     pipeline, so the attribute strings map to the shader's branch selectors in one place;
//   - the coordinate operand's DECODE PRECISION (the GRID_FP32 / FLOW_FP32 specialization
//     constants) and the warp fold's scalar, which is baked at the precision of the flow operand it
//     multiplies rather than the node's;
//   - the per-pixel tap resolution the shared taps header runs, including the empty-source-plane
//     class: a plane with a zero extent has no in-range tap in ANY padding mode.
//
// Each rule names its shader mirror; tests/test_gridsample_rule.cpp pins both sides.
#pragma once
#include "vknn/dtype.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace vknn {

    /// Sampler MODE (specialization constant 0), mirrored as MODE in every gridsample*.comp.
    constexpr uint32_t kGridSampleModeBilinear = 0;
    constexpr uint32_t kGridSampleModeNearest  = 1;
    constexpr uint32_t kGridSampleModeCubic    = 2;

    /// Padding PADMODE (specialization constant 1), mirrored as PADMODE in shaders/gridsample_taps.glsl.
    constexpr uint32_t kGridSamplePadZeros      = 0;
    constexpr uint32_t kGridSamplePadBorder     = 1;
    constexpr uint32_t kGridSamplePadReflection = 2;

    /// Out-of-range marker a tap resolver returns instead of a column index / row offset; a tap
    /// carrying it contributes zero. Mirrored as GS_TAP_OOB in shaders/gridsample_taps.glsl.
    constexpr int kGridSampleTapOob = -1;

    /// Taps per axis in cubic mode (floor-1 .. floor+2). Mirrored as GS_CUBIC_TAPS.
    constexpr int kGridSampleCubicTaps = 4;

    /// Base buffer bindings, before the epilogue's extra operand buffers: the plain kernel binds
    /// source, grid, dest; the warp kernel binds source, flow, base grid, dest. Mirrored as the
    /// binding slots and the PW_EPI_BASE of each variant.
    constexpr int kGridSamplePlainBuffers = 3;
    constexpr int kGridSampleWarpBuffers  = 4;

    /// The ONNX `mode` attribute as the shader's MODE selector. Unknown spellings sample bilinear,
    /// the ONNX default.
    inline uint32_t gridSampleModeCode(const std::string &mode) {
        if (mode == "nearest")
        {
            return kGridSampleModeNearest;
        }
        if (mode == "cubic" || mode == "bicubic")
        {
            return kGridSampleModeCubic;
        }
        return kGridSampleModeBilinear;
    }

    /// The ONNX `padding_mode` attribute as the shader's PADMODE selector. Unknown spellings pad
    /// with zeros, the ONNX default.
    inline uint32_t gridSamplePadCode(const std::string &pad) {
        if (pad == "border")
        {
            return kGridSamplePadBorder;
        }
        if (pad == "reflection")
        {
            return kGridSamplePadReflection;
        }
        return kGridSamplePadZeros;
    }

    /// GRID_FP32 for the plain kernel: a CONSTANT grid is uploaded fp32 by the op itself, and a
    /// runtime grid binds its own activation buffer — fp32 exactly when pinGridSampleGridFp32
    /// pinned its storage.
    inline bool gridSampleGridWordsFp32(bool gridIsInitializer, bool gridStoreFp32) {
        return gridIsInitializer || gridStoreFp32;
    }

    /// FLOW_FP32 for the warp kernel: the flow always binds its own NC4HW4 activation buffer (the
    /// op uploads only the base grid), so it decodes fp32 exactly when pinSampleCoordFp32 pinned a
    /// runtime flow's storage.
    inline bool gridSampleFlowWordsFp32(bool flowIsInitializer, bool flowStoreFp32) {
        return !flowIsInitializer && flowStoreFp32;
    }

    /// The warp fold's scalar as the kernel receives it in pc.scale, baked at the precision of the
    /// FLOW OPERAND it multiplies (gridSampleFlowWordsFp32 above), not the node's.
    ///
    /// An fp16 flow reproduces the standalone Mul the fold replaces: that Mul's scalar operand was
    /// itself an initializer stored fp16, so the narrowing (floatToHalfSat, saturating out of range
    /// like an imported constant) keeps the fused output byte-identical, and the kernel rounds the
    /// product. An fp32 flow decodes full-precision coordinates because pinSampleCoordFp32 pinned
    /// the cone, and the fp32 CPU oracle multiplies by the exact attribute value: narrowing the
    /// scalar there would put back the ~2^-11 relative coordinate error the pin removes.
    inline float gridSampleWarpScale(float warpScale, bool sessionFp16, bool flowWordsFp32) {
        const bool flowNarrowsToFp16 = sessionFp16 && !flowWordsFp32;
        return flowNarrowsToFp16 ? halfToFloat(floatToHalfSat(warpScale)) : warpScale;
    }

    /// Reflect a continuous coordinate back into [lo, hi] with the ONNX `reflection` rule: the axis
    /// bounces off each edge, folding with period 2*(hi-lo). A degenerate axis (hi <= lo) collapses
    /// to lo. Mirrors reflectc() in shaders/gridsample_taps.glsl.
    inline float gridSampleReflect(float x, float lo, float hi) {
        if (hi <= lo)
        {
            return lo;
        }
        const float range = hi - lo;
        float       t     = std::fmod(x - lo, 2.0f * range);
        if (t < 0.0f)
        {
            t += 2.0f * range;
        }
        if (t > range)
        {
            t = 2.0f * range - t;
        }
        return lo + t;
    }

    /// A source plane with a zero extent holds no pixel at all. Mirrors gridSampleSourceEmpty() in
    /// shaders/gridsample_taps.glsl.
    inline bool gridSampleSourceEmpty(int hin, int win) {
        return hin <= 0 || win <= 0;
    }

    /// Whether a resolved tap can carry kGridSampleTapOob, i.e. whether the tap read has to test
    /// for it. The zeros mode reports out-of-range taps by construction, and an empty source plane
    /// has no in-range tap in ANY padding mode. Mirrors gridSampleTapsCanBeOob() in
    /// shaders/gridsample_taps.glsl, which every variant's tapAt() calls.
    inline bool gridSampleTapsCanBeOob(uint32_t padCode, int hin, int win) {
        return padCode == kGridSamplePadZeros || gridSampleSourceEmpty(hin, win);
    }

    /// The column a tap at `px` reads, or kGridSampleTapOob when it contributes zero. Mirrors
    /// resolveTapX().
    inline int gridSampleResolveTapColumn(int px, uint32_t padCode, int win, int alignCorners) {
        if (win <= 0)
        {
            // No column exists, in any padding mode: the border/reflection clamp below would run
            // over the empty range [0, -1] and the tap would index past an empty source.
            return kGridSampleTapOob;
        }
        if (padCode == kGridSamplePadZeros)
        {
            if (px < 0 || px >= win)
            {
                return kGridSampleTapOob;
            }
        } else if (padCode == kGridSamplePadReflection)
        {
            const float lo = alignCorners == 1 ? 0.0f : -0.5f;
            const float hi = alignCorners == 1 ? (float) (win - 1) : (float) win - 0.5f;
            px             = std::min(std::max((int) std::lround(gridSampleReflect((float) px, lo, hi)), 0), win - 1);
        } else
        {
            px = std::min(std::max(px, 0), win - 1);
        }
        return px;
    }

    /// The row OFFSET (row * win) a tap at `py` reads, or kGridSampleTapOob when it contributes
    /// zero. Mirrors resolveTapRow().
    inline int gridSampleResolveTapRow(int py, uint32_t padCode, int hin, int win, int alignCorners) {
        if (hin <= 0)
        {
            // No row exists, in any padding mode (see gridSampleResolveTapColumn).
            return kGridSampleTapOob;
        }
        if (padCode == kGridSamplePadZeros)
        {
            if (py < 0 || py >= hin)
            {
                return kGridSampleTapOob;
            }
        } else if (padCode == kGridSamplePadReflection)
        {
            const float lo = alignCorners == 1 ? 0.0f : -0.5f;
            const float hi = alignCorners == 1 ? (float) (hin - 1) : (float) hin - 0.5f;
            py             = std::min(std::max((int) std::lround(gridSampleReflect((float) py, lo, hi)), 0), hin - 1);
        } else
        {
            py = std::min(std::max(py, 0), hin - 1);
        }
        return py * win;
    }

    /// The offset a resolved tap reads inside its channel block, or kGridSampleTapOob when the tap
    /// contributes zero. Mirrors tapAt() in every gridsample*.comp variant.
    inline int gridSampleTapOffset(uint32_t padCode, int hin, int win, int column, int rowOffset) {
        if (gridSampleTapsCanBeOob(padCode, hin, win) && (column < 0 || rowOffset < 0))
        {
            return kGridSampleTapOob;
        }
        return rowOffset + column;
    }

} // namespace vknn
