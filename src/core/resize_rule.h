// ONNX Resize geometry, shared by the CPU oracle and the Vulkan kernel.
//
// The two backends must resolve identical source indices and identical tap weights, or the GPU
// silently returns a different picture from the oracle with no fallback to announce it. This header
// holds the wire codes, the bound on the geometry both sides resolve exactly, and the rules they
// evaluate; the resize shaders carry a line-for-line copy of the rule bodies, since GLSL cannot
// include C++. The nearest index has two spellings: a 64-bit reference and the 32-bit form the
// shaders can execute, held equal over the whole accepted set by tests/test_resize_rule.cpp.
#pragma once
#include <cstdint>
#include <limits>
#include <string>

namespace vknn {

    /// Interpolation-mode wire codes, shared with the shader's `mode` push constant. These are the
    /// only three modes ONNX defines; vxResizeMode refuses anything else rather than defaulting.
    constexpr int kResizeModeNearest = 0;
    constexpr int kResizeModeLinear  = 1;
    constexpr int kResizeModeCubic   = 2;

    /// Coordinate-transformation wire codes, shared with the shader's `cm` push constant.
    constexpr int kResizeCoordHalfPixel        = 0; ///< ONNX default, and the fallback for an unknown string.
    constexpr int kResizeCoordAlignCorners     = 1;
    constexpr int kResizeCoordAsymmetric       = 2;
    constexpr int kResizeCoordPytorchHalfPixel = 3;

    /// Nearest-rounding wire codes (the ONNX nearest_mode attribute), shared with the shader's `nm`
    /// push constant. round_prefer_floor is the ONNX default and the fallback for an unknown
    /// string; floor is what a PyTorch `interpolate(mode="nearest")` export carries (paired with
    /// the asymmetric coordinate mode), and the two agree only when every sample lands off the
    /// rounding boundary - at a non-integer scale they differ by one source pixel.
    constexpr int kResizeNearestPreferFloor = 0;
    constexpr int kResizeNearestPreferCeil  = 1;
    constexpr int kResizeNearestFloor       = 2;
    constexpr int kResizeNearestCeil        = 3;

    /// Taps per axis in cubic mode: floor-1 .. floor+2.
    constexpr int kResizeCubicTaps = 4;

    /// Default of the ONNX cubic_coeff_a attribute (the PyTorch/OpenCV cubic-convolution kernel).
    constexpr float kResizeCubicCoeffDefault = -0.75f;

    /// Largest spatial extent (input or output, either axis) a Resize node may carry on either
    /// backend. The half_pixel denominator is 2 * outS, so this bound is what keeps that doubling
    /// inside the 32-bit signed arithmetic the GPU kernels evaluate; the CPU oracle is held to the
    /// same bound so both backends accept exactly the same set of shapes. One axis this long is
    /// 2^30 elements, past what either backend allocates for a whole tensor, so nothing a model
    /// carries approaches it -- and both ops refuse rather than truncating the extent into the
    /// `int` the push constants and the rule below take.
    constexpr int64_t kResizeMaxSpatialExtent = std::numeric_limits<int32_t>::max() / 2;

    /// Largest factor the narrow nearest rule multiplies directly: the greatest k whose square
    /// still fits a 32-bit signed int. Mirrored as RESIZE_NARROW_FACTOR_MAX in the four resize
    /// shaders; a factor pair past it takes vxResizeNearestSrcNarrow's bit-scanned path instead.
    constexpr int kResizeNarrowFactorMax = 46340;
    static_assert((int64_t) kResizeNarrowFactorMax * kResizeNarrowFactorMax <= std::numeric_limits<int32_t>::max(), "kResizeNarrowFactorMax squared must fit a 32-bit signed int");
    static_assert((int64_t) (kResizeNarrowFactorMax + 1) * (kResizeNarrowFactorMax + 1) > std::numeric_limits<int32_t>::max(), "kResizeNarrowFactorMax must be the largest such factor");

    /// Bits of the left factor the narrow nearest rule scans when the direct product would not fit.
    /// Mirrored as RESIZE_NARROW_PRODUCT_BITS in the four resize shaders. A factor is a spatial
    /// extent doubled at most once, so it is non-negative and this many bits describe it fully.
    constexpr int kResizeNarrowProductBits = 31;
    static_assert(kResizeNarrowProductBits == std::numeric_limits<int32_t>::digits, "the scan must cover every value bit of a 32-bit signed int");

    /// Interpolation mode of a Resize node's `mode` attribute, as a wire code.
    /// @throws Error(Unsupported) for any string that is not nearest / linear / cubic.
    int vxResizeMode(const std::string &s);
    /// Coordinate-transformation mode as a wire code; an unrecognized string is half_pixel.
    int vxResizeCoord(const std::string &s);
    /// Nearest-rounding mode as a wire code; an unrecognized string is round_prefer_floor.
    int vxResizeNearestMode(const std::string &s);
    /// Nearest-neighbour source index for output position `d`, in exact integer arithmetic. The
    /// reference form: 64-bit products, so the rule reads as the ratio of integers it is. An input
    /// axis of zero extent has no source pixel and resolves to 0.
    int vxResizeNearestSrc(int d, int outS, int inS, int coordMode, int nearestMode);
    /// The same nearest-neighbour source index, evaluated in 32-bit signed arithmetic only -- the
    /// form the GPU kernels can execute, since GLSL has no guaranteed 64-bit integer. Equal to
    /// vxResizeNearestSrc for every extent up to kResizeMaxSpatialExtent, which is the whole set
    /// both ops accept, so the two backends resolve the same source pixel by construction. The four
    /// resize shaders carry a line-for-line copy of this body.
    int vxResizeNearestSrcNarrow(int d, int outS, int inS, int coordMode, int nearestMode);
    /// Cubic-convolution weights for the four taps at fractional position `t`, coefficient `a`.
    void vxResizeCubicWeights(float t, float a, float w[kResizeCubicTaps]);

} // namespace vknn
