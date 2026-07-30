// ONNX Resize geometry, shared by the CPU oracle and the Vulkan kernel.
//
// The two backends must resolve identical source indices and identical tap weights, or the GPU
// silently returns a different picture from the oracle with no fallback to announce it. This header
// holds the wire codes and the rules that both sides evaluate; shaders/resize.comp carries a
// line-for-line copy of the two rule bodies, since GLSL cannot include C++.
#pragma once
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

    /// Interpolation mode of a Resize node's `mode` attribute, as a wire code.
    /// @throws Error(Unsupported) for any string that is not nearest / linear / cubic.
    int vxResizeMode(const std::string &s);
    /// Coordinate-transformation mode as a wire code; an unrecognized string is half_pixel.
    int vxResizeCoord(const std::string &s);
    /// Nearest-rounding mode as a wire code; an unrecognized string is round_prefer_floor.
    int vxResizeNearestMode(const std::string &s);
    /// Nearest-neighbour source index for output position `d`, in exact integer arithmetic.
    int vxResizeNearestSrc(int d, int outS, int inS, int coordMode, int nearestMode);
    /// Cubic-convolution weights for the four taps at fractional position `t`, coefficient `a`.
    void vxResizeCubicWeights(float t, float a, float w[kResizeCubicTaps]);

} // namespace vknn
