// Resize / Upsample (spatial), NCHW reference. nearest + linear(bilinear) + cubic; coordinate
// transform modes half_pixel / align_corners / asymmetric / pytorch_half_pixel. The output H/W derive from
// the RUNTIME input shape plus the `sizes` (input[3], preferred) or `scales` (input[2]) parameter,
// mirroring the Resize arm of inferShapes; the pre-opset-10 Upsample form carries neither input
// and falls back to the static graph desc. Only the H and W axes are resampled; N and C pass
// through unchanged, so each (n, c) plane is resized independently against the source (x.h * x.w)
// plane.
#include "backend/cpu/cpu_backend.h"
#include "core/resize_rule.h"
#include "vknn/op.h"
#include <algorithm>
#include <cmath>

namespace vknn {

    // Interpolation-mode codes shared with the shader: 0 = nearest, 1 = linear/bilinear, 2 = cubic.
    // These are the only three ONNX defines. An unrecognized string is refused rather than mapped to
    // a default: `cubic` used to land on nearest here, which is not an approximation of cubic but a
    // different picture -- cosine ~0.49 against onnxruntime, silently, on a model that reported no
    // fallback and no unsupported op.
    int vxResizeMode(const std::string &s) {
        if (s == "linear" || s == "bilinear")
        {
            return kResizeModeLinear;
        }
        if (s == "cubic" || s == "bicubic")
        {
            return kResizeModeCubic;
        }
        if (s != "nearest")
        {
            throw Error(Status::Unsupported, "Resize: unrecognized interpolation mode '" + s + "' (expects nearest, linear, or cubic)");
        }
        return kResizeModeNearest;
    }
    // Cubic-convolution weights for the 4 taps at floor-1 .. floor+2, at fractional position t.
    // `a` is the ONNX cubic_coeff_a attribute (-0.75 by default, the PyTorch/OpenCV kernel).
    // Identical body in shaders/resize.comp -- both must produce the same four floats in the same
    // order, since the tap sum below is order-observable in fp32.
    void vxResizeCubicWeights(float t, float a, float w[4]) {
        const float t1 = 1.f + t, t2 = 2.f - t, tc = 1.f - t;
        w[0] = a * t1 * t1 * t1 - 5.f * a * t1 * t1 + 8.f * a * t1 - 4.f * a;
        w[1] = (a + 2.f) * t * t * t - (a + 3.f) * t * t + 1.f;
        w[2] = (a + 2.f) * tc * tc * tc - (a + 3.f) * tc * tc + 1.f;
        w[3] = a * t2 * t2 * t2 - 5.f * a * t2 * t2 + 8.f * a * t2 - 4.f * a;
    }
    // Coordinate-transformation-mode code shared with the shader: 0 = half_pixel (the ONNX
    // default and the fallback for any unrecognized string), 1 = align_corners, 2 = asymmetric,
    // 3 = pytorch_half_pixel. srcCoord() dispatches on this code.
    int vxResizeCoord(const std::string &s) {
        if (s == "align_corners")
        {
            return kResizeCoordAlignCorners;
        }
        if (s == "asymmetric")
        {
            return kResizeCoordAsymmetric;
        }
        if (s == "pytorch_half_pixel")
        {
            return kResizeCoordPytorchHalfPixel;
        }
        return kResizeCoordHalfPixel;
    }
    // Map an output pixel index `d` (along one spatial axis of length `outS`) back to a fractional
    // source coordinate in the input axis of length `inS`, per the ONNX Resize coordinate-transform
    // formulas. `scale = outS / inS` is the resize ratio for the modes expressed in terms of it.
    // Nearest-neighbor rounds this value; bilinear splits it into floor + fractional weight.
    static float srcCoord(int d, int outS, int inS, int coordMode) {
        float scale = (float) outS / (float) inS;
        if (coordMode == kResizeCoordAlignCorners)
        {
            // align_corners: endpoints coincide, so d maps linearly over [0, inS-1]. A degenerate
            // outS==1 has no interval to span and collapses to 0 (avoids the /(outS-1) divide-by-zero).
            return outS > 1 ? (float) d * (inS - 1) / (outS - 1) : 0.f; // align_corners
        }
        if (coordMode == kResizeCoordAsymmetric)
        {
            return (float) d / scale; // asymmetric: pixel corners align at the origin, no half-pixel shift
        }
        if (coordMode == kResizeCoordPytorchHalfPixel)
        {
            // pytorch_half_pixel: same half-pixel formula as mode 0, but a single-pixel output axis
            // (outS==1) maps to 0 rather than -0.5.
            return outS > 1 ? ((float) d + 0.5f) / scale - 0.5f : 0.f; // pytorch_half_pixel
        }
        // half_pixel (ONNX default): sample at each output pixel's center, converted to the input's
        // pixel-center coordinate system, so the +0.5 / -0.5 shifts bracket the /scale rescale.
        return ((float) d + 0.5f) / scale - 0.5f; // half_pixel
    }

    // Nearest-neighbour source index for output position `d`, in exact integer arithmetic. Every
    // rule srcCoord() evaluates is a ratio of integers in (d, inS, outS), so round_prefer_floor is
    // decided here without ever forming the fractional coordinate.
    //
    // The float form cannot decide it. An integer downsample factor under half_pixel puts EVERY
    // output pixel exactly on the .5 tie (fy = 2d + 0.5 for a halving), and the GPU evaluates the
    // divide through a reciprocal, so one ulp at the tie moves the sample a whole pixel: the GPU
    // read x[2d+1] where this oracle read x[2d], on every pixel of every channel. Integers make the
    // index a pure function of the shape -- same on both backends, on any driver. The products stay
    // exact while outS * inS < 2^30, past any spatial extent the planner admits.
    int vxResizeNearestSrc(int d, int outS, int inS, int coordMode) {
        int64_t num, den;
        if (coordMode == kResizeCoordAlignCorners) // align_corners: d * (inS-1) / (outS-1)
        {
            num = (int64_t) d * (inS - 1);
            den = outS - 1;
        } else if (coordMode == kResizeCoordAsymmetric) // asymmetric: d / scale = d * inS / outS
        {
            num = (int64_t) d * inS;
            den = outS;
        } else // half_pixel / pytorch_half_pixel: ((2d+1) * inS - outS) / (2 * outS)
        {
            num = (int64_t) (2 * d + 1) * inS - outS;
            den = 2 * (int64_t) outS;
        }
        if (den <= 0 || (coordMode == kResizeCoordPytorchHalfPixel && outS <= 1))
        {
            return 0; // a single-pixel output axis spans no interval (srcCoord guards it the same way)
        }
        int64_t q = num / den, r = num % den;
        if (r < 0)
        {
            --q; // C division truncates toward zero; the rule needs floor
            r += den;
        }
        return (int) (2 * r > den ? q + 1 : q); // round_prefer_floor: bump only strictly past the midpoint
    }

    namespace {
        struct ResizeCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                NCHW            x = NCHW::from(X.shape);
                // ONNX Resize inputs: X, roi, scales, sizes (roi unused here; scales/sizes optional
                // but one of the two is present). `sizes` gives the output dims directly; `scales`
                // multiplies the runtime input dims (truncating, as inferShapes does). Both are read
                // for the H/W axes only — N and C pass through the kernel unchanged.
                auto param = [&](int idx) -> const RtTensor * {
                    return idx < (int) pwCoreInputs(node) && node.inputs[idx] != kNoTensor ? &ctx.t(node.inputs[idx]) : nullptr;
                };
                int64_t OH = 0, OW = 0;
                if (X.shape.size() == 4)
                {
                    if (const RtTensor *sz = param(3); sz && sz->elems() == 4)
                    {
                        OH = sz->dtype == DType::Int64 ? sz->host.i64()[2] : (int64_t) sz->host.f32()[2];
                        OW = sz->dtype == DType::Int64 ? sz->host.i64()[3] : (int64_t) sz->host.f32()[3];
                    } else if (const RtTensor *sc = param(2); sc && sc->elems() == 4 && sc->dtype != DType::Int64)
                    {
                        OH = (int64_t) (x.h * sc->host.f32()[2]);
                        OW = (int64_t) (x.w * sc->host.f32()[3]);
                    }
                }
                Shape os;
                if (OH > 0 && OW > 0)
                {
                    os = {x.n, x.c, OH, OW};
                } else
                {
                    // Upsample form (no scales/sizes input): the graph desc carries the output shape.
                    os = ctx.graph->desc(node.outputs[0]).shape;
                    if (os.size() != 4)
                    {
                        throw Error(Status::Unsupported, "Resize '" + node.name + "': no sizes/scales input and a rank-" + std::to_string(os.size()) + " output desc (expects the spatial NCHW form)");
                    }
                    OH = os[2];
                    OW = os[3];
                }
                int   mode      = vxResizeMode(node.attr.gets("mode", "nearest"));
                int   coordMode = vxResizeCoord(node.attr.gets("coordinate_transformation_mode", "half_pixel"));
                float cubicA    = node.attr.getf("cubic_coeff_a", kResizeCubicCoeffDefault);
                // exclude_outside=1 zeroes the taps that fall outside the input and renormalizes the
                // axis; the default keeps them and reads the border pixel instead.
                bool         excludeOutside = node.attr.geti("exclude_outside", 0) != 0;
                float       *y              = cpu::allocOut(Y, os);
                const float *xd             = X.host.f32();
                auto         clampi         = [](int v, int lo, int hi) {
                    return v < lo ? lo : (v > hi ? hi : v);
                };
                // Cubic taps for one axis: the four weights, already zeroed/renormalized for
                // exclude_outside, and the first tap index (floor - 1, unclamped).
                auto cubicAxis = [&](float f, int inS, float w[kResizeCubicTaps]) {
                    int first = (int) std::floor(f) - 1;
                    vxResizeCubicWeights(f - std::floor(f), cubicA, w);
                    if (excludeOutside)
                    {
                        float sum = 0.f;
                        for (int k = 0; k < kResizeCubicTaps; ++k)
                        {
                            int t = first + k;
                            w[k]  = (t < 0 || t >= inS) ? 0.f : w[k];
                            sum += w[k];
                        }
                        for (int k = 0; k < kResizeCubicTaps && sum != 0.f; ++k)
                        {
                            w[k] /= sum;
                        }
                    }
                    return first;
                };
                // Walk N and C outermost; xc / yc point at the start of the current source and
                // destination H*W planes (row-major, so element (row, col) sits at row*width + col).
                for (int64_t n = 0; n < x.n; ++n)
                {
                    for (int64_t c = 0; c < x.c; ++c)
                    {
                        const float *xc = xd + (n * x.c + c) * x.h * x.w;
                        float       *yc = y + (n * x.c + c) * OH * OW;
                        for (int oy = 0; oy < OH; ++oy)
                        {
                            float fy = srcCoord(oy, OH, (int) x.h, coordMode);
                            for (int ox = 0; ox < OW; ++ox)
                            {
                                float fx = srcCoord(ox, OW, (int) x.w, coordMode);
                                float v;
                                if (mode == kResizeModeNearest)
                                { // nearest (round_prefer_floor), decided in integers
                                    int iy = vxResizeNearestSrc(oy, (int) OH, (int) x.h, coordMode);
                                    int ix = vxResizeNearestSrc(ox, (int) OW, (int) x.w, coordMode);
                                    // Clamp to the valid input range so half_pixel's negative edge
                                    // coordinates (and the align_corners tail) read the border pixel.
                                    iy = clampi(iy, 0, (int) x.h - 1);
                                    ix = clampi(ix, 0, (int) x.w - 1);
                                    v  = xc[iy * x.w + ix];
                                } else if (mode == kResizeModeCubic)
                                {
                                    // Cubic convolution: 4 taps per axis at floor-1..floor+2, weights
                                    // separable in (fy, fx). Out-of-range taps read the border pixel
                                    // (or drop out entirely under exclude_outside, handled in
                                    // cubicAxis). The row-then-column summation order is fixed: it is
                                    // observable in fp32 and the shader repeats it exactly.
                                    float wy[kResizeCubicTaps], wx[kResizeCubicTaps];
                                    int   firstY = cubicAxis(fy, (int) x.h, wy);
                                    int   firstX = cubicAxis(fx, (int) x.w, wx);
                                    v            = 0.f;
                                    for (int ty = 0; ty < kResizeCubicTaps; ++ty)
                                    {
                                        const float *row = xc + (size_t) clampi(firstY + ty, 0, (int) x.h - 1) * x.w;
                                        float        acc = 0.f;
                                        for (int tx = 0; tx < kResizeCubicTaps; ++tx)
                                        {
                                            acc += wx[tx] * row[clampi(firstX + tx, 0, (int) x.w - 1)];
                                        }
                                        v += wy[ty] * acc;
                                    }
                                } else
                                { // bilinear
                                    // Floor gives the top-left neighbor (iy0, ix0); wy/wx are the
                                    // fractional distances into the 2x2 cell toward the bottom-right.
                                    int   iy0 = (int) std::floor(fy), ix0 = (int) std::floor(fx);
                                    float wy = fy - iy0, wx = fx - ix0;
                                    // Clamp all four corner indices independently so out-of-range
                                    // samples (edges) fold onto the nearest in-bounds pixel while the
                                    // interpolation weights below stay unchanged.
                                    int iy1 = clampi(iy0 + 1, 0, (int) x.h - 1), ix1 = clampi(ix0 + 1, 0, (int) x.w - 1);
                                    int cy0 = clampi(iy0, 0, (int) x.h - 1), cx0 = clampi(ix0, 0, (int) x.w - 1);
                                    // Four corners: a=top-left, b=top-right, cc=bottom-left, d=bottom-right.
                                    float a = xc[cy0 * x.w + cx0], b = xc[cy0 * x.w + ix1];
                                    float cc = xc[iy1 * x.w + cx0], d = xc[iy1 * x.w + ix1];
                                    // Bilinear blend: the four bilinear weights ((1-wy)/(wy) x (1-wx)/(wx))
                                    // sum to 1; this fixed left-to-right summation order is observable in fp32.
                                    v = a * (1 - wy) * (1 - wx) + b * (1 - wy) * wx + cc * wy * (1 - wx) + d * wy * wx;
                                }
                                yc[oy * OW + ox] = v;
                            }
                        }
                    }
                }
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Resize, ResizeCpu);
} // namespace vknn
