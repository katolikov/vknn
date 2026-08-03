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
    // index a pure function of the shape -- same on both backends, on any driver.
    //
    // This is the reference form: 64-bit products, so the rule reads as the ratio of integers it
    // is. vxResizeNearestSrcNarrow below is the same rule in the 32-bit arithmetic the GPU kernels
    // can execute, and the two agree over every extent the ops accept (tests/test_resize_rule.cpp).
    // Nearest-rounding-mode code shared with the shader: 0 = round_prefer_floor (the ONNX default
    // and the fallback for any unrecognized string), 1 = round_prefer_ceil, 2 = floor, 3 = ceil.
    // vxResizeNearestSrc dispatches on this code after the exact-integer coordinate transform.
    int vxResizeNearestMode(const std::string &s) {
        if (s == "round_prefer_ceil")
        {
            return kResizeNearestPreferCeil;
        }
        if (s == "floor")
        {
            return kResizeNearestFloor;
        }
        if (s == "ceil")
        {
            return kResizeNearestCeil;
        }
        return kResizeNearestPreferFloor;
    }
    int vxResizeNearestSrc(int d, int outS, int inS, int coordMode, int nearestMode) {
        if (inS <= 0)
        {
            return 0; // an input axis of zero extent holds no pixel any rounding could land on
        }
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
            // Every factor widens before it is combined: an output axis may run to
            // kResizeMaxSpatialExtent, and 2 * d + 1 leaves a 32-bit int well before that.
            num = (2 * (int64_t) d + 1) * inS - outS;
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
        // The four ONNX nearest_mode roundings over the exact fraction q + r/den. The prefer
        // variants differ only on the exact midpoint; floor/ceil differ from them at every
        // non-integer sample position, which is every position of a non-integer scale.
        switch (nearestMode)
        {
            case kResizeNearestPreferCeil:
                return (int) (2 * r >= den ? q + 1 : q);
            case kResizeNearestFloor:
                return (int) q;
            case kResizeNearestCeil:
                return (int) (r > 0 ? q + 1 : q);
            default:
                return (int) (2 * r > den ? q + 1 : q); // round_prefer_floor: bump only strictly past the midpoint
        }
    }

    // floor(a * b / den) into `quotient` and the remainder in [0, den) into `remainder`, using
    // 32-bit signed arithmetic only -- the arithmetic a GLSL kernel has, since Vulkan does not
    // guarantee 64-bit integers in shader code. Preconditions, all established by the nearest rule
    // below: a >= 0, b >= 0, den > 0, and a <= den (the left factor is an output position and the
    // denominator the output extent it is measured against, doubled together in the half_pixel
    // modes).
    //
    // Splitting b into whole multiples of den plus a residue below den leaves a product of two
    // factors that are each at most den. The direct path takes that product whenever both factors
    // fit kResizeNarrowFactorMax; past that, the bit-scanned path accumulates it from the top bit
    // of `a` down, holding the running remainder inside [0, den) so no intermediate ever leaves
    // the range one int holds, and the quotient it builds is bounded by a * residue / den < a.
    // Identical body in the four resize shaders (narrowMulDiv).
    static void resizeNarrowMulDiv(int a, int b, int den, int &quotient, int &remainder) {
        const int wholeSteps = b / den, residue = b - wholeSteps * den;
        if (a <= kResizeNarrowFactorMax && residue <= kResizeNarrowFactorMax)
        {
            const int product = a * residue, tileQuotient = product / den;
            quotient  = a * wholeSteps + tileQuotient;
            remainder = product - tileQuotient * den;
            return;
        }
        int tileQuotient = 0, rem = 0;
        for (int bit = kResizeNarrowProductBits - 1; bit >= 0; --bit)
        {
            // Double the running (quotient, remainder). `headroom` is what the remainder still has
            // before it reaches den, so comparing against it doubles without ever forming 2 * rem.
            const int headroom = den - rem;
            tileQuotient += tileQuotient;
            if (rem >= headroom)
            {
                rem -= headroom;
                ++tileQuotient;
            } else
            {
                rem += rem;
            }
            // Fold in one more copy of the residue where this bit of `a` is set, carrying the same way.
            if (((a >> bit) & 1) != 0)
            {
                const int slack = den - residue;
                if (rem >= slack)
                {
                    rem -= slack;
                    ++tileQuotient;
                } else
                {
                    rem += residue;
                }
            }
        }
        quotient  = a * wholeSteps + tileQuotient;
        remainder = rem;
    }

    int vxResizeNearestSrcNarrow(int d, int outS, int inS, int coordMode, int nearestMode) {
        if (inS <= 0)
        {
            return 0; // an input axis of zero extent holds no pixel any rounding could land on
        }
        // Each coordinate transform is one fraction (leftFactor * rightFactor - shift) / den. The
        // shift stays outside the product so resizeNarrowMulDiv sees two non-negative factors.
        int leftFactor, rightFactor, den, shift;
        if (coordMode == kResizeCoordAlignCorners) // align_corners: d * (inS-1) / (outS-1)
        {
            leftFactor  = d;
            rightFactor = inS - 1;
            den         = outS - 1;
            shift       = 0;
        } else if (coordMode == kResizeCoordAsymmetric) // asymmetric: d / scale = d * inS / outS
        {
            leftFactor  = d;
            rightFactor = inS;
            den         = outS;
            shift       = 0;
        } else // half_pixel / pytorch_half_pixel: ((2d+1) * inS - outS) / (2 * outS)
        {
            leftFactor  = 2 * d + 1;
            rightFactor = inS;
            den         = 2 * outS;
            shift       = outS;
        }
        if (den <= 0 || (coordMode == kResizeCoordPytorchHalfPixel && outS <= 1))
        {
            return 0; // a single-pixel output axis spans no interval (srcCoord guards it the same way)
        }
        int quotient, remainder;
        resizeNarrowMulDiv(leftFactor, rightFactor, den, quotient, remainder);
        // Apply the numerator's shift after the division. It is smaller than den, so it can borrow
        // at most one whole step, which is the floor correction the rule needs.
        remainder -= shift;
        if (remainder < 0)
        {
            remainder += den;
            --quotient;
        }
        // The four ONNX nearest_mode roundings over the exact fraction quotient + remainder/den.
        // The midpoint tests are spelled against den - remainder rather than 2 * remainder, which
        // for a denominator near the top of the int range would not fit.
        switch (nearestMode)
        {
            case kResizeNearestPreferCeil:
                return remainder >= den - remainder ? quotient + 1 : quotient;
            case kResizeNearestFloor:
                return quotient;
            case kResizeNearestCeil:
                return remainder > 0 ? quotient + 1 : quotient;
            default:
                return remainder > den - remainder ? quotient + 1 : quotient; // round_prefer_floor
        }
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
                // Whether this node carries a resize parameter at all. A `sizes` or `scales` input
                // that is PRESENT decides the output geometry even when it resolves to a
                // zero-extent axis: a zero there is a request for an empty output, not the
                // pre-opset-10 Upsample form, which carries neither input and is the only case the
                // graph desc stands in for.
                bool haveResizeParam = false;
                if (X.shape.size() == 4)
                {
                    if (const RtTensor *sz = param(3); sz && sz->elems() == 4)
                    {
                        OH              = sz->dtype == DType::Int64 ? sz->host.i64()[2] : (int64_t) sz->host.f32()[2];
                        OW              = sz->dtype == DType::Int64 ? sz->host.i64()[3] : (int64_t) sz->host.f32()[3];
                        haveResizeParam = true;
                    } else if (const RtTensor *sc = param(2); sc && sc->elems() == 4 && sc->dtype != DType::Int64)
                    {
                        OH              = (int64_t) (x.h * sc->host.f32()[2]);
                        OW              = (int64_t) (x.w * sc->host.f32()[3]);
                        haveResizeParam = true;
                    }
                }
                Shape os;
                if (haveResizeParam)
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
                if (OH < 0 || OW < 0)
                {
                    throw Error(Status::Unsupported, "Resize '" + node.name + "': output spatial extent " + std::to_string(OH) + "x" + std::to_string(OW) + " is negative (an axis length is a count)");
                }
                if (OH > kResizeMaxSpatialExtent || OW > kResizeMaxSpatialExtent || x.h > kResizeMaxSpatialExtent || x.w > kResizeMaxSpatialExtent)
                {
                    throw Error(Status::Unsupported, "Resize '" + node.name + "': spatial extent " + std::to_string(x.h) + "x" + std::to_string(x.w) + " -> " + std::to_string(OH) + "x" + std::to_string(OW) + " exceeds the exactly-resolvable bound " + std::to_string(kResizeMaxSpatialExtent) + " (past it the source index has no exact form on the GPU kernels, which resolve it in 32-bit integers)");
                }
                // A source plane with no elements has no pixel to sample. Every arm clamps its tap
                // indices into [0, extent-1], which is an empty range here, so resampling would
                // read off the front of an empty buffer rather than fold onto a border pixel. An
                // empty output needs no source and stays legal.
                if ((x.h <= 0 || x.w <= 0) && OH > 0 && OW > 0)
                {
                    throw Error(Status::Unsupported, "Resize '" + node.name + "': input spatial extent " + std::to_string(x.h) + "x" + std::to_string(x.w) + " holds no pixels, so the requested " + std::to_string(OH) + "x" + std::to_string(OW) + " output has nothing to sample");
                }
                int   mode        = vxResizeMode(node.attr.gets("mode", "nearest"));
                int   coordMode   = vxResizeCoord(node.attr.gets("coordinate_transformation_mode", "half_pixel"));
                int   nearestMode = vxResizeNearestMode(node.attr.gets("nearest_mode", "round_prefer_floor"));
                float cubicA      = node.attr.getf("cubic_coeff_a", kResizeCubicCoeffDefault);
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
                                { // nearest (nearest_mode rounding), decided in integers
                                    int iy = vxResizeNearestSrc(oy, (int) OH, (int) x.h, coordMode, nearestMode);
                                    int ix = vxResizeNearestSrc(ox, (int) OW, (int) x.w, coordMode, nearestMode);
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
