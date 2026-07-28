// Resize / Upsample (spatial), NCHW reference. nearest + linear(bilinear); coordinate transform
// modes half_pixel / align_corners / asymmetric / pytorch_half_pixel. The output H/W derive from
// the RUNTIME input shape plus the `sizes` (input[3], preferred) or `scales` (input[2]) parameter,
// mirroring the Resize arm of inferShapes; the pre-opset-10 Upsample form carries neither input
// and falls back to the static graph desc. Only the H and W axes are resampled; N and C pass
// through unchanged, so each (n, c) plane is resized independently against the source (x.h * x.w)
// plane.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>
#include <cmath>

namespace vknn {

    // mode codes shared with the shader: 0 = nearest, 1 = linear/bilinear. Any unrecognized
    // interpolation-mode string falls back to nearest.
    int vxResizeMode(const std::string &s) {
        return s == "linear" || s == "bilinear" ? 1 : 0;
    }
    // Coordinate-transformation-mode code shared with the shader: 0 = half_pixel (the ONNX
    // default and the fallback for any unrecognized string), 1 = align_corners, 2 = asymmetric,
    // 3 = pytorch_half_pixel. srcCoord() dispatches on this code.
    int vxResizeCoord(const std::string &s) {
        if (s == "align_corners")
        {
            return 1;
        }
        if (s == "asymmetric")
        {
            return 2;
        }
        if (s == "pytorch_half_pixel")
        {
            return 3;
        }
        return 0; // half_pixel
    }
    // Map an output pixel index `d` (along one spatial axis of length `outS`) back to a fractional
    // source coordinate in the input axis of length `inS`, per the ONNX Resize coordinate-transform
    // formulas. `scale = outS / inS` is the resize ratio for the modes expressed in terms of it.
    // Nearest-neighbor rounds this value; bilinear splits it into floor + fractional weight.
    static float srcCoord(int d, int outS, int inS, int coordMode) {
        float scale = (float) outS / (float) inS;
        if (coordMode == 1)
        {
            // align_corners: endpoints coincide, so d maps linearly over [0, inS-1]. A degenerate
            // outS==1 has no interval to span and collapses to 0 (avoids the /(outS-1) divide-by-zero).
            return outS > 1 ? (float) d * (inS - 1) / (outS - 1) : 0.f; // align_corners
        }
        if (coordMode == 2)
        {
            return (float) d / scale; // asymmetric: pixel corners align at the origin, no half-pixel shift
        }
        if (coordMode == 3)
        {
            // pytorch_half_pixel: same half-pixel formula as mode 0, but a single-pixel output axis
            // (outS==1) maps to 0 rather than -0.5.
            return outS > 1 ? ((float) d + 0.5f) / scale - 0.5f : 0.f; // pytorch_half_pixel
        }
        // half_pixel (ONNX default): sample at each output pixel's center, converted to the input's
        // pixel-center coordinate system, so the +0.5 / -0.5 shifts bracket the /scale rescale.
        return ((float) d + 0.5f) / scale - 0.5f; // half_pixel
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
                int          mode      = vxResizeMode(node.attr.gets("mode", "nearest"));
                int          coordMode = vxResizeCoord(node.attr.gets("coordinate_transformation_mode", "half_pixel"));
                float       *y         = cpu::allocOut(Y, os);
                const float *xd        = X.host.f32();
                auto         clampi    = [](int v, int lo, int hi) {
                    return v < lo ? lo : (v > hi ? hi : v);
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
                                if (mode == 0)
                                { // nearest (round_prefer_floor)
                                    // round_prefer_floor: floor, then bump up only when strictly past
                                    // the midpoint, so an exact .5 rounds down (toward the floor).
                                    int iy = (int) std::floor(fy);
                                    if (fy - iy > 0.5f)
                                    {
                                        iy++;
                                    }
                                    int ix = (int) std::floor(fx);
                                    if (fx - ix > 0.5f)
                                    {
                                        ix++;
                                    }
                                    // Clamp to the valid input range so half_pixel's negative edge
                                    // coordinates (and the align_corners tail) read the border pixel.
                                    iy = clampi(iy, 0, (int) x.h - 1);
                                    ix = clampi(ix, 0, (int) x.w - 1);
                                    v  = xc[iy * x.w + ix];
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
