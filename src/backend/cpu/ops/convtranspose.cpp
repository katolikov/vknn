// Scalar reference ConvTranspose2D (transposed / fractionally-strided convolution, a.k.a. deconv).
// This is the correctness oracle the GPU path is checked against, so it stays simple and readable.
//
// Written in GATHER form: each output element loops the kernel taps and, for the taps that land on a
// valid input pixel under the stride, accumulates X*W. Equivalent to the usual scatter form but with
// one write per output, no races. Weight layout is ONNX ConvTranspose: W[Cin, Cout/group, kH, kW].
#include "backend/cpu/cpu_backend.h"
#include "core/conv_geom.h"

namespace vknn {
    namespace {

        struct ConvTransposeCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X       = ctx.t(node.inputs[0]);
                const RtTensor &W       = ctx.t(node.inputs[1]);
                const bool      hasBias = pwCoreInputs(node) > 2 && node.inputs[2] != kNoTensor;
                const RtTensor *B       = hasBias ? &ctx.t(node.inputs[2]) : nullptr;
                RtTensor       &Y       = ctx.t(node.outputs[0]);

                NCHW    x     = NCHW::from(X.shape);
                // ONNX weight layout is W[Cin, Cout/group, kH, kW]: dim 1 is output channels PER
                // GROUP (not total Cout), so outC below is recovered as outCg * group.
                int64_t outCg = W.shape[1], kh = W.shape[2], kw = W.shape[3];
                auto    ints = [&](const char *k, std::vector<int64_t> d) {
                    const auto &v = node.attr.getints(k);
                    return v.empty() ? d : v;
                };
                auto    strides = ints("strides", {1, 1});
                auto    dil     = ints("dilations", {1, 1});
                int64_t group   = node.attr.geti("group", 1);
                int64_t sh = strides[0], sw = strides[1];
                int64_t dh = dil[0], dw = dil[1];
                int64_t outC = outCg * group;

                // geom resolves auto_pad / output_shape into the final output extent plus the BEGIN
                // pads (pt, pl); the gather loop uses only the begin pads to place the input window,
                // as the end pads affect nothing but outH/outW (already folded in here).
                ConvTransposeGeom geom = convTransposeGeom(x.h, x.w, kh, kw, node.attr);
                int64_t           outH = geom.outH, outW = geom.outW;
                int64_t           pt = geom.padH, pl = geom.padW;

                float       *y  = cpu::allocOut(Y, {x.n, outC, outH, outW});
                const float *xd = X.host.f32();
                const float *wd = W.host.f32();
                const float *bd = B ? B->host.f32() : nullptr;

                // Grouped ConvTranspose: the Cin channels split into `group` contiguous bands of inCg,
                // and each band produces its own band of outCg output channels. Output channel oc thus
                // reads only the input channels [icStart, icStart+inCg) belonging to the same group.
                int64_t inCg = x.c / group; // input channels per group
                for (int64_t n = 0; n < x.n; ++n)
                {
                    for (int64_t oc = 0; oc < outC; ++oc)
                    {
                        int64_t g       = oc / outCg; // group index
                        int64_t ocLocal = oc % outCg; // output channel within the group
                        int64_t icStart = g * inCg;   // first input channel of the group
                        float   bias    = bd ? bd[oc] : 0.f;
                        for (int64_t oy = 0; oy < outH; ++oy)
                        {
                            for (int64_t ox = 0; ox < outW; ++ox)
                            {
                                float acc = bias;
                                // Invert the scatter relation oy = iy*sh - pt + ky*dh for each tap ky:
                                // the input row that could have written oy through tap ky satisfies
                                // ny := oy + pt - ky*dh == iy*sh. ny must be a non-negative multiple of
                                // the stride, else this output falls in a fractional-stride "hole" that
                                // no input row reaches through this tap, and the tap contributes nothing.
                                for (int64_t ky = 0; ky < kh; ++ky)
                                {
                                    int64_t ny = oy + pt - ky * dh; // = iy * sh
                                    if (ny < 0 || ny % sh != 0)
                                    {
                                        continue;
                                    }
                                    int64_t iy = ny / sh;
                                    if (iy >= x.h) // input row exists only up to x.h-1
                                    {
                                        continue;
                                    }
                                    // Same inversion along the width axis: nx == ix*sw must land on a
                                    // stride multiple and inside [0, x.w) for the tap to hit input col ix.
                                    for (int64_t kx = 0; kx < kw; ++kx)
                                    {
                                        int64_t nx = ox + pl - kx * dw; // = ix * sw
                                        if (nx < 0 || nx % sw != 0)
                                        {
                                            continue;
                                        }
                                        int64_t ix = nx / sw;
                                        if (ix >= x.w)
                                        {
                                            continue;
                                        }
                                        for (int64_t ic = 0; ic < inCg; ++ic)
                                        {
                                            // Row-major NCHW address of X[n, gic, iy, ix] and of the
                                            // weight W[gic, ocLocal, ky, kx] (Cin-major per ONNX layout).
                                            int64_t      gic = icStart + ic;
                                            const float *xch = xd + ((n * x.c + gic) * x.h + iy) * x.w + ix;
                                            const float *wv  = wd + ((gic * outCg + ocLocal) * kh + ky) * kw + kx;
                                            acc += (*xch) * (*wv);
                                        }
                                    }
                                }
                                y[((n * outC + oc) * outH + oy) * outW + ox] = acc;
                            }
                        }
                    }
                }
                cpu::applyAct(y, Y.elems(), node.fusedAct, node.actLo, node.actHi);
            }
        };

    } // namespace

    VKNN_REGISTER_CPU_OP(OpType::ConvTranspose, ConvTransposeCpu);

} // namespace vknn
