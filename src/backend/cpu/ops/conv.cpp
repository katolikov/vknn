// Scalar reference Conv2D. Handles plain convolution, depthwise (group==channels) and 1x1
// pointwise through the same loop. This is the correctness oracle the GPU path is checked
// against, so it stays simple and readable rather than fast.
#include "backend/cpu/cpu_backend.h"

namespace vknn {
    namespace {

        struct ConvCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                const RtTensor &W = ctx.t(node.inputs[1]);
                // ONNX Conv input[2] is an optional bias. Bound the read by pwCoreInputs (fused
                // pointwise-chain operands are appended past the op's own inputs) and exclude the case
                // where index 2 aliases the fused-residual tensor, which is added in the epilogue below
                // rather than as a per-output-channel bias.
                const bool      hasBias = pwCoreInputs(node) > 2 && node.inputs[2] != kNoTensor && node.inputs[2] != node.fusedResidual;
                const RtTensor *B       = hasBias ? &ctx.t(node.inputs[2]) : nullptr;
                RtTensor       &Y       = ctx.t(node.outputs[0]);

                NCHW x = NCHW::from(X.shape);
                // Weight layout is ONNX [M, C/group, kH, kW]: outC output channels, inCg = input
                // channels per group, and the kh*kw spatial kernel. 1-D convs arrive normalized to
                // this form by normalizeConv1d (kW == 1); a weight that is still rank 3 (a runtime-
                // produced weight the pass cannot normalize) is rejected rather than mis-indexed.
                if (W.shape.size() < 4)
                {
                    throw Error(Status::Unsupported, "Conv '" + node.name + "': weight rank " + std::to_string(W.shape.size()) + " (expects the normalized [M, C/group, kH, kW] form)");
                }
                int64_t outC = W.shape[0], inCg = W.shape[1], kh = W.shape[2], kw = W.shape[3];
                // Attribute lists shorter than their 2-spatial-dim length (a malformed graph) pad
                // with identity values instead of indexing past the vector.
                auto    ints = [&](const char *k, std::vector<int64_t> d) {
                    std::vector<int64_t> v = node.attr.getints(k);
                    if (v.empty())
                    {
                        return d;
                    }
                    while (v.size() < d.size())
                    {
                        v.push_back(d[v.size()]);
                    }
                    return v;
                };
                auto    strides = ints("strides", {1, 1});
                auto    pads    = ints("pads", {0, 0, 0, 0});
                auto    dil     = ints("dilations", {1, 1});
                int64_t group   = node.attr.geti("group", 1);
                int64_t sh = strides[0], sw = strides[1];
                int64_t pt = pads[0], pl = pads[1]; // begin pads (top, left); pads[2]/pads[3] are the end pads
                int64_t dh = dil[0], dw = dil[1];

                // ONNX output spatial size: the dilated kernel spans dh*(kh-1)+1 rows, so the last valid
                // window origin is (padded_h - span); integer division by the stride (floor) plus one
                // gives the number of window positions. pads[2]/pads[3] are the bottom/right end pads.
                int64_t outH = (x.h + pt + pads[2] - (dh * (kh - 1) + 1)) / sh + 1;
                int64_t outW = (x.w + pl + pads[3] - (dw * (kw - 1) + 1)) / sw + 1;

                // A rank-3 input is a normalized 1-D conv (h carries the single spatial dim, w == 1,
                // kw == 1 so outW == 1): the output keeps the input's rank, mirroring inferShapes.
                const bool   oneD = X.shape.size() == 3 && outW == 1;
                float       *y    = cpu::allocOut(Y, oneD ? Shape {x.n, outC, outH} : Shape {x.n, outC, outH, outW});
                const float *xd = X.host.f32();
                const float *wd = W.host.f32();
                const float *bd = B ? B->host.f32() : nullptr;

                int64_t outCg = outC / group; // output channels per group
                for (int64_t n = 0; n < x.n; ++n)
                {
                    for (int64_t oc = 0; oc < outC; ++oc)
                    {
                        // Grouped conv partitions channels: output channel oc belongs to group g and
                        // reads only that group's inCg input channels, starting at icStart. group==1 is
                        // plain conv (icStart==0); group==inChannels with inCg==1 is depthwise.
                        int64_t g       = oc / outCg;
                        int64_t icStart = g * inCg;
                        float   bias    = bd ? bd[oc] : 0.f;
                        for (int64_t oy = 0; oy < outH; ++oy)
                        {
                            for (int64_t ox = 0; ox < outW; ++ox)
                            {
                                float   acc = bias;
                                // Top-left input coordinate of this output's receptive field, shifted
                                // by the begin pad so a negative value lands in the padded margin.
                                int64_t iy0 = oy * sh - pt;
                                int64_t ix0 = ox * sw - pl;
                                for (int64_t ic = 0; ic < inCg; ++ic)
                                {
                                    // Base of channel (icStart+ic) of input image n, and the matching
                                    // weight plane [oc, ic] — both dense-row-major over the H*W / kh*kw
                                    // spatial grids.
                                    const float *xch = xd + ((n * x.c + (icStart + ic)) * x.h) * x.w;
                                    const float *wch = wd + ((oc * inCg + ic) * kh) * kw;
                                    for (int64_t ky = 0; ky < kh; ++ky)
                                    {
                                        int64_t iy = iy0 + ky * dh;
                                        // Out-of-bounds taps fall on the implicit zero pad and drop out
                                        // of the sum (zero contribution), so skip them entirely.
                                        if (iy < 0 || iy >= x.h)
                                        {
                                            continue;
                                        }
                                        for (int64_t kx = 0; kx < kw; ++kx)
                                        {
                                            int64_t ix = ix0 + kx * dw;
                                            if (ix < 0 || ix >= x.w)
                                            {
                                                continue;
                                            }
                                            acc += xch[iy * x.w + ix] * wch[ky * kw + kx];
                                        }
                                    }
                                }
                                // Store into Y[n, oc, oy, ox], flat NCHW row-major.
                                y[((n * outC + oc) * outH + oy) * outW + ox] = acc;
                            }
                        }
                    }
                }
                // Fused epilogue: out = act(conv + residual). The residual is a same-shaped skip
                // connection added elementwise before the activation, matching the order the fused
                // GPU kernel applies so both backends agree bit-for-bit.
                if (node.fusedResidual != kNoTensor)
                {
                    const float *rd = ctx.t(node.fusedResidual).host.f32();
                    int64_t      n  = Y.elems();
                    for (int64_t i = 0; i < n; ++i)
                    {
                        y[i] += rd[i];
                    }
                }
                cpu::applyAct(y, Y.elems(), node.fusedAct, node.actLo, node.actHi);
            }
        };

    } // namespace

    VKNN_REGISTER_CPU_OP(OpType::Conv, ConvCpu);

} // namespace vknn
