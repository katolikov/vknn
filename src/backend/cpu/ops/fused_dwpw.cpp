// Fused depthwise + 1x1-project convolution (CPU oracle), the fused form of an inverted-residual
// block: a per-channel depthwise conv, its activation, then a 1x1 pointwise projection mixing the
// E channels down to Cout, an optional activation, and an optional residual add. Computing both
// stages in one op keeps the E-channel depthwise result (dwOut) in a scratch buffer instead of a
// materialized tensor. This is the correctness oracle the GPU FusedDwPw path is checked against, so
// it stays a plain scalar reference rather than fast.
//
// inputs: [0] X (expanded activation, NCHW with C==E), [1] dw_w [E,1,KH,KW] (one KHxKW kernel per
// channel), [2] dw_b [E] (optional), [3] pw_w [Cout,E,1,1] (the 1x1 projection as a Cout-by-E
// matrix), [4] pw_b [Cout] (optional); node.fusedResidual is an optional [N,Cout,OH,OW] tensor.
// node.subOp is the depthwise-stage ActType; node.fusedAct is the projection-stage activation.
#include "backend/cpu/cpu_backend.h"
#include "core/conv_geom.h"
#include "vknn/op.h"
#include <algorithm>
#include <vector>

namespace vknn {
    namespace {
        struct FusedDwPwCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X  = ctx.t(node.inputs[0]);
                const RtTensor &DW = ctx.t(node.inputs[1]);
                const RtTensor &PW = ctx.t(node.inputs[3]);
                const float    *db = node.inputs[2] != kNoTensor ? ctx.t(node.inputs[2]).host.f32() : nullptr;
                const float    *pb = node.inputs[4] != kNoTensor ? ctx.t(node.inputs[4]).host.f32() : nullptr;
                RtTensor       &Y  = ctx.t(node.outputs[0]);
                NCHW            x  = NCHW::from(X.shape);
                // E = expanded (depthwise) channel count = input C; Cout = projected channels = pw_w
                // rows; KH/KW = depthwise kernel extent from the [E,1,KH,KW] weight.
                int64_t         E = x.c, Cout = PW.shape[0], KH = DW.shape[2], KW = DW.shape[3];
                auto            a = [&](const char *k, std::vector<int64_t> d) {
                    const auto &v = node.attr.getints(k);
                    return v.empty() ? d : v;
                };
                // Depthwise conv geometry through the shared forward helper (core/conv_geom.h): the
                // node carries the depthwise Conv's attrs, so auto_pad resolves the same way there.
                // pads are [top, left, bottom, right] (begin then end).
                auto               st  = a("strides", {1, 1}), dil = a("dilations", {1, 1});
                ConvGeom           geo = convGeom(x.h, x.w, KH, KW, node.attr);
                auto               pad = geo.pads();
                int64_t            OH  = geo.outH;
                int64_t            OW  = geo.outW;
                const float       *xd  = X.host.f32();
                const float       *dw  = DW.host.f32();
                const float       *pw  = PW.host.f32();
                Shape              os  = {x.n, Cout, OH, OW};
                float             *y   = cpu::allocOut(Y, os);
                const float       *res = node.fusedResidual != kNoTensor ? ctx.t(node.fusedResidual).host.f32() : nullptr;
                // Per-batch scratch for the depthwise stage, laid out [E, OH, OW]; reused across n so
                // the E intermediate channels never become a full tensor.
                std::vector<float> dwOut(E * OH * OW);
                for (int64_t n = 0; n < x.n; ++n)
                {
                    // Stage 1: depthwise conv. Each output channel e convolves only input channel e
                    // (group == E), accumulating in fp32 from the optional per-channel bias.
                    for (int64_t e = 0; e < E; ++e)
                    {
                        for (int64_t oy = 0; oy < OH; ++oy)
                        {
                            for (int64_t ox = 0; ox < OW; ++ox)
                            {
                                float acc = db ? db[e] : 0.f;
                                for (int64_t ky = 0; ky < KH; ++ky)
                                {
                                    // Map kernel row to the strided, dilated, pad-shifted input row;
                                    // rows outside [0, x.h) are the zero-pad border, so skip them.
                                    int64_t iy = oy * st[0] - pad[0] + ky * dil[0];
                                    if (iy < 0 || iy >= x.h)
                                    {
                                        continue;
                                    }
                                    for (int64_t kx = 0; kx < KW; ++kx)
                                    {
                                        int64_t ix = ox * st[1] - pad[1] + kx * dil[1];
                                        if (ix < 0 || ix >= x.w)
                                        {
                                            continue;
                                        }
                                        // X[n,e,iy,ix] * dw_w[e,0,ky,kx]; dw_w has one channel of
                                        // input so its stride is (e*KH + ky)*KW + kx.
                                        acc += xd[((n * E + e) * x.h + iy) * x.w + ix] * dw[(e * KH + ky) * KW + kx];
                                    }
                                }
                                dwOut[(e * OH + oy) * OW + ox] = acc;
                            }
                        }
                    }
                    // Depthwise-stage activation applied in place to dwOut. subOp is an ActType code;
                    // the 0, 6 are the Clip lo/hi bounds, consulted only when subOp == Clip.
                    cpu::applyAct(dwOut.data(), E * OH * OW, (ActType) node.subOp, 0, 6);
                    // Stage 2: 1x1 pointwise projection. Treating each of the OH*OW spatial positions
                    // as a column, pw_w [Cout, E] maps the E depthwise channels to Cout output
                    // channels: y[c, p] = pw_b[c] + sum_e pw_w[c, e] * dwOut[e, p].
                    for (int64_t c = 0; c < Cout; ++c)
                    {
                        for (int64_t p = 0; p < OH * OW; ++p)
                        {
                            float acc = pb ? pb[c] : 0.f;
                            for (int64_t e = 0; e < E; ++e)
                            {
                                acc += pw[c * E + e] * dwOut[e * OH * OW + p];
                            }
                            int64_t oi = (n * Cout + c) * OH * OW + p;
                            // Optional fused residual (elementwise, same NCHW shape as Y), added
                            // before the projection activation per inverted-residual semantics.
                            if (res)
                            {
                                acc += res[oi];
                            }
                            y[oi] = acc;
                        }
                    }
                }
                // Projection-stage activation over the whole output, with its clamp/param bounds.
                cpu::applyAct(y, Y.elems(), node.fusedAct, node.actLo, node.actHi);
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::FusedDwPw, FusedDwPwCpu);
} // namespace vknn
