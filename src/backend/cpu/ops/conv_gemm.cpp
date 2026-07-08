// ConvGemm reference: the Conv lowered by lowerConv, with weights repacked [K][Cout]
// (K = Cin*KH*KW, k = (ky*KW + kx)*Cin + ic, channel-fastest — the same order lowerConv packs and
// the GPU kernel gathers). Plain fp32 convolution loops indexed through the repacked layout; the
// executor's epilogue hook applies any attached pointwise unit afterwards.
#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/parallel.h"
#include "core/conv_geom.h"

namespace vknn {
    namespace {

        struct ConvGemmCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X  = ctx.t(node.inputs[0]);
                const RtTensor &Wt = ctx.t(node.inputs[1]);
                const float    *bias = nullptr;
                // Bias presence bounds by pwCoreInputs: inputs appended past it are fused-unit operands.
                if (pwCoreInputs(node) > 2 && node.inputs[2] != kNoTensor)
                {
                    bias = ctx.t(node.inputs[2]).host.f32();
                }
                auto a = [&](const char *k, std::vector<int64_t> d) {
                    const auto &v = node.attr.getints(k);
                    return v.empty() ? d : v;
                };
                auto    kk = a("kernel_shape", {1, 1}), st = a("strides", {1, 1});
                auto    dl = a("dilations", {1, 1});
                int64_t N = X.shape[0], C = X.shape[1], H = X.shape[2], W = X.shape[3];
                int64_t Cout = Wt.shape[1], KH = kk[0], KW = kk[1];
                // Shared forward geometry (core/conv_geom.h): resolves auto_pad into begin/end pads.
                ConvGeom geo = convGeom(H, W, KH, KW, node.attr);
                auto     pd  = geo.pads();
                int64_t  OH = geo.outH, OW = geo.outW;

                RtTensor    &Y = ctx.t(node.outputs[0]);
                float       *y = cpu::allocOut(Y, {N, Cout, OH, OW});
                const float *x = X.host.f32();
                const float *w = Wt.host.f32();

                // One output plane per (image, output channel): disjoint stores, and each acc sums over
                // its own ic/ky/kx within a single iteration, so the planes partition across threads
                // without touching any accumulation order.
                cpu::parallelFor(cpu::threadCount(ctx.config), 0, N * Cout, cpu::minChunkForWork(OH * OW * C * KH * KW), [&](int64_t planeBegin, int64_t planeEnd) {
                    for (int64_t plane = planeBegin; plane < planeEnd; ++plane)
                    {
                        int64_t n  = plane / Cout;
                        int64_t oc = plane % Cout;
                        for (int64_t oy = 0; oy < OH; ++oy)
                        {
                            for (int64_t ox = 0; ox < OW; ++ox)
                            {
                                float acc = bias ? bias[oc] : 0.f;
                                for (int64_t ic = 0; ic < C; ++ic)
                                {
                                    for (int64_t ky = 0; ky < KH; ++ky)
                                    {
                                        int64_t iy = oy * st[0] - pd[0] + ky * dl[0];
                                        if (iy < 0 || iy >= H)
                                        {
                                            continue;
                                        }
                                        for (int64_t kx = 0; kx < KW; ++kx)
                                        {
                                            int64_t ix = ox * st[1] - pd[1] + kx * dl[1];
                                            if (ix < 0 || ix >= W)
                                            {
                                                continue;
                                            }
                                            int64_t k = (ky * KW + kx) * C + ic; // lowerConv's channel-fastest pack order
                                            acc += x[((n * C + ic) * H + iy) * W + ix] * w[k * Cout + oc];
                                        }
                                    }
                                }
                                y[((n * Cout + oc) * OH + oy) * OW + ox] = acc;
                            }
                        }
                    }
                });
                cpu::applyAct(y, Y.elems(), node.fusedAct, node.actLo, node.actHi);
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::ConvGemm, ConvGemmCpu);
} // namespace vknn
