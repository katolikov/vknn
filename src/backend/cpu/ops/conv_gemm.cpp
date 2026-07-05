// ConvGemm reference: the Conv lowered by lowerConv, with weights repacked [K][Cout]
// (K = Cin*KH*KW, k = (ic*KH + ky)*KW + kx). Plain fp32 convolution loops indexed through the
// repacked layout; the executor's epilogue hook applies any attached pointwise unit afterwards.
#include "backend/cpu/cpu_backend.h"

namespace vknn {
    namespace {

        struct ConvGemmCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X  = ctx.t(node.inputs[0]);
                const RtTensor &Wt = ctx.t(node.inputs[1]);
                const float    *bias = nullptr;
                if (node.inputs.size() > 2 && node.inputs[2] != kNoTensor)
                {
                    bias = ctx.t(node.inputs[2]).host.f32();
                }
                auto a = [&](const char *k, std::vector<int64_t> d) {
                    const auto &v = node.attr.getints(k);
                    return v.empty() ? d : v;
                };
                auto    kk = a("kernel_shape", {1, 1}), st = a("strides", {1, 1});
                auto    pd = a("pads", {0, 0, 0, 0}), dl = a("dilations", {1, 1});
                int64_t N = X.shape[0], C = X.shape[1], H = X.shape[2], W = X.shape[3];
                int64_t Cout = Wt.shape[1], KH = kk[0], KW = kk[1];
                int64_t OH = (H + pd[0] + pd[2] - (dl[0] * (KH - 1) + 1)) / st[0] + 1;
                int64_t OW = (W + pd[1] + pd[3] - (dl[1] * (KW - 1) + 1)) / st[1] + 1;

                RtTensor    &Y = ctx.t(node.outputs[0]);
                float       *y = cpu::allocOut(Y, {N, Cout, OH, OW});
                const float *x = X.host.f32();
                const float *w = Wt.host.f32();

                for (int64_t n = 0; n < N; ++n)
                {
                    for (int64_t oc = 0; oc < Cout; ++oc)
                    {
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
                                            int64_t k = (ic * KH + ky) * KW + kx;
                                            acc += x[((n * C + ic) * H + iy) * W + ix] * w[k * Cout + oc];
                                        }
                                    }
                                }
                                y[((n * Cout + oc) * OH + oy) * OW + ox] = acc;
                            }
                        }
                    }
                }
                cpu::applyAct(y, Y.elems(), node.fusedAct, node.actLo, node.actHi);
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::ConvGemm, ConvGemmCpu);
} // namespace vknn
