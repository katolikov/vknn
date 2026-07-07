// Windowed AveragePool2D (scalar reference). ONNX count_include_pad selects the divisor.
#include "backend/cpu/cpu_backend.h"
#include "core/conv_geom.h"
#include <algorithm>

namespace vknn {
    namespace {

        struct AvgPoolCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X    = ctx.t(node.inputs[0]);
                RtTensor       &Y    = ctx.t(node.outputs[0]);
                NCHW            x    = NCHW::from(X.shape);
                auto            ints = [&](const char *k, std::vector<int64_t> d) {
                    const auto &v = node.attr.getints(k);
                    return v.empty() ? d : v;
                };
                auto    ks = ints("kernel_shape", {1, 1});
                auto    st = ints("strides", {1, 1});
                // Pads and output extent through the shared pool geometry (core/conv_geom.h), which
                // resolves auto_pad into begin/end pads: pt/pl shift the window origin, while the end
                // pads only widen the output extent (already folded into oh/ow).
                ConvGeom geo = poolGeom(x.h, x.w, node.attr);
                int64_t  kh = ks[0], kw = ks[1], sh = st[0], sw = st[1], pt = geo.padT, pl = geo.padL;
                bool     incPad = node.attr.geti("count_include_pad", 0) != 0;
                int64_t  oh     = geo.outH;
                int64_t  ow     = geo.outW;

                float       *y  = cpu::allocOut(Y, {x.n, x.c, oh, ow});
                const float *xd = X.host.f32();
                for (int64_t n = 0; n < x.n; ++n)
                {
                    for (int64_t c = 0; c < x.c; ++c)
                    {
                        const float *xc = xd + (n * x.c + c) * x.h * x.w;
                        for (int64_t oy = 0; oy < oh; ++oy)
                        {
                            for (int64_t ox = 0; ox < ow; ++ox)
                            {
                                // Sum the in-bounds window cells; cnt tracks how many were valid so
                                // count_include_pad can choose between the two ONNX divisors below.
                                float   acc = 0;
                                int64_t cnt = 0;
                                for (int64_t ky = 0; ky < kh; ++ky)
                                {
                                    // Map output row oy back to the input row for kernel tap ky:
                                    // origin oy*sh minus the top pad, then advance by ky. Rows that
                                    // land in the padding are skipped (an implicit zero addend).
                                    int64_t iy = oy * sh - pt + ky;
                                    if (iy < 0 || iy >= x.h)
                                    {
                                        continue;
                                    }
                                    for (int64_t kx = 0; kx < kw; ++kx)
                                    {
                                        int64_t ix = ox * sw - pl + kx;
                                        if (ix < 0 || ix >= x.w)
                                        {
                                            continue;
                                        }
                                        acc += xc[iy * x.w + ix];
                                        ++cnt;
                                    }
                                }
                                // count_include_pad divisor: full kernel area kh*kw treats padded
                                // cells as zero contributors; otherwise divide by the valid count
                                // only (max(.,1) guards an all-padding window against divide-by-zero).
                                float denom                            = incPad ? (float) (kh * kw) : (float) std::max<int64_t>(cnt, 1);
                                y[((n * x.c + c) * oh + oy) * ow + ox] = acc / denom;
                            }
                        }
                    }
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::AvgPool, AvgPoolCpu);
} // namespace vknn
