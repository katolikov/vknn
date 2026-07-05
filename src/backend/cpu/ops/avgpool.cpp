// Windowed AveragePool2D (scalar reference). ONNX count_include_pad selects the divisor.
#include "backend/cpu/cpu_backend.h"
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
                auto    ks  = ints("kernel_shape", {1, 1});
                auto    st  = ints("strides", {1, 1});
                // ONNX `pads` is [begin_h, begin_w, end_h, end_w]: pt/pl shift the window origin,
                // while the end pads (pad[2]/pad[3]) only widen the valid output-size range.
                auto    pad = ints("pads", {0, 0, 0, 0});
                int64_t kh = ks[0], kw = ks[1], sh = st[0], sw = st[1], pt = pad[0], pl = pad[1];
                bool    incPad = node.attr.geti("count_include_pad", 0) != 0;
                // Standard pooling output extent (floor mode): a window of size k slides over the
                // padded input with step s, yielding floor((in + begin + end - k)/s) + 1 positions.
                int64_t oh     = (x.h + pt + pad[2] - kh) / sh + 1;
                int64_t ow     = (x.w + pl + pad[3] - kw) / sw + 1;

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
