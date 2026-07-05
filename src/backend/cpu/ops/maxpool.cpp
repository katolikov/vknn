// Windowed MaxPool2D (scalar reference), NCHW fp32: each output cell is the maximum over the
// in-bounds cells of its kernel window. Padded cells are skipped rather than treated as -inf,
// so an all-padding window would leave the initial -inf sentinel (matching the AveragePool
// origin/extent conventions in avgpool.cpp).
#include "backend/cpu/cpu_backend.h"
#include <limits>

namespace vknn {
    namespace {

        struct MaxPoolCpu: CpuOp {
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
                // Standard pooling output extent (floor mode): a window of size k slides over the
                // padded input with step s, yielding floor((in + begin + end - k)/s) + 1 positions.
                int64_t oh = (x.h + pt + pad[2] - kh) / sh + 1;
                int64_t ow = (x.w + pl + pad[3] - kw) / sw + 1;

                float       *y  = cpu::allocOut(Y, {x.n, x.c, oh, ow});
                const float *xd = X.host.f32();
                // Pooling is per-channel: the (n, c) planes are independent, so xc points at the
                // start of one row-major H*W plane and pooling stays within it.
                for (int64_t n = 0; n < x.n; ++n)
                {
                    for (int64_t c = 0; c < x.c; ++c)
                    {
                        const float *xc = xd + (n * x.c + c) * x.h * x.w;
                        for (int64_t oy = 0; oy < oh; ++oy)
                        {
                            for (int64_t ox = 0; ox < ow; ++ox)
                            {
                                // -inf identity so the first valid cell always wins; it survives only
                                // if every kernel tap falls in the padding (an empty window).
                                float m = -std::numeric_limits<float>::infinity();
                                for (int64_t ky = 0; ky < kh; ++ky)
                                {
                                    // Map output row oy back to the input row for kernel tap ky:
                                    // origin oy*sh minus the top pad, then advance by ky. Rows that
                                    // land in the padding are skipped (they contribute nothing).
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
                                        m = std::max(m, xc[iy * x.w + ix]);
                                    }
                                }
                                // Flatten (n, c, oy, ox) to a row-major offset into the output plane.
                                y[((n * x.c + c) * oh + oy) * ow + ox] = m;
                            }
                        }
                    }
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::MaxPool, MaxPoolCpu);
} // namespace vknn
