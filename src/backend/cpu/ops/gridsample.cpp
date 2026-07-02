// GridSample (ONNX, 2D): sample X[N,C,Hin,Win] at normalized coords from grid[N,Hout,Wout,2].
// bilinear/nearest/cubic; padding zeros/border/reflection; align_corners. NCHW fp32 reference oracle.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>
#include <cmath>

namespace vknn {
    namespace {

        static double reflectCoord(double x, double lo, double hi) {
            if (hi <= lo)
            {
                return lo;
            }
            double rng = hi - lo, t = std::fmod(x - lo, 2 * rng);
            if (t < 0)
            {
                t += 2 * rng;
            }
            if (t > rng)
            {
                t = 2 * rng - t;
            }
            return lo + t;
        }

        struct GridSampleCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X    = ctx.t(node.inputs[0]);
                const RtTensor &G    = ctx.t(node.inputs[1]);
                RtTensor       &Y    = ctx.t(node.outputs[0]);
                NCHW            x    = NCHW::from(X.shape);
                int             Hout = (int) G.shape[1], Wout = (int) G.shape[2];
                std::string     mode    = node.attr.gets("mode", "linear");
                bool            nearest = (mode == "nearest");
                bool            cubic   = (mode == "cubic" || mode == "bicubic");
                std::string     pad     = node.attr.gets("padding_mode", "zeros");
                int             align   = (int) node.attr.geti("align_corners", 0);
                int             a = align ? 1 : 0, b = 1 - a;

                float       *y      = cpu::allocOut(Y, {x.n, x.c, (int64_t) Hout, (int64_t) Wout});
                const float *xd     = X.host.f32();
                const float *gd     = G.host.f32();
                auto         unnorm = [&](double g, int S) {
                    return ((1.0 + g) * (S - a) - b) * 0.5;
                };
                auto handle = [&](double c, int S) { // map continuous coord per padding mode
                    if (pad == "reflection")
                    {
                        return reflectCoord(c, align ? 0.0 : -0.5, align ? (S - 1.0) : (S - 0.5));
                    }
                    return c; // zeros/border handled at fetch
                };
                auto fetch = [&](int n, int ch, int px, int py) -> float {
                    if (pad == "reflection")
                    {
                        // Reflect each tap (cubic reaches +-2 px past the mapped coordinate, where a
                        // clamp diverges from the reflected pixel; for linear/nearest the two agree).
                        px = (int) std::llround(reflectCoord(px, align ? 0.0 : -0.5, align ? (x.w - 1.0) : (x.w - 0.5)));
                        py = (int) std::llround(reflectCoord(py, align ? 0.0 : -0.5, align ? (x.h - 1.0) : (x.h - 0.5)));
                        px = std::min(std::max(px, 0), (int) x.w - 1);
                        py = std::min(std::max(py, 0), (int) x.h - 1);
                    } else if (pad == "border")
                    {
                        px = std::min(std::max(px, 0), (int) x.w - 1);
                        py = std::min(std::max(py, 0), (int) x.h - 1);
                    } else if (px < 0 || px >= (int) x.w || py < 0 || py >= (int) x.h)
                    { return 0.f; }
                    return xd[((n * x.c + ch) * x.h + py) * x.w + px];
                };

                for (int64_t n = 0; n < x.n; ++n)
                {
                    for (int oy = 0; oy < Hout; ++oy)
                    {
                        for (int ox = 0; ox < Wout; ++ox)
                        {
                            int64_t gi = ((n * Hout + oy) * Wout + ox) * 2;
                            double  ix = handle(unnorm(gd[gi + 0], (int) x.w), (int) x.w);
                            double  iy = handle(unnorm(gd[gi + 1], (int) x.h), (int) x.h);
                            if (nearest)
                            {
                                int rx = (int) std::floor(ix + 0.5), ry = (int) std::floor(iy + 0.5);
                                for (int64_t c = 0; c < x.c; ++c)
                                {
                                    y[((n * x.c + c) * Hout + oy) * Wout + ox] = fetch((int) n, (int) c, rx, ry);
                                }
                            } else if (cubic)
                            {
                                // Cubic convolution (alpha = -0.75, the ONNX/PyTorch kernel): 4 taps per
                                // axis at floor-1..floor+2, separable weights k(1+t), k(t), k(1-t), k(2-t).
                                int    x0 = (int) std::floor(ix), y0 = (int) std::floor(iy);
                                double tx = ix - x0, ty = iy - y0;
                                auto   wts = [](double t, double w[4]) {
                                    constexpr double A = -0.75;
                                    double           t1 = 1.0 + t, t2 = 2.0 - t;
                                    w[0] = A * t1 * t1 * t1 - 5 * A * t1 * t1 + 8 * A * t1 - 4 * A;
                                    w[1] = (A + 2) * t * t * t - (A + 3) * t * t + 1;
                                    w[2] = (A + 2) * (1 - t) * (1 - t) * (1 - t) - (A + 3) * (1 - t) * (1 - t) + 1;
                                    w[3] = A * t2 * t2 * t2 - 5 * A * t2 * t2 + 8 * A * t2 - 4 * A;
                                };
                                double wx[4], wy[4];
                                wts(tx, wx);
                                wts(ty, wy);
                                for (int64_t c = 0; c < x.c; ++c)
                                {
                                    double acc = 0;
                                    for (int j = 0; j < 4; ++j)
                                    {
                                        for (int i2 = 0; i2 < 4; ++i2)
                                        {
                                            acc += wy[j] * wx[i2] * fetch((int) n, (int) c, x0 - 1 + i2, y0 - 1 + j);
                                        }
                                    }
                                    y[((n * x.c + c) * Hout + oy) * Wout + ox] = (float) acc;
                                }
                            } else
                            {
                                int    x0 = (int) std::floor(ix), y0 = (int) std::floor(iy);
                                double wx = ix - x0, wy = iy - y0;
                                for (int64_t c = 0; c < x.c; ++c)
                                {
                                    float v00 = fetch((int) n, (int) c, x0, y0), v01 = fetch((int) n, (int) c, x0 + 1, y0);
                                    float v10 = fetch((int) n, (int) c, x0, y0 + 1), v11 = fetch((int) n, (int) c, x0 + 1, y0 + 1);
                                    y[((n * x.c + c) * Hout + oy) * Wout + ox] = (float) ((1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11));
                                }
                            }
                        }
                    }
                }
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::GridSample, GridSampleCpu);
} // namespace vknn
