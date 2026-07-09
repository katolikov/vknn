// GridSample (ONNX, 2D): sample X[N,C,Hin,Win] at normalized coords from grid[N,Hout,Wout,2].
// bilinear/nearest/cubic; padding zeros/border/reflection; align_corners. NCHW fp32 reference oracle.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>
#include <cmath>

namespace vknn {
    namespace {

        /// Reflect a continuous coordinate back into [lo, hi] with the ONNX `reflection` padding rule:
        /// the axis bounces off each edge, so a coordinate d past an edge lands d inside it, folding
        /// with period 2*(hi-lo). Degenerate axes (hi <= lo) collapse to lo.
        static double reflectCoord(double x, double lo, double hi) {
            if (hi <= lo)
            {
                return lo;
            }
            // Fold into one period [0, 2*rng); the second half of the period mirrors the first,
            // giving the bounce back toward lo.
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
                const RtTensor &X = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                NCHW            x = NCHW::from(X.shape);
                // Coordinate source. The plain op reads a materialized normalized grid
                // grid[N,Hout,Wout,2] (input 1). The fused warp op (fuseGridSampleWarp) instead reads
                // an NCHW flow[N,2,Hout,Wout] (input 1) + a base grid[Nb,Hout,Wout,2] (input 2) and a
                // scalar scale, computing coord = base + scale*flow per output location. The arithmetic
                // reproduces the split Mul(float)+Add(float) it replaces exactly, so the two paths are
                // byte-identical on this fp32 oracle.
                const bool      warp  = node.attr.has("warp");
                const RtTensor &G     = ctx.t(node.inputs[1]);
                int             Hout  = warp ? (int) G.shape[2] : (int) G.shape[1];
                int             Wout  = warp ? (int) G.shape[3] : (int) G.shape[2];
                const float    *gd    = warp ? nullptr : G.host.f32();
                const float    *flowd = warp ? G.host.f32() : nullptr;                     // [N,2,Hout,Wout]
                const float    *based = warp ? ctx.t(node.inputs[2]).host.f32() : nullptr; // [Nb,Hout,Wout,2]
                const float     scale = warp ? node.attr.getf("warp_scale", 1.f) : 0.f;
                const bool      baseBroadcastN = warp && ctx.t(node.inputs[2]).shape[0] == 1;
                // Normalized grid value at (n,oy,ox) for coordinate axis c (0=x, 1=y).
                auto            gridVal        = [&](int64_t n, int oy, int ox, int c) -> double {
                    if (warp)
                    {
                        int64_t bn = baseBroadcastN ? 0 : n;
                        float   b  = based[((bn * Hout + oy) * Wout + ox) * 2 + c];
                        // The product is rounded to float on its own line before the add: the split
                        // path stores fsc = float(flow*scale) then adds, so a single b + flow*scale
                        // expression (contractable to an FMA, one rounding) would diverge by a ULP.
                        float fsc = flowd[((n * 2 + c) * Hout + oy) * Wout + ox] * scale;
                        return (double) (b + fsc);
                    }
                    return (double) gd[((n * Hout + oy) * Wout + ox) * 2 + c];
                };
                std::string     mode    = node.attr.gets("mode", "linear");
                bool            nearest = (mode == "nearest");
                bool            cubic   = (mode == "cubic" || mode == "bicubic");
                std::string     pad     = node.attr.gets("padding_mode", "zeros");
                int             align   = (int) node.attr.geti("align_corners", 0);
                // Denormalization selectors for `unnorm` below. align_corners=1 maps grid [-1,1] onto
                // pixel centers [0, S-1] (a=1, b=0); align_corners=0 maps onto [-0.5, S-0.5], i.e. the
                // outer edges of the border pixels (a=0, b=1).
                int             a = align ? 1 : 0, b = 1 - a;

                float       *y      = cpu::allocOut(Y, {x.n, x.c, (int64_t) Hout, (int64_t) Wout});
                const float *xd     = X.host.f32();
                // Grid value g in [-1, 1] -> input coordinate. Expands to (g+1)/2*(S-1) when
                // align_corners (a=1,b=0) and ((g+1)*S - 1)/2 otherwise (a=0,b=1).
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
                            double  ix = handle(unnorm(gridVal(n, oy, ox, 0), (int) x.w), (int) x.w);
                            double  iy = handle(unnorm(gridVal(n, oy, ox, 1), (int) x.h), (int) x.h);
                            if (nearest)
                            {
                                // Round half up (floor(x+0.5)), the ONNX nearest convention: exact .5
                                // ties resolve toward +inf rather than to-even.
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
                                // Bilinear (default): the four integer neighbors of (ix,iy) with
                                // fractional weights, interpolated along x then blended along y.
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
