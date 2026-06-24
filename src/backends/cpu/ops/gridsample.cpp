// GridSample (ONNX, 2D): sample X[N,C,Hin,Win] at normalized coords from grid[N,Hout,Wout,2].
// bilinear/nearest; padding zeros/border/reflection; align_corners. NCHW fp32 reference oracle.
#include <algorithm>
#include <cmath>
#include "backends/cpu/cpu_backend.h"
#include "vx/op.h"

namespace vx {
namespace {

static double reflectCoord(double x, double lo, double hi) {
  if (hi <= lo) return lo;
  double rng = hi - lo, t = std::fmod(x - lo, 2 * rng);
  if (t < 0) t += 2 * rng;
  if (t > rng) t = 2 * rng - t;
  return lo + t;
}

struct GridSampleCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    const RtTensor& G = ctx.t(node.inputs[1]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    NCHW x = NCHW::from(X.shape);
    int Hout = (int)G.shape[1], Wout = (int)G.shape[2];
    std::string mode = node.attr.gets("mode", "linear");
    bool nearest = (mode == "nearest");
    std::string pad = node.attr.gets("padding_mode", "zeros");
    int align = (int)node.attr.geti("align_corners", 0);
    int a = align ? 1 : 0, b = 1 - a;

    float* y = cpu::allocOut(Y, {x.n, x.c, (int64_t)Hout, (int64_t)Wout});
    const float* xd = X.host.f32();
    const float* gd = G.host.f32();
    auto unnorm = [&](double g, int S) { return ((1.0 + g) * (S - a) - b) * 0.5; };
    auto handle = [&](double c, int S) {  // map continuous coord per padding mode
      if (pad == "reflection") return reflectCoord(c, align ? 0.0 : -0.5, align ? (S - 1.0) : (S - 0.5));
      return c;  // zeros/border handled at fetch
    };
    auto fetch = [&](int n, int ch, int px, int py) -> float {
      if (pad == "border" || pad == "reflection") {
        px = std::min(std::max(px, 0), (int)x.w - 1);
        py = std::min(std::max(py, 0), (int)x.h - 1);
      } else if (px < 0 || px >= (int)x.w || py < 0 || py >= (int)x.h) {
        return 0.f;
      }
      return xd[((n * x.c + ch) * x.h + py) * x.w + px];
    };

    for (int64_t n = 0; n < x.n; ++n)
      for (int oy = 0; oy < Hout; ++oy)
        for (int ox = 0; ox < Wout; ++ox) {
          int64_t gi = ((n * Hout + oy) * Wout + ox) * 2;
          double ix = handle(unnorm(gd[gi + 0], (int)x.w), (int)x.w);
          double iy = handle(unnorm(gd[gi + 1], (int)x.h), (int)x.h);
          if (nearest) {
            int rx = (int)std::floor(ix + 0.5), ry = (int)std::floor(iy + 0.5);
            for (int64_t c = 0; c < x.c; ++c)
              y[((n * x.c + c) * Hout + oy) * Wout + ox] = fetch((int)n, (int)c, rx, ry);
          } else {
            int x0 = (int)std::floor(ix), y0 = (int)std::floor(iy);
            double wx = ix - x0, wy = iy - y0;
            for (int64_t c = 0; c < x.c; ++c) {
              float v00 = fetch((int)n, (int)c, x0, y0), v01 = fetch((int)n, (int)c, x0 + 1, y0);
              float v10 = fetch((int)n, (int)c, x0, y0 + 1), v11 = fetch((int)n, (int)c, x0 + 1, y0 + 1);
              y[((n * x.c + c) * Hout + oy) * Wout + ox] =
                  (float)((1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11));
            }
          }
        }
  }
};
}  // namespace
VX_REGISTER_CPU_OP(OpType::kGridSample, GridSampleCpu);
}  // namespace vx
