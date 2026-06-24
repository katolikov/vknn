// Resize / Upsample (spatial), NCHW reference. nearest + linear(bilinear); coordinate transform
// modes half_pixel / align_corners / asymmetric / pytorch_half_pixel. Output shape from inferShapes.
#include <algorithm>
#include <cmath>
#include "backends/cpu/cpu_backend.h"
#include "vx/op.h"

namespace vx {

// mode codes shared with the shader
int vxResizeMode(const std::string& s) { return s == "linear" || s == "bilinear" ? 1 : 0; }
int vxResizeCoord(const std::string& s) {
  if (s == "align_corners") return 1;
  if (s == "asymmetric") return 2;
  if (s == "pytorch_half_pixel") return 3;
  return 0;  // half_pixel
}
static float srcCoord(int d, int outS, int inS, int cm) {
  float scale = (float)outS / (float)inS;
  if (cm == 1) return outS > 1 ? (float)d * (inS - 1) / (outS - 1) : 0.f;  // align_corners
  if (cm == 2) return (float)d / scale;                                    // asymmetric
  if (cm == 3) return outS > 1 ? ((float)d + 0.5f) / scale - 0.5f : 0.f;   // pytorch_half_pixel
  return ((float)d + 0.5f) / scale - 0.5f;                                 // half_pixel
}

namespace {
struct ResizeCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    Shape os = ctx.graph->desc(node.outputs[0]).shape;
    NCHW x = NCHW::from(X.shape);
    int OH = (int)os[2], OW = (int)os[3];
    int mode = vxResizeMode(node.attr.gets("mode", "nearest"));
    int cm = vxResizeCoord(node.attr.gets("coordinate_transformation_mode", "half_pixel"));
    float* y = cpu::allocOut(Y, os);
    const float* xd = X.host.f32();
    auto clampi = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };
    for (int64_t n = 0; n < x.n; ++n)
      for (int64_t c = 0; c < x.c; ++c) {
        const float* xc = xd + (n * x.c + c) * x.h * x.w;
        float* yc = y + (n * x.c + c) * OH * OW;
        for (int oy = 0; oy < OH; ++oy) {
          float fy = srcCoord(oy, OH, (int)x.h, cm);
          for (int ox = 0; ox < OW; ++ox) {
            float fx = srcCoord(ox, OW, (int)x.w, cm);
            float v;
            if (mode == 0) {  // nearest (round_prefer_floor)
              int iy = (int)std::floor(fy); if (fy - iy > 0.5f) iy++;
              int ix = (int)std::floor(fx); if (fx - ix > 0.5f) ix++;
              iy = clampi(iy, 0, (int)x.h - 1); ix = clampi(ix, 0, (int)x.w - 1);
              v = xc[iy * x.w + ix];
            } else {  // bilinear
              int iy0 = (int)std::floor(fy), ix0 = (int)std::floor(fx);
              float wy = fy - iy0, wx = fx - ix0;
              int iy1 = clampi(iy0 + 1, 0, (int)x.h - 1), ix1 = clampi(ix0 + 1, 0, (int)x.w - 1);
              int cy0 = clampi(iy0, 0, (int)x.h - 1), cx0 = clampi(ix0, 0, (int)x.w - 1);
              float a = xc[cy0 * x.w + cx0], b = xc[cy0 * x.w + ix1];
              float cc = xc[iy1 * x.w + cx0], d = xc[iy1 * x.w + ix1];
              v = a * (1 - wy) * (1 - wx) + b * (1 - wy) * wx + cc * wy * (1 - wx) + d * wy * wx;
            }
            yc[oy * OW + ox] = v;
          }
        }
      }
  }
};
}  // namespace
VX_REGISTER_CPU_OP(OpType::kResize, ResizeCpu);
}  // namespace vx
