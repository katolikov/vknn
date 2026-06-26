// Windowed AveragePool2D (scalar reference). ONNX count_include_pad selects the divisor.
#include <algorithm>

#include "backends/cpu/cpu_backend.h"

namespace vknn {
namespace {

struct AvgPoolCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    NCHW x = NCHW::from(X.shape);
    auto ints = [&](const char* k, std::vector<int64_t> d) {
      const auto& v = node.attr.getints(k);
      return v.empty() ? d : v;
    };
    auto ks = ints("kernel_shape", {1, 1});
    auto st = ints("strides", {1, 1});
    auto pad = ints("pads", {0, 0, 0, 0});
    int64_t kh = ks[0], kw = ks[1], sh = st[0], sw = st[1], pt = pad[0], pl = pad[1];
    bool incPad = node.attr.geti("count_include_pad", 0) != 0;
    int64_t oh = (x.h + pt + pad[2] - kh) / sh + 1;
    int64_t ow = (x.w + pl + pad[3] - kw) / sw + 1;

    float* y = cpu::allocOut(Y, {x.n, x.c, oh, ow});
    const float* xd = X.host.f32();
    for (int64_t n = 0; n < x.n; ++n)
      for (int64_t c = 0; c < x.c; ++c) {
        const float* xc = xd + (n * x.c + c) * x.h * x.w;
        for (int64_t oy = 0; oy < oh; ++oy)
          for (int64_t ox = 0; ox < ow; ++ox) {
            float acc = 0;
            int64_t cnt = 0;
            for (int64_t ky = 0; ky < kh; ++ky) {
              int64_t iy = oy * sh - pt + ky;
              if (iy < 0 || iy >= x.h)
                continue;
              for (int64_t kx = 0; kx < kw; ++kx) {
                int64_t ix = ox * sw - pl + kx;
                if (ix < 0 || ix >= x.w)
                  continue;
                acc += xc[iy * x.w + ix];
                ++cnt;
              }
            }
            float denom = incPad ? (float)(kh * kw) : (float)std::max<int64_t>(cnt, 1);
            y[((n * x.c + c) * oh + oy) * ow + ox] = acc / denom;
          }
      }
  }
};

}  // namespace
VKNN_REGISTER_CPU_OP(OpType::kAvgPool, AvgPoolCpu);
}  // namespace vknn
