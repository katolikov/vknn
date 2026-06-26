// Fused depthwise-3x3 + 1x1-project (CPU oracle). inputs: exp, dw_w[E,1,KH,KW], dw_b[E],
// pw_w[Cout,E,1,1], pw_b[Cout], (residual). subOp = dw ActType; fusedAct = project act.
#include <algorithm>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
namespace {
struct FusedDwPwCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    const RtTensor& DW = ctx.t(node.inputs[1]);
    const RtTensor& PW = ctx.t(node.inputs[3]);
    const float* db = node.inputs[2] != kNoTensor ? ctx.t(node.inputs[2]).host.f32() : nullptr;
    const float* pb = node.inputs[4] != kNoTensor ? ctx.t(node.inputs[4]).host.f32() : nullptr;
    RtTensor& Y = ctx.t(node.outputs[0]);
    NCHW x = NCHW::from(X.shape);
    int64_t E = x.c, Cout = PW.shape[0], KH = DW.shape[2], KW = DW.shape[3];
    auto a = [&](const char* k, std::vector<int64_t> d) {
      const auto& v = node.attr.getints(k);
      return v.empty() ? d : v;
    };
    auto st = a("strides", {1, 1}), pad = a("pads", {0, 0, 0, 0}), dil = a("dilations", {1, 1});
    int64_t OH = (x.h + pad[0] + pad[2] - (dil[0] * (KH - 1) + 1)) / st[0] + 1;
    int64_t OW = (x.w + pad[1] + pad[3] - (dil[1] * (KW - 1) + 1)) / st[1] + 1;
    const float* xd = X.host.f32();
    const float* dw = DW.host.f32();
    const float* pw = PW.host.f32();
    Shape os = {x.n, Cout, OH, OW};
    float* y = cpu::allocOut(Y, os);
    const float* res =
        node.fusedResidual != kNoTensor ? ctx.t(node.fusedResidual).host.f32() : nullptr;
    std::vector<float> dwOut(E * OH * OW);
    for (int64_t n = 0; n < x.n; ++n) {
      // depthwise
      for (int64_t e = 0; e < E; ++e)
        for (int64_t oy = 0; oy < OH; ++oy)
          for (int64_t ox = 0; ox < OW; ++ox) {
            float acc = db ? db[e] : 0.f;
            for (int64_t ky = 0; ky < KH; ++ky) {
              int64_t iy = oy * st[0] - pad[0] + ky * dil[0];
              if (iy < 0 || iy >= x.h)
                continue;
              for (int64_t kx = 0; kx < KW; ++kx) {
                int64_t ix = ox * st[1] - pad[1] + kx * dil[1];
                if (ix < 0 || ix >= x.w)
                  continue;
                acc += xd[((n * E + e) * x.h + iy) * x.w + ix] * dw[(e * KH + ky) * KW + kx];
              }
            }
            dwOut[(e * OH + oy) * OW + ox] = acc;
          }
      cpu::applyAct(dwOut.data(), E * OH * OW, (ActType)node.subOp, 0, 6);  // dw act
      // project 1x1 (+ residual + act)
      for (int64_t c = 0; c < Cout; ++c)
        for (int64_t p = 0; p < OH * OW; ++p) {
          float acc = pb ? pb[c] : 0.f;
          for (int64_t e = 0; e < E; ++e)
            acc += pw[c * E + e] * dwOut[e * OH * OW + p];
          int64_t oi = (n * Cout + c) * OH * OW + p;
          if (res)
            acc += res[oi];
          y[oi] = acc;
        }
    }
    cpu::applyAct(y, Y.elems(), node.fusedAct, node.actLo, node.actHi);  // project act
  }
};
}  // namespace
VKNN_REGISTER_CPU_OP(OpType::kFusedDwPw, FusedDwPwCpu);
}  // namespace vknn
