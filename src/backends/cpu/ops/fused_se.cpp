// Fused SE middle (CPU oracle): reads pooled avg [N,C,1,1], FC1(relu) -> FC2 -> hardsigmoid -> scale.
#include <algorithm>
#include "backends/cpu/cpu_backend.h"
#include "vx/op.h"

namespace vx {
namespace {
struct FusedSeCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& A = ctx.t(node.inputs[0]);     // pooled avg [N,C,1,1]
    const RtTensor& W1 = ctx.t(node.inputs[1]);
    const RtTensor& W2 = ctx.t(node.inputs[3]);
    const float* b1 = node.inputs[2] != kNoTensor ? ctx.t(node.inputs[2]).host.f32() : nullptr;
    const float* b2 = node.inputs[4] != kNoTensor ? ctx.t(node.inputs[4]).host.f32() : nullptr;
    RtTensor& Y = ctx.t(node.outputs[0]);
    NCHW x = NCHW::from(A.shape);
    int64_t N = x.n, C = x.c, Cr = W1.shape[0];
    const float* avg = A.host.f32();
    const float* w1 = W1.host.f32();
    const float* w2 = W2.host.f32();
    float a = node.actLo, b = node.actHi;
    float* y = cpu::allocOut(Y, {N, C, 1, 1});
    std::vector<float> s1(Cr);
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t j = 0; j < Cr; ++j) {
        double s = b1 ? b1[j] : 0.0;
        for (int64_t c = 0; c < C; ++c) s += (double)w1[j * C + c] * avg[n * C + c];
        s1[j] = s > 0 ? (float)s : 0.f;
      }
      for (int64_t k = 0; k < C; ++k) {
        double s = b2 ? b2[k] : 0.0;
        for (int64_t j = 0; j < Cr; ++j) s += (double)w2[k * Cr + j] * s1[j];
        y[n * C + k] = std::min(std::max(a * (float)s + b, 0.f), 1.f);
      }
    }
  }
};
}  // namespace
VX_REGISTER_CPU_OP(OpType::kFusedSE, FusedSeCpu);
}  // namespace vx
