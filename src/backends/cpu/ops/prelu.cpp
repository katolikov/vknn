// PRelu: y = x>0 ? x : slope*x, slope per-channel (or scalar). slope is initializer input[1].
#include "backends/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
namespace {
struct PReluCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    const RtTensor& S = ctx.t(node.inputs[1]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    NCHW x = NCHW::from(X.shape);
    int64_t hw = x.h * x.w, n = X.elems();
    float* y = cpu::allocOut(Y, X.shape);
    const float* xd = X.host.f32();
    const float* s = S.host.f32();
    int64_t nslope = S.elems();
    for (int64_t i = 0; i < n; ++i) {
      int64_t c = (i / hw) % x.c;
      float sl = nslope == 1 ? s[0] : s[c];
      y[i] = xd[i] > 0 ? xd[i] : sl * xd[i];
    }
  }
};
}  // namespace
VKNN_REGISTER_CPU_OP(OpType::kPRelu, PReluCpu);
}  // namespace vknn
