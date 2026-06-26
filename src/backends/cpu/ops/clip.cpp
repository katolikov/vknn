// Clip(x, min, max). MobileNetV2 uses this as ReLU6 (min=0, max=6). Opset 11/12 pass min/max
// as inputs; older models pass them as attributes - handle both.
#include <limits>

#include "backends/cpu/cpu_backend.h"

namespace vx {
namespace {

struct ClipCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    float lo = -std::numeric_limits<float>::infinity();
    float hi = std::numeric_limits<float>::infinity();
    if (node.inputs.size() > 1 && node.inputs[1] != kNoTensor)
      lo = ctx.t(node.inputs[1]).host.f32()[0];
    if (node.inputs.size() > 2 && node.inputs[2] != kNoTensor)
      hi = ctx.t(node.inputs[2]).host.f32()[0];
    if (node.attr.has("min"))
      lo = node.attr.getf("min", lo);
    if (node.attr.has("max"))
      hi = node.attr.getf("max", hi);
    int64_t n = X.elems();
    float* y = cpu::allocOut(Y, X.shape);
    const float* x = X.host.f32();
    for (int64_t i = 0; i < n; ++i) {
      float v = x[i];
      y[i] = v < lo ? lo : (v > hi ? hi : v);
    }
  }
};

}  // namespace
VX_REGISTER_CPU_OP(OpType::kClip, ClipCpu);
}  // namespace vx
