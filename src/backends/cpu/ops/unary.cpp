// Elementwise unary family (Sigmoid/Tanh/HardSwish/HardSigmoid/LeakyRelu/Elu/Abs/Neg/Exp/Log/
// Sqrt/Floor/Ceil/Relu). One op, switched on node.subOp; params (alpha/beta) in actLo/actHi.
#include <cmath>
#include "backends/cpu/cpu_backend.h"
#include "vx/op.h"

namespace vx {
namespace {

static float unary(float x, int op, float a, float b) {
  switch (op) {
    case kUSigmoid: return 1.f / (1.f + std::exp(-x));
    case kUTanh: return std::tanh(x);
    case kUHardSwish: return x * std::min(std::max(x + 3.f, 0.f), 6.f) / 6.f;
    case kUHardSigmoid: return std::min(std::max(a * x + b, 0.f), 1.f);
    case kULeakyRelu: return x > 0 ? x : a * x;
    case kUElu: return x > 0 ? x : a * (std::exp(x) - 1.f);
    case kUAbs: return std::fabs(x);
    case kUNeg: return -x;
    case kUExp: return std::exp(x);
    case kULog: return std::log(x);
    case kUSqrt: return std::sqrt(x);
    case kUFloor: return std::floor(x);
    case kUCeil: return std::ceil(x);
    case kURelu: return x > 0 ? x : 0;
  }
  return x;
}

struct UnaryCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int64_t n = X.elems();
    float* y = cpu::allocOut(Y, X.shape);
    const float* x = X.host.f32();
    for (int64_t i = 0; i < n; ++i) y[i] = unary(x[i], node.subOp, node.actLo, node.actHi);
  }
};

}  // namespace
VX_REGISTER_CPU_OP(OpType::kUnary, UnaryCpu);
}  // namespace vx
