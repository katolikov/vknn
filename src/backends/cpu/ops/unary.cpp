// Elementwise unary family (Sigmoid/Tanh/HardSwish/HardSigmoid/LeakyRelu/Elu/Abs/Neg/Exp/Log/
// Sqrt/Floor/Ceil/Relu). One op, switched on node.subOp; params (alpha/beta) in actLo/actHi.
#include <cmath>

#include "backends/cpu/cpu_backend.h"
#include "vx/op.h"

namespace vx {
namespace {

static float unary(float x, UnaryType op, float a, float b) {
  switch (op) {
    case UnaryType::kSigmoid:
      return 1.f / (1.f + std::exp(-x));
    case UnaryType::kTanh:
      return std::tanh(x);
    case UnaryType::kHardSwish:
      return x * std::min(std::max(x + 3.f, 0.f), 6.f) / 6.f;
    case UnaryType::kHardSigmoid:
      return std::min(std::max(a * x + b, 0.f), 1.f);
    case UnaryType::kLeakyRelu:
      return x > 0 ? x : a * x;
    case UnaryType::kElu:
      return x > 0 ? x : a * (std::exp(x) - 1.f);
    case UnaryType::kAbs:
      return std::fabs(x);
    case UnaryType::kNeg:
      return -x;
    case UnaryType::kExp:
      return std::exp(x);
    case UnaryType::kLog:
      return std::log(x);
    case UnaryType::kSqrt:
      return std::sqrt(x);
    case UnaryType::kFloor:
      return std::floor(x);
    case UnaryType::kCeil:
      return std::ceil(x);
    case UnaryType::kRelu:
      return x > 0 ? x : 0;
    case UnaryType::kSiLU:
      return x / (1.f + std::exp(-x));
    case UnaryType::kErf:
      return std::erf(x);
    case UnaryType::kCos:
      return std::cos(x);
    case UnaryType::kSin:
      return std::sin(x);
    case UnaryType::kReciprocal:
      return 1.f / x;
    case UnaryType::kSoftplus:
      return std::max(x, 0.f) + std::log1p(std::exp(-std::fabs(x)));
    case UnaryType::kInvalid:
      break;
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
    for (int64_t i = 0; i < n; ++i)
      y[i] = unary(x[i], (UnaryType)node.subOp, node.actLo, node.actHi);
  }
};

}  // namespace
VX_REGISTER_CPU_OP(OpType::kUnary, UnaryCpu);
}  // namespace vx
