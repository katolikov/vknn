// Constant: emit the node's stored value (int64 or float vector).
#include "backends/cpu/cpu_backend.h"

namespace vx {
namespace {

struct ConstantCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    RtTensor& Y = ctx.t(node.outputs[0]);
    auto it = node.attr.map.find("value");
    if (it != node.attr.map.end() && it->second.kind == Attr::kInts) {
      const auto& v = it->second.ints;
      int64_t* y = cpu::allocOutI64(Y, {(int64_t)v.size()});
      for (size_t i = 0; i < v.size(); ++i) y[i] = v[i];
    } else if (it != node.attr.map.end() && it->second.kind == Attr::kFloats) {
      const auto& v = it->second.floats;
      float* y = cpu::allocOut(Y, {(int64_t)v.size()});
      for (size_t i = 0; i < v.size(); ++i) y[i] = v[i];
    } else {
      cpu::allocOutI64(Y, {0});
    }
  }
  bool supportsDType(DType) const override { return true; }
};

}  // namespace
VX_REGISTER_CPU_OP(OpType::kConstant, ConstantCpu);
}  // namespace vx
