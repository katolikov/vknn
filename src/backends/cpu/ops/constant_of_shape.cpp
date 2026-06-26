// ConstantOfShape: input is an int64 1-D shape tensor; output is a tensor of that shape filled with
// the `value` attribute (default float 0). Usually const-folded, but registered so the parser
// accepts it and so a runtime shape input still works. Emits float (the canonical compute dtype);
// integer `value` attrs are emitted as int64.
#include "backends/cpu/cpu_backend.h"
#include "vx/op.h"

namespace vx {
namespace {

struct ConstantOfShapeCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& S = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    // The shape operand is an int64 vector; derive the output shape from it.
    int64_t r = S.elems();
    Shape out;
    if (S.dtype == DType::kInt64) {
      const int64_t* s = S.host.i64();
      for (int64_t i = 0; i < r; ++i) out.push_back(s[i]);
    } else {
      const float* s = S.host.f32();
      for (int64_t i = 0; i < r; ++i) out.push_back((int64_t)s[i]);
    }
    if (out.empty()) out = {1};  // scalar fill

    auto it = node.attr.map.find("value");
    bool intVal = it != node.attr.map.end() && it->second.kind == Attr::kInts;
    int64_t n = numElements(out);
    if (intVal) {
      int64_t v = it->second.ints.empty() ? 0 : it->second.ints[0];
      int64_t* y = cpu::allocOutI64(Y, out);
      for (int64_t i = 0; i < n; ++i) y[i] = v;
    } else {
      float v = (it != node.attr.map.end() && !it->second.floats.empty()) ? it->second.floats[0] : 0.f;
      float* y = cpu::allocOut(Y, out);
      for (int64_t i = 0; i < n; ++i) y[i] = v;
    }
  }
  bool supportsDType(DType) const override { return true; }
};

}  // namespace
VX_REGISTER_CPU_OP(OpType::kConstantOfShape, ConstantOfShapeCpu);
}  // namespace vx
