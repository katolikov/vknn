// Concat along an axis (same-rank inputs).
#include "backends/cpu/cpu_backend.h"
#include <cstring>

namespace vx {
namespace {

struct ConcatCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    RtTensor& Y = ctx.t(node.outputs[0]);
    int64_t axis = node.attr.geti("axis", 0);
    const RtTensor& first = ctx.t(node.inputs[0]);
    int64_t rank = (int64_t)first.shape.size();
    if (axis < 0) axis += rank;
    Shape out = first.shape;
    int64_t total = 0;
    for (TensorId in : node.inputs) total += ctx.t(in).shape[axis];
    out[axis] = total;
    auto blockElems = [&](const Shape& s) {
      int64_t b = 1;
      for (int64_t i = axis; i < (int64_t)s.size(); ++i) b *= s[i];
      return b;
    };
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= first.shape[i];
    bool isI64 = first.dtype == DType::kInt64;
    if (isI64) {
      int64_t* y = cpu::allocOutI64(Y, out);
      int64_t outBlock = blockElems(out), off = 0;
      for (TensorId in : node.inputs) {
        const RtTensor& T = ctx.t(in);
        int64_t bk = blockElems(T.shape);
        for (int64_t o = 0; o < outer; ++o)
          std::memcpy(y + o * outBlock + off, T.host.i64() + o * bk, bk * sizeof(int64_t));
        off += bk;
      }
    } else {
      float* y = cpu::allocOut(Y, out);
      int64_t outBlock = blockElems(out), off = 0;
      for (TensorId in : node.inputs) {
        const RtTensor& T = ctx.t(in);
        int64_t bk = blockElems(T.shape);
        for (int64_t o = 0; o < outer; ++o)
          std::memcpy(y + o * outBlock + off, T.host.f32() + o * bk, bk * sizeof(float));
        off += bk;
      }
    }
  }
  bool supportsDType(DType) const override { return true; }
};

}  // namespace
VX_REGISTER_CPU_OP(OpType::kConcat, ConcatCpu);
}  // namespace vx
