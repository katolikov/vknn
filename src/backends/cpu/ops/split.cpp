// Split a tensor along an axis into N outputs (generic N-D, dtype-agnostic).
#include "backends/cpu/cpu_backend.h"
#include "vx/op.h"

namespace vx {
namespace {
struct SplitCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    int rank = (int)X.shape.size();
    int64_t axis = node.attr.geti("axis", 0);
    if (axis < 0) axis += rank;
    int64_t nout = (int64_t)node.outputs.size();
    // outer = product of dims before axis; inner = product after axis
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; ++i) outer *= X.shape[i];
    for (int i = (int)axis + 1; i < rank; ++i) inner *= X.shape[i];
    bool i64 = X.dtype == DType::kInt64;
    int64_t off = 0;  // running offset along axis
    for (int64_t k = 0; k < nout; ++k) {
      if (node.outputs[k] == kNoTensor) continue;
      RtTensor& Y = ctx.t(node.outputs[k]);
      Shape os = ctx.graph->desc(node.outputs[k]).shape;
      int64_t seg = os[axis];
      float* yf = i64 ? nullptr : cpu::allocOut(Y, os);
      int64_t* yi = i64 ? cpu::allocOutI64(Y, os) : nullptr;
      for (int64_t o = 0; o < outer; ++o)
        for (int64_t s = 0; s < seg; ++s) {
          int64_t srcBase = (o * X.shape[axis] + off + s) * inner;
          int64_t dstBase = (o * seg + s) * inner;
          if (i64) { const int64_t* x = X.host.i64(); for (int64_t j = 0; j < inner; ++j) yi[dstBase + j] = x[srcBase + j]; }
          else { const float* x = X.host.f32(); for (int64_t j = 0; j < inner; ++j) yf[dstBase + j] = x[srcBase + j]; }
        }
      off += seg;
    }
  }
  bool supportsDType(DType) const override { return true; }
};
}  // namespace
VX_REGISTER_CPU_OP(OpType::kSplit, SplitCpu);
}  // namespace vx
