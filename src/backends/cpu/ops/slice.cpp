// Slice (generic N-D). starts/ends required; axes/steps optional. From attrs (opset<10) or
// initializer inputs[1..4] (opset 10+). dtype-agnostic for shape-path.
#include <algorithm>
#include "backends/cpu/cpu_backend.h"
#include "vx/op.h"

namespace vx {
namespace {
struct SliceCpu : CpuOp {
  static std::vector<int64_t> rd(const Node& n, ExecContext& ctx, const char* attr, int idx) {
    const auto& a = n.attr.getints(attr);
    if (!a.empty()) return a;
    if (idx < (int)n.inputs.size() && n.inputs[idx] != kNoTensor) {
      const RtTensor& t = ctx.t(n.inputs[idx]);
      return std::vector<int64_t>(t.host.i64(), t.host.i64() + t.elems());
    }
    return {};
  }
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int rank = (int)X.shape.size();
    auto starts = rd(node, ctx, "starts", 1), ends = rd(node, ctx, "ends", 2);
    auto axes = rd(node, ctx, "axes", 3), steps = rd(node, ctx, "steps", 4);
    std::vector<int64_t> begin(rank, 0), step(rank, 1);
    Shape out = X.shape;
    for (size_t k = 0; k < starts.size(); ++k) {
      int ax = (int)(axes.empty() ? (int64_t)k : axes[k]); if (ax < 0) ax += rank;
      if (ax < 0 || ax >= rank) continue;
      int64_t dim = X.shape[ax], sp = k < steps.size() ? steps[k] : 1;
      int64_t st = starts[k] < 0 ? starts[k] + dim : starts[k];
      int64_t en = ends[k] < 0 ? ends[k] + dim : ends[k];
      st = std::max<int64_t>(0, std::min(st, dim)); en = std::max<int64_t>(0, std::min(en, dim));
      begin[ax] = st; step[ax] = sp;
      out[ax] = sp > 0 ? std::max<int64_t>(0, (en - st + sp - 1) / sp) : 0;
    }
    std::vector<int64_t> inStride(rank, 1), outStride(rank, 1);
    for (int i = rank - 2; i >= 0; --i) { inStride[i] = inStride[i + 1] * X.shape[i + 1]; outStride[i] = outStride[i + 1] * out[i + 1]; }
    int64_t elems = numElements(out);
    bool i64 = X.dtype == DType::kInt64;
    const float* xf = i64 ? nullptr : X.host.f32();
    const int64_t* xi = i64 ? X.host.i64() : nullptr;
    float* yf = i64 ? nullptr : cpu::allocOut(Y, out);
    int64_t* yi = i64 ? cpu::allocOutI64(Y, out) : nullptr;
    for (int64_t oi = 0; oi < elems; ++oi) {
      int64_t rem = oi, inf = 0;
      for (int i = 0; i < rank; ++i) { int64_t c = rem / outStride[i]; rem %= outStride[i]; inf += (begin[i] + c * step[i]) * inStride[i]; }
      if (i64) yi[oi] = xi[inf]; else yf[oi] = xf[inf];
    }
  }
  bool supportsDType(DType) const override { return true; }
};
}  // namespace
VX_REGISTER_CPU_OP(OpType::kSlice, SliceCpu);
}  // namespace vx
