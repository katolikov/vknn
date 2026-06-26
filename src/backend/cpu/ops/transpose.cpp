// Transpose / Permute (generic N-D, dtype-agnostic). CPU-only; channel-permuting layouts make a
// packed NC4HW4 kernel a scatter, so this runs in canonical NCHW with boundary converts.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
namespace {
struct TransposeCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int rank = (int)X.shape.size();
    std::vector<int64_t> perm = node.attr.getints("perm");
    if ((int)perm.size() != rank) {
      perm.clear();
      for (int i = rank - 1; i >= 0; --i)
        perm.push_back(i);
    }
    for (auto& p : perm)
      if (p < 0)
        p += rank;
    Shape out(rank);
    for (int i = 0; i < rank; ++i)
      out[i] = X.shape[perm[i]];
    std::vector<int64_t> inStride(rank, 1), outStride(rank, 1);
    for (int i = rank - 2; i >= 0; --i) {
      inStride[i] = inStride[i + 1] * X.shape[i + 1];
      outStride[i] = outStride[i + 1] * out[i + 1];
    }
    int64_t elems = numElements(out);
    bool i64 = X.dtype == DType::kInt64;
    const float* xf = i64 ? nullptr : X.host.f32();
    const int64_t* xi = i64 ? X.host.i64() : nullptr;
    float* yf = i64 ? nullptr : cpu::allocOut(Y, out);
    int64_t* yi = i64 ? cpu::allocOutI64(Y, out) : nullptr;
    for (int64_t oi = 0; oi < elems; ++oi) {
      int64_t rem = oi, inf = 0;
      for (int i = 0; i < rank; ++i) {
        int64_t c = rem / outStride[i];
        rem %= outStride[i];
        inf += c * inStride[perm[i]];
      }
      if (i64)
        yi[oi] = xi[inf];
      else
        yf[oi] = xf[inf];
    }
  }
  bool supportsDType(DType) const override { return true; }
};
}  // namespace
VKNN_REGISTER_CPU_OP(OpType::kTranspose, TransposeCpu);
}  // namespace vknn
