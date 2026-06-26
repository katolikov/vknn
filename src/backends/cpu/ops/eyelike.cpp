// EyeLike: output has the same 2-D shape as the input, with ones on a diagonal (offset by attr `k`)
// and zeros elsewhere. The input values are ignored — only its shape matters. Usually const-folded.
#include "backends/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
namespace {

struct EyeLikeCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    Shape s = X.shape;
    // EyeLike is defined for 2-D tensors; fall back to a [n,n] view if rank != 2.
    int64_t rows, cols;
    if (s.size() == 2) {
      rows = s[0];
      cols = s[1];
    } else {
      int64_t n = X.elems();
      rows = n;
      cols = n > 0 ? 1 : 0;
      s = {rows, cols};
    }
    int64_t k = node.attr.geti("k", 0);  // diagonal offset
    float* y = cpu::allocOut(Y, s);
    for (int64_t i = 0; i < rows; ++i)
      for (int64_t j = 0; j < cols; ++j)
        y[i * cols + j] = (j - i == k) ? 1.f : 0.f;
  }
  bool supportsDType(DType) const override { return true; }
};

}  // namespace
VKNN_REGISTER_CPU_OP(OpType::kEyeLike, EyeLikeCpu);
}  // namespace vknn
