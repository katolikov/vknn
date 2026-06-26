// Reshape using the shape tensor in inputs[1]. Handles 0 (copy the matching input dim) and -1
// (infer the remaining size).
#include <algorithm>

#include "backends/cpu/cpu_backend.h"

namespace vknn {
namespace {

struct ReshapeCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    const RtTensor& S = ctx.t(node.inputs[1]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int64_t rank = S.elems();
    const int64_t* sd = S.host.i64();
    Shape out(rank);
    int64_t known = 1, inferIdx = -1;
    for (int64_t i = 0; i < rank; ++i) {
      int64_t d = sd[i];
      if (d == 0)
        d = (i < (int64_t)X.shape.size()) ? X.shape[i] : 1;
      out[i] = d;
      if (d == -1)
        inferIdx = i;
      else
        known *= d;
    }
    if (inferIdx >= 0)
      out[inferIdx] = X.elems() / std::max<int64_t>(known, 1);
    cpu::copyAs(X, Y, out);
  }
  bool supportsDType(DType) const override { return true; }
};

}  // namespace
VKNN_REGISTER_CPU_OP(OpType::kReshape, ReshapeCpu);
}  // namespace vknn
