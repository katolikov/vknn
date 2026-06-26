// Softmax over the axis range [axis, rank). Standard max-subtract for numerical stability.
#include <algorithm>
#include <cmath>

#include "backends/cpu/cpu_backend.h"

namespace vknn {
namespace {

struct SoftmaxCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int64_t axis = node.attr.geti("axis", -1);
    int64_t rank = (int64_t)X.shape.size();
    if (axis < 0)
      axis += rank;
    int64_t inner = 1;
    for (int64_t i = axis; i < rank; ++i)
      inner *= X.shape[i];
    int64_t outer = X.elems() / inner;
    float* y = cpu::allocOut(Y, X.shape);
    const float* x = X.host.f32();
    for (int64_t o = 0; o < outer; ++o) {
      const float* xr = x + o * inner;
      float* yr = y + o * inner;
      float mx = xr[0];
      for (int64_t i = 1; i < inner; ++i)
        mx = std::max(mx, xr[i]);
      double sum = 0;
      for (int64_t i = 0; i < inner; ++i) {
        yr[i] = std::exp(xr[i] - mx);
        sum += yr[i];
      }
      for (int64_t i = 0; i < inner; ++i)
        yr[i] = (float)(yr[i] / sum);
    }
  }
};

}  // namespace
VKNN_REGISTER_CPU_OP(OpType::kSoftmax, SoftmaxCpu);
}  // namespace vknn
