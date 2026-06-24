// Gather along axis 0. That's all the MobileNetV2 classifier preamble needs (indexing a shape
// vector); enough to keep the shape math on the CPU while the heavy ops run on the GPU.
#include "backends/cpu/cpu_backend.h"
#include <algorithm>
#include <cstring>

namespace vx {
namespace {

struct GatherCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& D = ctx.t(node.inputs[0]);
    const RtTensor& I = ctx.t(node.inputs[1]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int64_t nidx = I.elems();
    const int64_t* idx = I.host.i64();
    int64_t outer = D.shape.empty() ? 0 : D.shape[0];
    int64_t inner = D.elems() / std::max<int64_t>(outer, 1);
    Shape outShape;
    bool scalarIndex = I.shape.empty() || (I.shape.size() == 1 && I.shape[0] == 1 && nidx == 1);
    if (scalarIndex) {
      for (size_t i = 1; i < D.shape.size(); ++i) outShape.push_back(D.shape[i]);
      if (outShape.empty()) outShape = {1};
    } else {
      outShape.push_back(nidx);
      for (size_t i = 1; i < D.shape.size(); ++i) outShape.push_back(D.shape[i]);
    }
    if (D.dtype == DType::kInt64) {
      int64_t* y = cpu::allocOutI64(Y, outShape);
      const int64_t* d = D.host.i64();
      for (int64_t k = 0; k < nidx; ++k) {
        int64_t src = (idx[k] < 0 ? idx[k] + outer : idx[k]);
        std::memcpy(y + k * inner, d + src * inner, inner * sizeof(int64_t));
      }
    } else {
      float* y = cpu::allocOut(Y, outShape);
      const float* d = D.host.f32();
      for (int64_t k = 0; k < nidx; ++k) {
        int64_t src = (idx[k] < 0 ? idx[k] + outer : idx[k]);
        std::memcpy(y + k * inner, d + src * inner, inner * sizeof(float));
      }
    }
  }
  bool supportsDType(DType) const override { return true; }
};

}  // namespace
VX_REGISTER_CPU_OP(OpType::kGather, GatherCpu);
}  // namespace vx
