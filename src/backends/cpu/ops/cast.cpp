// Cast: change dtype. vknn computes in fp32 and carries int64 for shape paths, so we support
// float<->int64 conversions (other integer widths map onto these). Shape unchanged.
#include "backends/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
namespace {
struct CastCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int64_t to = node.attr.geti("to", 1);  // ONNX TensorProto: 1=FLOAT, 7=INT64, 6=INT32
    int64_t n = X.elems();
    bool inI64 = X.dtype == DType::kInt64;
    bool outI64 = (to == 7 || to == 6 || to == 5 || to == 3);  // integer targets -> carry as int64
    if (outI64) {
      int64_t* y = cpu::allocOutI64(Y, X.shape);
      if (inI64) {
        const int64_t* x = X.host.i64();
        for (int64_t i = 0; i < n; ++i)
          y[i] = x[i];
      } else {
        const float* x = X.host.f32();
        for (int64_t i = 0; i < n; ++i)
          y[i] = (int64_t)x[i];
      }
    } else {
      float* y = cpu::allocOut(Y, X.shape);
      if (inI64) {
        const int64_t* x = X.host.i64();
        for (int64_t i = 0; i < n; ++i)
          y[i] = (float)x[i];
      } else {
        const float* x = X.host.f32();
        for (int64_t i = 0; i < n; ++i)
          y[i] = x[i];
      }
    }
  }
  bool supportsDType(DType) const override { return true; }
};
}  // namespace
VKNN_REGISTER_CPU_OP(OpType::kCast, CastCpu);
}  // namespace vknn
