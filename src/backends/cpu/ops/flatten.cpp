// Flatten to 2D: dims [0,axis) collapse to rows, [axis,end) to columns.
#include "backends/cpu/cpu_backend.h"

namespace vx {
namespace {

struct FlattenCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int64_t axis = node.attr.geti("axis", 1);
    int64_t rank = (int64_t)X.shape.size();
    if (axis < 0)
      axis += rank;
    int64_t d0 = 1, d1 = 1;
    for (int64_t i = 0; i < rank; ++i)
      (i < axis ? d0 : d1) *= X.shape[i];
    cpu::copyAs(X, Y, {d0, d1});
  }
  bool supportsDType(DType) const override { return true; }
};

}  // namespace
VX_REGISTER_CPU_OP(OpType::kFlatten, FlattenCpu);
}  // namespace vx
