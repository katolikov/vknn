// Squeeze: remove size-1 dims (given `axes`, or all size-1 dims when axes is absent). Pure metadata
// reshape - data is untouched, so it's a byte copy with a new shape. axes from the attribute
// (opset<13) or the int64 input[1] (opset>=13). dtype-agnostic.
#include <algorithm>
#include "backends/cpu/cpu_backend.h"
#include "vx/op.h"

namespace vx {
namespace {

struct SqueezeCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int rank = (int)X.shape.size();
    std::vector<int64_t> axes = node.attr.getints("axes");
    if (axes.empty() && node.inputs.size() > 1 && node.inputs[1] != kNoTensor) {
      const RtTensor& A = ctx.t(node.inputs[1]);
      axes.assign(A.host.i64(), A.host.i64() + A.elems());
    }
    std::vector<bool> drop(rank, false);
    if (axes.empty()) {
      for (int k = 0; k < rank; ++k) drop[k] = (X.shape[k] == 1);  // remove every size-1 dim
    } else {
      for (int64_t ax : axes) { if (ax < 0) ax += rank; if (ax >= 0 && ax < rank) drop[ax] = true; }
    }
    Shape out;
    for (int k = 0; k < rank; ++k) if (!drop[k]) out.push_back(X.shape[k]);
    if (out.empty()) out.push_back(1);  // scalar -> [1]
    cpu::copyAs(X, Y, out);
  }
  bool supportsDType(DType) const override { return true; }
};

}  // namespace
VX_REGISTER_CPU_OP(OpType::kSqueeze, SqueezeCpu);
}  // namespace vx
