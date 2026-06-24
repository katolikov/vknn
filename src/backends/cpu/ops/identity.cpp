// Identity: pass the tensor through unchanged.
#include "backends/cpu/cpu_backend.h"

namespace vx {
namespace {

struct IdentityCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    Y.shape = X.shape;
    Y.dtype = X.dtype;
    Y.host = X.host;
    Y.hostValid = true;
    Y.deviceValid = false;
  }
};

}  // namespace
VX_REGISTER_CPU_OP(OpType::kIdentity, IdentityCpu);
}  // namespace vx
