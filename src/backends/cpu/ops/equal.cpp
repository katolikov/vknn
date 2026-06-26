// ONNX Equal (A == B -> 1.0/0.0) with NumPy-style broadcasting. Output is canonical fp32 (1.0/0.0)
// so it can feed a downstream Where over fp32 tensors. Inputs may be fp32 or int64 (Equal often runs
// on int64 shape tensors); each operand is read through its own dtype so the comparison is exact.
#include <algorithm>
#include "backends/cpu/cpu_backend.h"
#include "vx/op.h"

namespace vx {
namespace {

struct EqualCpu : CpuOp {
  // Accept fp32 + int64 (shape tensors) like the rest of the broadcasting elementwise ops.
  bool supportsDType(DType dt) const override {
    return dt == DType::kFloat32 || dt == DType::kInt64 || dt == DType::kInt32;
  }
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& A = ctx.t(node.inputs[0]);
    const RtTensor& B = ctx.t(node.inputs[1]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    const Shape &sa = A.shape, &sb = B.shape;
    size_t rank = std::max(sa.size(), sb.size());
    Shape out(rank, 1);
    auto dimOf = [&](const Shape& s, size_t i) -> int64_t {
      size_t off = rank - s.size();
      return i < off ? 1 : s[i - off];
    };
    for (size_t i = 0; i < rank; ++i) out[i] = std::max(dimOf(sa, i), dimOf(sb, i));
    int64_t n = numElements(out);
    std::vector<int64_t> oa(rank), ob(rank);
    int64_t sA = 1, sB = 1;
    for (int i = (int)rank - 1; i >= 0; --i) {
      oa[i] = (dimOf(sa, i) == 1) ? 0 : sA;
      ob[i] = (dimOf(sb, i) == 1) ? 0 : sB;
      sA *= dimOf(sa, i);
      sB *= dimOf(sb, i);
    }
    // Read each operand in its native dtype; compare in double so int64 magnitudes stay exact.
    auto val = [](const RtTensor& T, int64_t i) -> double {
      return T.dtype == DType::kInt64 ? (double)T.host.i64()[i] : (double)T.host.f32()[i];
    };
    float* y = cpu::allocOut(Y, out);  // canonical fp32 output (1.0 / 0.0)
    for (int64_t lin = 0; lin < n; ++lin) {
      int64_t ia = 0, ib = 0;
      for (size_t d = 0; d < rank; ++d) {
        int64_t stride = 1;
        for (size_t e = d + 1; e < rank; ++e) stride *= out[e];
        int64_t id = (lin / stride) % out[d];
        ia += id * oa[d];
        ib += id * ob[d];
      }
      y[lin] = (val(A, ia) == val(B, ib)) ? 1.0f : 0.0f;
    }
  }
};

}  // namespace
VX_REGISTER_CPU_OP(OpType::kEqual, EqualCpu);
}  // namespace vx
