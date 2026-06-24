// Elementwise binary family (Mul/Sub/Div/Max/Min/Pow) with NumPy-style broadcasting.
#include <algorithm>
#include <cmath>
#include "backends/cpu/cpu_backend.h"
#include "vx/op.h"

namespace vx {
namespace {

static float binary(float a, float b, int op) {
  switch (op) {
    case kBMul: return a * b;
    case kBSub: return a - b;
    case kBDiv: return a / b;
    case kBMax: return std::max(a, b);
    case kBMin: return std::min(a, b);
    case kBPow: return std::pow(a, b);
  }
  return a + b;
}

struct BinaryCpu : CpuOp {
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
    float* y = cpu::allocOut(Y, out);
    const float* a = A.host.f32();
    const float* b = B.host.f32();
    std::vector<int64_t> oa(rank), ob(rank);
    int64_t sA = 1, sB = 1;
    for (int i = (int)rank - 1; i >= 0; --i) {
      oa[i] = (dimOf(sa, i) == 1) ? 0 : sA;
      ob[i] = (dimOf(sb, i) == 1) ? 0 : sB;
      sA *= dimOf(sa, i);
      sB *= dimOf(sb, i);
    }
    for (int64_t lin = 0; lin < n; ++lin) {
      int64_t ia = 0, ib = 0;
      for (size_t d = 0; d < rank; ++d) {
        int64_t stride = 1;
        for (size_t e = d + 1; e < rank; ++e) stride *= out[e];
        int64_t id = (lin / stride) % out[d];
        ia += id * oa[d];
        ib += id * ob[d];
      }
      y[lin] = binary(a[ia], b[ib], node.subOp);
    }
  }
};

}  // namespace
VX_REGISTER_CPU_OP(OpType::kBinary, BinaryCpu);
}  // namespace vx
