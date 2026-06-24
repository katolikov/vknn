// Elementwise add with NumPy-style broadcasting. Equal-shape inputs (the residual connections)
// take a NEON fast path; broadcasting falls back to the general index walk.
#include "backends/cpu/cpu_backend.h"
#include <algorithm>
#include "vx/logging.h"
#if defined(VXRT_ENABLE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#define VX_HAS_NEON 1
#endif

namespace vx {
namespace {

struct AddCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& A = ctx.t(node.inputs[0]);
    const RtTensor& B = ctx.t(node.inputs[1]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    const Shape &sa = A.shape, &sb = B.shape;

    if (sa == sb) {  // residual add: same shape, vectorizable
      int64_t n = A.elems();
      float* y = cpu::allocOut(Y, sa);
      const float* a = A.host.f32();
      const float* b = B.host.f32();
      int64_t i = 0;
#if defined(VX_HAS_NEON)
      for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
#endif
      for (; i < n; ++i) y[i] = a[i] + b[i];
      return;
    }

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
      y[lin] = a[ia] + b[ib];
    }
  }
};

}  // namespace
VX_REGISTER_CPU_OP(OpType::kAdd, AddCpu);
}  // namespace vx
