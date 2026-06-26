// Einsum CPU reference for the three equations the YoNoSplat encoder uses:
//   "i,j->ij"            outer product (RoPE freq table = positions (x) inv_freq)   [102x]
//   "...ab,...b->...a"   batched mat-vec (intrinsics_inv @ ray coords)              [2x]
//   "bij,bnjk->bnik"     batched matmul w/ broadcast on n (SE3 pose transform)      [1x]
// A general N-operand einsum isn't needed; these three cover the model.
#include <string>

#include "backends/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
namespace {

static std::string stripw(const std::string& s) {
  std::string r;
  for (char c : s)
    if (c != ' ' && c != '\t')
      r += c;
  return r;
}

struct EinsumCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    std::string eq = stripw(node.attr.gets("equation", ""));
    const RtTensor& A = ctx.t(node.inputs[0]);
    const RtTensor& B = ctx.t(node.inputs[1]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    const float* a = A.host.f32();
    const float* b = B.host.f32();

    if (eq == "i,j->ij") {
      int64_t I = A.elems(), J = B.elems();
      float* y = cpu::allocOut(Y, {I, J});
      for (int64_t i = 0; i < I; ++i)
        for (int64_t j = 0; j < J; ++j)
          y[i * J + j] = a[i] * b[j];
      return;
    }
    if (eq == "...ab,...b->...a") {
      const Shape& as = A.shape;
      int ar = (int)as.size();
      int64_t aN = as[ar - 2], bN = as[ar - 1];
      int64_t batch = 1;
      for (int k = 0; k < ar - 2; ++k)
        batch *= as[k];
      int64_t bBatch = 1;
      for (int k = 0; k + 1 < (int)B.shape.size(); ++k)
        bBatch *= B.shape[k];
      Shape out(as.begin(), as.end() - 1);  // [..., a]
      float* y = cpu::allocOut(Y, out);
      for (int64_t bi = 0; bi < batch; ++bi) {
        const float* Ap = a + bi * aN * bN;
        const float* Bp = b + (bBatch == 1 ? 0 : bi % bBatch) * bN;
        float* Yp = y + bi * aN;
        for (int64_t ii = 0; ii < aN; ++ii) {
          float s = 0;
          for (int64_t jj = 0; jj < bN; ++jj)
            s += Ap[ii * bN + jj] * Bp[jj];
          Yp[ii] = s;
        }
      }
      return;
    }
    if (eq == "bij,bnjk->bnik") {
      const Shape& as = A.shape;
      const Shape& bs = B.shape;
      int64_t Bb = as[0], I = as[1], J = as[2], N = bs[1], K = bs[3];
      float* y = cpu::allocOut(Y, {Bb, N, I, K});
      for (int64_t bb = 0; bb < Bb; ++bb)
        for (int64_t n = 0; n < N; ++n)
          for (int64_t i = 0; i < I; ++i)
            for (int64_t k = 0; k < K; ++k) {
              float s = 0;
              for (int64_t j = 0; j < J; ++j)
                s += a[(bb * I + i) * J + j] * b[((bb * N + n) * J + j) * K + k];
              y[((bb * N + n) * I + i) * K + k] = s;
            }
      return;
    }
    // Unhandled equation: pass input through (keeps the graph runnable; not hit by this model).
    int64_t n = A.elems();
    float* y = cpu::allocOut(Y, A.shape);
    for (int64_t i = 0; i < n; ++i)
      y[i] = a[i];
  }
};

}  // namespace
VKNN_REGISTER_CPU_OP(OpType::kEinsum, EinsumCpu);
}  // namespace vknn
