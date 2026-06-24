// Gemm: Y = alpha*op(A)*op(B) + beta*C. The inner contraction gets a NEON path for the common
// classifier shape (transB=1, A not transposed); everything else uses the scalar fallback.
#include "backends/cpu/cpu_backend.h"
#if defined(VXRT_ENABLE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#define VX_HAS_NEON 1
#endif

namespace vx {
namespace {

struct GemmCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& A = ctx.t(node.inputs[0]);
    const RtTensor& Bt = ctx.t(node.inputs[1]);
    const bool hasC = node.inputs.size() > 2 && node.inputs[2] != kNoTensor;
    const RtTensor* C = hasC ? &ctx.t(node.inputs[2]) : nullptr;
    RtTensor& Y = ctx.t(node.outputs[0]);

    float alpha = node.attr.getf("alpha", 1.f);
    float beta = node.attr.getf("beta", 1.f);
    int64_t transA = node.attr.geti("transA", 0);
    int64_t transB = node.attr.geti("transB", 0);

    int64_t M = transA ? A.shape[1] : A.shape[0];
    int64_t K = transA ? A.shape[0] : A.shape[1];
    int64_t N = transB ? Bt.shape[0] : Bt.shape[1];

    float* y = cpu::allocOut(Y, {M, N});
    const float* a = A.host.f32();
    const float* b = Bt.host.f32();
    const float* c = C ? C->host.f32() : nullptr;
    int64_t cN = C ? (C->shape.size() == 1 ? C->shape[0] : numElements(C->shape)) : 0;

    for (int64_t m = 0; m < M; ++m)
      for (int64_t n = 0; n < N; ++n) {
        float acc = 0;
        int64_t k = 0;
#if defined(VX_HAS_NEON)
        if (transB && !transA) {
          const float* arow = a + m * K;
          const float* brow = b + n * K;
          float32x4_t v = vdupq_n_f32(0.f);
          for (; k + 4 <= K; k += 4) v = vmlaq_f32(v, vld1q_f32(arow + k), vld1q_f32(brow + k));
          acc = vaddvq_f32(v);
        }
#endif
        for (; k < K; ++k) {
          float av = transA ? a[k * M + m] : a[m * K + k];
          float bv = transB ? b[n * K + k] : b[k * N + n];
          acc += av * bv;
        }
        acc *= alpha;
        if (c) acc += beta * (cN == N ? c[n] : c[(m * N + n) % cN]);
        y[m * N + n] = acc;
      }
    cpu::applyAct(y, Y.elems(), node.fusedAct, node.actLo, node.actHi);
  }
};

}  // namespace

VX_REGISTER_CPU_OP(OpType::kGemm, GemmCpu);

}  // namespace vx
