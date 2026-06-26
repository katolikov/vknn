// ONNX MatMul: general batched N-D matmul with NumPy broadcasting on the leading batch dims.
//   A[...,M,K] @ B[...,K,N] -> [...,M,N].  A 1-D operand [K] is promoted to [1,K] (then the
//   prepended 1 is dropped from the result); a 1-D B [K] is promoted to [K,1] (and dropped).
// CPU reference: a straight triple loop over the broadcasted batch, then M,N,K. Host buffers are
// canonical NCHW fp32, so this is the oracle the host tests validate against.
#include <algorithm>
#include <vector>

#include "backends/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
namespace {

struct MatMulCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& A = ctx.t(node.inputs[0]);
    const RtTensor& B = ctx.t(node.inputs[1]);
    RtTensor& Y = ctx.t(node.outputs[0]);

    // Promote 1-D operands: A[K] -> [1,K]; B[K] -> [K,1]. Track whether we prepended/appended a
    // dim so it can be stripped from the output (NumPy matmul semantics).
    Shape sa = A.shape, sb = B.shape;
    bool aWas1D = sa.size() == 1, bWas1D = sb.size() == 1;
    if (aWas1D)
      sa = {1, sa[0]};
    if (bWas1D)
      sb = {sb[0], 1};

    int64_t M = sa[sa.size() - 2], K = sa[sa.size() - 1];
    int64_t Kb = sb[sb.size() - 2], N = sb[sb.size() - 1];
    (void)Kb;  // K == Kb by construction of a valid graph

    // Broadcast the batch dims (everything before the trailing 2) to a common shape.
    int64_t aBatchRank = (int64_t)sa.size() - 2, bBatchRank = (int64_t)sb.size() - 2;
    int64_t batchRank = std::max(aBatchRank, bBatchRank);
    Shape batch(batchRank, 1);
    auto aDim = [&](int64_t i) -> int64_t {  // i in [0,batchRank)
      int64_t off = batchRank - aBatchRank;
      return i < off ? 1 : sa[i - off];
    };
    auto bDim = [&](int64_t i) -> int64_t {
      int64_t off = batchRank - bBatchRank;
      return i < off ? 1 : sb[i - off];
    };
    int64_t batchElems = 1;
    for (int64_t i = 0; i < batchRank; ++i) {
      batch[i] = std::max(aDim(i), bDim(i));
      batchElems *= batch[i];
    }

    // Per-batch-dim element strides into A's and B's matrix stacks (0 on a broadcast dim).
    std::vector<int64_t> aBatchStride(batchRank, 0), bBatchStride(batchRank, 0);
    int64_t sA = M * K, sB = K * N;
    for (int64_t i = batchRank - 1; i >= 0; --i) {
      aBatchStride[i] = (aDim(i) == 1) ? 0 : sA;
      bBatchStride[i] = (bDim(i) == 1) ? 0 : sB;
      sA *= aDim(i);
      sB *= bDim(i);
    }

    // Output shape = batch ++ [M,N], with the promoted dims stripped back out.
    Shape out = batch;
    if (!aWas1D)
      out.push_back(M);
    out.push_back(N);
    if (bWas1D)
      out.pop_back();
    if (out.empty())
      out.push_back(1);  // scalar dot product -> [1]

    float* y = cpu::allocOut(Y, out);
    const float* a = A.host.f32();
    const float* b = B.host.f32();

    for (int64_t bi = 0; bi < batchElems; ++bi) {
      // Decode the batch index into per-dim coords to find the A/B base offsets.
      int64_t aBase = 0, bBase = 0, rem = bi;
      for (int64_t i = batchRank - 1; i >= 0; --i) {
        int64_t c = rem % batch[i];
        rem /= batch[i];
        aBase += c * aBatchStride[i];
        bBase += c * bBatchStride[i];
      }
      const float* am = a + aBase;
      const float* bm = b + bBase;
      float* ym = y + bi * M * N;
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
          float acc = 0.f;
          for (int64_t k = 0; k < K; ++k)
            acc += am[m * K + k] * bm[k * N + n];
          ym[m * N + n] = acc;
        }
      }
    }
  }
};

}  // namespace
VKNN_REGISTER_CPU_OP(OpType::kMatMul, MatMulCpu);
}  // namespace vknn
