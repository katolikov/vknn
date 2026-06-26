// Elementwise binary family (Mul/Sub/Div/Max/Min/Pow) with NumPy-style broadcasting.
#include <algorithm>
#include <cmath>

#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
namespace {

static float binary(float a, float b, BinaryType op) {
  switch (op) {
    case BinaryType::kMul:
      return a * b;
    case BinaryType::kSub:
      return a - b;
    case BinaryType::kDiv:
      return a / b;
    case BinaryType::kMax:
      return std::max(a, b);
    case BinaryType::kMin:
      return std::min(a, b);
    case BinaryType::kPow:
      return std::pow(a, b);
    default:
      break;
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
    for (size_t i = 0; i < rank; ++i)
      out[i] = std::max(dimOf(sa, i), dimOf(sb, i));
    int64_t n = numElements(out);
    auto strides = [&](std::vector<int64_t>& oa, std::vector<int64_t>& ob) {
      int64_t sA = 1, sB = 1;
      for (int i = (int)rank - 1; i >= 0; --i) {
        oa[i] = (dimOf(sa, i) == 1) ? 0 : sA;
        ob[i] = (dimOf(sb, i) == 1) ? 0 : sB;
        sA *= dimOf(sa, i);
        sB *= dimOf(sb, i);
      }
    };
    auto idx = [&](const std::vector<int64_t>& oa, const std::vector<int64_t>& ob, int64_t lin,
                   int64_t& ia, int64_t& ib) {
      ia = ib = 0;
      for (size_t d = 0; d < rank; ++d) {
        int64_t stride = 1;
        for (size_t e = d + 1; e < rank; ++e)
          stride *= out[e];
        int64_t id = (lin / stride) % out[d];
        ia += id * oa[d];
        ib += id * ob[d];
      }
    };
    // Shape arithmetic is int64 (Shape/Gather feed Add/Div/Mul to compute slice/reshape bounds).
    // Compute it in int64 so const-folding stays exact — reading those bytes as float corrupts
    // them.
    if (A.dtype == DType::kInt64 || B.dtype == DType::kInt64) {
      int64_t* y = cpu::allocOutI64(Y, out);
      std::vector<int64_t> oa(rank), ob(rank);
      strides(oa, ob);
      auto val = [](const RtTensor& T, int64_t i) {
        return T.dtype == DType::kInt64 ? T.host.i64()[i] : (int64_t)T.host.f32()[i];
      };
      for (int64_t lin = 0; lin < n; ++lin) {
        int64_t ia, ib;
        idx(oa, ob, lin, ia, ib);
        int64_t av = val(A, ia), bv = val(B, ib);
        switch ((BinaryType)node.subOp) {
          case BinaryType::kMul:
            y[lin] = av * bv;
            break;
          case BinaryType::kSub:
            y[lin] = av - bv;
            break;
          case BinaryType::kDiv:
            y[lin] = bv ? av / bv : 0;
            break;
          case BinaryType::kMax:
            y[lin] = std::max(av, bv);
            break;
          case BinaryType::kMin:
            y[lin] = std::min(av, bv);
            break;
          default:
            y[lin] = av + bv;
            break;
        }
      }
      return;
    }
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
        for (size_t e = d + 1; e < rank; ++e)
          stride *= out[e];
        int64_t id = (lin / stride) % out[d];
        ia += id * oa[d];
        ib += id * ob[d];
      }
      y[lin] = binary(a[ia], b[ib], (BinaryType)node.subOp);
    }
  }
};

}  // namespace
VKNN_REGISTER_CPU_OP(OpType::kBinary, BinaryCpu);
}  // namespace vknn
