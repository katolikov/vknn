// ONNX Where (cond ? X : Y) with full NumPy-style broadcasting over all three inputs. cond is bool/
// uint8 in ONNX but arrives here as fp32 (or int64); it is treated as "true" iff != 0. X and Y are
// the value operands. Output dtype follows X/Y: the dynamic-shape subgraph runs Where on INT64
// shape vectors (e.g. Where(Equal(dim,-1), input_shape, target)), so reading/writing those as fp32
// would reinterpret the int bytes as garbage — the value operands are read in their native dtype.
#include <algorithm>

#include "backends/cpu/cpu_backend.h"
#include "vx/op.h"

namespace vx {
namespace {

struct WhereCpu : CpuOp {
  bool supportsDType(DType dt) const override {
    return dt == DType::kFloat32 || dt == DType::kInt64 || dt == DType::kInt32;
  }
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& C = ctx.t(node.inputs[0]);
    const RtTensor& X = ctx.t(node.inputs[1]);
    const RtTensor& Yv = ctx.t(node.inputs[2]);
    RtTensor& Out = ctx.t(node.outputs[0]);
    const Shape &sc = C.shape, &sx = X.shape, &sy = Yv.shape;
    size_t rank = std::max(sc.size(), std::max(sx.size(), sy.size()));
    Shape out(rank, 1);
    auto dimOf = [&](const Shape& s, size_t i) -> int64_t {
      size_t off = rank - s.size();
      return i < off ? 1 : s[i - off];
    };
    for (size_t i = 0; i < rank; ++i)
      out[i] = std::max(dimOf(sc, i), std::max(dimOf(sx, i), dimOf(sy, i)));
    int64_t n = numElements(out);
    std::vector<int64_t> oc(rank), ox(rank), oy(rank);
    int64_t sC = 1, sX = 1, sY = 1;
    for (int i = (int)rank - 1; i >= 0; --i) {
      oc[i] = (dimOf(sc, i) == 1) ? 0 : sC;
      ox[i] = (dimOf(sx, i) == 1) ? 0 : sX;
      oy[i] = (dimOf(sy, i) == 1) ? 0 : sY;
      sC *= dimOf(sc, i);
      sX *= dimOf(sx, i);
      sY *= dimOf(sy, i);
    }
    // cond read through its native dtype (bool/uint8 imported as fp32; int64 shape masks possible).
    auto condTrue = [](const RtTensor& T, int64_t i) -> bool {
      return T.dtype == DType::kInt64 ? T.host.i64()[i] != 0 : T.host.f32()[i] != 0.0f;
    };
    // broadcast-index helper: maps a linear output index to (cond, X, Y) source offsets.
    auto offsets = [&](int64_t lin, int64_t& ic, int64_t& ix, int64_t& iy) {
      ic = ix = iy = 0;
      for (size_t d = 0; d < rank; ++d) {
        int64_t stride = 1;
        for (size_t e = d + 1; e < rank; ++e)
          stride *= out[e];
        int64_t id = (lin / stride) % out[d];
        ic += id * oc[d];
        ix += id * ox[d];
        iy += id * oy[d];
      }
    };
    // Output type follows the value operands (int64 for the shape-arithmetic Where).
    bool i64 = X.dtype == DType::kInt64 && Yv.dtype == DType::kInt64;
    if (i64) {
      int64_t* o = cpu::allocOutI64(Out, out);
      const int64_t* x = X.host.i64();
      const int64_t* y = Yv.host.i64();
      for (int64_t lin = 0; lin < n; ++lin) {
        int64_t ic, ix, iy;
        offsets(lin, ic, ix, iy);
        o[lin] = condTrue(C, ic) ? x[ix] : y[iy];
      }
    } else {
      float* o = cpu::allocOut(Out, out);
      const float* x = X.host.f32();
      const float* y = Yv.host.f32();
      for (int64_t lin = 0; lin < n; ++lin) {
        int64_t ic, ix, iy;
        offsets(lin, ic, ix, iy);
        o[lin] = condTrue(C, ic) ? x[ix] : y[iy];
      }
    }
  }
};

}  // namespace
VX_REGISTER_CPU_OP(OpType::kWhere, WhereCpu);
}  // namespace vx
