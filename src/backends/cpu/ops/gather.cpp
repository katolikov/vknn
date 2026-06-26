// ONNX Gather along an arbitrary axis: out = data[:axis] + indices.shape + data[axis+1:]. A scalar
// index (rank-0, stored here as [1] with one element) removes the gathered axis. Used both for the
// classifier-preamble shape math (axis-0 vectors) AND the transformer attention Q/K/V split, which
// gathers a single index along axis 2 of the permuted qkv tensor — so honoring `axis` is required
// for correctness (an axis-0-only gather silently corrupts every non-axis-0 Gather + its const-fold).
#include "backends/cpu/cpu_backend.h"
#include <algorithm>

namespace vx {
namespace {

struct GatherCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& D = ctx.t(node.inputs[0]);
    const RtTensor& I = ctx.t(node.inputs[1]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int64_t rank = (int64_t)D.shape.size();
    int64_t axis = node.attr.geti("axis", 0);
    if (axis < 0) axis += rank;
    axis = std::max<int64_t>(0, std::min<int64_t>(axis, rank > 0 ? rank - 1 : 0));

    int64_t nidx = I.elems();
    // Index dtype varies: const int64 (attention Q/K/V) or a runtime float activation (RoPE).
    auto indexAt = [&](int64_t k) -> int64_t {
      return I.dtype == DType::kInt64 ? I.host.i64()[k] : (int64_t)I.host.f32()[k];
    };
    int64_t axisSize = (rank > 0) ? D.shape[axis] : 1;
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= D.shape[i];
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) inner *= D.shape[i];

    // Scalar index (rank-0, or the importer's [1]-of-1 form) removes the axis; otherwise the
    // indices' own shape is spliced in at `axis`.
    bool scalarIndex = I.shape.empty() || (I.shape.size() == 1 && I.shape[0] == 1 && nidx == 1);
    Shape outShape;
    for (int64_t i = 0; i < axis; ++i) outShape.push_back(D.shape[i]);
    if (!scalarIndex)
      for (int64_t v : I.shape) outShape.push_back(v);
    for (int64_t i = axis + 1; i < rank; ++i) outShape.push_back(D.shape[i]);
    if (outShape.empty()) outShape = {1};

    auto copy = [&](auto* y, const auto* d) {
      for (int64_t o = 0; o < outer; ++o)
        for (int64_t k = 0; k < nidx; ++k) {
          int64_t ik = indexAt(k);
          int64_t src = ik < 0 ? ik + axisSize : ik;
          const auto* sp = d + (o * axisSize + src) * inner;
          auto* dp = y + (o * nidx + k) * inner;
          for (int64_t j = 0; j < inner; ++j) dp[j] = sp[j];
        }
    };
    if (D.dtype == DType::kInt64) {
      int64_t* y = cpu::allocOutI64(Y, outShape);
      copy(y, D.host.i64());
    } else {
      float* y = cpu::allocOut(Y, outShape);
      copy(y, D.host.f32());
    }
  }
  bool supportsDType(DType) const override { return true; }
};

}  // namespace
VX_REGISTER_CPU_OP(OpType::kGather, GatherCpu);
}  // namespace vx
