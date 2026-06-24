// Reduce family (ReduceMean/Sum/Max/Min/Prod), generic N-D, NCHW fp32. axes from attr or input[1].
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include "backends/cpu/cpu_backend.h"
#include "vx/op.h"

namespace vx {
namespace {
struct ReduceCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int rank = (int)X.shape.size();
    std::vector<int64_t> axes = node.attr.getints("axes");
    if (axes.empty() && node.inputs.size() > 1 && node.inputs[1] != kNoTensor) {
      const RtTensor& A = ctx.t(node.inputs[1]);
      axes.assign(A.host.i64(), A.host.i64() + A.elems());
    }
    if (axes.empty()) for (int i = 0; i < rank; ++i) axes.push_back(i);
    std::set<int> ax;
    for (int64_t a : axes) ax.insert((int)(a < 0 ? a + rank : a));
    int op = node.subOp;  // ReduceType
    Shape out = ctx.graph->desc(node.outputs[0]).shape;
    int64_t outElems = numElements(out), n = X.elems();
    std::vector<int64_t> inStride(rank, 1);
    for (int i = rank - 2; i >= 0; --i) inStride[i] = inStride[i + 1] * X.shape[i + 1];
    // output strides over the NON-reduced axes (in input order)
    std::vector<int64_t> kept;
    for (int i = 0; i < rank; ++i) if (!ax.count(i)) kept.push_back(i);
    std::vector<int64_t> outStrideK(kept.size(), 1);
    for (int i = (int)kept.size() - 2; i >= 0; --i) outStrideK[i] = outStrideK[i + 1] * X.shape[kept[i + 1]];
    float init = op == kRMax ? -std::numeric_limits<float>::infinity()
               : op == kRMin ? std::numeric_limits<float>::infinity()
               : op == kRProd ? 1.f : 0.f;
    std::vector<float> acc(outElems, init);
    std::vector<int64_t> cnt(outElems, 0);
    const float* x = X.host.f32();
    for (int64_t i = 0; i < n; ++i) {
      int64_t rem = i, oi = 0;
      for (size_t k = 0; k < kept.size(); ++k) { int64_t c = (i / inStride[kept[k]]) % X.shape[kept[k]]; oi += c * outStrideK[k]; }
      float v = x[i];
      if (op == kRMax) acc[oi] = std::max(acc[oi], v);
      else if (op == kRMin) acc[oi] = std::min(acc[oi], v);
      else if (op == kRProd) acc[oi] *= v;
      else acc[oi] += v;
      cnt[oi]++;
      (void)rem;
    }
    float* y = cpu::allocOut(Y, out);
    for (int64_t i = 0; i < outElems; ++i) y[i] = (op == kRMean && cnt[i]) ? acc[i] / cnt[i] : acc[i];
  }
};
}  // namespace
VX_REGISTER_CPU_OP(OpType::kReduce, ReduceCpu);
}  // namespace vx
