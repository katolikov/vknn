// Pad (constant/edge/reflect), generic N-D. pads = [begin..., end...] (2*rank) from attr or
// input[1]; constant value from attr "value" or input[2].
#include <algorithm>

#include "backends/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
namespace {
struct PadCpu : CpuOp {
  static std::vector<int64_t> readI64(const Node& n, ExecContext& ctx, const char* attr, int idx) {
    const auto& a = n.attr.getints(attr);
    if (!a.empty())
      return a;
    if (idx < (int)n.inputs.size() && n.inputs[idx] != kNoTensor) {
      const RtTensor& t = ctx.t(n.inputs[idx]);
      int64_t m = t.elems();
      return std::vector<int64_t>(t.host.i64(), t.host.i64() + m);
    }
    return {};
  }
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int rank = (int)X.shape.size();
    std::vector<int64_t> pads = readI64(node, ctx, "pads", 1);
    std::string mode = node.attr.gets("mode", "constant");
    float cval = node.attr.getf("value", 0.f);
    if (node.inputs.size() > 2 && node.inputs[2] != kNoTensor)
      cval = ctx.t(node.inputs[2]).host.f32()[0];
    Shape out = ctx.graph->desc(node.outputs[0]).shape;
    std::vector<int64_t> inStride(rank, 1), outStride(rank, 1);
    for (int i = rank - 2; i >= 0; --i) {
      inStride[i] = inStride[i + 1] * X.shape[i + 1];
      outStride[i] = outStride[i + 1] * out[i + 1];
    }
    int64_t elems = numElements(out);
    float* y = cpu::allocOut(Y, out);
    const float* x = X.host.f32();
    auto reflect = [](int64_t i, int64_t n) {
      if (n == 1)
        return (int64_t)0;
      int64_t p = 2 * (n - 1);
      i = ((i % p) + p) % p;
      return i < n ? i : p - i;
    };
    for (int64_t oi = 0; oi < elems; ++oi) {
      int64_t rem = oi, inf = 0;
      bool oob = false;
      for (int i = 0; i < rank; ++i) {
        int64_t oc = rem / outStride[i];
        rem %= outStride[i];
        int64_t ic = oc - (pads.empty() ? 0 : pads[i]);
        if (ic < 0 || ic >= X.shape[i]) {
          if (mode == "edge")
            ic = std::min<int64_t>(std::max<int64_t>(ic, 0), X.shape[i] - 1);
          else if (mode == "reflect")
            ic = reflect(ic, X.shape[i]);
          else {
            oob = true;
            break;
          }
        }
        inf += ic * inStride[i];
      }
      y[oi] = oob ? cval : x[inf];
    }
  }
};
}  // namespace
VKNN_REGISTER_CPU_OP(OpType::kPad, PadCpu);
}  // namespace vknn
