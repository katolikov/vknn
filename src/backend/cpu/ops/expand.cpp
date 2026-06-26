// Expand (ONNX): broadcast X to the shape given by the int64 `shape` input (numpy broadcasting:
// the target is the broadcast of X.shape and `shape`, so output dims can exceed both).
// dtype-agnostic. out[i] = in[ sum_k (outCoord_k % inDim_k) * inStride_k ] with input right-aligned
// into the output.
#include <algorithm>

#include "backend/cpu/cpu_backend.h"
#include "import/passes.h"  // readI64Param
#include "vknn/op.h"

namespace vknn {
namespace {

struct ExpandCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    const Shape& in = X.shape;
    // target shape from inputs[1] (graph initializer or runtime int64 tensor)
    std::vector<int64_t> tgt = readI64Param(*ctx.graph, node, "shape", 1);
    if (tgt.empty() && node.inputs.size() > 1 && node.inputs[1] != kNoTensor) {
      const RtTensor& S = ctx.t(node.inputs[1]);
      tgt.assign(S.host.i64(), S.host.i64() + S.elems());
    }
    int rank = (int)std::max(in.size(), tgt.size());
    // right-align both into rank, then numpy-broadcast each dim (max of the two; 1 broadcasts).
    Shape out(rank, 1), pin(rank, 1);
    for (int k = 0; k < (int)in.size(); ++k)
      pin[rank - (int)in.size() + k] = in[k];
    for (int k = 0; k < rank; ++k) {
      int64_t t = (k >= rank - (int)tgt.size()) ? tgt[k - (rank - (int)tgt.size())] : 1;
      out[k] = std::max<int64_t>(pin[k], t < 0 ? 1 : t);
    }
    std::vector<int64_t> inStride(rank, 0), outStride(rank, 1);
    int64_t acc = 1;
    for (int k = rank - 1; k >= 0; --k) {
      inStride[k] = (pin[k] == 1) ? 0 : acc;
      acc *= pin[k];
    }
    for (int k = rank - 2; k >= 0; --k)
      outStride[k] = outStride[k + 1] * out[k + 1];
    int64_t elems = numElements(out);
    bool i64 = X.dtype == DType::kInt64;
    const float* xf = i64 ? nullptr : X.host.f32();
    const int64_t* xi = i64 ? X.host.i64() : nullptr;
    float* yf = i64 ? nullptr : cpu::allocOut(Y, out);
    int64_t* yi = i64 ? cpu::allocOutI64(Y, out) : nullptr;
    for (int64_t oi = 0; oi < elems; ++oi) {
      int64_t rem = oi, inf = 0;
      for (int k = 0; k < rank; ++k) {
        int64_t c = rem / outStride[k];
        rem %= outStride[k];
        inf += c * inStride[k];
      }
      if (i64)
        yi[oi] = xi[inf];
      else
        yf[oi] = xf[inf];
    }
  }
  bool supportsDType(DType) const override { return true; }
};

}  // namespace
VKNN_REGISTER_CPU_OP(OpType::kExpand, ExpandCpu);
}  // namespace vknn
