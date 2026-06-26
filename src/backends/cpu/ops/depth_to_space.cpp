// DepthToSpace: [N,C,H,W] -> [N, C/(b*b), H*b, W*b]. Rearranges blocks of depth into spatial.
// modes (ONNX): DCR (depth-column-row, default) and CRD (column-row-depth). NCHW fp32 reference.
//   out[n, c, h, w], with ih=h/b, bh=h%b, iw=w/b, bw=w%b, C2 = C/(b*b):
//     DCR: in channel = (bh*b + bw) * C2 + c
//     CRD: in channel = c * (b*b) + (bh*b + bw)
#include "backends/cpu/cpu_backend.h"
#include "vx/op.h"

namespace vx {
namespace {
struct DepthToSpaceCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    NCHW x = NCHW::from(X.shape);
    int64_t b = node.attr.geti("blocksize", 1);
    if (b < 1) b = 1;
    bool crd = node.attr.gets("mode", "DCR") == "CRD";
    int64_t C2 = x.c / (b * b), OH = x.h * b, OW = x.w * b;
    Shape outShape = {x.n, C2, OH, OW};
    float* y = cpu::allocOut(Y, outShape);
    const float* in = X.host.f32();
    int64_t inHW = x.h * x.w;
    for (int64_t n = 0; n < x.n; ++n) {
      for (int64_t c = 0; c < C2; ++c) {
        for (int64_t h = 0; h < OH; ++h) {
          int64_t ih = h / b, bh = h % b;
          for (int64_t w = 0; w < OW; ++w) {
            int64_t iw = w / b, bw = w % b;
            int64_t blk = bh * b + bw;
            int64_t ic = crd ? (c * (b * b) + blk) : (blk * C2 + c);
            int64_t inIdx = ((n * x.c + ic) * x.h + ih) * x.w + iw;
            int64_t outIdx = ((n * C2 + c) * OH + h) * OW + w;
            y[outIdx] = in[inIdx];
            (void)inHW;
          }
        }
      }
    }
  }
};
}  // namespace
VX_REGISTER_CPU_OP(OpType::kDepthToSpace, DepthToSpaceCpu);
}  // namespace vx
