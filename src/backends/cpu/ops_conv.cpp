// vxrt — CPU reference Conv2D (general / depthwise / pointwise) and Gemm.
#include "cpu_backend.h"
#include "vx/logging.h"
#include <vector>

namespace vx {
namespace {

struct ConvCpuOp : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    const RtTensor& W = ctx.t(node.inputs[1]);
    const bool hasBias = node.inputs.size() > 2 && node.inputs[2] != kNoTensor;
    const RtTensor* B = hasBias ? &ctx.t(node.inputs[2]) : nullptr;
    RtTensor& Y = ctx.t(node.outputs[0]);

    NCHW x = NCHW::from(X.shape);
    // weight [outC, inC/group, kh, kw]
    int64_t outC = W.shape[0], inCg = W.shape[1], kh = W.shape[2], kw = W.shape[3];
    auto ints = [&](const char* k, std::vector<int64_t> d) {
      const auto& v = node.attr.getints(k); return v.empty() ? d : v;
    };
    auto strides = ints("strides", {1, 1});
    auto pads = ints("pads", {0, 0, 0, 0});
    auto dil = ints("dilations", {1, 1});
    int64_t group = node.attr.geti("group", 1);
    int64_t sh = strides[0], sw = strides[1];
    int64_t pt = pads[0], pl = pads[1];
    int64_t dh = dil[0], dw = dil[1];

    int64_t outH = (x.h + pt + pads[2] - (dh * (kh - 1) + 1)) / sh + 1;
    int64_t outW = (x.w + pl + pads[3] - (dw * (kw - 1) + 1)) / sw + 1;

    float* y = cpu::allocOut(Y, {x.n, outC, outH, outW});
    const float* xd = X.host.f32();
    const float* wd = W.host.f32();
    const float* bd = B ? B->host.f32() : nullptr;

    int64_t outCg = outC / group;     // output channels per group
    for (int64_t n = 0; n < x.n; ++n)
      for (int64_t oc = 0; oc < outC; ++oc) {
        int64_t g = oc / outCg;
        int64_t icStart = g * inCg;
        float bias = bd ? bd[oc] : 0.f;
        for (int64_t oy = 0; oy < outH; ++oy)
          for (int64_t ox = 0; ox < outW; ++ox) {
            float acc = bias;
            int64_t iy0 = oy * sh - pt;
            int64_t ix0 = ox * sw - pl;
            for (int64_t ic = 0; ic < inCg; ++ic) {
              const float* xch = xd + ((n * x.c + (icStart + ic)) * x.h) * x.w;
              const float* wch = wd + ((oc * inCg + ic) * kh) * kw;
              for (int64_t ky = 0; ky < kh; ++ky) {
                int64_t iy = iy0 + ky * dh;
                if (iy < 0 || iy >= x.h) continue;
                for (int64_t kx = 0; kx < kw; ++kx) {
                  int64_t ix = ix0 + kx * dw;
                  if (ix < 0 || ix >= x.w) continue;
                  acc += xch[iy * x.w + ix] * wch[ky * kw + kx];
                }
              }
            }
            y[((n * outC + oc) * outH + oy) * outW + ox] = acc;
          }
      }
    cpu::applyAct(y, Y.elems(), node.fusedAct, node.actLo, node.actHi);
  }
};

struct GemmCpuOp : CpuOp {
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

    // A: [M,K] (or [K,M] if transA), B: [K,N] (or [N,K] if transB)
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
        for (int64_t k = 0; k < K; ++k) {
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

VX_REGISTER_CPU_OP(OpType::kConv, ConvCpuOp);
VX_REGISTER_CPU_OP(OpType::kGemm, GemmCpuOp);

}  // namespace vx
