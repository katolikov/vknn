// vxrt — CPU reference: Clip, Relu, Add, pooling, Softmax, BatchNorm, Identity.
#include "cpu_backend.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include "vx/logging.h"
#if defined(VXRT_ENABLE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#define VX_HAS_NEON 1
#endif

namespace vx {
namespace {

// Clip (opset 11/12): min/max are inputs[1], inputs[2] (optional). Also handles attrs.
struct ClipCpuOp : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    float lo = -std::numeric_limits<float>::infinity();
    float hi = std::numeric_limits<float>::infinity();
    if (node.inputs.size() > 1 && node.inputs[1] != kNoTensor) lo = ctx.t(node.inputs[1]).host.f32()[0];
    if (node.inputs.size() > 2 && node.inputs[2] != kNoTensor) hi = ctx.t(node.inputs[2]).host.f32()[0];
    if (node.attr.has("min")) lo = node.attr.getf("min", lo);
    if (node.attr.has("max")) hi = node.attr.getf("max", hi);
    int64_t n = X.elems();
    float* y = cpu::allocOut(Y, X.shape);
    const float* x = X.host.f32();
    for (int64_t i = 0; i < n; ++i) { float v = x[i]; y[i] = v < lo ? lo : (v > hi ? hi : v); }
  }
};

struct ReluCpuOp : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int64_t n = X.elems();
    float* y = cpu::allocOut(Y, X.shape);
    const float* x = X.host.f32();
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] > 0 ? x[i] : 0;
  }
};

// Elementwise add with NumPy-style broadcasting (sufficient for residuals + bias-add).
struct AddCpuOp : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& A = ctx.t(node.inputs[0]);
    const RtTensor& B = ctx.t(node.inputs[1]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    const Shape& sa = A.shape, & sb = B.shape;
    // Fast path: identical shapes (residual adds) -> NEON 4-wide.
    if (sa == sb) {
      int64_t n = A.elems();
      float* y = cpu::allocOut(Y, sa);
      const float* a = A.host.f32();
      const float* b = B.host.f32();
      int64_t i = 0;
#if defined(VX_HAS_NEON)
      VX_LOG(kDebug) << "NEON Add kernel (" << n << " elems)";
      for (; i + 4 <= n; i += 4)
        vst1q_f32(y + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
#endif
      for (; i < n; ++i) y[i] = a[i] + b[i];
      return;
    }
    // result shape = broadcast
    size_t rank = std::max(sa.size(), sb.size());
    Shape out(rank, 1);
    auto dimOf = [&](const Shape& s, size_t i) -> int64_t {
      size_t off = rank - s.size();
      return i < off ? 1 : s[i - off];
    };
    for (size_t i = 0; i < rank; ++i) out[i] = std::max(dimOf(sa, i), dimOf(sb, i));
    int64_t n = numElements(out);
    float* y = cpu::allocOut(Y, out);
    const float* a = A.host.f32();
    const float* b = B.host.f32();
    // strides
    std::vector<int64_t> oa(rank), ob(rank), os(rank);
    int64_t sA = 1, sB = 1;
    for (int i = (int)rank - 1; i >= 0; --i) {
      oa[i] = (dimOf(sa, i) == 1) ? 0 : sA;
      ob[i] = (dimOf(sb, i) == 1) ? 0 : sB;
      sA *= dimOf(sa, i);
      sB *= dimOf(sb, i);
    }
    std::vector<int64_t> idx(rank, 0);
    for (int64_t lin = 0; lin < n; ++lin) {
      int64_t ia = 0, ib = 0, rem = lin;
      for (size_t d = 0; d < rank; ++d) {
        int64_t stride = 1;
        for (size_t e = d + 1; e < rank; ++e) stride *= out[e];
        int64_t id = (rem / stride) % out[d];
        ia += id * oa[d];
        ib += id * ob[d];
      }
      y[lin] = a[ia] + b[ib];
    }
  }
};

struct GlobalAvgPoolCpuOp : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    NCHW x = NCHW::from(X.shape);
    float* y = cpu::allocOut(Y, {x.n, x.c, 1, 1});
    const float* xd = X.host.f32();
    int64_t hw = x.h * x.w;
    for (int64_t n = 0; n < x.n; ++n)
      for (int64_t c = 0; c < x.c; ++c) {
        const float* p = xd + (n * x.c + c) * hw;
        double s = 0;
        for (int64_t i = 0; i < hw; ++i) s += p[i];
        y[n * x.c + c] = (float)(s / hw);
      }
  }
};

struct SoftmaxCpuOp : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int64_t axis = node.attr.geti("axis", -1);
    int64_t rank = (int64_t)X.shape.size();
    if (axis < 0) axis += rank;
    int64_t inner = 1;
    for (int64_t i = axis; i < rank; ++i) inner *= X.shape[i];
    int64_t outer = X.elems() / inner;
    float* y = cpu::allocOut(Y, X.shape);
    const float* x = X.host.f32();
    for (int64_t o = 0; o < outer; ++o) {
      const float* xr = x + o * inner;
      float* yr = y + o * inner;
      float mx = xr[0];
      for (int64_t i = 1; i < inner; ++i) mx = std::max(mx, xr[i]);
      double sum = 0;
      for (int64_t i = 0; i < inner; ++i) { yr[i] = std::exp(xr[i] - mx); sum += yr[i]; }
      for (int64_t i = 0; i < inner; ++i) yr[i] = (float)(yr[i] / sum);
    }
  }
};

// BatchNormalization (inference): y = (x-mean)/sqrt(var+eps)*scale + bias.
// Used to validate the BN->Conv fold pass; MobileNetV2 has BN pre-folded.
struct BatchNormCpuOp : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    const float* scale = ctx.t(node.inputs[1]).host.f32();
    const float* bias = ctx.t(node.inputs[2]).host.f32();
    const float* mean = ctx.t(node.inputs[3]).host.f32();
    const float* var = ctx.t(node.inputs[4]).host.f32();
    float eps = node.attr.getf("epsilon", 1e-5f);
    RtTensor& Y = ctx.t(node.outputs[0]);
    NCHW x = NCHW::from(X.shape);
    float* y = cpu::allocOut(Y, X.shape);
    const float* xd = X.host.f32();
    int64_t hw = x.h * x.w;
    for (int64_t n = 0; n < x.n; ++n)
      for (int64_t c = 0; c < x.c; ++c) {
        float a = scale[c] / std::sqrt(var[c] + eps);
        float b = bias[c] - mean[c] * a;
        const float* p = xd + (n * x.c + c) * hw;
        float* q = y + (n * x.c + c) * hw;
        for (int64_t i = 0; i < hw; ++i) q[i] = p[i] * a + b;
      }
  }
};

struct IdentityCpuOp : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    Y.shape = X.shape; Y.dtype = X.dtype;
    Y.host = X.host; Y.hostValid = true; Y.deviceValid = false;
  }
};

}  // namespace

VX_REGISTER_CPU_OP(OpType::kClip, ClipCpuOp);
VX_REGISTER_CPU_OP(OpType::kRelu, ReluCpuOp);
VX_REGISTER_CPU_OP(OpType::kAdd, AddCpuOp);
VX_REGISTER_CPU_OP(OpType::kGlobalAvgPool, GlobalAvgPoolCpuOp);
VX_REGISTER_CPU_OP(OpType::kSoftmax, SoftmaxCpuOp);
VX_REGISTER_CPU_OP(OpType::kBatchNorm, BatchNormCpuOp);
VX_REGISTER_CPU_OP(OpType::kIdentity, IdentityCpuOp);

}  // namespace vx
