// vxrt — CPU reference shape/tensor-manipulation ops: Reshape, Flatten, Shape,
// Constant, Gather, Unsqueeze, Concat. These let the MobileNetV2 classifier preamble
// (Shape->Gather->Unsqueeze->Concat->Reshape) run directly; the Vulkan path constant-folds it.
#include "cpu_backend.h"
#include <algorithm>
#include <cstring>

namespace vx {
namespace {

// copy raw bytes, preserving dtype, into Y with the given shape
static void copyAs(const RtTensor& X, RtTensor& Y, const Shape& shape) {
  Y.shape = shape;
  Y.dtype = X.dtype;
  Y.host.resizeElems(numElements(shape), X.dtype);
  Y.hostValid = true;
  Y.deviceValid = false;
  std::memcpy(Y.host.bytes.data(), X.host.bytes.data(),
              std::min(Y.host.bytes.size(), X.host.bytes.size()));
}

struct ReshapeCpuOp : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    const RtTensor& S = ctx.t(node.inputs[1]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int64_t rank = S.elems();
    const int64_t* sd = S.host.i64();
    Shape out(rank);
    int64_t known = 1, inferIdx = -1;
    for (int64_t i = 0; i < rank; ++i) {
      int64_t d = sd[i];
      if (d == 0) d = (i < (int64_t)X.shape.size()) ? X.shape[i] : 1;  // copy
      out[i] = d;
      if (d == -1) inferIdx = i; else known *= d;
    }
    if (inferIdx >= 0) out[inferIdx] = X.elems() / std::max<int64_t>(known, 1);
    copyAs(X, Y, out);
  }
  bool supportsDType(DType) const override { return true; }
};

struct FlattenCpuOp : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int64_t axis = node.attr.geti("axis", 1);
    int64_t rank = (int64_t)X.shape.size();
    if (axis < 0) axis += rank;
    int64_t d0 = 1, d1 = 1;
    for (int64_t i = 0; i < rank; ++i) (i < axis ? d0 : d1) *= X.shape[i];
    copyAs(X, Y, {d0, d1});
  }
  bool supportsDType(DType) const override { return true; }
};

struct ShapeCpuOp : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int64_t r = (int64_t)X.shape.size();
    int64_t* y = cpu::allocOutI64(Y, {r});
    for (int64_t i = 0; i < r; ++i) y[i] = X.shape[i];
  }
};

struct ConstantCpuOp : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    RtTensor& Y = ctx.t(node.outputs[0]);
    auto it = node.attr.map.find("value");
    if (it != node.attr.map.end() && it->second.kind == Attr::kInts) {
      const auto& v = it->second.ints;
      int64_t* y = cpu::allocOutI64(Y, {(int64_t)v.size()});
      for (size_t i = 0; i < v.size(); ++i) y[i] = v[i];
    } else if (it != node.attr.map.end() && it->second.kind == Attr::kFloats) {
      const auto& v = it->second.floats;
      float* y = cpu::allocOut(Y, {(int64_t)v.size()});
      for (size_t i = 0; i < v.size(); ++i) y[i] = v[i];
    } else {
      cpu::allocOutI64(Y, {0});
    }
  }
  bool supportsDType(DType) const override { return true; }
};

// Gather along axis 0 (sufficient for shape-vector indexing in the classifier preamble).
struct GatherCpuOp : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& D = ctx.t(node.inputs[0]);
    const RtTensor& I = ctx.t(node.inputs[1]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    int64_t axis = node.attr.geti("axis", 0);
    (void)axis;  // axis 0 assumed
    int64_t nidx = I.elems();
    const int64_t* idx = I.host.i64();
    int64_t outer = D.shape.empty() ? 0 : D.shape[0];
    int64_t inner = D.elems() / std::max<int64_t>(outer, 1);
    Shape outShape;
    if (I.shape.empty() || (I.shape.size() == 1 && I.shape[0] == 1 && nidx == 1)) {
      // scalar index -> drop axis
      for (size_t i = 1; i < D.shape.size(); ++i) outShape.push_back(D.shape[i]);
      if (outShape.empty()) outShape = {1};
    } else {
      outShape.push_back(nidx);
      for (size_t i = 1; i < D.shape.size(); ++i) outShape.push_back(D.shape[i]);
    }
    if (D.dtype == DType::kInt64) {
      int64_t* y = cpu::allocOutI64(Y, outShape);
      const int64_t* d = D.host.i64();
      for (int64_t k = 0; k < nidx; ++k) {
        int64_t src = (idx[k] < 0 ? idx[k] + outer : idx[k]);
        std::memcpy(y + k * inner, d + src * inner, inner * sizeof(int64_t));
      }
    } else {
      float* y = cpu::allocOut(Y, outShape);
      const float* d = D.host.f32();
      for (int64_t k = 0; k < nidx; ++k) {
        int64_t src = (idx[k] < 0 ? idx[k] + outer : idx[k]);
        std::memcpy(y + k * inner, d + src * inner, inner * sizeof(float));
      }
    }
  }
  bool supportsDType(DType) const override { return true; }
};

struct UnsqueezeCpuOp : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    std::vector<int64_t> axes = node.attr.getints("axes");
    if (axes.empty() && node.inputs.size() > 1 && node.inputs[1] != kNoTensor) {
      const RtTensor& A = ctx.t(node.inputs[1]);
      for (int64_t i = 0; i < A.elems(); ++i) axes.push_back(A.host.i64()[i]);
    }
    Shape out = X.shape;
    std::sort(axes.begin(), axes.end());
    for (int64_t ax : axes) {
      if (ax < 0) ax += (int64_t)out.size() + 1;
      out.insert(out.begin() + std::min<int64_t>(ax, out.size()), 1);
    }
    copyAs(X, Y, out);
  }
  bool supportsDType(DType) const override { return true; }
};

struct ConcatCpuOp : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    RtTensor& Y = ctx.t(node.outputs[0]);
    int64_t axis = node.attr.geti("axis", 0);
    const RtTensor& first = ctx.t(node.inputs[0]);
    int64_t rank = (int64_t)first.shape.size();
    if (axis < 0) axis += rank;
    Shape out = first.shape;
    int64_t total = 0;
    for (TensorId in : node.inputs) total += ctx.t(in).shape[axis];
    out[axis] = total;
    // outer = product of dims before axis, block = product from axis
    auto blockElems = [&](const Shape& s) {
      int64_t b = 1; for (int64_t i = axis; i < (int64_t)s.size(); ++i) b *= s[i]; return b;
    };
    int64_t outer = 1; for (int64_t i = 0; i < axis; ++i) outer *= first.shape[i];
    bool isI64 = first.dtype == DType::kInt64;
    if (isI64) {
      int64_t* y = cpu::allocOutI64(Y, out);
      int64_t outBlock = blockElems(out);
      int64_t off = 0;
      for (TensorId in : node.inputs) {
        const RtTensor& T = ctx.t(in);
        int64_t bk = blockElems(T.shape);
        for (int64_t o = 0; o < outer; ++o)
          std::memcpy(y + o * outBlock + off, T.host.i64() + o * bk, bk * sizeof(int64_t));
        off += bk;
      }
    } else {
      float* y = cpu::allocOut(Y, out);
      int64_t outBlock = blockElems(out);
      int64_t off = 0;
      for (TensorId in : node.inputs) {
        const RtTensor& T = ctx.t(in);
        int64_t bk = blockElems(T.shape);
        for (int64_t o = 0; o < outer; ++o)
          std::memcpy(y + o * outBlock + off, T.host.f32() + o * bk, bk * sizeof(float));
        off += bk;
      }
    }
  }
  bool supportsDType(DType) const override { return true; }
};

}  // namespace

VX_REGISTER_CPU_OP(OpType::kReshape, ReshapeCpuOp);
VX_REGISTER_CPU_OP(OpType::kFlatten, FlattenCpuOp);
VX_REGISTER_CPU_OP(OpType::kShape, ShapeCpuOp);
VX_REGISTER_CPU_OP(OpType::kConstant, ConstantCpuOp);
VX_REGISTER_CPU_OP(OpType::kGather, GatherCpuOp);
VX_REGISTER_CPU_OP(OpType::kUnsqueeze, UnsqueezeCpuOp);
VX_REGISTER_CPU_OP(OpType::kConcat, ConcatCpuOp);

}  // namespace vx
