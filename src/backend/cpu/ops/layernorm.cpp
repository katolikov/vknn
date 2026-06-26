// LayerNormalization (ONNX opset 17). Inputs: X, Scale(gamma), B(bias, optional).
// Normalizes over the axes from `axis` to the end: per outer row, y =
// (x-mean)/sqrt(var+eps)*gamma+beta. CPU reference (canonical NCHW fp32 host buffers); validated
// against by the host tests.
#include <cmath>

#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
namespace {

struct LayerNormCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    const RtTensor& G = ctx.t(node.inputs[1]);
    bool hasBeta = node.inputs.size() > 2 && node.inputs[2] != kNoTensor;
    const float* beta = hasBeta ? ctx.t(node.inputs[2]).host.f32() : nullptr;

    RtTensor& Y = ctx.t(node.outputs[0]);
    const Shape& shape = X.shape;
    int rank = (int)shape.size();
    int64_t axis = node.attr.geti("axis", -1);
    if (axis < 0)
      axis += rank;
    if (axis < 0)
      axis = 0;

    int64_t norm = 1;
    for (int k = (int)axis; k < rank; ++k)
      norm *= shape[k];
    if (norm < 1)
      norm = 1;
    int64_t outer = X.elems() / norm;
    float eps = node.attr.getf("epsilon", 1e-5f);

    float* y = cpu::allocOut(Y, shape);
    const float* x = X.host.f32();
    const float* gamma = G.host.f32();
    for (int64_t r = 0; r < outer; ++r) {
      const float* xr = x + r * norm;
      float* yr = y + r * norm;
      double mean = 0.0;
      for (int64_t j = 0; j < norm; ++j)
        mean += xr[j];
      mean /= (double)norm;
      double var = 0.0;
      for (int64_t j = 0; j < norm; ++j) {
        double c = xr[j] - mean;
        var += c * c;
      }
      var /= (double)norm;
      float inv = (float)(1.0 / std::sqrt(var + eps));
      for (int64_t j = 0; j < norm; ++j) {
        float v = ((float)(xr[j] - mean)) * inv * gamma[j];
        if (hasBeta)
          v += beta[j];
        yr[j] = v;
      }
    }
  }
};

}  // namespace
VKNN_REGISTER_CPU_OP(OpType::kLayerNorm, LayerNormCpu);
}  // namespace vknn
