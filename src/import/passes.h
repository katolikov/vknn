// graph optimization passes (backend-agnostic, operate on NCHW IR).
#pragma once
#include "vx/graph.h"

namespace vx {

// Resolve dynamic batch to `batch` and infer concrete shapes for all tensors possible.
void inferShapes(Graph& g, int64_t batch = 1);
// Constant-fold ops whose inputs are all known constants (shape arithmetic, scalar Binary, etc.)
// into initializers (requires inferShapes first). Returns the number of nodes folded.
int constFold(Graph& g);
// Fold BatchNormalization that follows a Conv into the conv weights/bias.
void foldBatchNorm(Graph& g);
// Fuse Clip(relu6)/Relu following Conv/Gemm into the producer's fused activation.
void fuseActivations(Graph& g);
// Fuse a residual Add(conv, x) into the (1x1) conv's epilogue: out = act(conv + x).
void fuseResidualAdd(Graph& g);
// Fuse a Squeeze-Excite scale chain (GAP->FC->relu->FC->hardsigmoid) into one kFusedSE node.
void fuseSqueezeExcite(Graph& g);
// Fuse a depthwise-3x3 conv followed by a 1x1 project conv into one kFusedDwPw kernel.
void fuseDwPw(Graph& g);
// Fuse Mul(x,HardSigmoid(x))=HardSwish / Mul(x,Sigmoid(x))=SiLU into the conv epilogue or one
// unary.
void fuseSwish(Graph& g);
// Remove Identity nodes, rewiring consumers to the input.
void eliminateIdentity(Graph& g);
// Remove nodes whose outputs are unused (keeps graph outputs alive).
void eliminateDeadNodes(Graph& g);
// Run the standard pipeline used before backend planning.
void runStandardPasses(Graph& g, int64_t batch = 1);
// Read an int64 list param from a node attribute or an initializer input (Slice/Pad/Reduce style).
std::vector<int64_t> readI64Param(const Graph& g, const Node& nd, const char* attrName,
                                  int inputIdx);
// Insert ConvertLayout nodes + mark tensors gpuFlat so the generic head ops run on the Vulkan
// backend in a flat row-major layout (Transpose/Slice/Concat/Binary/Softmax). No-op for graphs
// without such ops. Run after backend-agnostic passes, before backend planning.
void insertLayoutConverts(Graph& g);

}  // namespace vx
