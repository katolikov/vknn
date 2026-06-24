// graph optimization passes (backend-agnostic, operate on NCHW IR).
#pragma once
#include "vx/graph.h"

namespace vx {

// Resolve dynamic batch to `batch` and infer concrete shapes for all tensors possible.
void inferShapes(Graph& g, int64_t batch = 1);
// Constant-fold shape-computation ops (Shape/Constant/Gather/Unsqueeze/Concat) into
// initializers (requires inferShapes first). Leaves data-path ops intact.
void constFold(Graph& g);
// Fold BatchNormalization that follows a Conv into the conv weights/bias.
void foldBatchNorm(Graph& g);
// Fuse Clip(relu6)/Relu following Conv/Gemm into the producer's fused activation.
void fuseActivations(Graph& g);
// Fuse a residual Add(conv, x) into the (1x1) conv's epilogue: out = act(conv + x).
void fuseResidualAdd(Graph& g);
// Remove Identity nodes, rewiring consumers to the input.
void eliminateIdentity(Graph& g);
// Remove nodes whose outputs are unused (keeps graph outputs alive).
void eliminateDeadNodes(Graph& g);
// Run the standard pipeline used before backend planning.
void runStandardPasses(Graph& g, int64_t batch = 1);

}  // namespace vx
