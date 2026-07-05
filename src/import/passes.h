// Graph optimization/lowering passes over the NCHW IR. Their combined output is the serialized .vxm
// the runtime executes, so any change to a pass's logic changes every compiled model.
//
// This header is an unordered index of pass entry points, not their run order. Pass ordering and the
// interleaved inferShapes/constFold rounds are load-bearing and defined in one place --
// runStandardPasses (run_standard_passes.cpp) for the backend-agnostic set, then insertLayoutConverts
// and markFp32 at load once the backend is chosen. Most passes are pure IR-to-IR rewrites; the two
// layout/dtype passes at the bottom are Vulkan-backend-oriented and run after backend selection.
#pragma once
#include "vknn/graph.h"

namespace vknn {

    // Resolve dynamic batch to `batch` and infer concrete shapes for all tensors possible.
    void inferShapes(Graph &g, int64_t batch = 1);
    // Constant-fold ops whose inputs are all known constants (shape arithmetic, scalar Binary, etc.)
    // into initializers (requires inferShapes first). Returns the number of nodes folded.
    int constFold(Graph &g);
    // Fold BatchNormalization that follows a Conv into the conv weights/bias.
    void foldBatchNorm(Graph &g);
    // Fold Clip(relu6)/Relu following Conv/Gemm/Add onto the producer's fusedAct. Runs only as the
    // prerequisite of the experimental SE/DwPw fusions, whose matchers read fusedAct; the general
    // pointwise fusion owns activation folding otherwise.
    void fuseActivations(Graph &g);
    // Fuse a Squeeze-Excite scale chain (GAP->FC->relu->FC->hardsigmoid) into one kFusedSE node.
    void fuseSqueezeExcite(Graph &g);
    // Fuse a depthwise-3x3 conv followed by a 1x1 project conv into one kFusedDwPw kernel.
    void fuseDwPw(Graph &g);
    // The general fusion: grow each maximal same-shape per-element region (fanout included) and
    // emit it as one fused unit — a producer epilogue or a standalone FusedPointwise node, with
    // extra output streams for values consumed outside the region. Subsumes activation, residual-
    // add, swish-diamond, and matmul-bias folding. Runs last; on by default. In the default fast
    // mode the swish/residual/bias patterns emit the kernels' native fp32-accumulator epilogues
    // (old-main speed; bytes differ from unfused there); strict=true keeps every step rounded, so
    // fused == unfused is byte-identical — the byte gate compiles with --strict-fuse.
    void fusePointwiseChains(Graph &g, bool strictFuse);

    // Lower Reduce(Mean) over the spatial dims of a rank-4 tensor (keepdims) to GlobalAvgPool — the
    // ResNet classifier-head pattern with a dedicated NC4HW4 kernel. Needs resolved input ranks.
    void lowerReduceToGap(Graph &g);
    // Rewrite ConvTranspose (stride s, kernel k%s==0) as a stride-1 Conv (Cout*s*s channels) +
    // DepthToSpace(s), replacing the memory-bound gather deconv with a tiled conv. Weight rearrange is
    // exact; device output stays at the fp16 floor. Needs resolved input spatial dims.
    void subpixelConvTranspose(Graph &g);
    // Lower each non-Winograd, non-1x1 KxK Conv (group 1, constant weight, static shapes) to a
    // ConvGemm node with the weights repacked [K][Cout] — one LDS-tiled implicit-GEMM kernel instead
    // of the streaming direct conv. Deterministic; fp16-floor equivalent to Conv (the accumulation
    // order shifts, as Winograd's does). Needs resolved shapes; runs before pointwise fusion so
    // trailing units fold onto the ConvGemm.
    void lowerConv(Graph &g);
    // Remove Identity nodes, rewiring consumers to the input.
    void eliminateIdentity(Graph &g);
    // Remove nodes whose outputs are unused (keeps graph outputs alive).
    void eliminateDeadNodes(Graph &g);
    // Drop initializer payloads no node/output references (folded-chain intermediates, Cast-copied
    // weights' originals) so they are neither serialized to the .vxm nor uploaded at load.
    void pruneDeadInitializers(Graph &g);
    // Options for the standard pass pipeline (compile time), exposed by the model compiler as flags.
    struct PassOptions {
        int64_t batch               = 1;
        bool    fuseSqueezeExcite   = false; // fuse the SE squeeze->FC->scale chain (experimental)
        bool    fuseDwPw            = false; // fuse depthwise-3x3 + 1x1-project (experimental)
        bool    fusePointwiseChains = true;  // the general pointwise-region fusion (default on)
        bool    strictFuse          = false; // rounded steps everywhere: fused == unfused byte-identical
                                             // (the byte-gate mode; default fast mode uses the kernels'
                                             // native fp32-accumulator swish/residual/bias epilogues)
        bool    lowerConv           = false; // non-Winograd KxK Conv -> ConvGemm (experimental: the
                                             // v1 64x64x16 kernel loses to the direct conv on
                                             // classifier-CNN shapes — opt in per model, measure)
        bool    dumpBig             = false; // debug: log tensors > 50M elements after shape inference

        // Optimization-level preset (vknn_compile -O0..-O3). Individual fuse flags override on top.
        //   O0 = no optional fusion (reference output, one kernel per op)
        //   O1 = the default production set: the general pointwise fusion (bit-exact)
        //   O2/O3 = + the experimental squeeze-excite and dwpw-pair fusions (situational; can
        //           regress on some models — measure before shipping a model with them).
        //   ConvGemm lowering stays opt-in (--lower-conv) at every level until its kernel is tuned.
        static PassOptions forOptLevel(int level) {
            PassOptions o;
            o.fusePointwiseChains = level >= 1;
            o.fuseSqueezeExcite   = level >= 2;
            o.fuseDwPw            = level >= 2;
            return o;
        }
    };

    // Run the standard pipeline used before backend planning.
    void runStandardPasses(Graph &g, const PassOptions &opt = {});
    // Read an int64 list param from a node attribute or an initializer input (Slice/Pad/Reduce style).
    std::vector<int64_t> readI64Param(const Graph &g, const Node &nd, const char *attrName, int inputIdx);
    // Insert ConvertLayout nodes + mark tensors gpuFlat so the generic head ops run on the Vulkan
    // backend in a flat row-major layout (Transpose/Slice/Concat/Binary/Softmax). No-op for graphs
    // without such ops. Run after backend-agnostic passes, before backend planning.
    void insertLayoutConverts(Graph &g);

    // Mark activation tensors named by `substrs` (comma list) as fp32 storage and insert ConvertDtype
    // nodes at the fp16/fp32 frontier (Config::fp32Tensors). Runs at load, after insertLayoutConverts.
    void markFp32(Graph &g, const std::string &substrs);

} // namespace vknn
