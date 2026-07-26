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
#include <map>
#include <set>
#include <string>
#include <vector>

namespace vknn {

    // Resolve each graph input's dynamic (negative) dims, then infer concrete shapes for all tensors
    // possible. A dim is resolved from `declared` (keyed by input tensor name) when that input has a
    // declared shape; otherwise a symbolic axis whose dim_param(s) are all present in `bindings` is
    // resolved by evaluating its expression (Config::dimBindings / --dim NAME=VALUE), the leading (batch)
    // axis falls back to `batch`, and any OTHER dynamic axis that stays unresolved is a hard error
    // (@throws Error{InvalidArgument}) that LISTS the unbound symbol names (or, for an axis with no
    // recorded symbol, the input and axis) so the caller can bind them -- substituting `batch` into a
    // dynamic spatial/feature axis silently compiles the model to a 1x1 plan, which is the class of bug
    // this refuses. `declared` and `bindings` may be null (no declarations/bindings, the batch-only path).
    // Precedence per still-dynamic axis: `declared` (per-tensor override) > symbol binding > batch
    // fallback (leading axis) > error. A declared entry's rank must match the input's rank and its dims
    // must be non-negative. The pass is idempotent: a dim resolved on an earlier call is already positive
    // and is left untouched (the lookups and the error fire only while a dim is still dynamic).
    void inferShapes(Graph &g, int64_t batch = 1, const std::map<std::string, Shape> *declared = nullptr, const std::map<std::string, int64_t> *bindings = nullptr);
    // The per-node forward shape rule of inferShapes, callable on its own. constFold interleaves it
    // with value folding in one program-order walk so a folded Reshape/Slice/Expand target resolves
    // its consumer's shape (and everything behind it) within the SAME fold pass, instead of one
    // fold/infer alternation per dependent block. Idempotent; fills only shapes derivable from
    // resolved inputs.
    void inferNodeShape(Graph &g, Node &nd);
    // Normalize 1-D Convs (rank-3 constant weight, 1-spatial-dim attributes) to the canonical 2-D
    // geometry every conv consumer indexes: weight [M,C/g,k] -> [M,C/g,k,1], strides/dilations/
    // kernel_shape/pads extended with the W dim's identity values. Activation ranks are unchanged.
    // Runs before the first inferShapes so conv shape inference only ever sees 2-D geometry.
    void normalizeConv1d(Graph &g);
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
    // mode units are fp32-chained (pw_relax): the entry rounds to the producer's store byte, the
    // steps run unrounded in fp32 registers, and the unit rounds once per stored stream — faster
    // than per-step rounding and at least as accurate as the unfused fp16 graph on every chain.
    // strict=true keeps every step rounded, so fused == unfused is byte-identical — the byte gate
    // compiles with --strict-fuse.
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
    // Collapse ONNX quantized graphs (QDQ form) to plain float. DequantizeLinear over
    // all-initializer inputs folds to an fp32 initializer ((x - zero_point) * scale in double,
    // per-tensor or per-axis via the axis attribute), and a QuantizeLinear->DequantizeLinear
    // activation sandwich with matching scale/zero_point drops, consumers rewired to the float
    // producer. Residual activation q/dq nodes (mismatched sandwiches, graph-boundary q/dq) stay
    // in place; no kernel exists for them, so they surface through the support report. Idempotent.
    void dequantizeGraph(Graph &g);
    // Remove Identity nodes, rewiring consumers to the input.
    void eliminateIdentity(Graph &g);
    // Remove inference-mode Dropout nodes (training_mode absent or a constant false, mask output
    // absent or unconsumed), rewiring consumers to the input. A Dropout that is not provably
    // inference-mode, or whose mask is consumed, stays in place and is unsupported downstream.
    void eliminateDropout(Graph &g);
    // Remove nodes whose outputs are unused (keeps graph outputs alive).
    void eliminateDeadNodes(Graph &g);
    // Drop initializer payloads no node/output references (folded-chain intermediates, Cast-copied
    // weights' originals) so they are neither serialized to the .vxm nor uploaded at load.
    void pruneDeadInitializers(Graph &g);
    // Fuse cross-bucket hand-offs in a multi-graph model: when bucket A's graph OUTPUT feeds bucket B's
    // graph INPUT of the same name, copy A's subgraph into B so the value stays inside one graph on the
    // GPU (instead of a host round-trip that folds the producer op to the CPU); a fully-absorbed A is
    // deleted. Model-agnostic (VLM embed->decoder or any chained export); non-chained buckets and
    // differently-named recurrences (KV cache: present.* vs past_key_values.*) are left untouched. For a
    // vision-language model it also adds an image-capable copy of the decoder whose image-token rows are
    // filled from the encoder features by an on-GPU ScatterND (the position-dependent splice the host used
    // to do); the original decoder stays as the text-only path.
    void fuseBucketBoundaries(std::vector<Graph> &buckets, std::vector<std::string> &names);
    // The shape set for an automatic bucket that is one with-past decode graph recompiled at several
    // token columns per step. A decode bucket fixes the whole cache geometry (input_ids [1,1],
    // past_key_values.*.{key,value} [1, kvHeads, C, headDim], a mask spanning the C past slots plus
    // the step's tokens, position_ids); widening it to [1, width] gives a bucket that processes
    // `width` tokens against the same cache under one static plan. Two compiles use it: the
    // chunk-prefill bucket at kChunkPrefillTokens (io_link.h) and the speculative-verification
    // bucket at kSpecVerifyTokens (spec_decode.h).
    struct WidenedDecodePlan {
        size_t                       decodeBucket = 0; ///< Bucket whose geometry the shapes derive from.
        std::map<std::string, Shape> shapes;           ///< Full concrete shape per widened-bucket graph input.
        std::string                  label;            ///< Bucket label for the .vxm bucket list.
        int64_t                      cacheSlots = 0;   ///< Compiled context length C, from the past inputs.
    };
    // Scan `buckets` for a with-past decode graph (input_ids [1,1], position_ids [1,1],
    // attention_mask, past_key_values.0.key, logits + present.0.key outputs) and fill `plan` with
    // that bucket's index, its context length, and one full concrete shape per graph input at
    // `width` token columns: ids and positions become [1, width], the mask widens to past + width
    // columns, every other input keeps its decode shape. Leaves `plan->label` empty — the caller
    // names the role. False when no decode bucket qualifies, when the compiled context is shorter
    // than `width` (the runtime could never run a full window), or when the mask layout is neither
    // the 2-D [1, C+1] nor the 4-D [1, x, 1, C+1] convention. Callers apply their own duplicate
    // policy on top; this decides geometry only.
    bool planWidenedDecodeBucket(const std::vector<Graph> &buckets, int64_t width, WidenedDecodePlan *plan);
    using ChunkPrefillPlan = WidenedDecodePlan;
    // The chunk-prefill bucket's plan: planWidenedDecodeBucket at kChunkPrefillTokens, so a
    // variable-length prompt prefills as ceil(T / kChunkPrefillTokens) fixed-shape passes over the
    // cached KV. Additionally false when a chunk-capable prefill bucket (input_ids [1, S],
    // 1 < S <= kChunkPrefillTokens over the same cache shape) already exists, which would only
    // duplicate the plan.
    bool planChunkPrefillBucket(const std::vector<Graph> &buckets, ChunkPrefillPlan *plan);
    using SpecVerifyPlan = WidenedDecodePlan;
    // The speculative-verification bucket's plan: planWidenedDecodeBucket at kSpecVerifyTokens, so a
    // greedy speculative round verifies kSpecDraftTokens proposals plus their anchor token in ONE
    // target forward (spec_decode.h). Additionally false when the compile already carries a bucket
    // at exactly that width over the same cache (a recompile, or a hand-declared bucket that already
    // serves the verification pass). Emitted AFTER the chunk-prefill bucket in one compile: the
    // chunk plan refuses when any 1 < S <= kChunkPrefillTokens bucket exists, which the narrower
    // verification bucket would otherwise satisfy.
    bool planSpecVerifyBucket(const std::vector<Graph> &buckets, SpecVerifyPlan *plan);
    // Options for the standard pass pipeline (compile time), exposed by the model compiler as flags.
    struct PassOptions {
        int64_t batch = 1;
        // Declared concrete shapes for graph inputs, keyed by input tensor name. An input listed here
        // has its dynamic (negative) dims resolved from its declared shape; an input absent here falls
        // back to `batch` on its leading axis and errors on any other dynamic axis (see inferShapes).
        // Empty = the batch-only path, so existing callers that set only `batch` are unchanged.
        std::map<std::string, Shape> inputShapes;
        // Symbolic-dimension bindings keyed by ONNX dim_param name (e.g. "past_sequence_length" -> 256).
        // A dynamic input axis whose dim_param expression resolves entirely from these bindings is filled
        // automatically by inferShapes; `inputShapes` (per-tensor) overrides a binding for that tensor.
        // Empty = the batch-only path.
        std::map<std::string, int64_t> dimBindings;
        bool                           fuseSqueezeExcite   = false; // fuse the SE squeeze->FC->scale chain (experimental)
        bool                           fuseDwPw            = false; // fuse depthwise KxK + 1x1-project into FusedDwPw
                                                                    // (experimental: the fp16-rounded LDS intermediate matches
                                                                    // the unfused store bit-for-bit on the CPU oracle and at
                                                                    // fp32, but the fp16 GPU path still diverges from the
                                                                    // unfused graph — opt in with --fuse-dwpw and measure)
        bool                           fusePointwiseChains = true;  // the general pointwise-region fusion (default on)
        bool                           fuseGridSampleWarp  = true;  // fold a scaled-flow + base-grid coordinate chain into
                                                                    // GridSample (bit-exact; default on, part of O1)
        bool strictFuse = false;                                    // rounded steps everywhere: fused == unfused byte-identical
                                                                    // (the byte-gate mode; default fast mode fp32-chains each
                                                                    // unit and rounds once per stored stream)
        bool lowerConv = false;                                     // non-Winograd KxK Conv -> ConvGemm (experimental: the
                                                                    // v1 64x64x16 kernel loses to the direct conv on
                                                                    // classifier-CNN shapes — opt in per model, measure)
        bool dequantize = true;                                     // fold DequantizeLinear weights + collapse matching
                                                                    // QDQ sandwiches so quantized checkpoints run as
                                                                    // float graphs (--no-dequantize keeps the quantized
                                                                    // ops; they have no kernel and fail planning)
        bool dumpBig = false;                                       // debug: log tensors > 50M elements after shape inference

        // Optimization-level preset (vknn_compile -O0..-O3). Individual fuse flags override on top.
        //   O0 = no optional fusion (reference output, one kernel per op)
        //   O1 = the default production set: the general pointwise fusion (bit-exact)
        //   O2/O3 = + the experimental squeeze-excite and dwpw-pair fusions (situational; the
        //           dwpw pair still diverges from the unfused graph on the fp16 GPU path —
        //           measure before shipping a model with them).
        //   ConvGemm lowering stays opt-in (--lower-conv) at every level until its kernel is tuned.
        static PassOptions forOptLevel(int level) {
            PassOptions o;
            o.fusePointwiseChains = level >= 1;
            o.fuseGridSampleWarp  = level >= 1;
            o.fuseSqueezeExcite   = level >= 2;
            o.fuseDwPw            = level >= 2;
            return o;
        }
    };

    // Run the standard pipeline used before backend planning.
    void runStandardPasses(Graph &g, const PassOptions &opt = {});

    // Knobs for the -Os weight-quantization pass (quantize_weights.cpp). Structural defaults that
    // hold for any model — never tuned per model. All thresholds are generic eligibility rules:
    // small weights and shallow reductions (depthwise convs, tiny heads) stay fp16 by construction.
    struct QuantOptions {
        // Packed-weight bit width (--quant-bits): 4 = int4 (kWqFormatInt4, the -Os default), 8 =
        // int8 (kWqFormatInt8). One knob for every site; the AWQ outlier, min-MSE step, bias
        // correction, and guard machinery are shared across widths (core/quant_weights.h).
        int bits = 4;
        // LUT4 codebook format (--quant-bits lut4): the 4-bit payload holds indices into one
        // fitted 16-entry fp16 codebook per tensor (kWqFormatLut4) instead of a symmetric integer
        // alphabet. Requires bits == 4.
        bool lut4 = false;
        // Grouping/outlier defaults split by op CLASS (structural, never per-model): a MatMul weight
        // is read natively packed by the GPU (its bytes are decode traffic, and LLM residual streams
        // tolerate int4 well), so it takes the coarse traffic-optimal setting; Conv/Gemm weights are
        // dequantized to fp16 at load (file-size win only, zero runtime cost), so they take the fine
        // quality-optimal setting that deep conv stacks need.
        int64_t group           = 128;  // MatMul scale group size along the reduction axis
        double  outlierFrac     = 0.01; // MatMul fraction of activation-salient columns kept fp16 (AWQ)
        int64_t convGroup       = 32;   // Conv/Gemm scale group size
        double  convOutlierFrac = 0.05; // Conv/Gemm outlier fraction
        int     calibSamples    = 8;    // synthetic calibration samples when calibFiles is empty
        // Per-layer weighted relative weight-error bar; above it the layer stays fp16. Symmetric
        // int4 with grouped scales sits at ~8-15% weight-space error by construction (the OUTPUT
        // error is far smaller — independent per-column errors average out across the reduction), so
        // the bar catches anomalously quantization-hostile layers, not typical ones.
        double  maxLayerRelErr = 0.25;
        int64_t minElems       = 16384; // weights smaller than this stay fp16 (no size win)
        int64_t minK           = 256;   // reductions shallower than this stay fp16 (excludes depthwise)
        // Caller calibration data: each entry is ONE sample — raw .bin files in graph-input order
        // (the vknn_run_io convention). Empty = deterministic synthetic samples.
        std::vector<std::vector<std::string>> calibFiles;
    };
    // Totals from quantizeWeights, for the compiler's summary line.
    struct QuantStats {
        int64_t sites = 0, quantized = 0, guardKept = 0, outlierCols = 0;
        int64_t bytesBefore = 0, bytesAfter = 0;
        // Sites served from another bucket's already-quantized weight instead of being calibrated
        // and packed again (quantizeWeightsShared); always 0 for a single-graph quantization.
        int64_t shared     = 0;
        bool    calibrated = false;
    };
    // Quantize eligible MatMul/Gemm/Conv weights to the packed width opt.bits selects (int4 or
    // int8) with fp16 group scales, fp16 outlier columns, and calibrated min-MSE steps
    // (vknn_compile -Os). Layout/attribute contract in core/quant_weights.h (int4 authority:
    // core/quant_int4.h). Runs after runStandardPasses, before convertInitializersFp16.
    QuantStats quantizeWeights(Graph &g, const QuantOptions &opt);

    // Quantize ALL buckets of one compile together, returning one QuantStats per bucket in bucket
    // order. Calibration statistics are shape-dependent (they come from running the float graph on
    // samples sized by its input shapes), so quantizing each bucket on its own gives the same weight
    // a different packed payload per bucket: the buckets stop being one model, and the .vxm's
    // content-deduped initializer pool has to store every copy. Here each distinct weight — same
    // tensor name, same pre-quantization payload, same [K, N] geometry and op class — is calibrated
    // and packed exactly ONCE, and every bucket that shares it receives the identical payload,
    // scales, outlier columns, and bias correction. Buckets are visited longest-activation-first (the
    // most calibration rows per sample), so a shared weight carries the most representative bucket's
    // statistics; a weight no earlier bucket held is quantized by the first bucket that holds it.
    // One bucket is quantizeWeights(g, opt) exactly — payload-identical, sharing never applies
    // within a bucket. Every bucket must stay put for the duration of the call: a shared payload is
    // copied out of the bucket that produced it, not out of a cached duplicate.
    std::vector<QuantStats> quantizeWeightsShared(const std::vector<Graph *> &buckets, const QuantOptions &opt);

    // Byte totals from convertInitializersFp16, for the compiler's conversion summary line.
    struct Fp16ConvertStats {
        int64_t converted = 0, kept = 0, bytesBefore = 0, bytesAfter = 0;
    };
    // Convert every Float32 initializer payload to Float16 in place (vknn_compile --fp16), stamping
    // the tensor descs. Non-fp32 payloads (int64 shape tensors, ...) stay untouched. Runs after the
    // standard passes, immediately before saveGraphBin.
    Fp16ConvertStats convertInitializersFp16(Graph &g);

    // Read an int64 list param from a node attribute or an initializer input (Slice/Pad/Reduce style).
    std::vector<int64_t> readI64Param(const Graph &g, const Node &nd, const char *attrName, int inputIdx);
    // Insert ConvertLayout nodes + mark tensors gpuFlat so the generic head ops run on the Vulkan
    // backend in a flat row-major layout (Transpose/Slice/Concat/Binary/Softmax). No-op for graphs
    // without such ops. Run after backend-agnostic passes, before backend planning.
    void insertLayoutConverts(Graph &g);

    // Split a comma-separated pattern list into its non-empty entries, as typed (an exclude entry
    // keeps its leading '-'). Shared by markFp32's per-entry match accounting and the session's
    // zero-match warnings, so both enumerate the same entries.
    std::vector<std::string> splitPatternList(const std::string &patterns);

    // Mark activation tensors named by `substrs` (comma list) as fp32 storage and insert ConvertDtype
    // nodes at the fp16/fp32 frontier (Config::fp32Tensors). Runs at load, after insertLayoutConverts.
    // A non-null `matchedPatterns` collects every list entry (as typed, exclude '-' included) whose
    // substring occurs in an eligible tensor's name — zero-match accounting for the session's
    // load-end warning; the marking itself is unchanged.
    void markFp32(Graph &g, const std::string &substrs, std::set<std::string> *matchedPatterns = nullptr);

    // Pin every GPU Gather's runtime index (and the pure producer chain feeding it) to fp32 storage.
    // An index is an integer (token id / position) whose value can exceed the fp16 range, so an fp16
    // store would corrupt the lookup. Runs at load, after insertLayoutConverts, before markFp32.
    void pinGatherIndexFp32(Graph &g);

    // Pin every GPU GridSample's runtime grid (and the flat passthrough chain feeding it) to fp32
    // storage. The grid holds normalized sampling coordinates whose fp16 quantization drifts the
    // sample point by up to ~0.5 px at 1920-wide inputs (a direct warp/UV-quality loss); the shader
    // decodes the grid at its storage precision via the GRID_FP32 spec constant. Runs at load, after
    // insertLayoutConverts, before markFp32.
    void pinGridSampleGridFp32(Graph &g);

    // Fold chains of movement ops — a Transpose or Slice fed by another Transpose or Slice — into
    // ONE strided gather: the consumer reads the chain's source through the composed per-axis map
    // (stamped as view_stride/view_base attrs the CPU kernels and flat_gather geometry consume) and
    // the producer disappears, along with its materialized intermediate. Byte-identical: movement
    // kernels store loaded bytes verbatim, so the composed read yields exactly the bytes the
    // intermediate held. `fp32Pins` mirrors foldMatMulViews — a pinned intermediate keeps its
    // materialized form. Runs at load only, before insertLayoutConverts; never serialized. Returns
    // the number of folded producers.
    int foldMovementChains(Graph &g, const std::string &fp32Pins = "");

    // Fold Transpose/Expand/Reshape/Unsqueeze/Squeeze chains feeding a non-tiled-class MatMul
    // operand into per-axis stride attrs on the node (core/matmul_view.h) and rewire the operand to
    // the chain's source, so a GQA decode's repeat_kv broadcast and attention transposes never
    // materialize. Bit-identical (same values, same ascending-k order, kernel class preserved).
    // `fp32Pins` is the Config::fp32Tensors/mixedPrecisionFp32Tensors matcher the later markFp32
    // pass will apply: a chain touching a pinned tensor keeps its materialized form, since removing
    // the tensor would remove the fp32 store the pin exists for. Runs at load only, gated by
    // Hint::MatMulViewFold, before insertLayoutConverts; never serialized.
    void foldMatMulViews(Graph &g, const std::string &fp32Pins = "");

    // Fuse each rotate-half RoPE chain — the primitive expansion a contrib RotaryEmbedding lowers to
    // (Slice x1/x2 of the last-axis halves, Gather+Unsqueeze of a cos/sin table row by position, the
    // x1*cos-x2*sin / x1*sin+x2*cos rotate products as Binary nodes or the FusedPointwise units the
    // pointwise fusion built from them, Concat of the halves) — into ONE OpType::Rope node reading
    // the table rows directly. Matches by structure/shape only; a chain node carrying fused work
    // (pw epilogue, activation, bias/residual edge, extra outputs) or an attribute the fusion does
    // not fold refuses the site. `fp32Pins` mirrors foldMatMulViews: a site whose internal tensors
    // match a pin keeps its decomposed form. Runs at load only, gated by Hint::RopeFusion, before
    // foldMatMulViews; never serialized. Returns the number of sites fused.
    int fuseRope(Graph &g, const std::string &fp32Pins = "");
    // Fuse the decode-attention chain — MatMul(view) [-> scale/mask pointwise] -> Softmax ->
    // MatMul(view) [-> Transpose -> Reshape] — into one FusedAttention node
    // (core/fused_attention.h), so a decode step's attention core is one dispatch per layer and
    // the score/probability intermediates never touch memory. Consumes the operand-view stride
    // attrs foldMatMulViews composed, so it must run after that pass. Matches the M == 1
    // (single-query decode) form and the M > 1 form whose mask varies per query row (the
    // with-past chunk-prefill causal mask; the query axis is hosted as one more row dim);
    // maskless / row-broadcast-mask M > 1 chains and CNN graphs are untouched. Numerics-changing
    // (fp32 scores + softmax without the decomposed chain's fp16 round-trips); `fp32Pins` mirrors
    // foldMatMulViews — a chain whose erased tensors match the markFp32 set keeps its
    // decomposed form. Runs at load only, gated by Hint::FusedAttention; never serialized.
    void fuseDecodeAttention(Graph &g, const std::string &fp32Pins = "");

    // Fold the per-token KV-cache Concat feeding a FusedAttention node: the node reads the past
    // cache and the current rows as separate stride-addressed sources (token s < pastLen from the
    // past, the rest from the new rows), a present output that was the concat result is rewritten
    // to the new-rows tensor under the same name (the rows-only convention of io_link.h), and the
    // dead Concat falls to DCE — removing a whole-cache copy per decoded token. Values are
    // bit-identical; only the copy disappears. Runs at load after fuseDecodeAttention, gated by
    // Hint::KvConcatFold; never serialized. Returns the number of folded attention nodes.
    int foldFusedAttentionKvConcat(Graph &g);

} // namespace vknn
