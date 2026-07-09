#include "passes_internal.h"

namespace vknn {

    // The IR normalizes rank-0 tensors to [1] once a Constant node is folded, but ONNX Gather drops
    // the indexed axis ONLY for a true rank-0 index. Record scalar-ness on the Gather while the
    // information still exists: a rank-0 initializer keeps an empty shape, and a Constant node's
    // value attr keeps its original dims. inferShapes and GatherCpu read the tag.
    static void markScalarGatherIndices(Graph &g) {
        std::map<TensorId, const Node *> producer;
        for (const auto &n: g.nodes)
        {
            for (TensorId o: n.outputs)
            {
                if (o != kNoTensor)
                {
                    producer[o] = &n;
                }
            }
        }
        for (auto &n: g.nodes)
        {
            if (n.type != OpType::Gather || n.inputs.size() < 2 || n.inputs[1] == kNoTensor)
            {
                continue;
            }
            TensorId idx    = n.inputs[1];
            bool     scalar = false;
            if (g.isInitializer(idx))
            {
                scalar = g.desc(idx).shape.empty();
            } else if (auto it = producer.find(idx); it != producer.end() && it->second->type == OpType::Constant)
            {
                auto vit = it->second->attr.map.find("value");
                if (vit != it->second->attr.map.end() && vit->second.shape.empty())
                {
                    size_t nvals = vit->second.kind == Attr::Ints ? vit->second.ints.size() : vit->second.floats.size();
                    scalar       = nvals == 1;
                }
            }
            if (scalar)
            {
                Attr a;
                a.kind                   = Attr::Int;
                a.i                      = 1;
                n.attr.map["idx_scalar"] = a;
            }
        }
    }

    // Runs the full import-time optimization pipeline in place, rewriting `g` from the raw imported ONNX
    // graph into the lowered, fused, statically-shaped form that gets serialized to the .vxm the runtime
    // executes. The pass ORDER here is load-bearing and is the pipeline's central invariant: shape-producing
    // metadata is captured before the passes that erase it (scalar-Gather tags before const-fold, declared
    // output dtypes before the output-rewiring passes), shapes are re-inferred after every stage that can
    // change a rank, and the fold+infer fixpoint runs before the lowering/fusion passes so those passes only
    // ever see constant-resolved, statically-shaped chains. Pointwise-chain fusion runs LAST for the same
    // reason (it is opaque to constFold). Every stage is behavior-preserving w.r.t. output semantics, so the
    // pre-snapshotted declared output dtypes are restored at the end as authoritative. `opt` gates the
    // optional fusions and carries the batch size used for shape inference.
    void runStandardPasses(Graph &g, const PassOptions &opt) {
        int64_t batch = opt.batch;
        // Per-input declared shapes for the batch/spatial resolution in inferShapes (empty = the
        // batch-only path). Passed by pointer so every inferShapes round below resolves identically.
        const std::map<std::string, Shape> *declared = opt.inputShapes.empty() ? nullptr : &opt.inputShapes;
        // Symbolic-dim bindings (--dim) that resolve dynamic input axes from their dim_param names; passed
        // alongside `declared` to every inferShapes round below (empty = the batch-only path).
        const std::map<std::string, int64_t> *bindings = opt.dimBindings.empty() ? nullptr : &opt.dimBindings;
        // Snapshot each graph output's ONNX-declared dtype (from value_info, set by the builder) BEFORE
        // any pass runs. Two things would otherwise drop it: inferShapes overwrites a declared FLOAT16
        // output with the fp32 dtype of its internal producer, and fusion/elimination passes repoint
        // g.outputs[i] to a producer tensor that defaults to Float32. Either way the session's readback
        // then emits fp32 bytes for a FLOAT16/UINT8-declared output. Restored after all passes (below);
        // optimization preserves output semantics, so the declared dtype is authoritative. Indexed by
        // output slot -- passes rewire the value of g.outputs[i], never its count or order.
        std::vector<DType> declaredOutDtype(g.outputs.size(), DType::Float32);
        for (size_t i = 0; i < g.outputs.size(); ++i)
        {
            if (g.outputs[i] != kNoTensor)
            {
                declaredOutDtype[i] = g.desc(g.outputs[i]).dtype;
            }
        }
        markScalarGatherIndices(g); // before const-fold erases the Constant nodes' original ranks
        eliminateDropout(g);        // before shape inference: Dropout has no shape rule -- the eliminable
                                    // (inference-mode) form is an identity and is rewired past here
        normalizeConv1d(g);         // before shape inference: conv arms assume 2-D weight/attr geometry
        inferShapes(g, batch, declared, bindings);
        lowerOrtContribOps(g);      // ORT contrib ops (Skip/SimplifiedLayerNorm, RotaryEmbedding,
                                    // MultiHeadAttention, MatMulNBits) expand to primitives in a
                                    // fixpoint with shape inference; a scan-only no-op without them
        if (opt.dequantize)
        {
            dequantizeGraph(g); // QDQ checkpoints collapse to the float graph before any lowering
                                // pass sees them; the next inferShapes resolves the shapes the
                                // kernel-less q/dq nodes left unresolved
        }
        lowerReduceToGap(g); // needs input ranks; ReduceMean imports as generic Reduce
        inferShapes(g, batch, declared, bindings);
        eliminateIdentity(g);
        foldBatchNorm(g);
        lowerBatchNorm(g); // whatever foldBatchNorm left becomes a fusable per-channel Mul+Add
        // The experimental block-kernel fusions match on conv.fusedAct, so their prerequisite
        // activation fold runs only with them; the general pointwise fusion below owns activation
        // folding otherwise (and re-encodes any fusedAct these passes set as a unit step).
        if (opt.fuseSqueezeExcite || opt.fuseDwPw)
        {
            fuseActivations(g);
        }
        if (opt.fuseSqueezeExcite)
        {
            fuseSqueezeExcite(g);
        }
        if (opt.fuseDwPw)
        {
            fuseDwPw(g);
        }
        // Iterate fold+infer: folding a Shape/Gather/Concat chain turns a dynamic Reshape's shape input
        // into a constant, which lets the next inferShapes resolve that Reshape statically, which in turn
        // exposes more foldable shape ops downstream (YOLO's DFL/box-decode head). The loop runs until
        // constFold stops removing nodes; a transformer whose every block reads Shape() of its own
        // activations resolves roughly one block per round, so the round cap covers deep encoders
        // (folding removes nodes, so the loop terminates on its own well before the cap).
        for (int iter = 0; iter < 256; ++iter)
        {
            if (constFold(g) == 0)
            {
                break;
            }
            inferShapes(g, batch, declared, bindings);
        }
        eliminateFloatCast(g);   // drop float->float casts left by transformer import (post-fold)
        foldIntRoundtripCast(g); // collapse float->wide-int->float cast pairs to Unary(Trunc) so the
                                 // truncation joins the surrounding pointwise unit instead of splitting it
        eliminateDeadNodes(g);
        inferShapes(g, batch, declared, bindings);  // refresh shapes after fusion/folding
        lowerRMSNorm(g);        // Cast-free, shape-resolved decomposed RMSNorm chains -> one fp32-accumulate
                                // RMSNorm kernel; before pointwise fusion so a trailing residual Add can fold
        inferShapes(g, batch, declared, bindings);  // set the RMSNorm output shape (identity rule)
        lowerInstanceNorm(g);   // needs the fixpoint-resolved input shapes; the emitted spatial
                                // ReduceMeans recover as GlobalAvgPool in the pass below
        lowerReduceToGap(g);    // a late-resolving rank can expose the spatial-mean form
        lowerEinsum(g);        // batched einsums -> MatMul (needs the operand shapes resolved above)
        inferShapes(g, batch, declared, bindings); // resolve the inserted Unsqueeze/MatMul/Squeeze
        subpixelConvTranspose(g); // ConvTranspose -> Conv + DepthToSpace; runs on fully-resolved dims, before
        inferShapes(g, batch, declared, bindings);    // the pointwise fusion so trailing pointwise ops can still fold onto the Conv
        lowerGroupedConv(g); // general grouped Conv (1 < group < Cin) -> group-1 Convs + Concat, on resolved shapes
        inferShapes(g, batch, declared, bindings); // shape the inserted Slice/Conv/Concat parts
        if (opt.lowerConv)
        {
            lowerConv(g); // non-Winograd KxK Conv -> ConvGemm, on resolved shapes, before pointwise fusion
        }
        if (opt.fuseGridSampleWarp)
        {
            // Claim the scaled-flow + base-grid warp coordinate chain into GridSample before the
            // pointwise fusion would host the Add on the Transpose. Needs the resolved shapes above.
            fuseGridSampleWarp(g);
            inferShapes(g, batch, declared, bindings); // GridSample warp output shape from the flow input
        }
        // Pointwise-chain fusion runs LAST, after const-fold + shape resolution: the shape-computation
        // subgraph (Shape/Gather/Neg/Sqrt/... feeding dynamic Reshapes) is now folded to constants, so
        // fusion only ever sees statically-shaped float activation chains. Fusing earlier would replace a
        // foldable shape op with a FusedPointwise (opaque to constFold), leaving a dynamic shape
        // unresolved -> an empty shape propagates and downstream ops crash.
        if (opt.fusePointwiseChains)
        {
            fusePointwiseChains(g, opt.strictFuse);
            inferShapes(g, batch, declared, bindings); // set the FusedPointwise output shapes
        }
        pruneDeadInitializers(g); // after all rewiring: orphaned fold intermediates + Cast-copied weights
        // Restore the declared output dtypes dropped by the output-rewiring passes above (see snapshot).
        for (size_t i = 0; i < g.outputs.size(); ++i)
        {
            if (g.outputs[i] != kNoTensor)
            {
                g.desc(g.outputs[i]).dtype = declaredOutDtype[i];
            }
        }
        if (opt.dumpBig)
        {
            for (const Node &n: g.nodes)
            {
                for (TensorId o: n.outputs)
                {
                    if (o == kNoTensor)
                    {
                        continue;
                    }
                    int64_t ne = numElements(g.desc(o).shape);
                    if (ne > 50000000)
                    {
                        std::string sh;
                        for (int64_t d: g.desc(o).shape)
                        {
                            sh += std::to_string(d) + ",";
                        }
                        VKNN_WARN << "BIG tensor " << ne << " elems from " << opTypeName(n.type) << " " << n.name << " shape=[" << sh << "]";
                    }
                }
            }
        }
    }

} // namespace vknn
