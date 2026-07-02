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

    void runStandardPasses(Graph &g, const PassOptions &opt) {
        int64_t batch = opt.batch;
        markScalarGatherIndices(g); // before const-fold erases the Constant nodes' original ranks
        inferShapes(g, batch);
        lowerReduceToGap(g); // needs input ranks; ReduceMean imports as generic Reduce
        inferShapes(g, batch);
        eliminateIdentity(g);
        foldBatchNorm(g);
        fuseActivations(g);
        fuseResidualAdd(g);
        if (opt.fuseSwish)
        {
            fuseSwish(g); // HardSwish/SiLU into conv epilogue (default on)
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
        // exposes more foldable shape ops downstream (YOLO's DFL/box-decode head). Converges in a couple
        // rounds; the loop runs until constFold stops removing nodes.
        for (int iter = 0; iter < 8; ++iter)
        {
            if (constFold(g) == 0)
            {
                break;
            }
            inferShapes(g, batch);
        }
        eliminateFloatCast(g); // drop float->float casts left by transformer import (post-fold)
        fuseMatMulBias(g);     // fold Linear bias-Adds into the MatMul epilogue (Casts now gone)
        eliminateDeadNodes(g);
        inferShapes(g, batch); // refresh shapes after fusion/folding
        lowerReduceToGap(g);   // a late-resolving rank can expose the spatial-mean form
        lowerEinsum(g);        // batched einsums -> MatMul (needs the operand shapes resolved above)
        inferShapes(g, batch); // resolve the inserted Unsqueeze/MatMul/Squeeze
        // Pointwise-chain fusion runs LAST, after const-fold + shape resolution: the shape-computation
        // subgraph (Shape/Gather/Neg/Sqrt/... feeding dynamic Reshapes) is now folded to constants, so
        // fusion only ever sees statically-shaped float activation chains. Fusing earlier would replace a
        // foldable shape op with a FusedPointwise (opaque to constFold), leaving a dynamic shape
        // unresolved -> an empty shape propagates and downstream ops crash.
        if (opt.fusePointwiseChains)
        {
            fusePointwiseChains(g);
            inferShapes(g, batch); // set the FusedPointwise output shapes
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
