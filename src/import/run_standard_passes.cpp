#include "passes_internal.h"

namespace vknn {

    void runStandardPasses(Graph &g, const PassOptions &opt) {
        int64_t batch = opt.batch;
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
