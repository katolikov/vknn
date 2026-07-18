#include "passes_internal.h"

namespace vknn {

    // Fold the group-interleave channel-shuffle idiom (ShuffleNetV2) into one ChannelShuffle node:
    //
    //   Reshape  [N,C,(sp...)] -> [N,g,C/g,(sp...)]     (split channels into g groups)
    //   Transpose perm {0,2,1,(identity tail)}          (swap the two group axes)
    //   Reshape  [N,C/g,g,(sp...)] -> [N,C,(sp...)]     (merge back to one channel axis)
    //
    // Composition of the channel index: c splits as (i, j) with i = c / (C/g), j = c % (C/g); the
    // transpose reorders to (j, i); the merge flattens to c_out = j * g + i. Inverting for a gather,
    // output channel c reads input channel (c % g) * (C/g) + c / g — a pure permutation, so the fold
    // is bit-identical to the chain it replaces. The spatial tail (any rank, including none) rides
    // along untouched; the match is purely structural on resolved shapes, never model-specific.
    //
    // Structure mirrors fuseDwPw: read-count/sole-reader maps over the ORIGINAL node list, one walk
    // replacing each matching merge Reshape in place with the ChannelShuffle node (the final output
    // tensor id is unchanged, so consumers need no rewiring), then one compaction dropping the two
    // interior nodes. The Reshape shape-target initializers go dead and are collected by
    // pruneDeadInitializers at the end of runStandardPasses.
    //
    // Preconditions: shapes are resolved (runs after the const-fold/infer fixpoint). Each interior
    // tensor must have exactly ONE reader and not be a graph output or a fused residual/bias edge —
    // any other reader needs the intermediate materialized, so the chain is left alone.
    void fuseChannelShuffle(Graph &g) {
        std::vector<int> readCount(g.tensors.size(), 0);
        std::vector<int> soleReader(g.tensors.size(), -1);
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            const Node &n = g.nodes[i];
            for (TensorId in: n.inputs)
            {
                if (in != kNoTensor)
                {
                    readCount[in]++;
                    soleReader[in] = (int) i;
                }
            }
            // Fused residual/bias edges are reads outside the inputs list (rewireTensor's contract);
            // counting them keeps a tensor a fused edge consumes from being folded away under it.
            for (TensorId edge: {n.fusedResidual, n.fusedBias})
            {
                if (edge != kNoTensor)
                {
                    readCount[edge]++;
                    soleReader[edge] = (int) i;
                }
            }
        }
        std::set<TensorId> graphOutputs(g.outputs.begin(), g.outputs.end());
        // Index of the single node reading tensor t, or -1 when t has any other reader (a second
        // node, a fused edge, or a graph output) and must stay materialized.
        auto chainReader = [&](TensorId t) -> int {
            if (t == kNoTensor || readCount[t] != 1 || graphOutputs.count(t))
            {
                return -1;
            }
            return soleReader[t];
        };
        std::set<int> remove;
        int           fused = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &split = g.nodes[i]; // Reshape [N,C,(sp...)] -> [N,g,C/g,(sp...)]
            if (split.type != OpType::Reshape || remove.count((int) i) || split.inputs.empty() || split.inputs[0] == kNoTensor)
            {
                continue;
            }
            const Shape &inShape    = g.desc(split.inputs[0]).shape;
            const Shape &splitShape = g.desc(split.outputs[0]).shape;
            if (inShape.size() < 2 || splitShape.size() != inShape.size() + 1)
            {
                continue;
            }
            int64_t groupCount = splitShape[1]; // g
            int64_t groupWidth = splitShape[2]; // C/g
            if (splitShape[0] != inShape[0] || groupCount < 1 || groupWidth < 1 || groupCount * groupWidth != inShape[1])
            {
                continue;
            }
            bool spatialKept = true;
            for (size_t k = 2; k < inShape.size(); ++k)
            {
                spatialKept = spatialKept && splitShape[k + 1] == inShape[k];
            }
            if (!spatialKept)
            {
                continue;
            }
            int swapIdx = chainReader(split.outputs[0]);
            if (swapIdx < 0 || remove.count(swapIdx) || g.nodes[swapIdx].type != OpType::Transpose)
            {
                continue;
            }
            Node &swap = g.nodes[swapIdx]; // Transpose swapping axes 1 and 2, identity elsewhere
            if (swap.inputs.empty() || swap.inputs[0] != split.outputs[0])
            {
                continue;
            }
            const auto &perm = swap.attr.getints("perm");
            if (perm.size() != splitShape.size() || perm[0] != 0 || perm[1] != 2 || perm[2] != 1)
            {
                continue; // an absent perm defaults to full reversal, never this pattern
            }
            bool tailIdentity = true;
            for (size_t k = 3; k < perm.size(); ++k)
            {
                tailIdentity = tailIdentity && perm[k] == (int64_t) k;
            }
            if (!tailIdentity)
            {
                continue;
            }
            int mergeIdx = chainReader(swap.outputs[0]);
            if (mergeIdx < 0 || remove.count(mergeIdx) || g.nodes[mergeIdx].type != OpType::Reshape)
            {
                continue;
            }
            Node &merge = g.nodes[mergeIdx]; // Reshape back to the original [N,C,(sp...)]
            if (merge.inputs.empty() || merge.inputs[0] != swap.outputs[0] || g.desc(merge.outputs[0]).shape != inShape)
            {
                continue;
            }
            // The composition is exactly the group-interleave channel permutation: replace the merge
            // Reshape with one ChannelShuffle reading the chain's source. The final output tensor id
            // is unchanged, so downstream consumers need no rewiring.
            Node shuffle;
            shuffle.type    = OpType::ChannelShuffle;
            shuffle.name    = merge.name + "#shuffle";
            shuffle.inputs  = {split.inputs[0]};
            shuffle.outputs = {merge.outputs[0]};
            Attr groupsAttr;
            groupsAttr.kind            = Attr::Int;
            groupsAttr.i               = groupCount;
            shuffle.attr.map["groups"] = groupsAttr;
            g.nodes[mergeIdx]          = shuffle;
            remove.insert((int) i);
            remove.insert(swapIdx);
            fused++;
        }
        if (fused)
        {
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!remove.count((int) i))
                {
                    kept.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(kept);
            VKNN_INFO << "fuseChannelShuffle: fused " << fused << " channel-shuffle chain(s)";
        }
    }

} // namespace vknn
