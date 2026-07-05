#include "passes_internal.h"

namespace vknn {

    /// Dead-code elimination over the node list: drops every node that cannot reach a graph output.
    ///
    /// A tensor is *live* if it is a graph output or feeds (transitively) into one; a node is live if
    /// any of its outputs is live. The pass seeds liveness from `g.outputs`, propagates it backward
    /// across node input edges to a fixpoint, then keeps only the live nodes (in their original order).
    ///
    /// Precondition: node outputs uniquely produce their tensors and the node list is otherwise
    /// consistent. Postcondition: `g.nodes` contains exactly the nodes reachable from `g.outputs`,
    /// order-preserved; the tensor table and initializers are untouched (pruneDeadInitializers reclaims
    /// tensors that lost their last consumer). Only reads through `Node::inputs` propagate liveness, so
    /// run this AFTER the fusion passes have rewired any folded edges into the input lists.
    void eliminateDeadNodes(Graph &g) {
        // Roots of the backward walk: the graph's declared external outputs are live by definition.
        std::set<TensorId> live(g.outputs.begin(), g.outputs.end());
        bool               changed = true;
        // Reverse map tensor -> producing node. Built but not consulted by the fixpoint below; it holds
        // the last writer of each tensor for callers/diagnostics that share this producer convention.
        std::vector<int>   producer(g.tensors.size(), -1);
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId o: g.nodes[i].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[o] = (int) i;
                }
            }
        }
        // Backward liveness fixpoint: repeatedly, any node with a live output marks its inputs live.
        // Iterates until a full sweep adds nothing new, so liveness reaches every transitive producer
        // regardless of node ordering. Terminates because `live` only grows and is bounded by the tensor
        // count.
        while (changed)
        {
            changed = false;
            for (auto &nd: g.nodes)
            {
                bool nodeLive = false;
                for (TensorId o: nd.outputs)
                {
                    if (o != kNoTensor && live.count(o))
                    {
                        nodeLive = true;
                    }
                }
                if (!nodeLive)
                {
                    continue;
                }
                for (TensorId in: nd.inputs)
                {
                    if (in != kNoTensor && !live.count(in))
                    {
                        live.insert(in);
                        changed = true;
                    }
                }
            }
        }
        // Retention sweep: a node survives iff at least one of its outputs is live. Copies live nodes in
        // their original relative order, so the schedule the later passes see is unchanged.
        std::vector<Node> kept;
        int               removed = 0;
        for (auto &nd: g.nodes)
        {
            bool nodeLive = false;
            for (TensorId o: nd.outputs)
            {
                if (o != kNoTensor && live.count(o))
                {
                    nodeLive = true;
                }
            }
            if (nodeLive)
            {
                kept.push_back(nd);
            } else
            {
                removed++;
            }
        }
        if (removed)
        {
            g.nodes = std::move(kept);
            VKNN_INFO << "eliminateDeadNodes: removed " << removed << " node(s)";
        }
    }

} // namespace vknn
