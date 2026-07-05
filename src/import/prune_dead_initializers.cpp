#include "passes_internal.h"

namespace vknn {

    // Right-shift turning a byte count into mebibytes for the human-readable diagnostic.
    constexpr unsigned kBytesToMiBShift = 20;

    /// Drop initializer payloads no live tensor references.
    ///
    /// Const-folding materializes every intermediate of a folded chain (and an unbounded Cast fold
    /// copies whole weight tensors), so after the fold and rewiring passes a transformer import can
    /// carry gigabytes of orphaned payloads — a folded meshgrid's Expand/Unsqueeze steps, or the fp32
    /// original of every Cast-wrapped weight — which would otherwise all be serialized into the .vxm
    /// and uploaded at load.
    ///
    /// Reachability is defined by the final graph shape: a payload survives iff it is a graph
    /// input/output or feeds some node through a node input, a fused-residual edge, or a fused-bias
    /// edge (the fused edges are not always in the node's inputs list, so they are collected
    /// separately). Precondition: fold/fusion/rewiring have finished, so the reference set is stable.
    /// Postcondition: every retained entry in g.initializers is reachable, and each dropped tensor's
    /// desc has isInitializer cleared so it is no longer treated as a constant elsewhere.
    void pruneDeadInitializers(Graph &g) {
        std::set<TensorId> referenced(g.outputs.begin(), g.outputs.end());
        referenced.insert(g.inputs.begin(), g.inputs.end());
        for (const auto &nd: g.nodes)
        {
            for (TensorId in: nd.inputs)
            {
                if (in != kNoTensor)
                {
                    referenced.insert(in);
                }
            }
            if (nd.fusedResidual != kNoTensor)
            {
                referenced.insert(nd.fusedResidual);
            }
            if (nd.fusedBias != kNoTensor)
            {
                referenced.insert(nd.fusedBias);
            }
        }
        size_t freed = 0;
        int    count = 0;
        for (auto it = g.initializers.begin(); it != g.initializers.end();)
        {
            if (!referenced.count(it->first))
            {
                freed += it->second.bytes.size();
                g.desc(it->first).isInitializer = false;
                it                              = g.initializers.erase(it);
                count++;
            } else
            {
                ++it;
            }
        }
        if (count)
        {
            VKNN_INFO << "pruneDeadInitializers: dropped " << count << " orphaned payload(s), " << (freed >> kBytesToMiBShift) << " MB";
        }
    }

} // namespace vknn
