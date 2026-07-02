#include "passes_internal.h"

namespace vknn {

    // Drop initializer payloads no node references. Const-folding materializes every intermediate
    // of a folded chain (and an unbounded Cast fold copies whole weight tensors), so after the fold
    // and rewiring passes a transformer import can carry gigabytes of orphaned payloads — a folded
    // meshgrid's Expand/Unsqueeze steps, or the fp32 original of every Cast-wrapped weight — which
    // would otherwise all be serialized into the .vxm and uploaded at load.
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
            VKNN_INFO << "pruneDeadInitializers: dropped " << count << " orphaned payload(s), " << (freed >> 20) << " MB";
        }
    }

} // namespace vknn
