#include "passes_internal.h"

namespace vknn {

    void eliminateDeadNodes(Graph &g) {
        std::set<TensorId> live(g.outputs.begin(), g.outputs.end());
        bool               changed = true;
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
        // propagate liveness backward
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
