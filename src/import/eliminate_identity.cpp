#include "passes_internal.h"

namespace vknn {

    void eliminateIdentity(Graph &g) {
        // Drop Identity nodes by pointing their consumers (and any graph output) straight at the input.
        std::set<int> remove;
        int           n = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &id = g.nodes[i];
            if (id.type != OpType::Identity)
            {
                continue;
            }
            if (id.inputs.empty() || id.outputs.empty())
            {
                continue;
            }
            TensorId in = id.inputs[0], out = id.outputs[0];
            for (auto &nn: g.nodes)
            {
                for (TensorId &x: nn.inputs)
                {
                    if (x == out)
                    {
                        x = in;
                    }
                }
            }
            for (TensorId &go: g.outputs)
            {
                if (go == out)
                {
                    go = in;
                }
            }
            remove.insert((int) i);
            ++n;
        }
        if (n)
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
            VKNN_INFO << "eliminateIdentity: removed " << n << " Identity node(s)";
        }
    }

} // namespace vknn
