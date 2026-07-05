#include "passes_internal.h"

namespace vknn {

    // Drop ONNX Identity nodes, which copy their input tensor through unchanged. Each removal rewires
    // every consumer reference (node inputs and any graph output naming the Identity result) straight to
    // the Identity input, so the tensor is read from its original producer with no intervening copy.
    // Precondition: the Identity has both an input and an output; degenerate nodes missing either are
    // skipped and left in place rather than rewired against a nonexistent tensor. Postcondition: no
    // Identity node remains reachable through the rewired edges, and the removed nodes are dropped in a
    // single compaction pass that preserves the surviving node order (later passes depend on visitation
    // order, so kept nodes must stay in their original sequence).
    void eliminateIdentity(Graph &g) {
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
                // A malformed Identity with no input or no output has no source tensor to redirect
                // consumers to; leaving it intact is safe and avoids fabricating an edge.
                continue;
            }
            TensorId in = id.inputs[0], out = id.outputs[0];
            // Redirect every consumer of the Identity output (node inputs, then graph outputs) to the
            // Identity input tensor.
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
