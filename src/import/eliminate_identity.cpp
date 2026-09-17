#include "passes_internal.h"

namespace vknn {

    // Drop ONNX Identity nodes, which copy their input tensor through unchanged. A removal whose result
    // is not a graph output rewires every consumer of the Identity result straight to the Identity input,
    // so the tensor is read from its original producer with no intervening copy. A graph output keeps its
    // own tensor (callers address outputs by the declared name, and two outputs must never share one
    // readback buffer): when the input is an internal tensor written by a node, that producer and every
    // other reader of the input are rewired onto the declared output tensor instead; when the input is a
    // graph input, an initializer or another graph output, the Identity stays as the output's copy.
    // Precondition: the Identity has both an input and an output; degenerate nodes missing either are
    // skipped and left in place rather than rewired against a nonexistent tensor. Postcondition: every
    // removed Identity's result is reachable through the rewired edges, and the removed nodes are dropped
    // in a single compaction pass that preserves the surviving node order (later passes depend on
    // visitation order, so kept nodes must stay in their original sequence).
    void eliminateIdentity(Graph &g) {
        std::set<int> remove;
        int           n       = 0;
        auto          listHas = [](const std::vector<TensorId> &list, TensorId t) {
            return std::find(list.begin(), list.end(), t) != list.end();
        };
        // Rewire every read of `from` (node inputs and fused edges) to `to`.
        auto rewireReads = [&](TensorId from, TensorId to) {
            for (auto &nn: g.nodes)
            {
                for (TensorId &x: nn.inputs)
                {
                    if (x == from)
                    {
                        x = to;
                    }
                }
                for (TensorId *edge: {&nn.fusedResidual, &nn.fusedBias})
                {
                    if (*edge == from)
                    {
                        *edge = to;
                    }
                }
            }
        };
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &id = g.nodes[i];
            if (id.type != OpType::Identity)
            {
                continue;
            }
            if (id.inputs.empty() || id.outputs.empty() || id.inputs[0] == kNoTensor || id.outputs[0] == kNoTensor)
            {
                // A malformed Identity with no input or no output has no source tensor to redirect
                // consumers to; leaving it intact is safe and avoids fabricating an edge.
                continue;
            }
            const TensorId in = id.inputs[0], out = id.outputs[0];
            if (!listHas(g.outputs, out))
            {
                rewireReads(out, in);
                remove.insert((int) i);
                ++n;
                continue;
            }
            // The result is a graph output. Find the node writing the input, if any.
            Node *producer = nullptr;
            for (size_t p = 0; p < g.nodes.size() && !producer; ++p)
            {
                if (p != i && listHas(g.nodes[p].outputs, in))
                {
                    producer = &g.nodes[p];
                }
            }
            const bool internalInput = producer && !g.isInitializer(in) && !g.desc(in).isInput && !listHas(g.inputs, in) && !listHas(g.outputs, in);
            if (!internalInput)
            {
                continue; // a graph input, initializer or graph output source: the Identity is the copy
            }
            for (TensorId &o: producer->outputs)
            {
                if (o == in)
                {
                    o = out;
                }
            }
            rewireReads(in, out); // the Identity's own input included; the node is removed below
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
