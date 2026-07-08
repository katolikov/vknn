#include "passes_internal.h"
#include <cstring>
#include <map>

namespace vknn {

    namespace {

        // First element of an initializer, read through the tensor's dtype and widened to double.
        // @returns True when the payload holds at least one element of a decodable dtype.
        bool initializerScalar(const Graph &g, TensorId id, double &out) {
            auto it = g.initializers.find(id);
            if (it == g.initializers.end())
            {
                return false;
            }
            const HostBuffer &hb = it->second;
            switch (g.desc(id).dtype)
            {
                case DType::Float32:
                    if (hb.bytes.size() < 4)
                    {
                        return false;
                    }
                    out = (double) hb.f32()[0];
                    return true;
                case DType::Int64:
                    if (hb.bytes.size() < 8)
                    {
                        return false;
                    }
                    out = (double) hb.i64()[0];
                    return true;
                case DType::Int32: {
                    if (hb.bytes.size() < 4)
                    {
                        return false;
                    }
                    int32_t v;
                    std::memcpy(&v, hb.bytes.data(), 4);
                    out = (double) v;
                    return true;
                }
                case DType::UInt8: // ONNX BOOL imports as UInt8 normalized to 0/1
                case DType::Int8:
                    if (hb.bytes.empty())
                    {
                        return false;
                    }
                    out = (double) hb.bytes.data()[0];
                    return true;
                default:
                    return false;
            }
        }

        // True when the Dropout's training_mode input (inputs[2]) is absent or a constant false —
        // ONNX inference mode, in which the op passes its data input through unchanged. The
        // constant may be an initializer or a not-yet-folded Constant node's `value` attribute; a
        // runtime tensor (or an undecodable payload) is not provably false and yields false.
        bool trainingModeOff(const Graph &g, const std::map<TensorId, const Node *> &producer, const Node &nd) {
            if (nd.inputs.size() < 3 || nd.inputs[2] == kNoTensor)
            {
                return true;
            }
            TensorId tm = nd.inputs[2];
            double   v  = 0;
            if (initializerScalar(g, tm, v))
            {
                return v == 0.0;
            }
            if (auto it = producer.find(tm); it != producer.end() && it->second->type == OpType::Constant)
            {
                auto vit = it->second->attr.map.find("value");
                if (vit != it->second->attr.map.end())
                {
                    const Attr &a = vit->second;
                    if (a.kind == Attr::Ints && !a.ints.empty())
                    {
                        return a.ints[0] == 0;
                    }
                    if (a.kind == Attr::Floats && !a.floats.empty())
                    {
                        return a.floats[0] == 0.0f;
                    }
                }
            }
            return false;
        }

        // True when any node input, fused edge, or graph output reads tensor `t`.
        bool tensorConsumed(const Graph &g, TensorId t) {
            for (const auto &n: g.nodes)
            {
                for (TensorId x: n.inputs)
                {
                    if (x == t)
                    {
                        return true;
                    }
                }
                if (n.fusedResidual == t || n.fusedBias == t)
                {
                    return true;
                }
            }
            for (TensorId go: g.outputs)
            {
                if (go == t)
                {
                    return true;
                }
            }
            return false;
        }

    } // namespace

    // Drop inference-mode ONNX Dropout nodes, which pass their data input through unchanged (the
    // ratio input only parameterizes training-mode sampling). A node is removed when its
    // training_mode input is absent or a constant false AND its mask output is absent or
    // unconsumed; every consumer reference to the data output (node inputs, fused edges, graph
    // outputs) is rewired to the Dropout input via rewireTensor, so the tensor is read from its
    // original producer. A Dropout that is not provably inference-mode, or whose mask output is
    // consumed, stays in place — the pass never fabricates a mask — and fails backend planning as
    // unsupported. Removed nodes are dropped in a single compaction pass that preserves the
    // surviving node order (later passes depend on visitation order).
    void eliminateDropout(Graph &g) {
        std::map<TensorId, const Node *> producer;
        for (const auto &nn: g.nodes)
        {
            for (TensorId o: nn.outputs)
            {
                if (o != kNoTensor)
                {
                    producer[o] = &nn;
                }
            }
        }
        std::set<int> remove;
        int           n = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &dp = g.nodes[i];
            if (dp.type != OpType::Dropout)
            {
                continue;
            }
            if (dp.inputs.empty() || dp.inputs[0] == kNoTensor || dp.outputs.empty() || dp.outputs[0] == kNoTensor)
            {
                // A malformed Dropout with no data input or no data output has no source tensor to
                // redirect consumers to; leaving it intact is safe and avoids fabricating an edge.
                continue;
            }
            if (!trainingModeOff(g, producer, dp))
            {
                VKNN_WARN << "Dropout " << dp.name << " kept: training_mode is not a constant false (unsupported downstream)";
                continue;
            }
            if (dp.outputs.size() > 1 && dp.outputs[1] != kNoTensor && tensorConsumed(g, dp.outputs[1]))
            {
                VKNN_WARN << "Dropout " << dp.name << " kept: mask output is consumed (unsupported downstream)";
                continue;
            }
            rewireTensor(g, dp.outputs[0], dp.inputs[0]);
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
            VKNN_INFO << "eliminateDropout: removed " << n << " Dropout node(s)";
        }
    }

} // namespace vknn
