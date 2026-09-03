#include "core/squeeze_excite.h"
#include "passes_internal.h"

namespace vknn {
    namespace {
        // A Conv is a plain 1x1 projection when its weight is [Cout][Cin][1][1], it is ungrouped and it
        // neither pads nor strides (on the [N,C,1,1] pooled tensor any of those would change the shape).
        bool isPointwiseProjection(const Graph &g, const Node &conv) {
            if (conv.type != OpType::Conv || conv.inputs.size() < 2 || conv.inputs[1] == kNoTensor)
            {
                return false;
            }
            const Shape &w = g.desc(conv.inputs[1]).shape;
            if (w.size() != 4 || w[2] != 1 || w[3] != 1 || conv.attr.geti("group", 1) != 1)
            {
                return false;
            }
            for (int64_t p: conv.attr.getints("pads"))
            {
                if (p != 0)
                {
                    return false;
                }
            }
            for (int64_t s: conv.attr.getints("strides"))
            {
                if (s != 1)
                {
                    return false;
                }
            }
            return true;
        }
        bool isUnary(const Node &n, UnaryType u) {
            return n.type == OpType::Unary && (UnaryType) n.subOp == u;
        }
        // HardSigmoid(1/6, 1/2) is the ONNX HardSwish gate: x * HardSigmoid(x) == HardSwish(x).
        bool isHardSwishGate(const Node &n) {
            return isUnary(n, UnaryType::HardSigmoid) && n.actLo == 1.0f / 6.0f && n.actHi == 0.5f;
        }
        TensorId biasOf(const Node &conv) {
            return conv.inputs.size() > 2 ? conv.inputs[2] : kNoTensor;
        }
    } // namespace

    void fuseSqueezeExcite(Graph &g) {
        // Collapse the Squeeze-Excite scale chain
        //     GlobalAvgPool(X) -> Conv1x1 -> activation -> Conv1x1 -> gate
        // into ONE FusedSE node that pools X and emits the channel scale [N,C,1,1]; the broadcast Mul
        // that applies the scale to X stays. The activation is Relu, SiLU or HardSwish, present either
        // as the first conv's fusedAct epilogue (after fuseActivations), as its own Relu / Unary node, or
        // as the Sigmoid-Mul / HardSigmoid(1/6,1/2)-Mul diamond the importer leaves for SiLU / HardSwish.
        // The gate is HardSigmoid (alpha/beta kept in actLo/actHi) or Sigmoid. The match anchors on the
        // gate and walks producers upward.
        //
        // Precondition: every intermediate (the pooled tensor, both conv outputs, the activation) is
        // consumed only inside the chain, so removing the chain orphans no live tensor; the convs are
        // plain 1x1 projections; the widths fit the fused kernel (seShapeFits). A chain failing any of
        // these is left as it is and runs unfused.
        // Postcondition: the gate's node slot holds the FusedSE node (preserving the gate's output id so
        // the downstream Mul still reads it); the pool, both convs and the activation node(s) are gone.
        // Index every tensor by its producing node and its consumer count in one O(nodes*edges) pass,
        // so the upward producer walk and the consumer-count tests below are both O(1) lookups.
        std::vector<int> producer(g.tensors.size(), -1);
        std::vector<int> consumers(g.tensors.size(), 0);
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId o: g.nodes[i].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[o] = (int) i;
                }
            }
            for (TensorId in: g.nodes[i].inputs)
            {
                if (in != kNoTensor)
                {
                    consumers[in]++;
                }
            }
        }
        auto single = [&](TensorId t) {
            return t != kNoTensor && consumers[t] == 1;
        };
        std::set<int> remove;
        int           fused = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            const Node &gate = g.nodes[i];
            if (gate.type != OpType::Unary || seGateCode((UnaryType) gate.subOp) == kSeCodeNone || gate.inputs.empty())
            {
                continue;
            }
            const int p2 = producer[gate.inputs[0]];
            if (p2 < 0 || !isPointwiseProjection(g, g.nodes[p2]) || !single(g.nodes[p2].outputs[0]))
            {
                continue;
            }
            const Node &conv2 = g.nodes[p2];
            // The activation between the convs, in whichever form it survives at this point.
            const TensorId   act   = conv2.inputs[0];
            const int        pa    = producer[act];
            ActType          act1  = ActType::None;
            std::vector<int> actNodes; // the standalone activation node(s) the fusion removes
            int              p1    = -1;
            if (pa < 0)
            {
                continue;
            }
            const Node &a = g.nodes[pa];
            if (a.type == OpType::Conv)
            {
                // Epilogue form: the conv already carries the activation.
                if (seActCode(a.fusedAct) == kSeCodeNone || !single(act))
                {
                    continue;
                }
                act1 = a.fusedAct;
                p1   = pa;
            } else if (a.type == OpType::Relu || isUnary(a, UnaryType::SiLU) || isUnary(a, UnaryType::HardSwish))
            {
                if (!single(act) || a.inputs.empty())
                {
                    continue;
                }
                act1 = a.type == OpType::Relu ? ActType::Relu : isUnary(a, UnaryType::SiLU) ? ActType::SiLU : ActType::HardSwish;
                p1   = producer[a.inputs[0]];
                actNodes.push_back(pa);
                if (p1 < 0 || !single(a.inputs[0]))
                {
                    continue;
                }
            } else if (a.type == OpType::Binary && (BinaryType) a.subOp == BinaryType::Mul && a.inputs.size() == 2)
            {
                // Diamond form: Mul(u, s) with s = Sigmoid(u) (SiLU) or HardSigmoid(1/6, 1/2)(u) (HardSwish).
                if (!single(act))
                {
                    continue;
                }
                int      ps = -1;
                TensorId u  = kNoTensor;
                for (int side = 0; side < 2 && ps < 0; ++side)
                {
                    const TensorId s  = a.inputs[side];
                    const int      pc = producer[s];
                    if (pc >= 0 && (isUnary(g.nodes[pc], UnaryType::Sigmoid) || isHardSwishGate(g.nodes[pc])) && !g.nodes[pc].inputs.empty() &&
                        g.nodes[pc].inputs[0] == a.inputs[1 - side] && single(s))
                    {
                        ps = pc;
                        u  = a.inputs[1 - side];
                    }
                }
                if (ps < 0 || u == kNoTensor || consumers[u] != 2)
                {
                    continue; // u feeds exactly the sigmoid and the Mul
                }
                act1 = isUnary(g.nodes[ps], UnaryType::Sigmoid) ? ActType::SiLU : ActType::HardSwish;
                p1   = producer[u];
                actNodes.push_back(ps);
                actNodes.push_back(pa);
            } else
            {
                continue;
            }
            if (p1 < 0 || !isPointwiseProjection(g, g.nodes[p1]) || (g.nodes[p1].fusedAct != ActType::None && !actNodes.empty()))
            {
                continue; // a standalone activation on top of an epilogue is not one activation
            }
            const Node &conv1 = g.nodes[p1];
            const int   pg    = producer[conv1.inputs[0]];
            if (pg < 0 || g.nodes[pg].type != OpType::GlobalAvgPool || !single(conv1.inputs[0]) || g.nodes[pg].inputs.empty())
            {
                continue;
            }
            const Node &pool = g.nodes[pg];
            const Shape &w1  = g.desc(conv1.inputs[1]).shape; // [Cr][C][1][1]
            const Shape &w2  = g.desc(conv2.inputs[1]).shape; // [C][Cr][1][1]
            const int64_t Cr = w1[0], C = w1[1];
            if (w2[0] != C || w2[1] != Cr || !seShapeFits(C, Cr))
            {
                continue;
            }
            bool overlap = remove.count(p1) || remove.count(p2) || remove.count(pg);
            for (int an: actNodes)
            {
                overlap = overlap || remove.count(an);
            }
            if (overlap)
            {
                continue;
            }
            Node se;
            se.type     = OpType::FusedSE;
            se.name     = conv1.name + "#se";
            se.inputs   = {pool.inputs[0], conv1.inputs[1], biasOf(conv1), conv2.inputs[1], biasOf(conv2)};
            se.outputs  = {gate.outputs[0]}; // the scale tensor (the Mul still consumes it)
            se.fusedAct = act1;
            se.subOp    = gate.subOp; // the gate UnaryType
            se.actLo    = gate.actLo; // HardSigmoid alpha / beta (unused by Sigmoid)
            se.actHi    = gate.actHi;
            g.nodes[i]  = se; // the gate's slot becomes the fused node
            remove.insert(pg);
            remove.insert(p1);
            remove.insert(p2);
            for (int an: actNodes)
            {
                remove.insert(an);
            }
            fused++;
        }
        // Rebuild the node list once, skipping the removed indices: O(nodes) instead of an erase per
        // removed node, and a no-op (no copy) in the common case of a graph with no SE blocks.
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
            VKNN_INFO << "fuseSqueezeExcite: fused " << fused << " SE chain(s)";
        }
    }
} // namespace vknn
