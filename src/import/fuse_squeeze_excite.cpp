#include "passes_internal.h"

namespace vknn {

    void fuseSqueezeExcite(Graph &g) {
        // Collapse the Squeeze-Excite scale chain GlobalAvgPool -> Conv1x1(+relu) -> Conv1x1 ->
        // HardSigmoid into ONE kFusedSE node that emits the channel scale. The following
        // channel-broadcast Mul (scale * feature) is left intact. MobileNetV3 has ~11 of these tiny
        // multi-dispatch chains.
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
            Node &hs = g.nodes[i];
            if (hs.type != OpType::Unary || (UnaryType) hs.subOp != UnaryType::HardSigmoid)
            {
                continue;
            }
            int p2 = producer[hs.inputs[0]];
            if (p2 < 0 || g.nodes[p2].type != OpType::Conv || !single(g.nodes[p2].outputs[0]))
            {
                continue;
            }
            Node &conv2 = g.nodes[p2];
            int   p1    = producer[conv2.inputs[0]];
            if (p1 < 0 || g.nodes[p1].type != OpType::Conv || !single(g.nodes[p1].outputs[0]))
            {
                continue;
            }
            Node &conv1 = g.nodes[p1];
            if (conv1.fusedAct != ActType::Relu)
            {
                continue;
            }
            // Require a GlobalAvgPool feeding conv1, but KEEP it (its reduction is parallel). We fuse only
            // the tiny [N,C,1,1] middle: conv1(relu)->conv2->hardsigmoid -> one kernel that reads the
            // pooled avg directly. Fusing the pool into one workgroup regressed; this keeps GAP + Mul
            // parallel.
            int pg = producer[conv1.inputs[0]];
            if (pg < 0 || g.nodes[pg].type != OpType::GlobalAvgPool)
            {
                continue;
            }
            TensorId avg  = conv1.inputs[0]; // the pooled [N,C,1,1] tensor
            auto     bias = [](const Node &c) {
                return c.inputs.size() > 2 ? c.inputs[2] : kNoTensor;
            };
            Node se;
            se.type    = OpType::FusedSE;
            se.name    = conv1.name + "#se";
            se.inputs  = {avg, conv1.inputs[1], bias(conv1), conv2.inputs[1], bias(conv2)};
            se.outputs = {hs.outputs[0]}; // the scale tensor (Mul still consumes it)
            se.actLo   = hs.actLo;        // hardsigmoid alpha/beta
            se.actHi   = hs.actHi;
            g.nodes[i] = se;   // replace hardsigmoid node with the fused node
            remove.insert(p1); // remove conv1 + conv2 (gap stays)
            remove.insert(p2);
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
            VKNN_INFO << "fuseSqueezeExcite: fused " << fused << " SE chain(s)";
        }
    }


} // namespace vknn
