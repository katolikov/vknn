#include "passes_internal.h"

namespace vknn {

    // Redirect every reference to tensor `from` so it points at `to`: node inputs, the fused-residual
    // edge (which is not in the inputs list on every op), and graph outputs. Fusion passes that delete a
    // node and fold its output into a producer must use this; rewiring only node.inputs leaves a stale
    // fusedResidual edge dangling at a dead tensor, which crashes a conv residual read.
    void rewireTensor(Graph &g, TensorId from, TensorId to) {
        if (from == to || from == kNoTensor)
        {
            return;
        }
        for (auto &nn: g.nodes)
        {
            for (TensorId &in: nn.inputs)
            {
                if (in == from)
                {
                    in = to;
                }
            }
            if (nn.fusedResidual == from)
            {
                nn.fusedResidual = to;
            }
        }
        for (TensorId &go: g.outputs)
        {
            if (go == from)
            {
                go = to;
            }
        }
    }


    void fuseDwPw(Graph &g) {
        // Fuse depthwise-3x3 conv (D) -> 1x1 project conv (P) into one kFusedDwPw node, so the expanded
        // intermediate (D's output, the block's largest activation) never hits global memory and a
        // dispatch+barrier is removed. Only when D's output feeds ONLY P, D's activation is a plain
        // ActType (Relu/Relu6/Clip/None), and P is a stride-1 pad-0 group-1 pointwise conv.
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
        auto ints = [](const Node &n, const char *k, std::vector<int64_t> d) {
            const auto &v = n.attr.getints(k);
            return v.empty() ? d : v;
        };
        std::set<int> remove;
        int           fused = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &P = g.nodes[i];
            if (P.type != OpType::Conv)
            {
                continue;
            }
            const Shape &pw = g.desc(P.inputs[1]).shape; // [Cout, Cin, 1, 1]
            if (pw.size() != 4 || pw[2] != 1 || pw[3] != 1)
            {
                continue;
            }
            if (P.attr.geti("group", 1) != 1)
            {
                continue;
            }
            auto poolStrides = ints(P, "strides", {1, 1}), poolPads = ints(P, "pads", {0, 0, 0, 0});
            if (poolStrides[0] != 1 || poolStrides[1] != 1 || poolPads[0] || poolPads[1] || poolPads[2] || poolPads[3])
            {
                continue;
            }
            int prodIdx = producer[P.inputs[0]];
            if (prodIdx < 0 || g.nodes[prodIdx].type != OpType::Conv)
            {
                continue;
            }
            Node &D = g.nodes[prodIdx];
            if (consumers[D.outputs[0]] != 1)
            {
                continue; // D feeds only P
            }
            const Shape &dw        = g.desc(D.inputs[1]).shape; // [C,1,KH,KW]
            NCHW         dx        = NCHW::from(g.desc(D.inputs[0]).shape);
            bool         depthwise = (D.attr.geti("group", 1) == dx.c && dw.size() == 4 && dw[1] == 1);
            if (!depthwise)
            {
                continue;
            }
            // D's activation must be parameterless (None/Relu/Relu6); skip custom Clip and hardswish-dw.
            if (D.fusedAct != ActType::None && D.fusedAct != ActType::Relu && D.fusedAct != ActType::Relu6)
            {
                continue;
            }
            auto bias = [](const Node &c) {
                return c.inputs.size() > 2 ? c.inputs[2] : kNoTensor;
            };
            Node f;
            f.type   = OpType::FusedDwPw;
            f.name   = D.name + "#dwpw";
            f.inputs = {D.inputs[0], D.inputs[1], bias(D), P.inputs[1], bias(P)};
            if (P.fusedResidual != kNoTensor)
            {
                f.inputs.push_back(P.fusedResidual);
                f.fusedResidual = P.fusedResidual;
            }
            f.outputs  = {P.outputs[0]};
            f.attr     = D.attr;               // carry dw strides/pads/kernel for shape inference + kernel
            f.subOp    = (int32_t) D.fusedAct; // dw activation
            f.fusedAct = P.fusedAct;           // project activation
            f.actLo    = P.actLo;
            f.actHi    = P.actHi;
            g.nodes[i] = f;         // replace P with the fused node
            remove.insert(prodIdx); // remove D
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
            VKNN_INFO << "fuseDwPw: fused " << fused << " depthwise+project pair(s)";
        }
    }


} // namespace vknn
