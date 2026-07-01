#include "passes_internal.h"

namespace vknn {

    void fuseResidualAdd(Graph &g) {
        // Fuse  out = Add(pointwise_conv(x), residual)  into the conv's epilogue, so the conv writes
        // conv+residual directly (saves the Add's full read+write). Only for 1x1 stride-1 pad-0 group-1
        // convs (the ones our conv1x1/split-K kernels run and support a residual on).
        std::vector<int> producer(g.tensors.size(), -1);
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
        auto convEligible = [&](const Node &c) {
            if (c.type != OpType::Conv || c.fusedResidual != kNoTensor)
            {
                return false;
            }
            auto ints = [&](const char *k, std::vector<int64_t> d) {
                const auto &v = c.attr.getints(k);
                return v.empty() ? d : v;
            };
            auto k = ints("kernel_shape", {1, 1}), s = ints("strides", {1, 1});
            auto p = ints("pads", {0, 0, 0, 0});
            return c.attr.geti("group", 1) == 1 && k[0] == 1 && k[1] == 1 && s[0] == 1 && s[1] == 1 && p[0] == 0 && p[1] == 0 && p[2] == 0 && p[3] == 0;
        };
        std::set<int> remove;
        int           fused = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &add = g.nodes[i];
            if (add.type != OpType::Add || add.inputs.size() != 2)
            {
                continue;
            }
            int      p0 = producer[add.inputs[0]], p1 = producer[add.inputs[1]];
            int      ci       = -1;
            TensorId residual = kNoTensor;
            if (p0 >= 0 && convEligible(g.nodes[p0]))
            {
                ci       = p0;
                residual = add.inputs[1];
            } else if (p1 >= 0 && convEligible(g.nodes[p1]))
            {
                ci       = p1;
                residual = add.inputs[0];
            }
            if (ci < 0)
            {
                continue;
            }
            // the conv output must feed only this Add
            int consumers = 0;
            for (auto &nn: g.nodes)
            {
                for (TensorId in: nn.inputs)
                {
                    if (in == g.nodes[ci].outputs[0])
                    {
                        consumers++;
                    }
                }
            }
            for (TensorId go: g.outputs)
            {
                if (go == g.nodes[ci].outputs[0])
                {
                    consumers++;
                }
            }
            if (consumers != 1)
            {
                continue;
            }
            Node &conv         = g.nodes[ci];
            conv.fusedResidual = residual;
            conv.inputs.push_back(residual); // keep it live for DCE / buffer allocation / scheduling
            if (conv.fusedAct == ActType::None)
            { // carry any activation that was folded into the Add
                conv.fusedAct = add.fusedAct;
                conv.actLo    = add.actLo;
                conv.actHi    = add.actHi;
            }
            TensorId addOut = add.outputs[0], convOut = conv.outputs[0];
            for (auto &nn: g.nodes)
            {
                for (TensorId &in: nn.inputs)
                {
                    if (in == addOut)
                    {
                        in = convOut;
                    }
                }
            }
            for (TensorId &go: g.outputs)
            {
                if (go == addOut)
                {
                    go = convOut;
                }
            }
            remove.insert((int) i);
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
            VKNN_INFO << "fuseResidualAdd: fused " << fused << " residual Add(s) into Conv";
        }
    }

} // namespace vknn
