#include "passes_internal.h"

namespace vknn {

    void fuseActivations(Graph &g) {
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
        std::set<int> remove;
        int           fused = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &act = g.nodes[i];
            if (act.type != OpType::Clip && act.type != OpType::Relu)
            {
                continue;
            }
            int pi = producer[act.inputs[0]];
            if (pi < 0)
            {
                continue;
            }
            Node &prod = g.nodes[pi];
            if (prod.type != OpType::Conv && prod.type != OpType::Gemm && prod.type != OpType::Add)
            {
                continue;
            }
            if (prod.fusedAct != ActType::None)
            {
                continue;
            }
            // producer output must feed only this activation
            int consumers = 0;
            for (auto &nn: g.nodes)
            {
                for (TensorId in: nn.inputs)
                {
                    if (in == prod.outputs[0])
                    {
                        consumers++;
                    }
                }
            }
            for (TensorId go: g.outputs)
            {
                if (go == prod.outputs[0])
                {
                    consumers++;
                }
            }
            if (consumers != 1)
            {
                continue;
            }

            if (act.type == OpType::Relu)
            {
                prod.fusedAct = ActType::Relu;
            } else
            {
                float lo = 0, hi = 6; // default relu6
                if (act.inputs.size() > 1 && act.inputs[1] != kNoTensor && g.isInitializer(act.inputs[1]))
                {
                    lo = g.initializers[act.inputs[1]].f32()[0];
                }
                if (act.inputs.size() > 2 && act.inputs[2] != kNoTensor && g.isInitializer(act.inputs[2]))
                {
                    hi = g.initializers[act.inputs[2]].f32()[0];
                }
                if (act.attr.has("min"))
                {
                    lo = act.attr.getf("min", lo);
                }
                if (act.attr.has("max"))
                {
                    hi = act.attr.getf("max", hi);
                }
                prod.fusedAct = (lo == 0.f && hi == 6.f) ? ActType::Relu6 : ActType::Clip;
                prod.actLo    = lo;
                prod.actHi    = hi;
            }
            // rewire consumers of act output to producer output
            TensorId actOut = act.outputs[0], prodOut = prod.outputs[0];
            for (auto &nn: g.nodes)
            {
                for (TensorId &in: nn.inputs)
                {
                    if (in == actOut)
                    {
                        in = prodOut;
                    }
                }
            }
            for (TensorId &go: g.outputs)
            {
                if (go == actOut)
                {
                    go = prodOut;
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
            VKNN_INFO << "fuseActivations: fused " << fused << " activation(s) into Conv/Gemm";
        }
    }


} // namespace vknn
