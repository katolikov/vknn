#include "passes_internal.h"

namespace vknn {

    void foldBatchNorm(Graph &g) {
        // Fold BN(Conv(x)) -> Conv with adjusted weights/bias. MobileNetV2 ships BN pre-folded,
        // so this is typically a no-op here, but implemented for correctness on other models.
        int           folded = 0;
        std::set<int> remove;
        // map tensor -> producing node index
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

        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &bn = g.nodes[i];
            if (bn.type != OpType::BatchNorm)
            {
                continue;
            }
            int pi = producer[bn.inputs[0]];
            if (pi < 0 || g.nodes[pi].type != OpType::Conv)
            {
                continue;
            }
            Node &conv = g.nodes[pi];
            // only fold if conv output feeds only this BN
            int consumers = 0;
            for (auto &nn: g.nodes)
            {
                for (TensorId in: nn.inputs)
                {
                    if (in == conv.outputs[0])
                    {
                        consumers++;
                    }
                }
            }
            if (consumers != 1)
            {
                continue;
            }

            const auto &scale = g.initializers[bn.inputs[1]].f32();
            const auto &bias  = g.initializers[bn.inputs[2]].f32();
            const auto &mean  = g.initializers[bn.inputs[3]].f32();
            const auto &var   = g.initializers[bn.inputs[4]].f32();
            float       eps   = bn.attr.getf("epsilon", 1e-5f);
            int64_t     outC  = g.desc(conv.inputs[1]).shape[0];
            int64_t     perOC = numElements(g.desc(conv.inputs[1]).shape) / outC;
            HostBuffer &W     = g.initializers[conv.inputs[1]];
            // ensure bias exists
            TensorId   biasId = (conv.inputs.size() > 2 && conv.inputs[2] != kNoTensor) ? conv.inputs[2] : kNoTensor;
            HostBuffer biasBuf;
            if (biasId == kNoTensor)
            {
                biasBuf.resizeElems(outC, DType::Float32);
            }
            HostBuffer &Bb = (biasId == kNoTensor) ? biasBuf : g.initializers[biasId];
            for (int64_t oc = 0; oc < outC; ++oc)
            {
                float  a = scale[oc] / std::sqrt(var[oc] + eps);
                float *w = W.f32() + oc * perOC;
                for (int64_t k = 0; k < perOC; ++k)
                {
                    w[k] *= a;
                }
                Bb.f32()[oc] = (Bb.f32()[oc] - mean[oc]) * a + bias[oc];
            }
            if (biasId == kNoTensor)
            {
                TensorDesc d;
                d.name             = conv.name + "_bias";
                d.shape            = {outC};
                d.isInitializer    = true;
                TensorId nb        = g.addTensor(d);
                g.initializers[nb] = std::move(biasBuf);
                if (conv.inputs.size() < 3)
                {
                    conv.inputs.resize(3, kNoTensor);
                }
                conv.inputs[2] = nb;
            }
            // rewire: BN consumers now read conv output
            TensorId bnOut = bn.outputs[0], convOut = conv.outputs[0];
            for (auto &nn: g.nodes)
            {
                for (TensorId &in: nn.inputs)
                {
                    if (in == bnOut)
                    {
                        in = convOut;
                    }
                }
            }
            for (TensorId &go: g.outputs)
            {
                if (go == bnOut)
                {
                    go = convOut;
                }
            }
            remove.insert((int) i);
            folded++;
        }
        if (folded)
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
            VKNN_INFO << "foldBatchNorm: folded " << folded << " BN node(s) into Conv";
        }
    }


} // namespace vknn
