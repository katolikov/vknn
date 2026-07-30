#include "passes_internal.h"

namespace vknn {

    // Fold BatchNorm(Conv(x)) -> Conv(x) with rescaled weights and bias, so the inference-time BN affine
    // (a fixed per-output-channel scale+shift once mean/var/scale/bias are constants) is absorbed into the
    // Conv and its whole-tensor read+write dispatch disappears. Per output channel oc with
    // a = scale/sqrt(var+eps): W[oc] *= a and bias[oc] = (bias[oc] - mean[oc])*a + shift[oc]; the Conv gains
    // a synthesized bias if it had none. Eligible only when the BN's data input is produced by a Conv whose
    // output feeds ONLY this BN — a shared Conv output cannot be rewritten in place. Models that ship BN
    // already folded (e.g. MobileNetV2) never match, leaving this a no-op for them.
    void foldBatchNorm(Graph &g) {
        int           folded = 0;
        std::set<int> remove;
        // producer[t] = index of the node whose output is tensor t, or -1 for graph inputs/initializers.
        // Lets each BatchNorm find its data producer in O(1) below.
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
            // Precondition for an in-place rewrite: the Conv output must feed ONLY this BN. If any other
            // node (or a graph output) also reads it, scaling the Conv weights would corrupt that path,
            // so leave the BN standing and let the runtime execute it.
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

            // ONNX BatchNormalization inputs are (X, scale, shift, mean, var); here `bias` is the affine
            // shift term, distinct from the Conv bias synthesized below. All four are constant at inference.
            const auto &scale = g.initializers[bn.inputs[1]].f32();
            const auto &bias  = g.initializers[bn.inputs[2]].f32();
            const auto &mean  = g.initializers[bn.inputs[3]].f32();
            const auto &var   = g.initializers[bn.inputs[4]].f32();
            float       eps   = bn.attr.getf("epsilon", 1e-5f);
            // Conv weights are [outC, ...]; perOC is the element count of one output filter, so W.f32()
            // partitions into outC contiguous blocks of perOC scaled by that channel's fold factor a.
            int64_t     outC  = g.desc(conv.inputs[1]).shape[0];
            int64_t     perOC = numElements(g.desc(conv.inputs[1]).shape) / outC;
            HostBuffer &W     = g.initializers[conv.inputs[1]];
            // A Conv with no bias input needs one to receive the folded shift; synthesize a zero-filled
            // [outC] buffer now and register it as an initializer only after the fold succeeds (below).
            TensorId   biasId = (conv.inputs.size() > 2 && conv.inputs[2] != kNoTensor) ? conv.inputs[2] : kNoTensor;
            HostBuffer biasBuf;
            if (biasId == kNoTensor)
            {
                biasBuf.resizeElems(outC, DType::Float32);
            }
            // Bb aliases the existing Conv bias, or the fresh zero buffer — the fold formula is identical
            // either way (a missing bias is just bias 0).
            HostBuffer &Bb = (biasId == kNoTensor) ? biasBuf : g.initializers[biasId];
            for (int64_t oc = 0; oc < outC; ++oc)
            {
                // Fold factor: BN normalizes by sqrt(var+eps) then multiplies by scale, which distributes
                // over the linear Conv as a per-channel weight multiplier.
                float  a = scale[oc] / std::sqrt(var[oc] + eps);
                float *w = W.f32() + oc * perOC;
                for (int64_t k = 0; k < perOC; ++k)
                {
                    w[k] *= a;
                }
                // Shifted bias: the pre-existing Conv bias goes through the same normalize-and-scale as the
                // activations, then the BN shift is added on top.
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
            // The BN node is now redundant: redirect everything that read its output to the Conv output
            // (whose weights/bias already carry the affine), across both node inputs and graph outputs.
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
        // Compact out the folded BN nodes in one pass, preserving the relative order of the survivors so
        // later passes still see a valid topological ordering.
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
