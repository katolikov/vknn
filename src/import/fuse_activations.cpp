#include "passes_internal.h"

namespace vknn {

    /// Fold a trailing Relu / Clip into the compute op that produces its input, recording the clamp on
    /// the producer's `fusedAct` epilogue so the activation runs in the fp32 accumulator and the tensor
    /// is stored once — removing the activation's whole-tensor read + write and one fp16 requantization.
    /// Eligible only when the activation follows a Conv, Gemm, or Add that has no epilogue yet and whose
    /// output feeds nothing but this activation (checked against every node input AND the graph outputs,
    /// so a fused-away tensor can never still be consumed). Clip maps to Relu6 when its bounds are the
    /// [0, 6] default, otherwise to a general Clip carrying `actLo`/`actHi`; Relu maps to Relu.
    /// Postcondition: each fused activation node is dropped and its consumers are rewired to read the
    /// producer's output directly, leaving the graph semantically identical.
    void fuseActivations(Graph &g) {
        // producer[t] = index of the node that writes tensor t, or -1 for graph inputs / initializers.
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
            // Fusing rewrites the producer's output in place, so it is safe only when that tensor feeds
            // nothing but this activation: count every node input that reads it, plus a use as a graph
            // output (which must keep observing the pre-activation value), and require exactly one.
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
                // Clip bounds default to [0, 6] (Relu6) when neither is supplied. min/max arrive either
                // as initializer inputs (ONNX opset >= 11) or as attributes (older opsets); attributes
                // are applied last so an explicit attribute wins over an absent/initializer default.
                float lo = 0, hi = 6;
                // A bound is a scalar initializer; read element [0] only when the payload actually
                // holds it, so a rank-0 tensor that resolved to an empty buffer keeps the default
                // rather than dereferencing a null host pointer.
                if (act.inputs.size() > 1 && act.inputs[1] != kNoTensor && g.isInitializer(act.inputs[1]) && !g.initializers[act.inputs[1]].bytes.empty())
                {
                    lo = g.initializers[act.inputs[1]].f32()[0];
                }
                if (act.inputs.size() > 2 && act.inputs[2] != kNoTensor && g.isInitializer(act.inputs[2]) && !g.initializers[act.inputs[2]].bytes.empty())
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
