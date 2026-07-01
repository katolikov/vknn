#include "passes_internal.h"

namespace vknn {

    // Drop Cast nodes that convert float -> float. Storage precision is uniform across a segment, so a
    // float->float cast is a same-size buffer copy (CastOp is a vkCmdCopyBuffer) — a wasted dispatch,
    // barrier, and full intermediate round-trip. Transformer graphs emit hundreds (RoPE/attention/
    // layernorm chains). A forward dtype propagation, seeded from initializers and the float graph
    // inputs, gates the removal strictly to a float input and a float ONNX target so genuine
    // int<->float casts (shape / index paths) are left intact.
    void eliminateFloatCast(Graph &g) {
        auto onnxToIsFloat = [](int64_t to) {
            return to == 1 || to == 10 || to == 11;
        }; // FLOAT/FLOAT16/DOUBLE
        auto isFloat = [](DType d) {
            return d == DType::Float32 || d == DType::Float16;
        };
        // float-result math ops: their output is float regardless of an int-typed input.
        auto floatResult = [](OpType t) {
            switch (t)
            {
                case OpType::Conv:
                case OpType::Gemm:
                case OpType::MatMul:
                case OpType::Einsum:
                case OpType::Softmax:
                case OpType::LayerNorm:
                case OpType::BatchNorm:
                case OpType::Reduce:
                case OpType::GlobalAvgPool:
                case OpType::AvgPool:
                case OpType::Resize:
                case OpType::GridSample:
                case OpType::FusedSE:
                case OpType::FusedDwPw:
                    return true;
                default:
                    return false;
            }
        };
        std::vector<DType> dt(g.tensors.size(), DType::Float32);
        std::vector<char>  known(g.tensors.size(), 0);
        auto               setk = [&](TensorId id, DType d) {
            if (id >= 0 && id < (TensorId) dt.size())
            {
                dt[id]    = d;
                known[id] = 1;
            }
        };
        for (TensorId id = 0; id < (TensorId) g.tensors.size(); ++id)
        {
            if (g.tensors[id].isInitializer)
            {
                setk(id, g.tensors[id].dtype);
            }
        }
        for (TensorId id: g.inputs)
        {
            setk(id, DType::Float32); // model inputs are float (image / intrinsics / ...)
        }
        // Forward pass (nodes are topo-ordered after import): assign each output a dtype.
        for (const Node &nd: g.nodes)
        {
            if (nd.outputs.empty())
            {
                continue;
            }
            DType out;
            if (nd.type == OpType::Shape)
            {
                out = DType::Int64;
            } else if (nd.type == OpType::Cast)
            {
                out = onnxToIsFloat(nd.attr.geti("to", 1)) ? DType::Float32 : DType::Int64;
            } else if (nd.type == OpType::Equal)
            {
                out = DType::Int32; // boolean result, not float
            } else if (floatResult(nd.type))
            {
                out = DType::Float32;
            } else
            {
                // Elementwise / shape-movement ops carry their primary data input's dtype (Where: the X
                // branch, input[1]). Unknown input -> leave the output unknown (conservative).
                int      pidx = nd.type == OpType::Where ? 1 : 0;
                TensorId pin  = (int) nd.inputs.size() > pidx ? nd.inputs[pidx] : kNoTensor;
                if (pin == kNoTensor || pin >= (TensorId) known.size() || !known[pin])
                {
                    continue;
                }
                out = dt[pin];
            }
            for (TensorId o: nd.outputs)
            {
                setk(o, out);
            }
        }
        // Remove float->float casts, redirecting consumers to the cast's input (graph outputs untouched
        // so a named output is never renamed).
        std::set<int> remove;
        int           n = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &c = g.nodes[i];
            if (c.type != OpType::Cast || c.inputs.size() != 1 || c.outputs.size() != 1)
            {
                continue;
            }
            TensorId in = c.inputs[0], out = c.outputs[0];
            if (in == kNoTensor || !onnxToIsFloat(c.attr.geti("to", 1)))
            {
                continue;
            }
            if (in >= (TensorId) known.size() || !known[in] || !isFloat(dt[in]))
            {
                continue;
            }
            bool isGraphOut = false;
            for (TensorId go: g.outputs)
            {
                if (go == out)
                {
                    isGraphOut = true;
                    break;
                }
            }
            if (isGraphOut)
            {
                continue;
            }
            for (auto &nn: g.nodes)
            {
                for (TensorId &x: nn.inputs)
                {
                    if (x == out)
                    {
                        x = in;
                    }
                }
            }
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
            VKNN_INFO << "eliminateFloatCast: removed " << n << " float->float Cast node(s)";
        }
    }


} // namespace vknn
