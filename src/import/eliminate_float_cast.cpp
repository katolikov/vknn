#include "passes_internal.h"

namespace vknn {

    // Drop Cast nodes that convert float -> float. Storage precision is uniform across a segment, so a
    // float->float cast is a same-size buffer copy (CastOp is a vkCmdCopyBuffer) — a wasted dispatch,
    // barrier, and full intermediate round-trip. Transformer graphs emit hundreds (RoPE/attention/
    // layernorm chains). A forward dtype propagation, seeded from initializers and the float graph
    // inputs, gates the removal strictly to a float input and a float ONNX target so genuine
    // int<->float casts (shape / index paths) are left intact.
    void eliminateFloatCast(Graph &g) {
        // ONNX TensorProto.DataType codes for the float element types a Cast can target.
        constexpr int64_t kOnnxFloat    = 1;
        constexpr int64_t kOnnxFloat16  = 10;
        constexpr int64_t kOnnxDouble   = 11;
        auto              onnxToIsFloat = [](int64_t to) {
            return to == kOnnxFloat || to == kOnnxFloat16 || to == kOnnxDouble;
        };
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
        // Parallel per-tensor dtype lattice: dt[id] is meaningful only where known[id] is set. The
        // Float32 fill is an inert placeholder for not-yet-inferred tensors — every read below is
        // guarded by known[], so a tensor whose dtype is never resolved is treated as unknown, not
        // float, and its consumer casts are left in place.
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
            // Seed each input with its DECLARED dtype, not a blanket float. Integer inputs (token ids,
            // position_ids, attention_mask) are int64: their int->float Cast is a genuine conversion and
            // must survive, or the raw integer bytes reach a float consumer (e.g. the RoPE position feeding
            // the rotary MatMul) and read back as a near-zero denormal.
            DType d = g.tensors[id].dtype;
            // A uint8/int8 image input is converted to compute-precision float at the device boundary
            // (its device buffer is float), so a Cast(uint8/int8 -> float) reading it is a redundant
            // same-data copy (CastOp = vkCmdCopyBuffer over the C=4-padded plane). Seed it as float so
            // that cast is dropped; int64/int32 index inputs keep their type (their int->float cast is a
            // genuine conversion).
            if (d == DType::UInt8 || d == DType::Int8)
            {
                d = DType::Float32;
            }
            setk(id, d);
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
                // A Cast pins its output to the target's float-ness. Int64 here is a representative
                // "some integer type" marker, not a width claim: the removal test below only asks
                // isFloat(), so any non-float dtype is interchangeable.
                out = onnxToIsFloat(nd.attr.geti("to", 1)) ? DType::Float32 : DType::Int64;
            } else if (nd.type == OpType::Equal)
            {
                out = DType::Int32; // boolean result, not float
            } else if (nd.type == OpType::TopK)
            {
                // Per-output dtypes: values (output 0) carry the data input's dtype; indices
                // (output 1) are int64, so a Cast-to-float of the indices is a genuine int->float
                // cast and must be kept.
                if (nd.outputs.size() > 1)
                {
                    setk(nd.outputs[1], DType::Int64);
                }
                TensorId pin = nd.inputs.empty() ? kNoTensor : nd.inputs[0];
                if (pin != kNoTensor && pin < (TensorId) known.size() && known[pin])
                {
                    setk(nd.outputs[0], dt[pin]);
                }
                continue;
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
