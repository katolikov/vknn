#include "passes_internal.h"

namespace vknn {

    // Fold a value round-trip through an integer dtype -- Cast(float -> wide integer) immediately
    // followed by Cast(wide integer -> float) on the same sole-consumer tensor -- into one
    // Unary(Trunc). ONNX Cast to an integer type truncates toward zero, and casting the wide integer
    // back to float is exact, so the pair computes exactly trunc(x). Only the wide integer targets
    // (INT32/INT64/UINT32/UINT64) qualify: they truncate WITHOUT the modulo (INT8) or saturation
    // (UINT8) the narrow targets apply, so the fold is byte-identical to the two standalone Cast
    // kernels -- and to the CPU/GPU int round-trip -- for every finite value.
    //
    // The integer intermediate is a fusion barrier: the pointwise fuser grows a unit only through
    // float members, so a chain split solely by such a round-trip cannot span it and materializes as
    // separate kernels. Rewriting the pair to one Unary(Trunc) -- a float-in/float-out pointwise
    // member -- lets the surrounding pointwise chain fuse across it.
    //
    // Gating mirrors eliminateFloatCast: a forward dtype lattice seeded from the declared input and
    // initializer dtypes proves the FIRST cast reads a float value, so a genuine int->int narrowing or
    // an int shape/index tensor entering the first cast is never folded (its value would not survive
    // fp32 truncation, and an index tensor must not become a float activation). Runs after
    // eliminateFloatCast (which has already dropped the float->float casts), before the pointwise
    // fusion.
    void foldIntRoundtripCast(Graph &g) {
        // ONNX TensorProto.DataType codes.
        auto onnxIsFloat = [](int64_t to) {
            return to == 1 || to == 10 || to == 11; // FLOAT, FLOAT16, DOUBLE
        };
        // Wide integer targets: truncate toward zero with no modulo/saturation, so the round-trip back
        // to float is exactly trunc(x). The narrow targets (UINT8=2, INT8=3, UINT16=4, INT16=5,
        // BOOL=9) wrap or saturate and are deliberately excluded.
        auto onnxIsWideInt = [](int64_t to) {
            return to == 6 || to == 7 || to == 12 || to == 13; // INT32, INT64, UINT32, UINT64
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

        // Forward per-tensor dtype float-ness lattice (mirrors eliminateFloatCast): dt[id] is
        // meaningful only where known[id] is set; an unresolved tensor stays unknown and its casts are
        // left in place.
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
            setk(id, g.tensors[id].dtype);
        }
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
                out = onnxIsFloat(nd.attr.geti("to", 1)) ? DType::Float32 : DType::Int64;
            } else if (nd.type == OpType::Equal)
            {
                out = DType::Int32; // boolean result, not float
            } else if (nd.type == OpType::TopK)
            {
                if (nd.outputs.size() > 1)
                {
                    setk(nd.outputs[1], DType::Int64); // indices
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
                int      pidx = nd.type == OpType::Where ? 1 : 0; // Where: the X branch (input[1])
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

        // Producer / reader maps + graph-output set, built once against the original node order.
        std::vector<std::vector<int>> readers(g.tensors.size());
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId in: g.nodes[i].inputs)
            {
                if (in != kNoTensor && in < (TensorId) readers.size())
                {
                    readers[in].push_back((int) i);
                }
            }
        }
        std::vector<char> isGraphOut(g.tensors.size(), 0);
        for (TensorId o: g.outputs)
        {
            if (o != kNoTensor && o < (TensorId) isGraphOut.size())
            {
                isGraphOut[o] = 1;
            }
        }

        std::set<int> removed;
        int           folded = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            if (removed.count((int) i))
            {
                continue;
            }
            Node &c1 = g.nodes[i];
            if (c1.type != OpType::Cast || c1.inputs.size() != 1 || c1.outputs.size() != 1)
            {
                continue;
            }
            if (!onnxIsWideInt(c1.attr.geti("to", 1)))
            {
                continue;
            }
            TensorId in  = c1.inputs[0];
            TensorId mid = c1.outputs[0];
            if (in == kNoTensor || in >= (TensorId) known.size() || !known[in] || !isFloat(dt[in]))
            {
                continue; // first cast must read a proven float value, not a shape/index int
            }
            // The integer intermediate must exist only to be cast straight back to float: not a graph
            // output, and read by exactly one node (that second cast). Any other reader means the int
            // value is used as an integer and the truncation is not the whole story.
            if (mid == kNoTensor || mid >= (TensorId) isGraphOut.size() || isGraphOut[mid] || readers[mid].size() != 1)
            {
                continue;
            }
            int j = readers[mid][0];
            if (removed.count(j) || j == (int) i)
            {
                continue;
            }
            Node &c2 = g.nodes[j];
            if (c2.type != OpType::Cast || c2.inputs.size() != 1 || c2.outputs.size() != 1 || c2.inputs[0] != mid)
            {
                continue;
            }
            if (!onnxIsFloat(c2.attr.geti("to", 1)))
            {
                continue;
            }
            TensorId out = c2.outputs[0];
            if (out == kNoTensor)
            {
                continue;
            }
            // Repurpose the first cast as Unary(Trunc) producing the second cast's output tensor, and
            // drop the second cast. Consumers already read `out`; the int intermediate is left dead.
            c1.type     = OpType::Unary;
            c1.subOp    = (int32_t) UnaryType::Trunc;
            c1.actLo    = 0.f;
            c1.actHi    = 0.f;
            c1.fusedAct = ActType::None;
            c1.attr.map.clear(); // drop the Cast's "to" (and any opset-19 "saturate"); Unary reads none
            c1.outputs[0] = out;
            removed.insert(j);
            ++folded;
        }

        if (folded)
        {
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!removed.count((int) i))
                {
                    kept.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(kept);
            VKNN_INFO << "foldIntRoundtripCast: folded " << folded << " float->int->float Cast pair(s) into Unary(Trunc)";
        }
    }

} // namespace vknn
