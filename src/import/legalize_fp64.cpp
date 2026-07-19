// fp64 legalization: confine real double-precision tensors to the ops that compute in fp64, and
// insert an explicit narrowing Cast(fp64->fp32) on every edge into an op that does not.
//
// The engine computes the numerically-sensitive SVD / camera-head path (Cast, Det, Unary, Transpose,
// and the metadata reshapes) in genuine fp64; every other op computes in fp32. This pass runs after
// shape/dtype inference and after the fusion passes (which already skip fp64 -- pwFloatDtype excludes
// it), so an fp64 tensor that would otherwise reach an fp32-only kernel and be reinterpreted as fp32
// gets a real Cast to fp32 in front of that consumer instead. The narrowing is therefore visible in
// the graph (a Cast node), never a silent misread, and one Cast per fp64 tensor is shared by all its
// fp32 consumers. A fp64 tensor consumed only by fp64-capable ops (and fp64 graph outputs) is left
// untouched and keeps full precision end to end.
#include "passes_internal.h"
#include <map>

namespace vknn {

    namespace {
        // Ops whose CPU kernel computes in real fp64 for a Float64 input (so an fp64 tensor may flow
        // straight into them). Kept in sync with the fp64 paths in the CPU backend: Cast (the
        // fp32<->fp64 bridge), Det (fp64 determinant), Unary (fp64 activations incl. Sign), Binary
        // (fp64 elementwise arithmetic), Transpose (dtype-preserving gather), and the metadata reshapes
        // that copyAs relocates byte-for-byte. Every other op is narrowed to fp32 by an explicit Cast.
        bool fp64CapableOp(OpType t) {
            switch (t)
            {
                case OpType::Cast:
                case OpType::Det:
                case OpType::Unary:
                case OpType::Binary:
                case OpType::Add:
                case OpType::Identity:
                case OpType::Reshape:
                case OpType::Flatten:
                case OpType::Squeeze:
                case OpType::Unsqueeze:
                case OpType::Transpose:
                    return true;
                default:
                    return false;
            }
        }
        constexpr int64_t kOnnxFloat = 1; // ONNX TensorProto.DataType FLOAT
    } // namespace

    void legalizeFp64(Graph &g) {
        std::map<TensorId, TensorId> narrowed; // one fp32 narrowing per fp64 tensor, shared by consumers
        const size_t                 original = g.nodes.size();
        for (size_t ni = 0; ni < original; ++ni)
        {
            if (fp64CapableOp(g.nodes[ni].type))
            {
                continue; // this op reads fp64 directly
            }
            for (size_t k = 0; k < g.nodes[ni].inputs.size(); ++k)
            {
                const TensorId in = g.nodes[ni].inputs[k];
                if (in == kNoTensor || g.desc(in).dtype != DType::Float64)
                {
                    continue;
                }
                auto     it = narrowed.find(in);
                TensorId f32id;
                if (it != narrowed.end())
                {
                    f32id = it->second;
                } else
                {
                    TensorDesc d;
                    d.name  = g.desc(in).name + "_fp32";
                    d.shape = g.desc(in).shape;
                    d.dtype = DType::Float32;
                    f32id   = g.addTensor(d);
                    Node cast;
                    cast.type    = OpType::Cast;
                    cast.name    = g.desc(in).name + "_narrow_fp32";
                    cast.inputs  = {in};
                    cast.outputs = {f32id};
                    Attr a;
                    a.kind                = Attr::Int;
                    a.i                   = kOnnxFloat;
                    cast.attr.map["to"]   = a;
                    g.nodes.push_back(cast); // topoSort() below restores a valid execution order
                    narrowed[in] = f32id;
                }
                g.nodes[ni].inputs[k] = f32id;
            }
        }
        if (!narrowed.empty())
        {
            g.topoSort();
        }
    }

} // namespace vknn
