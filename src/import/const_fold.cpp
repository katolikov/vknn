#include "passes_internal.h"
#include <map>

namespace vknn {

    /// Evaluate every node whose inputs are all known constants on the CPU and replace it with an
    /// initializer holding the result, then drop the node from the graph. Nodes are visited in
    /// topological program order so a folded output feeds the constant set for later nodes in the
    /// same pass. Precondition: inferShapes has run, so shapes referenced by the classifier below
    /// are populated. Postcondition: every folded node is removed from g.nodes, its outputs are
    /// initializers with concrete shape/dtype, and the runtime never needs a backend for it.
    ///
    /// This exists to collapse the shape/index arithmetic that detection and transformer graphs
    /// build dynamically (Shape/Gather/Range/Expand feeding scalar Binary, Where/Equal selects,
    /// RoPE position vectors) into exact CPU-computed constants. Integer index/shape tensors in
    /// particular MUST fold: the GPU stores activations as float, where an int64 value reinterpreted
    /// as float corrupts, so those tensors are folded even when large while float outputs keep a
    /// small size bound to avoid baking a big all-const buffer into the model.
    ///
    /// @return the number of nodes folded and removed.
    int constFold(Graph &g) {
        std::set<TensorId> known;
        for (auto &kv: g.initializers)
        {
            known.insert(kv.first);
        }
        std::vector<RtTensor> pool(g.tensors.size());
        for (size_t i = 0; i < pool.size(); ++i)
        {
            pool[i].id    = (TensorId) i;
            pool[i].shape = g.tensors[i].shape;
            pool[i].dtype = g.tensors[i].dtype;
            if (g.isInitializer((TensorId) i))
            {
                pool[i].host      = g.initializers[i];
                pool[i].hostValid = true;
            }
        }
        ExecContext ctx;
        ctx.pool  = &pool;
        ctx.graph = &g;
        Config cfg;
        ctx.config = &cfg;

        // An INT8/UINT8 initializer keeps its native 1-byte lanes in the graph (materializeInitializers),
        // but the CPU kernels read every non-int64 operand through host.f32(), which on a 1-byte payload
        // reads past the buffer. Widen such an operand's pool entry to integer-valued fp32 host bytes --
        // the values the session's CPU pool loads -- right before a node that reads its values folds.
        // The pool entry is relabeled Float32 so label-sized copies (cpu::copyAs) stay consistent with
        // the widened bytes; `byteLabelOf` keeps the original label, which restoreMovedByteLabels puts
        // back on the result of a node that only moves data. Widening is lazy and per operand, so a
        // packed int4/uint8 weight no value-reading foldable node reads never quadruples in host memory.
        std::map<TensorId, DType> byteLabelOf;
        auto                      widenByteInitializerOperands = [&](const Node &nd) {
            for (TensorId in: nd.inputs)
            {
                if (in == kNoTensor || !pool[in].hostValid || (pool[in].dtype != DType::Int8 && pool[in].dtype != DType::UInt8))
                {
                    continue;
                }
                const DType    label     = pool[in].dtype;
                const size_t   laneCount = pool[in].host.bytes.size(); // one byte per element
                const uint8_t *lanes     = pool[in].host.bytes.data();
                HostBuffer     widened;
                widened.resizeElems((int64_t) laneCount, DType::Float32);
                float *values = widened.f32();
                for (size_t k = 0; k < laneCount; ++k)
                {
                    values[k] = label == DType::Int8 ? (float) (int8_t) lanes[k] : (float) lanes[k];
                }
                pool[in].host   = std::move(widened);
                pool[in].dtype  = DType::Float32;
                byteLabelOf[in] = label;
            }
        };
        // The input slots whose element values a data-movement node copies into its outputs unchanged
        // (the other inputs are shape/index/axis parameters), as [first, end). Empty for a node that
        // computes new values: its result keeps the dtype its kernel writes.
        auto movedDataSlots = [&](const Node &nd) -> std::pair<size_t, size_t> {
            static constexpr size_t kSingleDataOperand = 1; // movement ops reading their values from operand 0 alone
            switch (nd.type)
            {
                case OpType::Reshape:
                case OpType::Squeeze:
                case OpType::Unsqueeze:
                case OpType::Slice:
                case OpType::Transpose:
                case OpType::Expand:
                case OpType::Tile:
                case OpType::Gather:
                    return {0, std::min(kSingleDataOperand, nd.inputs.size())};
                case OpType::Concat:
                    return {0, pwCoreInputs(nd)};
                default:
                    return {0, 0};
            }
        };
        // A data-movement node whose every data operand was a widened INT8/UINT8 initializer of ONE label
        // stores exactly those integer values, so its fp32 result narrows back to 1-byte lanes under that
        // label: the fold yields what the graph declares (QuantizeLinear takes its output type and
        // saturation range from a zero_point's label) at the native 1-byte footprint.
        auto restoreMovedByteLabels = [&](const Node &nd) {
            const auto [firstSlot, endSlot] = movedDataSlots(nd);
            bool  sawData                   = false;
            DType label                     = DType::Float32;
            for (size_t slot = firstSlot; slot < endSlot; ++slot)
            {
                TensorId in = nd.inputs[slot];
                if (in == kNoTensor)
                {
                    continue;
                }
                auto it = byteLabelOf.find(in);
                if (it == byteLabelOf.end() || (sawData && it->second != label))
                {
                    return; // a non-byte or mixed-label operand: the result keeps its kernel dtype
                }
                label   = it->second;
                sawData = true;
            }
            if (!sawData)
            {
                return;
            }
            for (TensorId o: nd.outputs)
            {
                if (o == kNoTensor || pool[o].dtype != DType::Float32)
                {
                    continue;
                }
                RtTensor    &result = pool[o];
                const size_t count  = result.host.bytes.size() / sizeof(float);
                const float *values = result.host.f32();
                HostBuffer   narrowed;
                narrowed.resizeElems((int64_t) count, label);
                uint8_t *lanes = narrowed.bytes.data();
                for (size_t k = 0; k < count; ++k)
                {
                    lanes[k] = label == DType::Int8 ? (uint8_t) (int8_t) values[k] : (uint8_t) values[k];
                }
                result.host  = std::move(narrowed);
                result.dtype = label;
            }
        };

        std::set<int> removeNodes;
        auto          foldable = [&](const Node &nd) {
            switch (nd.type)
            {
                case OpType::Constant:
                    return true;
                case OpType::Shape:
                    return !g.desc(nd.inputs[0]).shape.empty(); // shape known
                // Any op whose every input is a known constant can be evaluated now. This collapses the
                // shape-arithmetic that detection heads (YOLO) build at runtime — Shape/Gather feeding scalar
                // Binary/Add to derive per-level strides — into plain constants, so those ops never need a
                // backend at all (neither CPU nor GPU).
                case OpType::Gather:
                case OpType::Unsqueeze:
                case OpType::Squeeze:
                case OpType::Concat:
                case OpType::Binary:
                case OpType::Add:
                case OpType::Reshape:
                case OpType::Slice:
                case OpType::Transpose:
                case OpType::Cast:
                case OpType::Reduce:
                // Integer-valued elementwise ops: shape/index arithmetic and packed-flag masks must
                // fold exactly on the CPU (an int64 value above the fp32 mantissa cannot survive the
                // GPU float lanes), so they fold unbounded like Binary.
                case OpType::Mod:
                case OpType::BitShift:
                case OpType::BitwiseAnd:
                case OpType::BitwiseOr:
                case OpType::BitwiseXor:
                case OpType::BitwiseNot: {
                    if (nd.inputs.empty())
                    {
                        return false;
                    }
                    for (TensorId in: nd.inputs)
                    {
                        if (in != kNoTensor && !known.count(in))
                        {
                            return false;
                        }
                    }
                    return true;
                }
                // The transformer's dynamic-shape subgraph computes Reshape/Expand/Tile target vectors with
                // float arithmetic that passes through Where/Equal (e.g. Where(Equal(dim,-1), computed, dim))
                // and ConstantOfShape. Fold these when all inputs are constant so the targets become readable
                // initializers — but bound the output size so a large all-const fill/select isn't baked in
                // (its shape is still inferred by inferShapes, and it runs at runtime).
                case OpType::Where:
                case OpType::Equal:
                case OpType::Greater:
                case OpType::GreaterEqual:
                case OpType::Less:
                case OpType::LessEqual:
                case OpType::And: // boolean mask logic: an all-constant mask folds like the compares feeding it
                case OpType::Or:
                case OpType::Xor:
                case OpType::Not:
                case OpType::EyeLike: // identity matrix is constant once the (now-known) shape is fixed
                case OpType::ConstantOfShape: {
                    if (nd.inputs.empty())
                    {
                        return false;
                    }
                    int64_t maxElems = 1;
                    for (TensorId in: nd.inputs)
                    {
                        if (in == kNoTensor)
                        {
                            continue;
                        }
                        if (!known.count(in))
                        {
                            return false;
                        }
                        maxElems = std::max(maxElems, numElements(g.desc(in).shape));
                    }
                    if (nd.type == OpType::ConstantOfShape)
                    {
                        // output size = product of the const shape-vector's values
                        auto it = g.initializers.find(nd.inputs[0]);
                        if (it == g.initializers.end())
                        {
                            return false;
                        }
                        const HostBuffer &hb  = it->second;
                        bool              i64 = g.desc(nd.inputs[0]).dtype == DType::Int64;
                        int64_t           r   = numElements(g.desc(nd.inputs[0]).shape);
                        if (r <= 0)
                        {
                            r = (int64_t) (hb.bytes.size() / (i64 ? 8 : 4));
                        }
                        int64_t prod = 1;
                        for (int64_t i = 0; i < r; ++i)
                        {
                            int64_t dv = i64 ? hb.i64()[i] : (int64_t) hb.f32()[i];
                            prod *= (dv > 0 ? dv : 1);
                        }
                        maxElems = prod;
                    }
                    return maxElems <= (1 << 16);
                }
                // Range folds when start/limit/delta are all constant: its output is an index/position
                // vector that must stay exact (an int64 range corrupts on the GPU float path), so bake
                // it like Expand/Tile, with the same int-aware size bound.
                case OpType::Range: {
                    if (nd.inputs.size() < 3)
                    {
                        return false;
                    }
                    double vals[3];
                    for (int i = 0; i < 3; ++i)
                    {
                        TensorId t = nd.inputs[i];
                        if (t == kNoTensor || !known.count(t))
                        {
                            return false;
                        }
                        auto it  = g.initializers.find(t);
                        bool i64 = g.desc(t).dtype == DType::Int64;
                        if (it == g.initializers.end() || it->second.bytes.size() < (i64 ? 8u : 4u))
                        {
                            return false;
                        }
                        vals[i] = i64 ? (double) it->second.i64()[0] : (double) it->second.f32()[0];
                    }
                    if (vals[2] == 0.0)
                    {
                        return false;
                    }
                    int64_t n     = std::max<int64_t>((int64_t) std::ceil((vals[1] - vals[0]) / vals[2]), 0);
                    DType   dt    = g.desc(nd.inputs[0]).dtype;
                    bool    isInt = dt == DType::Int64 || dt == DType::Int32;
                    return n <= (isInt ? (int64_t(1) << 26) : (int64_t(1) << 18));
                }
                // Expand/Tile of all-constant operands fold too (bounded). Required for integer index/shape
                // tensors such as the RoPE position arange (int64, built via Expand->Add->Reshape->Gather):
                // on the GPU float path the int64 positions corrupt to zeros and the rotary embedding loses
                // all position information, so folding computes the small constant index on the CPU exactly.
                case OpType::Expand:
                case OpType::Tile: {
                    if (nd.inputs.size() < 2)
                    {
                        return false;
                    }
                    for (TensorId in: nd.inputs)
                    {
                        if (in != kNoTensor && !known.count(in))
                        {
                            return false;
                        }
                    }
                    const Shape         &in  = g.desc(nd.inputs[0]).shape;
                    std::vector<int64_t> p   = readI64Param(g, nd, nd.type == OpType::Tile ? "repeats" : "shape", 1);
                    int64_t              out = 1;
                    if (nd.type == OpType::Tile)
                    {
                        for (size_t k = 0; k < in.size(); ++k)
                        {
                            out *= in[k] * std::max<int64_t>(k < p.size() ? p[k] : 1, 1);
                        }
                    } else
                    { // Expand: numpy broadcast of in.shape against the target
                        int rank = (int) std::max(in.size(), p.size());
                        for (int k = 0; k < rank; ++k)
                        {
                            int64_t a = (k >= rank - (int) in.size()) ? in[k - (rank - (int) in.size())] : 1;
                            int64_t b = (k >= rank - (int) p.size()) ? p[k - (rank - (int) p.size())] : 1;
                            out *= std::max<int64_t>(std::max(a, b), 1);
                        }
                    }
                    // Integer (index / shape) tensors must fold even when large: they cannot run on the GPU's
                    // float buffers, where an int64 value reinterpreted as float corrupts to ~0 (e.g. a large
                    // ScatterND upsample meshgrid index would come out all-zeros and scatter everything to
                    // token 0). Float tensors keep the small bound so a large all-const broadcast is not baked
                    // into the model.
                    DType idt   = g.desc(nd.inputs[0]).dtype;
                    bool  isInt = idt == DType::Int64 || idt == DType::Int32;
                    return out > 0 && out <= (isInt ? (int64_t(1) << 26) : (int64_t(1) << 18));
                }
                default:
                    return false;
            }
        };

        for (size_t ni = 0; ni < g.nodes.size(); ++ni)
        {
            Node &nd = g.nodes[ni];
            // Interleave the forward shape rule with folding: a target vector folded earlier in THIS
            // walk resolves this node's shape now, which lets a Shape() later in the walk fold in the
            // same pass. Without this each fold/infer alternation advanced one dependent block per
            // round, so a deep encoder needed dozens of full-graph rounds to converge.
            inferNodeShape(g, nd);
            if (!foldable(nd))
            {
                continue;
            }
            // ensure Shape's input has a shape-only RtTensor
            if (nd.type == OpType::Shape)
            {
                pool[nd.inputs[0]].shape = g.desc(nd.inputs[0]).shape;
            }
            auto op = CpuOpRegistry::instance().create(nd.type);
            if (!op)
            {
                continue;
            }
            if (nd.type != OpType::Shape)
            {
                widenByteInitializerOperands(nd); // Shape reads only the operand's shape, never its payload
            }
            try
            { op->run(nd, ctx); } catch (...)
            { continue; }
            restoreMovedByteLabels(nd);
            // The CPU ops normalize a rank-0 result to [1] (a zero-byte runtime buffer would be the
            // alternative), but on the fold path the TRUE ONNX rank is part of the value: a scalar
            // Gather-of-Shape that keeps the [1] turns the following Unsqueeze into rank 2, a Concat
            // of those into rank 2, and a Where/Equal against a genuine 1-D vector broadcasts the
            // shape VECTOR into a matrix — an Expand target of 257^4 elements instead of [1,1,S,T]
            // (the ORT transformer mask subgraph). A folded output is an INITIALIZER, where an empty
            // shape is the established rank-0 convention (imported scalar initializers carry it and
            // every payload reader recovers the element from the byte size), so restore the true
            // rank here, on both the desc and the pool entry later folds in this walk read.
            bool scalarOut = false;
            switch (nd.type)
            {
                case OpType::Gather:
                    // out rank = data.rank - 1 + index.rank; scalar iff 1-D data and rank-0 index
                    // (the desc-empty form, or the idx_scalar tag markScalarGatherIndices saved
                    // before the index Constant folded).
                    scalarOut = nd.inputs.size() >= 2 && (g.desc(nd.inputs[1]).shape.empty() || nd.attr.geti("idx_scalar", 0) != 0) &&
                                g.desc(nd.inputs[0]).shape.size() == 1;
                    break;
                case OpType::Squeeze: {
                    // Scalar iff every input dim is dropped (explicit axes covering the whole rank,
                    // or no axes over an all-ones shape).
                    const Shape         &in   = g.desc(nd.inputs[0]).shape;
                    std::vector<int64_t> axes = readI64Param(g, nd, "axes", 1);
                    if (in.empty())
                    {
                        scalarOut = true;
                        break;
                    }
                    int kept = 0;
                    for (int64_t k = 0; k < (int64_t) in.size(); ++k)
                    {
                        bool dropK = axes.empty() ? in[(size_t) k] == 1 : false;
                        for (int64_t ax: axes)
                        {
                            if (ax < 0)
                            {
                                ax += (int64_t) in.size();
                            }
                            dropK = dropK || ax == k;
                        }
                        kept += dropK ? 0 : 1;
                    }
                    scalarOut = kept == 0;
                    break;
                }
                // Scalar arithmetic between scalars stays scalar (the mask subgraph's
                // past+seq additions and casts sit between the Gathers and the Unsqueezes).
                case OpType::Add:
                case OpType::Binary:
                case OpType::Cast:
                case OpType::Where:
                case OpType::Equal:
                case OpType::Greater:
                case OpType::GreaterEqual:
                case OpType::Less:
                case OpType::LessEqual:
                case OpType::And:
                case OpType::Or:
                case OpType::Xor:
                case OpType::Not:
                case OpType::Mod:
                case OpType::BitShift:
                case OpType::BitwiseAnd:
                case OpType::BitwiseOr:
                case OpType::BitwiseXor:
                case OpType::BitwiseNot: {
                    scalarOut = !nd.inputs.empty();
                    for (TensorId in: nd.inputs)
                    {
                        if (in != kNoTensor && !g.desc(in).shape.empty())
                        {
                            scalarOut = false;
                        }
                    }
                    break;
                }
                default:
                    break;
            }
            for (TensorId o: nd.outputs)
            {
                if (o == kNoTensor)
                {
                    continue;
                }
                if (scalarOut && numElements(pool[o].shape) == 1)
                {
                    pool[o].shape = Shape {};
                }
                g.initializers[o]       = pool[o].host;
                g.desc(o).isInitializer = true;
                g.desc(o).shape         = pool[o].shape;
                g.desc(o).dtype         = pool[o].dtype;
                known.insert(o);
            }
            removeNodes.insert((int) ni);
        }
        if (!removeNodes.empty())
        {
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!removeNodes.count((int) i))
                {
                    kept.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(kept);
            VKNN_INFO << "constFold: folded " << removeNodes.size() << " shape-path node(s)";
        }
        return (int) removeNodes.size();
    }

} // namespace vknn
