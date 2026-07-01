#include "passes_internal.h"

namespace vknn {

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
                case OpType::Reduce: {
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
            try
            { op->run(nd, ctx); } catch (...)
            { continue; }
            for (TensorId o: nd.outputs)
            {
                if (o == kNoTensor)
                {
                    continue;
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
