#include "passes_internal.h"

namespace vknn {

    // Reduce(Mean) whose axes are exactly the spatial dims of a rank-4 input, with keepdims, is a
    // GlobalAveragePool: route it to the dedicated NC4HW4 kernel. Every other reduction stays a
    // generic Reduce (the flat kernel). The importer maps ONNX ReduceMean to Reduce unconditionally,
    // so this runs once input ranks are known.
    void lowerReduceToGap(Graph &g) {
        int               lowered = 0;
        std::vector<Node> reshapes; // appended after the walk so the node vector is not resized under it
        for (auto &n: g.nodes)
        {
            if (n.type != OpType::Reduce || (ReduceType) n.subOp != ReduceType::Mean)
            {
                continue;
            }
            if (n.inputs.empty() || n.inputs[0] == kNoTensor)
            {
                continue;
            }
            const Shape &in = g.desc(n.inputs[0]).shape;
            if (in.size() != 4)
            {
                continue;
            }
            // ONNX carries the reduction axes either as an attribute or as input 1 (opset 18+).
            std::vector<int64_t> axes = readI64Param(g, n, "axes", 1);
            if (axes.size() != 2)
            {
                continue;
            }
            // Normalize negatives against the rank-4 tensor, then require exactly the spatial dims:
            // {H, W} == {2, 3} of NCHW. Only then is the mean a GlobalAveragePool over each channel.
            int64_t a0 = axes[0] < 0 ? axes[0] + 4 : axes[0];
            int64_t a1 = axes[1] < 0 ? axes[1] + 4 : axes[1];
            if (std::min(a0, a1) != 2 || std::max(a0, a1) != 3)
            {
                continue;
            }
            // keepdims=0 drops the reduced axes, which GlobalAvgPool keeps as [N,C,1,1]. The values
            // are the same means either way, so the difference is metadata: pool, then Reshape to the
            // shape the graph declared. A Reshape is layout-agnostic and stores no bytes, so the
            // reduction still runs blocked instead of dragging the whole chain onto the flat path.
            const bool     keepDims = n.attr.geti("keepdims", 1) != 0;
            const TensorId out      = n.outputs[0];
            if (!keepDims)
            {
                TensorDesc pooled    = g.desc(out);
                pooled.name          = g.desc(out).name + "#pooled";
                pooled.shape         = {in[0], in[1], 1, 1};
                pooled.isOutput      = false;
                pooled.isInitializer = false;
                const TensorId mid   = g.addTensor(pooled);
                // Reshape reads its target from input 1, as an int64 initializer; a Reshape without
                // one has no shape to infer and every consumer downstream then works on an empty one.
                const Shape &decl = g.desc(out).shape;
                TensorDesc   sd;
                sd.name                = g.desc(out).name + "#shape";
                sd.shape               = {(int64_t) decl.size()};
                sd.dtype               = DType::Int64;
                sd.isInitializer       = true;
                const TensorId shapeId = g.addTensor(sd);
                HostBuffer     hb;
                hb.resizeElems((int64_t) decl.size(), DType::Int64);
                for (size_t k = 0; k < decl.size(); ++k)
                {
                    hb.i64()[k] = decl[k];
                }
                g.initializers[shapeId] = std::move(hb);
                Node shape;
                shape.type    = OpType::Reshape;
                shape.name    = n.name + "_keepdims";
                shape.inputs  = {mid, shapeId};
                shape.outputs = {out};
                n.outputs[0]  = mid;
                reshapes.push_back(std::move(shape));
            }
            n.type = OpType::GlobalAvgPool;
            n.inputs.resize(1); // drop an axes initializer input; GAP reads the data tensor only
            lowered++;
        }
        if (!reshapes.empty())
        {
            for (Node &r: reshapes)
            {
                g.nodes.push_back(std::move(r));
            }
            g.topoSort();
        }
        if (lowered)
        {
            VKNN_INFO << "lowerReduceToGap: lowered " << lowered << " spatial ReduceMean(s) to GlobalAvgPool";
        }
    }

} // namespace vknn
