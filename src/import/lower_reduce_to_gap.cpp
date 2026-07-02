#include "passes_internal.h"

namespace vknn {

    // Reduce(Mean) whose axes are exactly the spatial dims of a rank-4 input, with keepdims, is a
    // GlobalAveragePool: route it to the dedicated NC4HW4 kernel. Every other reduction stays a
    // generic Reduce (the flat kernel). The importer maps ONNX ReduceMean to Reduce unconditionally,
    // so this runs once input ranks are known.
    void lowerReduceToGap(Graph &g) {
        int lowered = 0;
        for (auto &n: g.nodes)
        {
            if (n.type != OpType::Reduce || (ReduceType) n.subOp != ReduceType::Mean)
            {
                continue;
            }
            if (n.inputs.empty() || n.inputs[0] == kNoTensor || n.attr.geti("keepdims", 1) == 0)
            {
                continue;
            }
            const Shape &in = g.desc(n.inputs[0]).shape;
            if (in.size() != 4)
            {
                continue;
            }
            std::vector<int64_t> axes = readI64Param(g, n, "axes", 1);
            if (axes.size() != 2)
            {
                continue;
            }
            int64_t a0 = axes[0] < 0 ? axes[0] + 4 : axes[0];
            int64_t a1 = axes[1] < 0 ? axes[1] + 4 : axes[1];
            if (std::min(a0, a1) != 2 || std::max(a0, a1) != 3)
            {
                continue;
            }
            n.type = OpType::GlobalAvgPool;
            n.inputs.resize(1); // drop an axes initializer input; GAP reads the data tensor only
            lowered++;
        }
        if (lowered)
        {
            VKNN_INFO << "lowerReduceToGap: lowered " << lowered << " spatial ReduceMean(s) to GlobalAvgPool";
        }
    }

} // namespace vknn
