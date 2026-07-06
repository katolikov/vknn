// Normalize 1-D convolutions to the engine's canonical 2-D geometry. Every shape consumer
// (inferShapes, the CPU/GPU conv kernels, lowerConv, the Winograd/autotune gates) indexes conv
// weights as [M, C/group, kH, kW] and the strides/pads/dilations attributes at their 2-spatial-dim
// lengths, so a 1-D conv (rank-3 weight, 1-length strides/dilations, 2-length pads) would read past
// those vectors. The single spatial dim maps to H — matching NCHW::from's rank-3 view [N,C,L] ->
// (n=N, c=C, h=L, w=1) — so the weight becomes [M, C/group, k, 1] (payload bytes unchanged; the
// trailing extent is 1) and the attributes gain the W dim's identity values. Activation tensors keep
// their ONNX rank 3 end to end; inferShapes and the CPU conv emit a rank-3 output for a rank-3 input.
#include "import/passes.h"
#include "vknn/logging.h"

namespace vknn {
    namespace {
        // Extend a 1-spatial-dim int-list attribute to its 2-D length, appending the W dim's
        // identity value: strides/dilations [v] -> [v, fill]; pads [begin, end] (ONNX 1-D layout)
        // -> [begin, fill, end, fill] (ONNX 2-D layout [beginH, beginW, endH, endW]).
        void extendAttr(Attributes &attr, const char *key, size_t oneD, int64_t fill) {
            auto it = attr.map.find(key);
            if (it == attr.map.end() || it->second.ints.size() != oneD)
            {
                return;
            }
            std::vector<int64_t> &v = it->second.ints;
            if (oneD == 2) // pads: interleave the W dim into begin/end pairs
            {
                v = {v[0], fill, v[1], fill};
            } else
            {
                v.push_back(fill);
            }
        }
    } // namespace

    void normalizeConv1d(Graph &g) {
        int count = 0;
        for (Node &n: g.nodes)
        {
            if (n.type != OpType::Conv || n.inputs.size() < 2 || n.inputs[1] == kNoTensor)
            {
                continue;
            }
            // Only constant weights carry a static rank to normalize; a runtime-produced rank-3
            // weight keeps its shape and the CPU conv rejects it with a clear error.
            if (!g.isInitializer(n.inputs[1]))
            {
                continue;
            }
            Shape &w = g.desc(n.inputs[1]).shape;
            if (w.size() != 3)
            {
                continue;
            }
            w.push_back(1); // [M, C/group, k] -> [M, C/group, k, 1]; row-major payload is identical
            extendAttr(n.attr, "strides", 1, 1);
            extendAttr(n.attr, "dilations", 1, 1);
            extendAttr(n.attr, "kernel_shape", 1, 1);
            extendAttr(n.attr, "pads", 2, 0);
            ++count;
        }
        if (count)
        {
            VKNN_INFO << "normalizeConv1d: normalized " << count << " 1-D conv(s) to 2-D geometry";
        }
    }

} // namespace vknn
