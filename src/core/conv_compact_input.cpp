#include "core/conv_compact_input.h"
#include "vknn/nchw.h"

namespace vknn {

    namespace {
        // Geometry the compact kernel (shaders/conv3x3_cin_lt4*) is specialized to.
        constexpr int64_t kCompactKernelExtent = 3;
        // NC4HW4's block width: at or above this the padded layout wastes nothing and the compact
        // path has nothing to win.
        constexpr int64_t kChannelBlock = 4;

        std::vector<int64_t> attrIntsOr(const Node &n, const char *key, std::vector<int64_t> fallback) {
            const auto &v = n.attr.getints(key);
            return v.empty() ? std::move(fallback) : v;
        }
    } // namespace

    bool convCompactInputEligible(const Graph &g, const Node &n) {
        if (n.type != OpType::Conv || n.inputs.size() < 2 || n.inputs[0] == kNoTensor || n.inputs[1] == kNoTensor)
        {
            return false;
        }
        const Shape &in = g.desc(n.inputs[0]).shape;
        if (in.size() != 4)
        {
            return false;
        }
        const NCHW x = NCHW::from(in);
        if (x.c < 1 || x.c >= kChannelBlock)
        {
            return false;
        }
        const Shape &ws = g.desc(n.inputs[1]).shape; // [Cout, Cin/group, KH, KW]
        if (ws.size() != 4 || ws[2] != kCompactKernelExtent || ws[3] != kCompactKernelExtent)
        {
            return false;
        }
        const auto dil = attrIntsOr(n, "dilations", {1, 1});
        return n.attr.geti("group", 1) == 1 && dil.size() == 2 && dil[0] == 1 && dil[1] == 1;
    }

    bool tensorWantsCompactConvInput(const Graph &g, TensorId tid) {
        if (tid == kNoTensor || g.isInitializer(tid))
        {
            return false;
        }
        bool anyConsumer = false;
        for (const Node &n: g.nodes)
        {
            for (TensorId o: n.outputs)
            {
                if (o == tid)
                {
                    return false; // produced on the GPU: the producer would pay a convert instead
                }
            }
            for (size_t i = 0; i < n.inputs.size(); ++i)
            {
                if (n.inputs[i] != tid)
                {
                    continue;
                }
                // Every reader must take it compactly, and only as the activation slot -- one reader
                // that wants NC4HW4 would just move the padding into a spliced convert.
                if (i != 0 || !convCompactInputEligible(g, n))
                {
                    return false;
                }
                anyConsumer = true;
            }
        }
        return anyConsumer;
    }

} // namespace vknn
