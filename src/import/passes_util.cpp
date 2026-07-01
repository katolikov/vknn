#include "passes_internal.h"

namespace vknn {

    // Read an int64 list parameter from either a node attribute (older opsets) or an initializer input
    // (opset 10+/13+ moved Slice/Pad/Reduce params to inputs). Returns empty if neither is present.
    std::vector<int64_t> readI64Param(const Graph &g, const Node &nd, const char *attrName, int inputIdx) {
        const auto &av = nd.attr.getints(attrName);
        if (!av.empty())
        {
            return av;
        }
        if (inputIdx >= 0 && inputIdx < (int) nd.inputs.size() && nd.inputs[inputIdx] != kNoTensor)
        {
            auto it = g.initializers.find(nd.inputs[inputIdx]);
            if (it != g.initializers.end())
            {
                const HostBuffer &hb = it->second;
                if (g.tensors[nd.inputs[inputIdx]].dtype == DType::Int64)
                {
                    int64_t n = (int64_t) hb.bytes.size() / 8;
                    return std::vector<int64_t>(hb.i64(), hb.i64() + n);
                }
                int64_t              n = (int64_t) hb.bytes.size() / 4;
                std::vector<int64_t> out;
                const float         *f = hb.f32();
                for (int64_t i = 0; i < n; ++i)
                {
                    out.push_back((int64_t) f[i]);
                }
                return out;
            }
        }
        return {};
    }

} // namespace vknn
