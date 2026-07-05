#include "passes_internal.h"

namespace vknn {

    /// Read an int64 list parameter that a graph op carries either as a named attribute (older opsets)
    /// or as a constant initializer input (opset 10+/13+ moved Slice/Pad/Reduce params to inputs).
    /// Passes call this so they resolve such params uniformly regardless of which encoding the source
    /// model used.
    ///
    /// Resolution order: the attribute wins if present; otherwise the input at @p inputIdx is read, but
    /// only when it names an initializer (a runtime-computed param stays unresolved and yields empty).
    ///
    /// The positional input read is bounded by pwCoreInputs(nd), not inputs.size(): fusePointwiseChains
    /// appends chain operands past pw_opbase, and one of those must never be misread as this op's param
    /// (e.g. a Reduce that kept `axes` as an attribute could otherwise pick up a fused operand at the
    /// axes-input slot).
    ///
    /// A float initializer is decoded by truncation to int64, tolerating models that encode an
    /// index/axis list in a float tensor.
    ///
    /// @param g        Graph whose initializers back any input-encoded param.
    /// @param nd       Node to read the parameter from.
    /// @param attrName Attribute name to try first.
    /// @param inputIdx Positional input index to fall back to, or negative to skip the input path.
    /// @returns The parameter values, or an empty vector when neither encoding supplies them.
    std::vector<int64_t> readI64Param(const Graph &g, const Node &nd, const char *attrName, int inputIdx) {
        const auto &av = nd.attr.getints(attrName);
        if (!av.empty())
        {
            return av;
        }
        if (inputIdx >= 0 && inputIdx < (int) pwCoreInputs(nd) && nd.inputs[inputIdx] != kNoTensor)
        {
            auto it = g.initializers.find(nd.inputs[inputIdx]);
            if (it != g.initializers.end())
            {
                const HostBuffer &hb = it->second;
                if (g.tensors[nd.inputs[inputIdx]].dtype == DType::Int64)
                {
                    // Native path: element count is the raw byte count over the 8-byte int64 stride.
                    int64_t n = (int64_t) hb.bytes.size() / 8;
                    return std::vector<int64_t>(hb.i64(), hb.i64() + n);
                }
                // Fallback: a float-typed initializer, decoded over the 4-byte float stride and
                // truncated element-wise to int64.
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
