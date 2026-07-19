#include "passes_internal.h"

namespace vknn {

    // Fold a Gather whose indices are a CONSTANT contiguous ascending run [s, s+1, ..., s+n-1] into
    // the equivalent Slice(starts=[s], ends=[s+n], axes=[axis], steps=[1]). The two ops read exactly
    // the same elements in the same order, so the fold is byte-identical on every backend -- and the
    // Slice form is what the runtime optimizes: an identity slice aliases its input outright, and a
    // leading-axis slice becomes a zero-copy sub-buffer view instead of a gather dispatch.
    //
    // Only rank-1 index tensors qualify: a rank-0 (scalar) index DROPS the gathered axis (the
    // idx_scalar convention), which Slice cannot express, and a rank>1 index reshapes the output.
    // The indices must be an int32/int64 initializer with at least one element, in bounds for the
    // gathered axis. Runs after the const-fold fixpoint (indices resolved to initializers), before
    // the layout/fusion passes that treat Slice specially.
    void foldGatherToSlice(Graph &g) {
        int folded = 0;
        for (Node &nd: g.nodes)
        {
            if (nd.type != OpType::Gather || nd.inputs.size() < 2 || nd.inputs[0] == kNoTensor || nd.inputs[1] == kNoTensor)
            {
                continue;
            }
            if (nd.attr.has("idx_scalar") || nd.attr.has("pw_steps"))
            {
                continue;
            }
            TensorId data = nd.inputs[0], idx = nd.inputs[1];
            if (!g.isInitializer(idx) || g.isInitializer(data))
            {
                continue;
            }
            const TensorDesc &id = g.desc(idx);
            if (id.shape.size() != 1 || (id.dtype != DType::Int32 && id.dtype != DType::Int64))
            {
                continue;
            }
            const Shape &ds = g.desc(data).shape;
            if (ds.empty())
            {
                continue;
            }
            int64_t axis = nd.attr.geti("axis", 0);
            if (axis < 0)
            {
                axis += (int64_t) ds.size();
            }
            if (axis < 0 || axis >= (int64_t) ds.size())
            {
                continue;
            }
            const auto          &payload = g.initializers.at(idx);
            std::vector<int64_t> vals;
            const int64_t        count = id.shape[0];
            if (payload.bytes.empty() || count < 1)
            {
                continue;
            }
            if (id.dtype == DType::Int64)
            {
                if ((int64_t) payload.bytes.size() < count * (int64_t) sizeof(int64_t))
                {
                    continue;
                }
                const int64_t *p = reinterpret_cast<const int64_t *>(payload.bytes.data());
                vals.assign(p, p + count);
            } else
            {
                if ((int64_t) payload.bytes.size() < count * (int64_t) sizeof(int32_t))
                {
                    continue;
                }
                const int32_t *p = reinterpret_cast<const int32_t *>(payload.bytes.data());
                vals.assign(p, p + count);
            }
            bool contiguous = true;
            for (int64_t k = 0; k < count; ++k)
            {
                int64_t v = vals[(size_t) k] < 0 ? vals[(size_t) k] + ds[(size_t) axis] : vals[(size_t) k];
                if (v != vals[0] + k || v < 0 || v >= ds[(size_t) axis])
                {
                    contiguous = false;
                    break;
                }
            }
            const int64_t start = vals[0] < 0 ? vals[0] + ds[(size_t) axis] : vals[0];
            if (!contiguous || start < 0 || start + count > ds[(size_t) axis])
            {
                continue;
            }
            nd.type = OpType::Slice;
            nd.inputs.assign(1, data); // the dead indices initializer falls to pruneDeadInitializers
            auto setInts = [&](const char *name, std::vector<int64_t> v) {
                Attr a;
                a.kind           = Attr::Ints;
                a.ints           = std::move(v);
                nd.attr.map[name] = a;
            };
            setInts("starts", {start});
            setInts("ends", {start + count});
            setInts("axes", {axis});
            setInts("steps", {1});
            ++folded;
        }
        if (folded)
        {
            VKNN_INFO << "foldGatherToSlice: folded " << folded << " constant-contiguous Gather(s)";
        }
    }

} // namespace vknn
