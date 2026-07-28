#include "passes.h"
#include "passes_internal.h"

namespace vknn {

    namespace {

        // Upper bound on fold rounds. Each round erases at least one node, so a chain of K movers
        // folds in K-1 rounds; the cap only guards against a rewrite bug looping forever.
        constexpr int kMovementFoldMaxRounds = 256;

        // Row-major strides of `s`: stride[i] = product of the dims after axis i.
        std::vector<int64_t> rowStridesOf(const Shape &s) {
            std::vector<int64_t> st(s.size(), 1);
            for (int i = (int) s.size() - 2; i >= 0; --i)
            {
                st[(size_t) i] = st[(size_t) i + 1] * s[(size_t) i + 1];
            }
            return st;
        }

        // The per-output-axis affine map of a movement node: source elements are read at
        // base + sum_k coord_k * stride[k], where coord walks the node's OUTPUT row-major axes and
        // the strides count elements of the node's INPUT. This is exactly the addressing the CPU
        // Transpose/Slice loops and the flat_gather shader evaluate, so two maps compose into one
        // without changing which element any output coordinate reads.
        struct MovementMap {
            std::vector<int64_t> stride;
            int64_t              base = 0;
            bool                 ok   = false;
        };

        // Extract the map of node `nd` relative to ITS input. A previously folded node carries the
        // map verbatim in view_stride/view_base; a plain Transpose permutes the input strides; a
        // plain Slice scales them by the step and folds the starts into the base — with the SAME
        // normalize/clamp semantics the CPU kernel and flat_gather apply, so the fold can never
        // disagree with the kernels it replaces.
        MovementMap movementMapOf(const Graph &g, const Node &nd) {
            MovementMap  m;
            const Shape &in  = g.desc(nd.inputs[0]).shape;
            const Shape &out = g.desc(nd.outputs[0]).shape;
            const int    r   = (int) in.size();
            if (r == 0 || (int) out.size() != r || numElements(in) <= 0 || numElements(out) <= 0)
            {
                return m;
            }
            const auto inStride = rowStridesOf(in);
            if (nd.attr.has("view_stride"))
            {
                m.stride = nd.attr.getints("view_stride");
                m.base   = nd.attr.geti("view_base", 0);
                m.ok     = (int) m.stride.size() == r;
                return m;
            }
            if (nd.type == OpType::Transpose)
            {
                std::vector<int64_t> perm = nd.attr.getints("perm");
                if ((int) perm.size() != r)
                {
                    perm.clear();
                    for (int i = r - 1; i >= 0; --i)
                    {
                        perm.push_back(i); // ONNX default: full axis reversal
                    }
                }
                m.stride.resize((size_t) r);
                for (int k = 0; k < r; ++k)
                {
                    int64_t p = perm[(size_t) k] < 0 ? perm[(size_t) k] + r : perm[(size_t) k];
                    if (p < 0 || p >= r)
                    {
                        return m;
                    }
                    m.stride[(size_t) k] = inStride[(size_t) p];
                }
                m.ok = true;
                return m;
            }
            if (nd.type == OpType::Slice)
            {
                auto starts = readI64Param(g, nd, "starts", 1), ends = readI64Param(g, nd, "ends", 2);
                auto axes = readI64Param(g, nd, "axes", 3), steps = readI64Param(g, nd, "steps", 4);
                if (starts.empty())
                {
                    return m; // params must be static to prove the map
                }
                std::vector<int64_t> begin(r, 0), step(r, 1);
                for (size_t k = 0; k < starts.size() && k < ends.size(); ++k)
                {
                    int ax = (int) ((axes.empty() || k >= axes.size()) ? (int64_t) k : axes[k]);
                    if (ax < 0)
                    {
                        ax += r;
                    }
                    if (ax < 0 || ax >= r)
                    {
                        continue;
                    }
                    const int64_t dim = in[(size_t) ax], sp = k < steps.size() ? steps[k] : 1;
                    if (sp <= 0)
                    {
                        return m; // negative steps reverse; the kernels here support positive only
                    }
                    int64_t st         = starts[k] < 0 ? starts[k] + dim : starts[k];
                    st                 = std::max<int64_t>(0, std::min(st, dim));
                    begin[(size_t) ax] = st;
                    step[(size_t) ax]  = sp;
                }
                m.stride.resize((size_t) r);
                for (int k = 0; k < r; ++k)
                {
                    m.stride[(size_t) k] = inStride[(size_t) k] * step[(size_t) k];
                    m.base += begin[(size_t) k] * inStride[(size_t) k];
                }
                m.ok = true;
                return m;
            }
            return m;
        }

        // A movement node the fold may touch: single data input/output, no fused work riding it.
        bool movementEligible(const Node &nd) {
            if (nd.type != OpType::Transpose && nd.type != OpType::Slice)
            {
                return false;
            }
            return nd.outputs.size() == 1 && nd.outputs[0] != kNoTensor && !nd.inputs.empty() && nd.inputs[0] != kNoTensor && !nd.attr.has("pw_steps") &&
                   !nd.attr.has("pw_outs") && nd.fusedResidual == kNoTensor;
        }

    } // namespace

    /// Fold chains of movement ops — a Transpose or Slice whose input comes from another Transpose
    /// or Slice (possibly itself already folded) — into ONE strided gather: the consumer reads the
    /// chain's SOURCE through the composed per-axis map, stamped as view_stride/view_base attrs that
    /// the CPU kernels and the flat_gather geometry consume directly, and the producer disappears.
    /// Byte-identical by construction: movement kernels store loaded bytes verbatim (no arithmetic,
    /// no re-rounding — an fp16 byte pattern passes through unchanged), so reading the source
    /// through the composed map yields exactly the bytes the materialized intermediate held. The
    /// composition is per-output-axis affine: a consumer Transpose permutes the producer's
    /// coefficients, a consumer Slice scales them by its step and accumulates its starts into the
    /// base. A producer output with another reader, a graph-output producer, or an fp32-pinned
    /// intermediate (`fp32Pins`, the markFp32 matcher — removing the tensor would remove the store
    /// the pin exists for) refuses the site. Runs at load only, before insertLayoutConverts; never
    /// serialized. Returns the number of folded producers.
    int foldMovementChains(Graph &g, const std::string &fp32Pins) {
        int foldedTotal = 0;
        for (int round = 0; round < kMovementFoldMaxRounds; ++round)
        {
            // producer node index per tensor + consumer counts, rebuilt per round (cheap: the pass
            // runs once at load and rounds are bounded by the longest movement chain).
            std::vector<int> producer(g.tensors.size(), -1);
            std::vector<int> consumers(g.tensors.size(), 0);
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                for (TensorId o: g.nodes[i].outputs)
                {
                    if (o != kNoTensor)
                    {
                        producer[(size_t) o] = (int) i;
                    }
                }
                for (TensorId in: g.nodes[i].inputs)
                {
                    if (in != kNoTensor)
                    {
                        ++consumers[(size_t) in];
                    }
                }
            }
            std::vector<char> isOut(g.tensors.size(), 0);
            for (TensorId o: g.outputs)
            {
                if (o != kNoTensor)
                {
                    isOut[(size_t) o] = 1;
                }
            }
            int           folded = 0;
            std::set<int> dead;
            for (auto &nd: g.nodes)
            {
                // The consumer must be a PLAIN (un-stamped) mover: its axes move the producer's
                // output coordinates one axis at a time, which is what makes the composition exact.
                // A stamped producer is fine — its attrs are already per-axis coefficients.
                if (!movementEligible(nd) || nd.attr.has("view_stride"))
                {
                    continue;
                }
                const TensorId mid = nd.inputs[0];
                const int      pi  = mid < (TensorId) producer.size() ? producer[(size_t) mid] : -1;
                if (pi < 0 || dead.count(pi))
                {
                    continue;
                }
                const Node &P = g.nodes[(size_t) pi];
                if (!movementEligible(P) || consumers[(size_t) mid] != 1 || isOut[(size_t) mid])
                {
                    continue;
                }
                if (!fp32Pins.empty() && (fp32NameMatch(g.desc(mid).name, fp32Pins) || g.desc(mid).storeFp32))
                {
                    continue; // the pin exists for the materialized store this fold would remove
                }
                MovementMap pm = movementMapOf(g, P);
                MovementMap fm = movementMapOf(g, nd);
                if (!pm.ok || !fm.ok)
                {
                    continue;
                }
                const Shape &midShape = g.desc(mid).shape;
                const int    r        = (int) midShape.size();
                if ((int) pm.stride.size() != r || (int) fm.stride.size() != r)
                {
                    continue;
                }
                // Compose from the consumer's OWN parameters — exact by construction, no stride
                // arithmetic to invert: a Transpose consumer permutes the producer's coefficients
                // (its output axis k walks the producer's output axis perm[k] by 1); a Slice
                // consumer scales axis k's coefficient by its step and folds its begin into the
                // base through the producer's coefficient for that axis.
                std::vector<int64_t> composed((size_t) r, 0);
                int64_t              composedBase = pm.base;
                if (nd.type == OpType::Transpose)
                {
                    std::vector<int64_t> perm = nd.attr.getints("perm");
                    if ((int) perm.size() != r)
                    {
                        perm.clear();
                        for (int i = r - 1; i >= 0; --i)
                        {
                            perm.push_back(i); // ONNX default: full axis reversal
                        }
                    }
                    bool permOk = true;
                    for (int k = 0; k < r; ++k)
                    {
                        int64_t p = perm[(size_t) k] < 0 ? perm[(size_t) k] + r : perm[(size_t) k];
                        if (p < 0 || p >= r)
                        {
                            permOk = false;
                            break;
                        }
                        composed[(size_t) k] = pm.stride[(size_t) p];
                    }
                    if (!permOk)
                    {
                        continue;
                    }
                } else
                {
                    auto starts = readI64Param(g, nd, "starts", 1), ends = readI64Param(g, nd, "ends", 2);
                    auto axes = readI64Param(g, nd, "axes", 3), steps = readI64Param(g, nd, "steps", 4);
                    if (starts.empty())
                    {
                        continue;
                    }
                    std::vector<int64_t> begin(r, 0), step(r, 1);
                    bool                 sliceOk = true;
                    for (size_t k = 0; k < starts.size() && k < ends.size(); ++k)
                    {
                        int ax = (int) ((axes.empty() || k >= axes.size()) ? (int64_t) k : axes[k]);
                        if (ax < 0)
                        {
                            ax += r;
                        }
                        if (ax < 0 || ax >= r)
                        {
                            continue;
                        }
                        const int64_t dim = midShape[(size_t) ax], sp = k < steps.size() ? steps[k] : 1;
                        if (sp <= 0)
                        {
                            sliceOk = false;
                            break;
                        }
                        int64_t st         = starts[k] < 0 ? starts[k] + dim : starts[k];
                        begin[(size_t) ax] = std::max<int64_t>(0, std::min(st, dim));
                        step[(size_t) ax]  = sp;
                    }
                    if (!sliceOk)
                    {
                        continue;
                    }
                    for (int k = 0; k < r; ++k)
                    {
                        composed[(size_t) k] = pm.stride[(size_t) k] * step[(size_t) k];
                        composedBase += begin[(size_t) k] * pm.stride[(size_t) k];
                    }
                }
                // Commit: the consumer reads the chain's source through the composed map.
                nd.inputs[0] = P.inputs[0];
                {
                    Attr a;
                    a.kind                     = Attr::Ints;
                    a.ints                     = composed;
                    nd.attr.map["view_stride"] = a;
                }
                {
                    Attr a;
                    a.kind                   = Attr::Int;
                    a.i                      = composedBase;
                    nd.attr.map["view_base"] = a;
                }
                dead.insert(pi);
                ++folded;
            }
            if (!folded)
            {
                break;
            }
            std::vector<Node> kept;
            kept.reserve(g.nodes.size());
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!dead.count((int) i))
                {
                    kept.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(kept);
            foldedTotal += folded;
        }
        if (foldedTotal)
        {
            VKNN_INFO << "foldMovementChains: folded " << foldedTotal << " movement producer(s) into strided reads";
        }
        return foldedTotal;
    }

} // namespace vknn
