#include "passes_internal.h"

namespace vknn {

    // Does this op run as a FLAT (row-major) GPU op rather than the NC4HW4 path? Mirrors the cases the
    // Vulkan supportsNode() can't do in NC4HW4: Transpose/Slice always; Softmax on a non-channel axis;
    // Concat that isn't 4D channel-axis 4-aligned; Binary/Add with a constant operand or a broadcast/
    // rank that the packed kernel doesn't handle.
    bool gpuFlatNode(const Graph &g, const Node &n) {
        auto sh = [&](TensorId t) -> const Shape & {
            return g.desc(t).shape;
        };
        switch (n.type)
        {
            case OpType::Transpose:
            case OpType::Slice:
            case OpType::Expand:       // numpy broadcast to a target shape, flat row-major gather
            case OpType::Tile:         // repeat each dim, flat row-major gather
            case OpType::LayerNorm:    // always flat: reduce over the trailing axes, row-major
            case OpType::DepthToSpace: // spatial<->channel index remap, flat row-major gather
            case OpType::Reduce:       // generic N-D reduce (incl. ReduceL2) runs flat
            case OpType::MatMul:       // batched N-D matmul runs on the flat row-major path
            case OpType::Where:        // cond?X:Y, broadcasting flat select
            case OpType::Equal:        // A==B, broadcasting flat compare
            case OpType::Greater:      // A>B,  broadcasting flat compare
            case OpType::GreaterEqual: // A>=B, broadcasting flat compare
            case OpType::Range:           // 1-D arange, flat fill
            case OpType::ConstantOfShape: // scalar fill, flat
                return true;
            case OpType::ConvTranspose: {
                // Flat row-major transposed conv (one thread per output element, gather form). Needs a
                // 4D input and constant weight (uploaded flat); anything else falls back to the CPU op.
                if (sh(n.inputs[0]).size() != 4)
                {
                    return false;
                }
                if (n.inputs.size() < 2 || !g.isInitializer(n.inputs[1]))
                {
                    return false;
                }
                return !(pwCoreInputs(n) > 2 && n.inputs[2] != kNoTensor && !g.isInitializer(n.inputs[2]));
            }
            case OpType::Pad: {
                // Flat row-major pad (constant/edge/reflect). Needs static pads (attr or a constant
                // input[1]) and rank within the flat limit; a runtime pad value falls back to CPU.
                if (sh(n.outputs[0]).size() > 8)
                {
                    return false;
                }
                std::string mode = n.attr.gets("mode", "constant");
                if (mode != "constant" && mode != "edge" && mode != "reflect")
                {
                    return false;
                }
                bool padsKnown = !n.attr.getints("pads").empty() || (n.inputs.size() > 1 && n.inputs[1] != kNoTensor && g.isInitializer(n.inputs[1]));
                if (!padsKnown)
                {
                    return false;
                }
                return !(n.inputs.size() > 2 && n.inputs[2] != kNoTensor && !g.isInitializer(n.inputs[2]));
            }
            case OpType::Gather:
                // Flat row-major gather along an axis; index may be constant or a runtime float activation
                // (RoPE).
                return n.inputs.size() >= 2;
            case OpType::Clip:
                return true; // flat elementwise clamp (geometry-tail Clip is rank-6, can't be NC4HW4)
            case OpType::Split: {
                // Channel-axis 4-aligned split stays NC4HW4 (contiguous block copy); any other split is flat.
                const Shape &in   = sh(n.inputs[0]);
                int          rank = (int) in.size();
                int64_t      axis = n.attr.geti("axis", 0);
                if (axis < 0)
                {
                    axis += rank;
                }
                if (rank == 4 && axis == 1)
                {
                    for (TensorId o: n.outputs)
                    {
                        if (o != kNoTensor && (sh(o).size() != 4 || sh(o)[1] % 4 != 0))
                        {
                            return true;
                        }
                    }
                    return false;
                }
                return true;
            }
            case OpType::ScatterND:
                // GPU flat scatter; index may be a constant or a runtime float activation.
                return n.inputs.size() >= 3;
            case OpType::Einsum: {
                // Only the "i,j->ij" outer product runs on the GPU; other equations use the CPU op.
                std::string eq;
                for (char c: n.attr.gets("equation", ""))
                {
                    if (c != ' ' && c != '\t')
                    {
                        eq += c;
                    }
                }
                return eq == "i,j->ij";
            }
            case OpType::Softmax: {
                if (n.inputs.empty())
                {
                    return false;
                }
                const Shape &s = sh(n.inputs[0]);
                if (s.size() < 2)
                {
                    return true;
                }
                int     rank = (int) s.size();
                int64_t axis = n.attr.geti("axis", -1);
                if (axis < 0)
                {
                    axis += rank;
                }
                NCHW    x     = NCHW::from(s);
                int64_t inner = 1;
                for (int k = (int) axis; k < rank; ++k)
                {
                    inner *= s[k];
                }
                return !(x.h * x.w == 1 && inner == x.c); // channel softmax stays NC4HW4
            }
            case OpType::Concat: {
                const Shape &o    = sh(n.outputs[0]);
                int          rank = (int) o.size();
                int64_t      axis = n.attr.geti("axis", 1);
                if (axis < 0)
                {
                    axis += rank;
                }
                if (rank != 4 || axis != 1)
                {
                    return true;
                }
                for (TensorId in: n.inputs)
                {
                    if (sh(in).size() != 4 || sh(in)[1] % 4 != 0)
                    {
                        return true;
                    }
                }
                return false;
            }
            case OpType::Binary:
            case OpType::Add: {
                if (n.inputs.size() != 2)
                {
                    return false;
                }
                if (g.isInitializer(n.inputs[0]) || g.isInitializer(n.inputs[1]))
                {
                    return true;
                }
                const Shape &a = sh(n.inputs[0]);
                const Shape &b = sh(n.inputs[1]);
                if (a.size() == 4 && b.size() == 4 && a == b)
                {
                    return false; // NC4HW4 same-shape
                }
                if (n.type == OpType::Binary)
                {
                    auto bc = [](const Shape &s, const Shape &f) {
                        return s.size() == 4 && f.size() == 4 && s[0] == f[0] && s[1] == f[1] && s[2] == 1 && s[3] == 1 && (f[2] > 1 || f[3] > 1);
                    };
                    if (bc(a, b) || bc(b, a))
                    {
                        return false; // NC4HW4 channel-broadcast
                    }
                }
                return true;
            }
            case OpType::FusedPointwise:
                // The fusion pass records the chain's own layout (all steps agree) in pw_flat.
                return n.attr.geti("pw_flat", 0) != 0;
            default:
                return false;
        }
    }


    // --- Global layout assignment (NC4HW4 vs flat) ------------------------------------------------------
    // Every GPU tensor runs in one layout. A FIXED op has a kernel in only one layout; a FLEXIBLE op is
    // bit-exact in either (layout is an index remap over the same math + fp16 rounding), so the assignment
    // is free to pick its layout to minimise total NC4HW4<->flat converts. AGNOSTIC reshape/cast pass their
    // input's layout through. The assignment is a deterministic pure function of the graph — required for
    // run-to-run bit-exactness.
    namespace {
        enum class LKind { FixedFlat, FixedNC4, Flexible, Agnostic };

        bool layoutAgnostic(const Node &n) {
            // metadata reshape / no-op copy: input and output bytes are identical, so it keeps its layout.
            return n.type == OpType::Reshape || n.type == OpType::Flatten || n.type == OpType::Squeeze || n.type == OpType::Unsqueeze || n.type == OpType::Cast;
        }

        LKind opLayoutKind(const Graph &g, const Node &n) {
            if (layoutAgnostic(n))
            {
                return LKind::Agnostic;
            }
            // FLEXIBLE candidates (pointwise ops the flat classifier rejects from NC4 only for lack of a
            // native NC4 kernel — e.g. const-operand Binary/Add — but which run bit-exactly through the
            // NC4 fused-pointwise kernel) are unpinned in a later stage. For now every op follows the
            // per-op classifier, so the assignment reproduces the previous per-op marking exactly.
            return gpuFlatNode(g, n) ? LKind::FixedFlat : LKind::FixedNC4;
        }
    } // namespace

    // Assign every tensor's gpuFlat flag (see the notes above). Runs at load, before the converts are
    // spliced in below.
    static void globalLayoutAssign(Graph &g) {
        // 1) seed fixed + agnostic layouts in topo order (producers precede consumers). A flat reshape is a
        //    plain row-major copy (valid for any shape); the NC4HW4 byte-copy is only valid when the channel
        //    count is unchanged (else the vec4 interleave shifts) — so an agnostic op is flat if its input
        //    is flat OR it changes the channel count.
        for (auto &nd: g.nodes)
        {
            LKind k = opLayoutKind(g, nd);
            bool  f;
            if (k == LKind::Agnostic && !nd.inputs.empty() && nd.inputs[0] != kNoTensor)
            {
                bool    inFlat = g.desc(nd.inputs[0]).gpuFlat;
                int64_t cin    = NCHW::from(g.desc(nd.inputs[0]).shape).c;
                int64_t cout   = NCHW::from(g.desc(nd.outputs[0]).shape).c;
                f              = inFlat || cin != cout;
            } else if (k == LKind::FixedNC4)
            {
                f = false;
            } else
            {
                f = gpuFlatNode(g, nd); // FixedFlat, or an agnostic op with no usable input
            }
            for (TensorId o: nd.outputs)
            {
                if (o != kNoTensor)
                {
                    g.desc(o).gpuFlat = f;
                }
            }
        }
        // 2) (byte-weighted propagation for FLEXIBLE ops lands here in the next stage.)
        // 3) NC4HW4 can only represent rank <= 4 (NCHW::from collapses rank>4 to (1,1,1,1)). Any tensor with
        //    rank > 4 MUST be a flat row-major buffer — including graph inputs with no producer (a multi-view
        //    image input [1,2,3,224,224], left NC4HW4, would be mis-packed and corrupt the graph).
        for (auto &t: g.tensors)
        {
            if (t.shape.size() > 4)
            {
                t.gpuFlat = true;
            }
        }
    }

    /// Splice ConvertLayout nodes onto the graph wherever a tensor is read in a layout (NC4HW4 vs flat
    /// row-major) different from the one it was produced in. First globalLayoutAssign() stamps every
    /// tensor's gpuFlat flag to minimise the total number of converts; then for each node input whose
    /// producer layout differs from what the consumer operates in, an already-converted copy is reused
    /// from `cache` or a fresh ConvertLayout is emitted. Graph outputs left in NC4HW4 also get a trailing
    /// flat convert so the host readback is a bulk copy rather than a per-element strided gather.
    ///
    /// Precondition: shapes are inferred and pointwise fusion has run (FusedPointwise carries its pw_flat
    /// layout); runs at load, before markFp32. Postcondition: adjacent producer/consumer layouts agree, so
    /// the runtime never mixes layouts across an edge. ConvertLayout is a lossless same-dtype reorder, so
    /// the compiled result is byte-identical to the pre-pass math; the assignment is a pure function of the
    /// graph, keeping the compiled .vxm bit-exact run to run. New nodes are appended and the graph is
    /// re-topo-sorted so each convert precedes its consumer.
    void insertLayoutConverts(Graph &g) {
        // Assign every tensor a layout (minimising converts), then for every node input whose layout differs
        // from what the consumer needs, splice in a ConvertLayout node.
        globalLayoutAssign(g);
        std::map<std::pair<TensorId, bool>, TensorId> cache; // (tensor, needFlat) -> converted tensor
        std::vector<Node>                             converts;
        int                                           n = 0;
        for (auto &nd: g.nodes)
        {
            if (nd.outputs.empty() || nd.outputs[0] == kNoTensor)
            {
                continue;
            }
            bool needFlat = g.desc(nd.outputs[0]).gpuFlat; // the format this node operates in
            for (size_t inIdx = 0; inIdx < nd.inputs.size(); ++inIdx)
            {
                TensorId &in = nd.inputs[inIdx];
                if (in == kNoTensor || g.isInitializer(in))
                {
                    continue; // constants handled inside flat ops
                }
                // GridSample's grid (input 1) is a flat [N,Hout,Wout,2] buffer regardless of the NC4HW4
                // data path — keep it flat so a runtime grid is not mis-packed as NC4HW4.
                bool wantFlat = (nd.type == OpType::GridSample && inIdx == 1) ? true : needFlat;
                if (g.desc(in).gpuFlat == wantFlat)
                {
                    continue;
                }
                auto key = std::make_pair(in, wantFlat);
                auto it  = cache.find(key);
                if (it == cache.end())
                {
                    TensorDesc d    = g.desc(in);
                    d.name          = g.desc(in).name + (wantFlat ? "#flat" : "#nc4") + std::to_string(n);
                    d.isInitializer = d.isInput = d.isOutput = false;
                    d.gpuFlat                                = wantFlat;
                    TensorId t2                              = g.addTensor(d);
                    Node     cv;
                    cv.type    = OpType::ConvertLayout;
                    cv.name    = "convert" + std::to_string(n++);
                    cv.subOp   = wantFlat ? 0 : 1; // 0: NC4HW4->flat, 1: flat->NC4HW4
                    cv.inputs  = {in};
                    cv.outputs = {t2};
                    converts.push_back(cv);
                    it = cache.emplace(key, t2).first;
                }
                in = it->second;
            }
        }
        // Graph outputs have no consumer to trigger a convert, so a conv/pool output stays NC4HW4 and the
        // host readback pays an expensive scalar NC4HW4->NCHW gather (per-element, strided, fp16->fp32).
        // Emit each NC4HW4 graph output in flat NCHW via a ConvertLayout: the GPU does the vectorized layout
        // gather and the host readback becomes a bulk copy. Lossless same-dtype reorder, so byte-identical.
        for (size_t oi = 0; oi < g.outputs.size(); ++oi)
        {
            TensorId out = g.outputs[oi];
            if (out == kNoTensor || g.isInitializer(out) || g.desc(out).gpuFlat)
            {
                continue;
            }
            TensorDesc d    = g.desc(out); // carries the model's output name, declared dtype, shape, storeFp32
            d.isInitializer = d.isInput = false;
            d.isOutput      = true;
            d.gpuFlat       = true;
            // The flat convert output KEEPS the model's declared output name (callers look up outputs by
            // name); the pre-convert tensor becomes an internal boundary and is renamed to stay unique.
            g.desc(out).name += "#nc4";
            g.desc(out).isOutput = false;
            TensorId t2 = g.addTensor(d);
            Node     cv;
            cv.type    = OpType::ConvertLayout;
            cv.name    = "convertout" + std::to_string(n++);
            cv.subOp   = 0; // NC4HW4 -> flat
            cv.inputs  = {out};
            cv.outputs = {t2};
            converts.push_back(std::move(cv));
            g.outputs[oi] = t2;
        }
        if (!converts.empty())
        {
            for (auto &c: converts)
            {
                g.nodes.push_back(std::move(c));
            }
            g.topoSort();
            VKNN_INFO << "insertLayoutConverts: inserted " << converts.size() << " layout convert(s)";
        }
    }

} // namespace vknn
