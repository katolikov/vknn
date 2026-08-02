#include "core/slice_bounds.h"
#include "passes_internal.h"
#include <iterator>

namespace vknn {

    // Channel-last permutation of a rank-4 NCHW tensor: [N,C,H,W] -> [N,H,W,C].
    static constexpr int64_t kNhwcPerm[] = {0, 2, 3, 1};

    // Channel bound for the NC4-reading channel-last kernel. The lane-per-(pixel, channel-block)
    // map wins where the July image-path class lives (measured -50% at C = 32 over a large spatial
    // map) and degrades sharply where the channel axis dominates the spatial one (measured 7x
    // SLOWER at a deep-channel, small-spatial embedding transpose): every output pixel's channel
    // run then reads C/4 blocks a full plane apart with no reuse. The bound keeps the kernel inside
    // its proven-win region; wider channels take the ConvertLayout + flat gather route, exactly as
    // before the kernel existed. Routing only - both routes store identical bytes.
    constexpr int64_t kTransposeNhwcMaxChannels = 32;

    bool transposeReadsNc4(const Graph &g, const Node &n) {
        if (n.type != OpType::Transpose || n.inputs.empty() || n.inputs[0] == kNoTensor || n.outputs.empty() || n.outputs[0] == kNoTensor)
        {
            return false;
        }
        if (n.attr.has("pw_steps") || n.attr.has("view_stride"))
        {
            return false; // a fused epilogue / folded movement chain runs through the flat gather
        }
        const Shape &in = g.desc(n.inputs[0]).shape;
        if (in.size() != std::size(kNhwcPerm) || g.isInitializer(n.inputs[0]))
        {
            return false;
        }
        if (g.desc(n.inputs[0]).gpuFlat)
        {
            // The packed read pays off only on an input that is ALREADY packed. Behind a flat
            // producer it would cost a full-tensor flat->NC4HW4 convert the flat gather never
            // needed, so the gather wins by exactly that convert. Both routes store identical
            // bytes. The layout assignment and the Vulkan op both ask this predicate over the final
            // gpuFlat facts, so they agree on which buffer the kernel is handed.
            return false;
        }
        if (in[1] > kTransposeNhwcMaxChannels)
        {
            return false; // outside the kernel's proven-win region (see kTransposeNhwcMaxChannels)
        }
        const auto &perm = n.attr.getints("perm");
        if (perm.size() != std::size(kNhwcPerm))
        {
            return false; // an absent perm is the full reverse, not channel-last
        }
        for (size_t k = 0; k < perm.size(); ++k)
        {
            if (perm[k] != kNhwcPerm[k])
            {
                return false;
            }
        }
        return true;
    }

    // Does this op run as a FLAT (row-major) GPU op rather than the NC4HW4 path? Mirrors the cases the
    // Vulkan supportsNode() cannot do in NC4HW4: Transpose always; Softmax on a non-channel axis;
    // Concat that isn't 4D channel-axis 4-aligned; Binary/Add with a constant operand or a broadcast/
    // rank that the packed kernel doesn't handle.
    //
    // The per-OpType layout fact lives in opDescriptor(): LayoutClass::Flat is unconditionally flat,
    // LayoutClass::Nc4 unconditionally NC4HW4. Only LayoutClass::ShapeDependent ops keep a per-node
    // predicate below; the switch handles exactly those (the anti-drift test asserts the two agree).
    //
    // A Transpose's OUTPUT is always flat; transposeReadsNc4 governs its INPUT separately.
    bool depthToSpaceIsNc4(const Graph &g, const Node &n) {
        if (n.type != OpType::DepthToSpace || n.inputs.empty() || n.inputs[0] == kNoTensor || n.outputs.empty() || n.outputs[0] == kNoTensor)
        {
            return false;
        }
        const Shape &in = g.desc(n.inputs[0]).shape, &out = g.desc(n.outputs[0]).shape;
        if (in.size() != 4 || out.size() != 4)
        {
            return false;
        }
        // Partly-filled blocks on either side are fine: the kernel reads its four source channels
        // through the SCALAR view of the input, so they may sit in different blocks, and it stops at
        // the last real output channel, leaving that block's remaining lanes zero the way every
        // blocked buffer carries its padding. What it does need is a whole number of output pixels
        // per input pixel, which the block size gives by construction.
        return in[1] > 0 && out[1] > 0;
    }

    bool sliceIsNc4(const Graph &g, const Node &n) {
        if (n.type != OpType::Slice || n.inputs.empty() || n.inputs[0] == kNoTensor || n.outputs.empty() || n.outputs[0] == kNoTensor)
        {
            return false;
        }
        // A folded epilogue or a folded movement chain has arithmetic/reindexing to do; only a plain
        // slice degenerates to a copy.
        if (n.attr.has("pw_steps") || n.attr.has("view_stride"))
        {
            return false;
        }
        const Shape &in = g.desc(n.inputs[0]).shape, &out = g.desc(n.outputs[0]).shape;
        if (in.size() != kNchwRank || out.size() != kNchwRank)
        {
            return false;
        }
        const auto starts = readI64Param(g, n, "starts", 1), ends = readI64Param(g, n, "ends", 2);
        const auto axes = readI64Param(g, n, "axes", 3), steps = readI64Param(g, n, "steps", 4);
        if (starts.size() != 1 || ends.size() != 1)
        {
            return false; // exactly one sliced axis
        }
        const int64_t axis = axes.empty() ? 0 : (axes[0] < 0 ? axes[0] + (int64_t) kNchwRank : axes[0]);
        const int64_t step = steps.empty() ? 1 : steps[0];
        if (axis != 1 || step != 1)
        {
            return false; // the channel axis, walked forward
        }
        // NC4HW4 groups four channels into one block, so a channel range is a run of WHOLE blocks
        // exactly when it starts and ends on a block boundary. Anything else would split a block
        // across the seam, which no byte copy can express.
        const SliceAxisBounds b = resolveSliceAxis(in[1], starts[0], ends[0], step);
        if (b.start % kNC4Block != 0 || b.count % kNC4Block != 0 || b.count <= 0)
        {
            return false;
        }
        // Every other axis must be taken whole: the copy moves contiguous block runs, not a box.
        return out[0] == in[0] && out[1] == b.count && out[2] == in[2] && out[3] == in[3];
    }

    bool reduceIsNc4(const Graph &g, const Node &n) {
        if (n.type != OpType::Reduce || n.inputs.empty() || n.inputs[0] == kNoTensor || n.outputs.empty() || n.outputs[0] == kNoTensor)
        {
            return false;
        }
        if (n.attr.has("view_stride"))
        {
            return false; // a folded movement chain reindexes; the blocked kernel reads the plane as stored
        }
        const Shape &in = g.desc(n.inputs[0]).shape, &out = g.desc(n.outputs[0]).shape;
        if (in.size() != kNchwRank || out.size() != kNchwRank)
        {
            return false;
        }
        // Exactly the spatial axes, keeping them: that is one reduction per channel, which is what a
        // channel block's four lanes each carry. Any other axis set would have to cross lanes.
        const std::vector<int64_t> axes = readI64Param(g, n, "axes", 1);
        if (axes.size() != 2)
        {
            return false;
        }
        const int64_t a0 = axes[0] < 0 ? axes[0] + (int64_t) kNchwRank : axes[0];
        const int64_t a1 = axes[1] < 0 ? axes[1] + (int64_t) kNchwRank : axes[1];
        if (std::min(a0, a1) != 2 || std::max(a0, a1) != 3)
        {
            return false;
        }
        return out[0] == in[0] && out[1] == in[1] && out[2] == 1 && out[3] == 1;
    }

    bool gpuFlatNode(const Graph &g, const Node &n) {
        auto sh = [&](TensorId t) -> const Shape & {
            return g.desc(t).shape;
        };
        LayoutClass lc = opDescriptor(n.type).layout;
        if (lc == LayoutClass::Flat)
        {
            return true;
        }
        if (lc == LayoutClass::Nc4)
        {
            return false;
        }
        // LayoutClass::ShapeDependent: the layout is a per-node function of shapes/attributes.
        switch (n.type)
        {
            case OpType::ConvTranspose: {
                // Flat row-major transposed conv (one thread per output element, gather form). Needs a
                // 4D input and a weight input; the weight/bias may be constant (uploaded flat) or a
                // runtime activation (bound at dispatch). A non-4D input falls back to the CPU op.
                // Mirrors the ConvTranspose gate in vkNodeGate.
                if (sh(n.inputs[0]).size() != 4)
                {
                    return false;
                }
                return n.inputs.size() >= 2 && n.inputs[1] != kNoTensor;
            }
            case OpType::Pad: {
                // Flat row-major pad (constant/edge/reflect). Needs static pads (attr or a constant
                // input[1]); the flat kernel decodes any rank (geometry in a plan SSBO). A runtime pad
                // VALUE runs on the GPU via flat_pad_rt (the value binds as an SSBO). A runtime pads
                // GEOMETRY falls back to CPU (data-dependent output shape). Mirrors the Pad gate in
                // vkNodeGate.
                std::string mode = n.attr.gets("mode", "constant");
                if (mode != "constant" && mode != "edge" && mode != "reflect")
                {
                    return false;
                }
                bool padsKnown = !n.attr.getints("pads").empty() || (n.inputs.size() > 1 && n.inputs[1] != kNoTensor && g.isInitializer(n.inputs[1]));
                return padsKnown;
            }
            case OpType::Gather:
                // Flat row-major gather along an axis; index may be constant or a runtime float activation
                // (RoPE).
                return n.inputs.size() >= 2;
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
            case OpType::Reduce:
                // A spatial reduction is one reduction per channel, which is exactly what a channel
                // block's four lanes carry -- so it reads the blocked buffer as stored. Every other
                // axis set has to cross lanes and keeps the flat kernel.
                return !reduceIsNc4(g, n);
            case OpType::Slice:
                // A block-aligned channel slice is a contiguous run of NC4HW4 blocks per batch, so it
                // copies buffer ranges (and often aliases outright) instead of gathering through flat.
                // Every other slice keeps the flat gather.
                return !sliceIsNc4(g, n);
            case OpType::DepthToSpace:
                // Both sides packed when every NC4HW4 block is fully occupied; otherwise the flat
                // row-major remap, with the layout pass converting at the boundary.
                return !depthToSpaceIsNc4(g, n);
            case OpType::ScatterND:
                // GPU flat scatter; index may be a constant or a runtime float activation.
                return n.inputs.size() >= 3;
            case OpType::TopK:
                // Per-slice selection along an axis (flat row-major), with a compile-time k (the `k`
                // attribute or a constant int64 input[1]); a runtime k stays on the CPU op. Mirrors the
                // TopK gate in vkNodeGate so a GPU-assigned TopK is always marked flat.
                if (sh(n.inputs[0]).empty())
                {
                    return false;
                }
                return n.attr.has("k") || (pwCoreInputs(n) > 1 && n.inputs[1] != kNoTensor && g.isInitializer(n.inputs[1]));
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
                // Only the concatenated parts decide the layout: inputs from pwCoreInputs on are
                // fused-epilogue operands (broadcast scales etc.), not parts.
                size_t parts = (size_t) pwCoreInputs(n);
                for (size_t e = 0; e < parts && e < n.inputs.size(); ++e)
                {
                    if (sh(n.inputs[e]).size() != 4 || sh(n.inputs[e])[1] % 4 != 0)
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
                // A constant operand is no longer a reason to run flat: binary.cpp packs it into the
                // blocked layout at prepare and uploads it, so the only question left is whether the
                // SHAPES are ones the blocked kernel indexes.
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
            case OpType::ChannelShuffle:
                // Kernels exist in BOTH layouts (channel_shuffle_flat / channel_shuffle_nc4), so the
                // node runs in whatever layout its input carries — globalLayoutAssign resolves it
                // through the Agnostic arm (channel count is unchanged, so the output simply adopts
                // the input's layout) and this predicate mirrors that assignment for direct callers.
                return !n.inputs.empty() && n.inputs[0] != kNoTensor && g.desc(n.inputs[0]).gpuFlat;
            default:
                // A ShapeDependent descriptor with no arm here (a mis-registration): fall back to the
                // NC4HW4 default, matching a plain Nc4 op.
                return false;
        }
    }

    /// Why this node runs flat, in a few words, for the load-time inventory below.
    ///
    /// The two causes are different work. A whole OP CLASS with no blocked kernel is a kernel to
    /// write; a SHAPE that the blocked kernel cannot express is a case to widen (or a graph to
    /// change). Naming which one applies per node is what turns "62 layout converts" into a task
    /// list, so the inventory reports the reason rather than only the count.
    const char *flatReason(const Graph &g, const Node &n) {
        if (opDescriptor(n.type).layout == LayoutClass::Flat)
        {
            return "op class has no blocked kernel";
        }
        switch (n.type)
        {
            case OpType::Binary:
            case OpType::Add:
                if (n.inputs.size() == 2 && (g.isInitializer(n.inputs[0]) || g.isInitializer(n.inputs[1])))
                {
                    return "constant operand (the blocked kernel binds activations only)";
                }
                return "operand shapes outside the blocked broadcast forms";
            case OpType::Slice:
                return "channel range not block-aligned, or carries an epilogue";
            case OpType::Concat:
                return "not a 4D channel concat with every part 4-aligned";
            case OpType::Split:
                return "not a 4D channel split with every part 4-aligned";
            case OpType::DepthToSpace:
                return "a partly filled channel block on one side";
            case OpType::Softmax:
                return "reduces an axis other than a full channel axis";
            case OpType::FusedPointwise:
                return "the unit was recorded flat (an operand had no blocked index)";
            case OpType::Pad:
            case OpType::Gather:
            case OpType::ScatterND:
            case OpType::TopK:
            case OpType::Einsum:
            case OpType::ConvTranspose:
                return "this form has only a flat kernel";
            default:
                return "shape-dependent rule resolved flat";
        }
    }

    /// Everything still running flat, by op type and reason. The inventory of what is left to move.
    void reportFlatInventory(const Graph &g) {
        std::map<std::string, int> byReason;
        int                        flatNodes = 0;
        for (const Node &n: g.nodes)
        {
            if (n.type == OpType::ConvertLayout || n.outputs.empty() || n.outputs[0] == kNoTensor || !g.desc(n.outputs[0]).gpuFlat)
            {
                continue;
            }
            ++flatNodes;
            byReason[std::string(opTypeName(n.type)) + ": " + flatReason(g, n)]++;
        }
        if (flatNodes == 0)
        {
            VKNN_INFO << "flat path: no node runs flat -- the whole graph is blocked";
            return;
        }
        std::vector<std::pair<std::string, int>> ranked(byReason.begin(), byReason.end());
        std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
            return a.second > b.second;
        });
        VKNN_INFO << "flat path: " << flatNodes << " node(s) still run flat";
        constexpr size_t kReasonsReported = 10;
        for (size_t i = 0; i < ranked.size() && i < kReasonsReported; ++i)
        {
            VKNN_INFO << "flat path:   " << ranked[i].second << "x  " << ranked[i].first;
        }
        if (ranked.size() > kReasonsReported)
        {
            VKNN_INFO << "flat path:   ... " << (ranked.size() - kReasonsReported) << " further reason(s)";
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
            // metadata reshape / no-op copy: input and output bytes are identical, so it keeps its
            // layout. ChannelShuffle is not a byte copy but has a kernel in BOTH layouts (a pure
            // index remap either way), so it equally adopts its input's layout — the channel count
            // is unchanged, which keeps the NC4HW4 arm of the agnostic rule valid.
            return n.type == OpType::Reshape || n.type == OpType::Flatten || n.type == OpType::Squeeze || n.type == OpType::Unsqueeze || n.type == OpType::Cast || n.type == OpType::ChannelShuffle;
        }

        /// The layout ONE reader operates a given input slot in — the exact rule the convert
        /// splicer applies below, so an assignment derived from it provably removes the convert
        /// instead of moving it.
        bool readerWantsFlat(const Graph &g, const Node &n, size_t inputIndex) {
            // GridSample's non-warp grid is always a flat [N,Hout,Wout,2] buffer, whatever layout the
            // data path runs in: a runtime grid left NC4HW4 would be mis-packed. A warp-mode
            // GridSample instead reads its NCHW flow (input 1) in the NC4HW4 activation layout (the
            // op computes coordinates from it directly), so it follows the node's own format.
            if (n.type == OpType::GridSample && inputIndex == 1 && !n.attr.has("warp"))
            {
                return true;
            }
            // A channel-last Transpose reads NC4HW4 directly (transposeReadsNc4): the packed vec4 is
            // four consecutive output channels, so the reindex is one coalesced pass and the
            // full-size ConvertLayout that would otherwise precede it never gets spliced in.
            if (inputIndex == 0 && transposeReadsNc4(g, n))
            {
                return false;
            }
            return n.outputs.empty() || n.outputs[0] == kNoTensor ? false : g.desc(n.outputs[0]).gpuFlat;
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
        auto seedFromProducers = [&g]() {
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
        };
        seedFromProducers();
        // 2) FLEXIBLE re-vote: a standalone FusedPointwise whose plan is expressible in BOTH
        //    layouts — a rank-4 run with no general-broadcast (class-2) operand, the exact
        //    nc4Ok/flatOk rule the compile-time fuser applied — runs bit-identically through
        //    fused_pw_flat and fused_pw_nc4 (same VM, same vknnRte16 stores), so its layout is a
        //    pure placement choice. Each such node adopts the layout that minimizes the element
        //    count crossing a convert on its full-size edges (entry, runtime operands, outputs read
        //    by non-agnostic consumers); agnostic readers follow for free and fixed readers vote
        //    with their classifier layout. Rounds alternate vote and re-seed until stable — the
        //    result stays a deterministic pure function of the graph.
        {
            // A step's bcast field of kPwBcastGeneral marks the general-broadcast operand class
            // that only the flat kernel addresses (record geometry: kPwStepInts/kPwStepBcastField,
            // include/vknn/op_type.h).
            constexpr int       kFlexVoteMaxRounds = 8; // cycles are byte-weight monotone; this only caps oscillation
            constexpr int64_t   kNc4Rank           = 4;
            std::vector<size_t> flexible;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                const Node &nd = g.nodes[i];
                if (nd.type != OpType::FusedPointwise || !nd.attr.has("pw_steps") || nd.outputs.empty() || nd.outputs[0] == kNoTensor)
                {
                    continue;
                }
                if ((int64_t) g.desc(nd.outputs[0]).shape.size() != kNc4Rank)
                {
                    continue; // rank != 4 has no NC4HW4 form: stays with the classifier
                }
                const auto &steps = nd.attr.getints("pw_steps");
                bool        nc4Ok = steps.size() % kPwStepInts == 0;
                for (size_t s = 0; nc4Ok && s + kPwStepBcastField < steps.size(); s += kPwStepInts)
                {
                    nc4Ok = steps[s + kPwStepBcastField] != kPwBcastGeneral;
                }
                if (nc4Ok)
                {
                    flexible.push_back(i);
                }
            }
            if (!flexible.empty())
            {
                std::vector<std::vector<size_t>> readers(g.tensors.size());
                for (size_t j = 0; j < g.nodes.size(); ++j)
                {
                    for (TensorId in: g.nodes[j].inputs)
                    {
                        if (in != kNoTensor && in < (TensorId) readers.size())
                        {
                            readers[(size_t) in].push_back(j);
                        }
                    }
                }
                std::set<size_t> flexSet(flexible.begin(), flexible.end());
                for (int round = 0; round < kFlexVoteMaxRounds; ++round)
                {
                    bool changed = false;
                    for (size_t i: flexible)
                    {
                        Node   &nd       = g.nodes[i];
                        int64_t flatCost = 0; // elements converted if this node runs FLAT
                        int64_t nc4Cost  = 0; // ... if it runs NC4HW4
                        auto    voteEdge = [&](TensorId t, bool neighborFlat) {
                            const int64_t w = numElements(g.desc(t).shape);
                            (neighborFlat ? nc4Cost : flatCost) += w;
                        };
                        for (TensorId in: nd.inputs)
                        {
                            if (in != kNoTensor && !g.isInitializer(in))
                            {
                                voteEdge(in, g.desc(in).gpuFlat);
                            }
                        }
                        for (TensorId o: nd.outputs)
                        {
                            if (o == kNoTensor)
                            {
                                continue;
                            }
                            for (size_t rj: readers[(size_t) o])
                            {
                                const Node &R = g.nodes[rj];
                                if (layoutAgnostic(R))
                                {
                                    continue; // adopts whatever this node chooses: no convert either way
                                }
                                voteEdge(o, flexSet.count(rj) ? R.attr.geti("pw_flat", 0) != 0 : gpuFlatNode(g, R));
                            }
                        }
                        // Ties keep the classifier's original choice (already in pw_flat): stability
                        // and determinism over churn.
                        const bool wantFlat = flatCost != nc4Cost ? flatCost < nc4Cost : nd.attr.geti("pw_flat", 0) != 0;
                        if (wantFlat != (nd.attr.geti("pw_flat", 0) != 0))
                        {
                            Attr a;
                            a.kind                 = Attr::Int;
                            a.i                    = wantFlat ? 1 : 0;
                            nd.attr.map["pw_flat"] = a;
                            changed                = true;
                        }
                    }
                    if (!changed)
                    {
                        break;
                    }
                    // Re-seed every tensor layout with the updated pw_flat facts so the next round's
                    // (and the convert splicer's) view of neighbor layouts is consistent.
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
                            f = gpuFlatNode(g, nd);
                        }
                        for (TensorId o: nd.outputs)
                        {
                            if (o != kNoTensor)
                            {
                                g.desc(o).gpuFlat = f;
                            }
                        }
                    }
                }
            }
        }
        // 2b) Graph inputs have no producer, so the seed above never reaches them and they keep the
        //    NC4HW4 default — which forces a full-tensor ConvertLayout in front of every flat reader.
        //    On a with-past decoder that is one convert of the WHOLE resident KV cache per layer per
        //    decode step. An input read exclusively in flat layout is instead assigned flat, so the
        //    host packs it flat once and the converts disappear; a mixed-layout or all-NC4HW4 input
        //    keeps the default, so no convert is ever merely relocated. ConvertLayout is a lossless
        //    same-dtype reorder, so the result is byte-identical either way, and the rule is a pure
        //    function of the graph (adoption only ever flips NC4HW4 -> flat, so the loop below is
        //    monotone and its fixed point is unique).
        {
            constexpr int kInputLayoutMaxRounds = 8; // monotone; this only caps pathological churn
            for (int round = 0; round < kInputLayoutMaxRounds; ++round)
            {
                std::set<TensorId> produced;
                for (const Node &nd: g.nodes)
                {
                    for (TensorId o: nd.outputs)
                    {
                        if (o != kNoTensor)
                        {
                            produced.insert(o);
                        }
                    }
                }
                std::map<TensorId, bool> allReadersFlat; // absent => no reader seen yet
                for (const Node &nd: g.nodes)
                {
                    for (size_t inIdx = 0; inIdx < nd.inputs.size(); ++inIdx)
                    {
                        TensorId in = nd.inputs[inIdx];
                        if (in == kNoTensor || g.isInitializer(in) || produced.count(in))
                        {
                            continue;
                        }
                        bool &all = allReadersFlat.emplace(in, true).first->second;
                        all       = all && readerWantsFlat(g, nd, inIdx);
                    }
                    for (TensorId edge: {nd.fusedResidual, nd.fusedBias})
                    {
                        if (edge == kNoTensor || g.isInitializer(edge) || produced.count(edge))
                        {
                            continue;
                        }
                        bool &all = allReadersFlat.emplace(edge, true).first->second;
                        all       = all && (!nd.outputs.empty() && nd.outputs[0] != kNoTensor && g.desc(nd.outputs[0]).gpuFlat);
                    }
                }
                bool changed = false;
                for (const auto &entry: allReadersFlat)
                {
                    TensorDesc &d = g.desc(entry.first);
                    if (entry.second && !d.gpuFlat)
                    {
                        d.gpuFlat = true;
                        changed   = true;
                    }
                }
                if (!changed)
                {
                    break;
                }
                seedFromProducers(); // an agnostic reader propagates the new input layout downstream
            }
        }
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
    namespace {

        /// Name the SEAMS the layout converts sit on, grouped by the pair of ops they bridge.
        ///
        /// A convert count alone says a graph pays for layout changes; it does not say where, and a
        /// profile that shows ConvertLayout among the top costs leaves nothing to act on. Every
        /// convert lies between one producer and one consumer, and the same pair usually recurs --
        /// so a handful of op pairs explains the whole bill, and each pair is a question with an
        /// answer ("can this consumer read the producer's layout?").
        void reportConvertSeams(const Graph &g) {
            std::vector<int> producer(g.tensors.size(), -1);
            for (int ni = 0; ni < (int) g.nodes.size(); ++ni)
            {
                for (TensorId o: g.nodes[ni].outputs)
                {
                    if (o != kNoTensor)
                    {
                        producer[(size_t) o] = ni;
                    }
                }
            }
            // seam -> (count, elements converted), keyed by "<producer op> -> <consumer op>".
            // The element total is what decides where effort goes: a dozen converts over reduced
            // scalars cost nothing, while a handful over full-resolution maps is the whole bill, and
            // a count alone ranks them identically.
            struct Seam {
                int     converts = 0;
                int64_t elements = 0;
            };
            std::map<std::string, Seam> seams;
            for (int ni = 0; ni < (int) g.nodes.size(); ++ni)
            {
                const Node &nd = g.nodes[(size_t) ni];
                if (nd.type != OpType::ConvertLayout || nd.inputs.empty() || nd.inputs[0] == kNoTensor)
                {
                    continue;
                }
                const int   src  = producer[(size_t) nd.inputs[0]];
                const char *from = src >= 0 ? opTypeName(g.nodes[(size_t) src].type) : "(graph input)";
                // Which side asked for flat. subOp 0 converts NC4HW4 -> flat (the CONSUMER wants
                // flat), 1 converts the other way (the PRODUCER was flat). Without this the pair
                // names two ops and leaves it open which of them has no blocked kernel.
                const char *direction = nd.subOp == 0 ? "   [consumer wants flat]" : "   [producer was flat]";
                // A convert may feed several consumers; each is its own seam, and a convert with no
                // consumer feeds a graph output.
                bool consumed = false;
                for (const Node &other: g.nodes)
                {
                    if (&other == &nd)
                    {
                        continue;
                    }
                    for (TensorId in: other.inputs)
                    {
                        if (in != kNoTensor && in == nd.outputs[0])
                        {
                            Seam &seam = seams[std::string(from) + " -> " + opTypeName(other.type) + direction];
                            seam.converts++;
                            seam.elements += numElements(g.desc(nd.outputs[0]).shape);
                            consumed = true;
                            break;
                        }
                    }
                }
                if (!consumed)
                {
                    Seam &seam = seams[std::string(from) + " -> (graph output)" + direction];
                    seam.converts++;
                    seam.elements += numElements(g.desc(nd.outputs[0]).shape);
                }
            }
            std::vector<std::pair<std::string, Seam>> ranked(seams.begin(), seams.end());
            // Ranked by ELEMENTS moved, not by how many converts there are.
            std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
                return a.second.elements > b.second.elements;
            });
            int64_t total = 0;
            for (const auto &r: ranked)
            {
                total += r.second.elements;
            }
            constexpr size_t kSeamsReported = 8;
            int64_t          shown          = 0;
            for (size_t i = 0; i < ranked.size() && i < kSeamsReported; ++i)
            {
                const Seam &s2 = ranked[i].second;
                shown += s2.elements;
                VKNN_INFO << "insertLayoutConverts:   " << s2.converts << "x  " << (total ? s2.elements * 100 / total : 0) << "% of converted elements  "
                          << ranked[i].first;
            }
            if (ranked.size() > kSeamsReported)
            {
                VKNN_INFO << "insertLayoutConverts:   ... " << (ranked.size() - kSeamsReported) << " further seam(s), " << (total ? (total - shown) * 100 / total : 0) << "% of converted elements";
            }
        }

    } // namespace

    void insertLayoutConverts(Graph &g) {
        // Assign every tensor a layout (minimising converts), then for every node input whose layout differs
        // from what the consumer needs, splice in a ConvertLayout node.
        globalLayoutAssign(g);
        std::map<std::pair<TensorId, bool>, TensorId> cache; // (tensor, needFlat) -> converted tensor
        std::vector<Node>                             converts;
        int                                           n = 0;
        // Route one tensor read through a ConvertLayout when its layout differs from the layout the
        // reader operates in, reusing an already-converted copy from `cache`. `ref` is the reader's
        // reference (a node input or a fused edge) and is rewired to the converted tensor in place.
        auto convertRead = [&](TensorId &ref, bool wantFlat) {
            if (ref == kNoTensor || g.isInitializer(ref))
            {
                return; // constants handled inside flat ops
            }
            if (g.desc(ref).gpuFlat == wantFlat)
            {
                return;
            }
            auto key = std::make_pair(ref, wantFlat);
            auto it  = cache.find(key);
            if (it == cache.end())
            {
                TensorDesc d    = g.desc(ref);
                d.name          = g.desc(ref).name + (wantFlat ? "#flat" : "#nc4") + std::to_string(n);
                d.isInitializer = d.isInput = d.isOutput = false;
                d.gpuFlat                                = wantFlat;
                TensorId t2                              = g.addTensor(d);
                Node     cv;
                cv.type    = OpType::ConvertLayout;
                cv.name    = "convert" + std::to_string(n++);
                cv.subOp   = wantFlat ? 0 : 1; // 0: NC4HW4->flat, 1: flat->NC4HW4
                cv.inputs  = {ref};
                cv.outputs = {t2};
                converts.push_back(cv);
                it = cache.emplace(key, t2).first;
            }
            ref = it->second;
        };
        for (auto &nd: g.nodes)
        {
            if (nd.outputs.empty() || nd.outputs[0] == kNoTensor)
            {
                continue;
            }
            bool needFlat = g.desc(nd.outputs[0]).gpuFlat; // the format this node operates in
            for (size_t inIdx = 0; inIdx < nd.inputs.size(); ++inIdx)
            {
                // readerWantsFlat is the ONE rule for which layout a reader operates an input slot
                // in; globalLayoutAssign derived the assignment from it, so the splicer has to ask
                // the same function or the two drift and a convert gets moved instead of removed.
                convertRead(nd.inputs[inIdx], readerWantsFlat(g, nd, inIdx));
            }
            // Fused residual/bias edges are reads outside the inputs list (rewireTensor's contract)
            // and the kernel decodes them in ITS layout world, so they take the same converts as any
            // input. An edge mirrored into inputs (a conv residual doubling at the bias slot; the conv
            // kernel tests inputs[2] != fusedResidual to tell the two apart) resolves through the same
            // cache entry, so the mirrored entry and the edge stay one tensor.
            convertRead(nd.fusedResidual, needFlat);
            convertRead(nd.fusedBias, needFlat);
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
            d.isOutput                  = true;
            d.gpuFlat                   = true;
            // The flat convert output KEEPS the model's declared output name (callers look up outputs by
            // name); the pre-convert tensor becomes an internal boundary and is renamed to stay unique.
            g.desc(out).name += "#nc4";
            g.desc(out).isOutput = false;
            TensorId t2          = g.addTensor(d);
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
            reportConvertSeams(g);
        }
        reportFlatInventory(g);
    }

} // namespace vknn
