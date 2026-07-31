// Rank collapse of pointwise runs beyond the fused plan's rank budget (collapsePwRunRanks).
//
// PwPlanner has no encoding for a run of rank above kPwMaxRank -- neither the flat plan (bounded
// broadcast rank) nor the NC4HW4 plan (rank 4 exactly) covers it -- so without the collapse a
// rank-5 elementwise chain never fuses and decays to standalone per-op dispatches. The pre-pass
// regroups adjacent run axes: two axes merge only when every data input of the region carries
// them BOTH full or BOTH broadcast (a NumPy-legal merge), run axes of extent 1 constrain nothing,
// and the cut lands immediately left of the conflicting axis, so the grouping is a pure function
// of the shapes. Regions rewire through "#pwrc" views: members connect DIRECTLY at the grouped
// shape, boundary runtime inputs collapse behind Reshape views (an initializer instead becomes a
// grouped-shape constant carrying the source's bytes), and expand Reshapes restore the full-rank
// tensors only at the region's true boundary -- including for a value read out of band through a
// fused residual/bias edge. These tests pin the merge rule, the no-legal-grouping refusal, rank<=4
// byte-stability (no "#pwrc" views appear), the mid-region and cross-region expand ordering, the
// grouped constants' payload ownership, and exact CPU execution of collapsed rank-5 and rank-6
// chains against plain right-aligned broadcast loops.
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    // Rank-5 multi-view run extents [N,V,C,H,W] -- the batch x views x channels x spatial family
    // that rank>4 elementwise blocks appear with in multi-view/attention-style graphs.
    constexpr int64_t kRunN = 2;
    constexpr int64_t kRunV = 3;
    constexpr int64_t kRunC = 4;
    constexpr int64_t kRunH = 5;
    constexpr int64_t kRunW = 6;
    // A broadcast (size-1) mask axis.
    constexpr int64_t kBcast = 1;

    const Shape kRun5 {kRunN, kRunV, kRunC, kRunH, kRunW};

    // MSVC at C++17 rejects designated initializers; a tiny helper names the tensor.
    TensorDesc namedDesc(const char *name) {
        TensorDesc d;
        d.name = name;
        return d;
    }

    // Distinct-per-position mask payload so a wrong axis stride reads a grossly different value.
    std::vector<float> maskPayload(const Shape &s) {
        std::vector<float> v((size_t) numElements(s));
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = 0.25f + 0.125f * (float) (i % 13);
        }
        return v;
    }

    // A built chain plus the ids the structural tests attach extra readers to.
    struct MaskChain {
        Graph                 g;
        std::vector<TensorId> masks;       // one initializer (or runtime activation) per mask shape
        std::vector<TensorId> stepOutputs; // one Binary output per mask shape, in order
    };

    // x[run] -> Abs -> chain of Binary ops, one per mask shape in `maskShapes`. Masks are
    // INITIALIZERS consumed directly by the Binary steps. Ops alternate Mul/Add so no two steps
    // merge. Rank-generic: the run may be the rank-5 shape under test or a rank-4 control.
    MaskChain buildConstMaskChain(const Shape &run, const std::vector<Shape> &maskShapes) {
        MaskChain  chain;
        Graph     &g = chain.g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = run;
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};

        TensorId t = g.addTensor(namedDesc("t"));
        Node     act;
        act.type    = OpType::Unary;
        act.name    = "producer";
        act.subOp   = (int) UnaryType::Abs;
        act.inputs  = {x};
        act.outputs = {t};
        g.nodes.push_back(act);

        TensorId cur = t;
        for (size_t k = 0; k < maskShapes.size(); ++k)
        {
            TensorDesc mi;
            mi.name                     = "m" + std::to_string(k);
            mi.shape                    = maskShapes[k];
            mi.isInitializer            = true;
            TensorId                 m  = g.addTensor(mi);
            const std::vector<float> mv = maskPayload(maskShapes[k]);
            HostBuffer               hb;
            hb.resizeElems((int64_t) mv.size(), DType::Float32);
            std::memcpy(hb.f32(), mv.data(), mv.size() * sizeof(float));
            g.initializers[m] = std::move(hb);
            chain.masks.push_back(m);

            TensorId y = g.addTensor(namedDesc(("y" + std::to_string(k)).c_str()));
            Node     bin;
            bin.type    = OpType::Binary;
            bin.name    = "mask_step_" + std::to_string(k);
            bin.subOp   = (int) (k % 2 == 0 ? BinaryType::Mul : BinaryType::Add);
            bin.inputs  = {cur, m};
            bin.outputs = {y};
            g.nodes.push_back(bin);
            chain.stepOutputs.push_back(y);
            cur = y;
        }
        g.outputs = {cur};
        return chain;
    }

    Graph buildConstMaskChainGraph(const Shape &run, const std::vector<Shape> &maskShapes) {
        return buildConstMaskChain(run, maskShapes).g;
    }

    // Same chain with the masks as RUNTIME graph inputs routed through their own Abs, so each mask
    // reaches the region as a non-scalar broadcast activation and enters through a collapse Reshape
    // rather than a grouped constant copy.
    MaskChain buildRuntimeMaskChain(const Shape &run, const std::vector<Shape> &maskShapes) {
        MaskChain  chain;
        Graph     &g = chain.g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = run;
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};

        TensorId t = g.addTensor(namedDesc("t"));
        Node     act;
        act.type    = OpType::Unary;
        act.name    = "producer";
        act.subOp   = (int) UnaryType::Abs;
        act.inputs  = {x};
        act.outputs = {t};
        g.nodes.push_back(act);

        TensorId cur = t;
        for (size_t k = 0; k < maskShapes.size(); ++k)
        {
            TensorDesc mi;
            mi.name    = "m" + std::to_string(k);
            mi.shape   = maskShapes[k];
            mi.isInput = true;
            TensorId m = g.addTensor(mi);
            g.inputs.push_back(m);

            TensorId mm = g.addTensor(namedDesc(("mm" + std::to_string(k)).c_str()));
            Node     mact;
            mact.type    = OpType::Unary;
            mact.name    = "mask_producer_" + std::to_string(k);
            mact.subOp   = (int) UnaryType::Abs;
            mact.inputs  = {m};
            mact.outputs = {mm};
            g.nodes.push_back(mact);
            chain.masks.push_back(mm);

            TensorId y = g.addTensor(namedDesc(("y" + std::to_string(k)).c_str()));
            Node     bin;
            bin.type    = OpType::Binary;
            bin.name    = "mask_step_" + std::to_string(k);
            bin.subOp   = (int) (k % 2 == 0 ? BinaryType::Mul : BinaryType::Add);
            bin.inputs  = {cur, mm};
            bin.outputs = {y};
            g.nodes.push_back(bin);
            chain.stepOutputs.push_back(y);
            cur = y;
        }
        g.outputs = {cur};
        return chain;
    }

    // True when some node writes `t` -- the invariant a value read anywhere in the graph must keep.
    bool hasProducer(const Graph &g, TensorId t) {
        for (const Node &nd: g.nodes)
        {
            for (TensorId o: nd.outputs)
            {
                if (o == t)
                {
                    return true;
                }
            }
        }
        return false;
    }

    // Index of the node writing `t`, or the node count when nothing does.
    size_t producerIndex(const Graph &g, TensorId t) {
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId o: g.nodes[i].outputs)
            {
                if (o == t)
                {
                    return i;
                }
            }
        }
        return g.nodes.size();
    }

    // Index of the first node reading `t` as an input, or the node count when nothing does.
    size_t firstReaderIndex(const Graph &g, TensorId t) {
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId in: g.nodes[i].inputs)
            {
                if (in == t)
                {
                    return i;
                }
            }
        }
        return g.nodes.size();
    }

    // x -> Abs -> Mul(m) -> Relu with m a same-shape RUNTIME graph input: the all-full rank-5
    // chain whose axes all merge, so it must collapse to a single flat axis and fuse.
    Graph buildSameShapeRuntimeChainGraph(const Shape &run) {
        Graph      g;
        TensorDesc xi;
        xi.name      = "x";
        xi.shape     = run;
        xi.isInput   = true;
        TensorId   x = g.addTensor(xi);
        TensorDesc mi;
        mi.name    = "m";
        mi.shape   = run;
        mi.isInput = true;
        TensorId m = g.addTensor(mi);
        g.inputs   = {x, m};

        TensorId t = g.addTensor(namedDesc("t"));
        Node     act;
        act.type    = OpType::Unary;
        act.name    = "producer";
        act.subOp   = (int) UnaryType::Abs;
        act.inputs  = {x};
        act.outputs = {t};
        g.nodes.push_back(act);

        TensorId y = g.addTensor(namedDesc("y"));
        Node     mul;
        mul.type    = OpType::Binary;
        mul.name    = "mask_mul";
        mul.subOp   = (int) BinaryType::Mul;
        mul.inputs  = {t, m};
        mul.outputs = {y};
        g.nodes.push_back(mul);

        TensorId z = g.addTensor(namedDesc("z"));
        Node     relu;
        relu.type    = OpType::Relu;
        relu.name    = "gate";
        relu.inputs  = {y};
        relu.outputs = {z};
        g.nodes.push_back(relu);

        g.outputs = {z};
        return g;
    }

    const Node *findFusedPointwise(const Graph &g) {
        for (const Node &nd: g.nodes)
        {
            if (nd.type == OpType::FusedPointwise)
            {
                return &nd;
            }
        }
        return nullptr;
    }

    // Right-aligned rank-`rank` reading of a mask shape (leading axes filled with 1).
    Shape rightAligned(const Shape &s, size_t rank) {
        Shape rs = s;
        rs.insert(rs.begin(), rank - rs.size(), kBcast);
        return rs;
    }

    // Flat mask element a right-aligned NumPy broadcast read yields at run position `idx`:
    // a broadcast axis reads index 0.
    size_t maskOffset(const Shape &aligned, const std::vector<int64_t> &idx) {
        size_t off = 0;
        for (size_t k = 0; k < aligned.size(); ++k)
        {
            off = off * (size_t) aligned[k] + (size_t) (aligned[k] == kBcast ? 0 : idx[k]);
        }
        return off;
    }
} // namespace

// An all-full rank-5 chain (same-shape runtime second input) merges every adjacent axis pair, so
// the region collapses to one flat axis and fuses into a single FusedPointwise unit. Before the
// collapse existed, PwPlanner refused the rank-5 run outright and no fused node appeared.
TEST(PwRankCollapse, SameShapeRank5ChainCollapsesAndFuses) {
    Graph g = buildSameShapeRuntimeChainGraph(kRun5);
    inferShapes(g, 1);
    fusePointwiseChains(g, /*strictFuse*/ false);

    const Node *unit = findFusedPointwise(g);
    ASSERT_NE(unit, nullptr) << "a rank-5 all-full elementwise chain must collapse and fuse";
    const Shape flatRun {numElements(kRun5)};
    EXPECT_EQ(g.desc(unit->outputs[0]).shape, flatRun) << "every axis pair is mergeable, so the coarsest grouping is one flat axis";
    ASSERT_TRUE(unit->attr.has("pw_steps"));
    EXPECT_EQ(unit->attr.getints("pw_steps").size(), (size_t) (3 * kPwStepInts)) << "Abs, Mul, and Relu must all encode into the one unit";

    // The full-rank value reappears only at the region's true boundary: an expand Reshape feeds
    // the graph output, whose declared shape is untouched.
    ASSERT_EQ(g.outputs.size(), 1u);
    EXPECT_EQ(g.desc(g.outputs[0]).shape, kRun5) << "the graph output keeps its full-rank contract";
    const Node *expand = nullptr;
    for (const Node &nd: g.nodes)
    {
        if (nd.type == OpType::Reshape && !nd.outputs.empty() && nd.outputs[0] == g.outputs[0])
        {
            expand = &nd;
        }
    }
    ASSERT_NE(expand, nullptr) << "an expand Reshape must regenerate the full-rank graph output";
    EXPECT_NE(expand->name.find("#pwrc"), std::string::npos);
}

// Mask [N,V,1,1,W] against [N,V,C,H,W]: the (C,H) pair is (1,1) for the mask so those axes merge;
// (V,C) is (full,1) and (H,W) is (1,full), so both of those boundaries stay cut. The coarsest
// grouping is [N*V, C*H, W], and the mask's grouped copy is [N*V, 1, W].
TEST(PwRankCollapse, PartialBroadcastMergesOnlyLegalAdjacentPairs) {
    Graph g = buildConstMaskChainGraph(kRun5, {{kRunN, kRunV, kBcast, kBcast, kRunW}});
    inferShapes(g, 1);
    fusePointwiseChains(g, /*strictFuse*/ false);

    const Node *unit = findFusedPointwise(g);
    ASSERT_NE(unit, nullptr) << "a rank-5 chain with a groupable broadcast mask must collapse and fuse";
    const Shape groupedRun {kRunN * kRunV, kRunC * kRunH, kRunW};
    EXPECT_EQ(g.desc(unit->outputs[0]).shape, groupedRun) << "only the (C,H) pair may merge; (V,C) and (H,W) are blocked by the mask";
    ASSERT_FALSE(unit->inputs.empty());
    EXPECT_EQ(g.desc(unit->inputs[0]).shape, groupedRun) << "the entry streams at the grouped run shape";

    const Shape groupedMask {kRunN * kRunV, kBcast, kRunW};
    bool        haveGroupedMask = false;
    for (TensorId in: unit->inputs)
    {
        haveGroupedMask = haveGroupedMask || (g.isInitializer(in) && g.desc(in).shape == groupedMask);
    }
    EXPECT_TRUE(haveGroupedMask) << "the mask initializer must enter as a grouped-shape copy " << shapeStr(groupedMask);
    EXPECT_EQ(g.desc(g.outputs[0]).shape, kRun5) << "the graph output keeps its full-rank contract";
}

// Mask [N,1,C,1,W] alternates full/broadcast on every axis, so every adjacent pair has the mask
// full on one axis and broadcast on the other -- no merge is legal, five groups exceed
// kPwMaxRank, and the region must stay at full rank, untouched and unfused.
TEST(PwRankCollapse, AlternatingBroadcastBlocksEveryMergeAndStaysUnchanged) {
    Graph g = buildConstMaskChainGraph(kRun5, {{kRunN, kBcast, kRunC, kBcast, kRunW}});
    inferShapes(g, 1);
    const size_t nodesBefore = g.nodes.size();
    fusePointwiseChains(g, /*strictFuse*/ false);

    EXPECT_EQ(g.nodes.size(), nodesBefore) << "a region with no legal grouping must not be rewired";
    for (const Node &nd: g.nodes)
    {
        EXPECT_NE(nd.type, OpType::FusedPointwise) << "a rank-5 run with no legal grouping has no fusable form";
        EXPECT_FALSE(nd.attr.has("pw_steps"));
        EXPECT_EQ(nd.name.find("#pwrc"), std::string::npos);
    }
    for (const TensorDesc &td: g.tensors)
    {
        EXPECT_EQ(td.name.find("#pwrc"), std::string::npos) << "no collapse view may appear for a refused region";
    }
    EXPECT_EQ(g.desc(g.outputs[0]).shape, kRun5);
}

// CPU execution of a collapsed and fused rank-5 chain must select the same element the plain
// right-aligned broadcast walk selects, at every position. The CPU fused kernel chains fp32 steps
// in the same order as the loops, so equality is exact.
TEST(PwRankCollapse, CpuCollapsedChainMatchesReferenceLoops) {
    const std::vector<Shape> masks {{kRunN, kRunV, kBcast, kBcast, kRunW}, {kBcast, kBcast, kBcast, kBcast, kRunW}};

    // The comparison below must exercise the collapsed fused unit, not an unfused fallback: prove
    // on an identically built graph that the pass yields a FusedPointwise at a collapsed rank.
    {
        Graph probe = buildConstMaskChainGraph(kRun5, masks);
        inferShapes(probe, 1);
        fusePointwiseChains(probe, /*strictFuse*/ false);
        const Node *unit = findFusedPointwise(probe);
        ASSERT_NE(unit, nullptr) << "the CPU comparison graph must fuse";
        ASSERT_LE((int) probe.desc(unit->outputs[0]).shape.size(), kPwMaxRank) << "the fused unit must run at a collapsed rank";
    }

    Graph  g = buildConstMaskChainGraph(kRun5, masks);
    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_NE(sess, nullptr);

    const size_t       total = (size_t) numElements(kRun5);
    std::vector<float> src(total);
    for (size_t i = 0; i < total; ++i)
    {
        src[i] = 0.5f + (float) (i % 11) * 0.125f;
    }
    std::vector<IOTensor> in, out;
    in.resize(1);
    in[0].name  = "x";
    in[0].shape = kRun5;
    in[0].dtype = DType::Float32;
    in[0].data.resize(src.size() * sizeof(float));
    std::memcpy(in[0].data.data(), src.data(), src.size() * sizeof(float));
    ASSERT_EQ(sess->run(in, out), Status::Ok);
    ASSERT_EQ(out.size(), 1u);

    std::vector<std::vector<float>> maskValues;
    for (const Shape &s: masks)
    {
        maskValues.push_back(maskPayload(s));
    }

    const float *got = out[0].f32();
    for (size_t i = 0; i < total; ++i)
    {
        // per-axis run indices of flat position i (row-major)
        std::vector<int64_t> idx(kRun5.size());
        size_t               rem = i;
        for (size_t k = kRun5.size(); k-- > 0;)
        {
            idx[k] = (int64_t) (rem % (size_t) kRun5[k]);
            rem /= (size_t) kRun5[k];
        }
        float ref = std::abs(src[i]);
        for (size_t k = 0; k < masks.size(); ++k)
        {
            const float mv = maskValues[k][maskOffset(rightAligned(masks[k], kRun5.size()), idx)];
            ref            = k % 2 == 0 ? ref * mv : ref + mv;
        }
        EXPECT_FLOAT_EQ(got[i], ref) << "flat position " << i;
    }
}

// An INTERIOR member's value read from outside the region -- and exported as a graph output --
// keeps its full-rank form: the expand Reshape lands mid-region, before the outside reader, and the
// interior edge to the next member still connects at the grouped shape.
TEST(PwRankCollapse, InteriorMemberValueReadOutsideIsExpandedBeforeItsReader) {
    MaskChain      chain    = buildConstMaskChain(kRun5, {{kRunN, kRunV, kBcast, kBcast, kRunW}, {kRunN, kRunV, kBcast, kBcast, kRunW}});
    Graph         &g        = chain.g;
    const TensorId interior = chain.stepOutputs[0];

    // A Reduce over the last axis is not pointwise-fusable, so it stays outside the region and its
    // read of the interior value is a real external use.
    TensorDesc ro;
    ro.name    = "reduced";
    TensorId r = g.addTensor(ro);
    Node     red;
    red.type    = OpType::Reduce;
    red.name    = "outside_reader";
    red.subOp   = (int) ReduceType::Sum;
    red.inputs  = {interior};
    red.outputs = {r};
    Attr axes;
    axes.kind            = Attr::Ints;
    axes.ints            = {(int64_t) kRun5.size() - 1};
    red.attr.map["axes"] = axes;
    Attr keep;
    keep.kind                = Attr::Int;
    keep.i                   = 0;
    red.attr.map["keepdims"] = keep;
    g.nodes.push_back(red);
    g.outputs.push_back(interior); // ... and the same value is a graph output, mid-region
    g.outputs.push_back(r);

    inferShapes(g, 1);
    fusePointwiseChains(g, /*strictFuse*/ false);

    ASSERT_TRUE(hasProducer(g, interior)) << "an externally read member value must keep a producer";
    EXPECT_EQ(g.desc(interior).shape, kRun5) << "the exported value keeps its full-rank contract";
    EXPECT_LT(producerIndex(g, interior), firstReaderIndex(g, interior)) << "the expand Reshape must precede the outside reader";
}

// A member value read OUT OF BAND -- through another node's fused residual edge, which is NOT
// mirrored into that node's inputs list -- is an external use like any other: the full-rank tensor
// must be regenerated, or the edge would point at a value nothing produces.
TEST(PwRankCollapse, MemberValueReadThroughAFusedResidualEdgeStaysProduced) {
    MaskChain      chain    = buildConstMaskChain(kRun5, {{kRunN, kRunV, kBcast, kBcast, kRunW}, {kRunN, kRunV, kBcast, kBcast, kRunW}});
    Graph         &g        = chain.g;
    const TensorId interior = chain.stepOutputs[0];

    // A conv over its own rank-4 activation; the residual edge is the only reference to the
    // interior rank-5 value, and fused edges live outside the inputs list by contract.
    TensorDesc ci;
    ci.name    = "c";
    ci.shape   = {1, kRunC, kRunH, kRunW};
    ci.isInput = true;
    TensorId c = g.addTensor(ci);
    g.inputs.push_back(c);

    TensorDesc wi;
    wi.name          = "cw";
    wi.shape         = {kRunC, kRunC, 1, 1};
    wi.isInitializer = true;
    TensorId   w     = g.addTensor(wi);
    HostBuffer hb;
    hb.resizeElems(kRunC * kRunC, DType::Float32);
    for (int64_t i = 0; i < kRunC * kRunC; ++i)
    {
        hb.f32()[i] = i % (kRunC + 1) == 0 ? 1.0f : 0.0f; // values are not the point
    }
    g.initializers[w] = std::move(hb);

    TensorId cy = g.addTensor(namedDesc("cy"));
    Node     conv;
    conv.type          = OpType::Conv;
    conv.name          = "residual_host";
    conv.inputs        = {c, w};
    conv.outputs       = {cy};
    conv.fusedResidual = interior;
    g.nodes.push_back(conv);
    g.outputs.push_back(cy);

    inferShapes(g, 1);
    fusePointwiseChains(g, /*strictFuse*/ false);

    const Node *host = nullptr;
    for (const Node &nd: g.nodes)
    {
        if (nd.type == OpType::Conv)
        {
            host = &nd;
        }
    }
    ASSERT_NE(host, nullptr);
    EXPECT_EQ(host->fusedResidual, interior) << "the collapse must not rewire an out-of-band edge";
    EXPECT_TRUE(hasProducer(g, interior)) << "a value read only through a fused edge still needs its producer";
    EXPECT_EQ(g.desc(interior).shape, kRun5) << "the residual edge reads the full-rank value";
}

// Two regions at different run shapes, the second reading the first's full-rank output as a
// broadcast operand: the second region's collapse Reshape of that tensor must land AFTER the expand
// Reshape that regenerates it (both want the same slot -- the first region's last member + 1).
TEST(PwRankCollapse, SecondRegionReadsTheFirstRegionsExpandedValueInOrder) {
    // Region 1 runs at [N,V,C,1,1]; region 2 at the full [N,V,C,H,W], with region 1's output as a
    // 1-or-full broadcast operand, so the two never merge into one region.
    const Shape    firstRun {kRunN, kRunV, kRunC, kBcast, kBcast};
    MaskChain      chain    = buildConstMaskChain(firstRun, {{kRunN, kRunV, kBcast, kBcast, kBcast}});
    Graph         &g        = chain.g;
    const TensorId firstOut = chain.stepOutputs.back();

    TensorDesc xi;
    xi.name     = "x2";
    xi.shape    = kRun5;
    xi.isInput  = true;
    TensorId x2 = g.addTensor(xi);
    g.inputs.push_back(x2);

    TensorId t2 = g.addTensor(namedDesc("t2"));
    Node     act;
    act.type    = OpType::Unary;
    act.name    = "second_producer";
    act.subOp   = (int) UnaryType::Abs;
    act.inputs  = {x2};
    act.outputs = {t2};
    g.nodes.push_back(act);

    TensorId y2 = g.addTensor(namedDesc("y2"));
    Node     bin;
    bin.type    = OpType::Binary;
    bin.name    = "second_step";
    bin.subOp   = (int) BinaryType::Mul;
    bin.inputs  = {t2, firstOut};
    bin.outputs = {y2};
    g.nodes.push_back(bin);
    g.outputs = {y2};

    inferShapes(g, 1);
    fusePointwiseChains(g, /*strictFuse*/ false);

    ASSERT_TRUE(hasProducer(g, firstOut)) << "the value the second region reads must be regenerated at full rank";
    EXPECT_EQ(g.desc(firstOut).shape, firstRun);
    const size_t expandAt = producerIndex(g, firstOut);
    const size_t readAt   = firstReaderIndex(g, firstOut);
    ASSERT_LT(readAt, g.nodes.size()) << "the second region must still read the full-rank value through its collapse view";
    EXPECT_LT(expandAt, readAt) << "an expand must precede the collapse view that reads it";
    EXPECT_EQ(g.nodes[readAt].type, OpType::Reshape) << "the second region enters through a collapse Reshape";
}

// A non-scalar RUNTIME broadcast operand enters through a collapse Reshape at the grouped shape --
// the runtime counterpart of the grouped constant copy -- and the values stay exact.
TEST(PwRankCollapse, RuntimeBroadcastBoundaryEntersThroughACollapseReshape) {
    const Shape    maskShape {kBcast, kBcast, kRunC, kRunH, kRunW};
    MaskChain      chain = buildRuntimeMaskChain(kRun5, {maskShape});
    Graph         &g     = chain.g;
    const TensorId mask  = chain.masks[0];
    inferShapes(g, 1);
    fusePointwiseChains(g, /*strictFuse*/ false);

    // (N,V) are broadcast and (C,H,W) full, so one cut falls between them: [N*V, C*H*W].
    const Shape groupedRun {kRunN * kRunV, kRunC * kRunH * kRunW};
    const Shape groupedMask {kBcast, kRunC * kRunH * kRunW};
    const Node *unit = findFusedPointwise(g);
    ASSERT_NE(unit, nullptr) << "a groupable runtime broadcast must still collapse and fuse";
    EXPECT_EQ(g.desc(unit->outputs[0]).shape, groupedRun);

    const Node *view = nullptr;
    for (const Node &nd: g.nodes)
    {
        if (nd.type == OpType::Reshape && !nd.inputs.empty() && nd.inputs[0] == mask)
        {
            view = &nd;
        }
    }
    ASSERT_NE(view, nullptr) << "a runtime broadcast boundary input must enter behind a collapse Reshape";
    EXPECT_EQ(g.desc(view->outputs[0]).shape, groupedMask);
    EXPECT_NE(g.desc(view->outputs[0]).name.find("#pwrc"), std::string::npos);
    EXPECT_NE(std::find(unit->inputs.begin(), unit->inputs.end(), view->outputs[0]), unit->inputs.end())
        << "the unit must read the grouped view, not the full-rank tensor";
}

// The same runtime-broadcast chain, executed: the collapsed unit must select the element the plain
// right-aligned broadcast walk selects, at every position.
TEST(PwRankCollapse, CpuRuntimeBroadcastCollapsedChainMatchesReferenceLoops) {
    const Shape maskShape {kBcast, kBcast, kRunC, kRunH, kRunW};
    Graph       g = buildRuntimeMaskChain(kRun5, {maskShape}).g;

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_NE(sess, nullptr);

    const size_t       total = (size_t) numElements(kRun5);
    std::vector<float> src(total);
    for (size_t i = 0; i < total; ++i)
    {
        src[i] = 0.5f + (float) (i % 11) * 0.125f;
    }
    const std::vector<float> maskValues = maskPayload(maskShape);

    std::vector<IOTensor> in, out;
    in.resize(2);
    in[0].name  = "x";
    in[0].shape = kRun5;
    in[0].dtype = DType::Float32;
    in[0].data.resize(src.size() * sizeof(float));
    std::memcpy(in[0].data.data(), src.data(), src.size() * sizeof(float));
    in[1].name  = "m0";
    in[1].shape = maskShape;
    in[1].dtype = DType::Float32;
    in[1].data.resize(maskValues.size() * sizeof(float));
    std::memcpy(in[1].data.data(), maskValues.data(), maskValues.size() * sizeof(float));
    ASSERT_EQ(sess->run(in, out), Status::Ok);
    ASSERT_EQ(out.size(), 1u);

    const float *got = out[0].f32();
    for (size_t i = 0; i < total; ++i)
    {
        std::vector<int64_t> idx(kRun5.size());
        size_t               rem = i;
        for (size_t k = kRun5.size(); k-- > 0;)
        {
            idx[k] = (int64_t) (rem % (size_t) kRun5[k]);
            rem /= (size_t) kRun5[k];
        }
        const float ref = std::abs(src[i]) * std::abs(maskValues[maskOffset(maskShape, idx)]);
        EXPECT_FLOAT_EQ(got[i], ref) << "flat position " << i;
    }
}

// Rank 6: the grouping rule is rank-generic, so a rank-6 run collapses by the same cuts and
// executes exactly. Mask [1,1,1,H,W,K] leaves one cut, between the broadcast head and the full
// tail.
TEST(PwRankCollapse, RankSixChainCollapsesAndMatchesReferenceLoops) {
    const int64_t kRunK = 3;
    const Shape   run6 {kRunN, kRunV, kRunC, kRunH, kRunW, kRunK};
    const Shape   maskShape {kBcast, kBcast, kBcast, kRunH, kRunW, kRunK};
    const Shape   groupedRun {kRunN * kRunV * kRunC, kRunH * kRunW * kRunK};

    {
        Graph probe = buildConstMaskChainGraph(run6, {maskShape});
        inferShapes(probe, 1);
        fusePointwiseChains(probe, /*strictFuse*/ false);
        const Node *unit = findFusedPointwise(probe);
        ASSERT_NE(unit, nullptr) << "a groupable rank-6 chain must collapse and fuse";
        EXPECT_EQ(probe.desc(unit->outputs[0]).shape, groupedRun);
        EXPECT_EQ(probe.desc(probe.outputs[0]).shape, run6) << "the graph output keeps its full-rank contract";
    }

    Graph  g = buildConstMaskChainGraph(run6, {maskShape});
    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_NE(sess, nullptr);

    const size_t       total = (size_t) numElements(run6);
    std::vector<float> src(total);
    for (size_t i = 0; i < total; ++i)
    {
        src[i] = 0.5f + (float) (i % 11) * 0.125f;
    }
    std::vector<IOTensor> in, out;
    in.resize(1);
    in[0].name  = "x";
    in[0].shape = run6;
    in[0].dtype = DType::Float32;
    in[0].data.resize(src.size() * sizeof(float));
    std::memcpy(in[0].data.data(), src.data(), src.size() * sizeof(float));
    ASSERT_EQ(sess->run(in, out), Status::Ok);
    ASSERT_EQ(out.size(), 1u);

    const std::vector<float> maskValues = maskPayload(maskShape);
    const float             *got        = out[0].f32();
    for (size_t i = 0; i < total; ++i)
    {
        std::vector<int64_t> idx(run6.size());
        size_t               rem = i;
        for (size_t k = run6.size(); k-- > 0;)
        {
            idx[k] = (int64_t) (rem % (size_t) run6[k]);
            rem /= (size_t) run6[k];
        }
        const float ref = std::abs(src[i]) * maskValues[maskOffset(maskShape, idx)];
        EXPECT_FLOAT_EQ(got[i], ref) << "flat position " << i;
    }
}

// The grouped constant is a reshaped reading of the SAME bytes, so when the collapse consumed the
// source's last reference the copy takes the payload outright: one payload, not two.
TEST(PwRankCollapse, GroupedConstantTakesThePayloadOfASourceNothingElseReads) {
    const Shape              maskShape {kRunN, kRunV, kBcast, kBcast, kRunW};
    MaskChain                chain   = buildConstMaskChain(kRun5, {maskShape});
    Graph                   &g       = chain.g;
    const TensorId           source  = chain.masks[0];
    const std::vector<float> payload = maskPayload(maskShape);
    inferShapes(g, 1);
    fusePointwiseChains(g, /*strictFuse*/ false);

    EXPECT_FALSE(g.isInitializer(source)) << "a source the collapse consumed entirely must not keep a second payload";

    const Node *unit = findFusedPointwise(g);
    ASSERT_NE(unit, nullptr);
    const Shape groupedMask {kRunN * kRunV, kBcast, kRunW};
    TensorId    grouped = kNoTensor;
    for (TensorId in: unit->inputs)
    {
        if (g.isInitializer(in) && g.desc(in).shape == groupedMask)
        {
            grouped = in;
        }
    }
    ASSERT_NE(grouped, kNoTensor) << "the mask must enter as a grouped-shape constant";
    ASSERT_EQ(g.initializers.at(grouped).bytes.size(), payload.size() * sizeof(float)) << "the grouped constant carries the source payload";
    const float *bytes = g.initializers.at(grouped).f32();
    for (size_t i = 0; i < payload.size(); ++i)
    {
        EXPECT_FLOAT_EQ(bytes[i], payload[i]) << "element " << i;
    }
}

// A source another consumer still reads at the full rank keeps its own payload: two shapes, two
// tensors, both correct.
TEST(PwRankCollapse, GroupedConstantKeepsASourceASecondConsumerStillReads) {
    const Shape    maskShape {kRunN, kRunV, kBcast, kBcast, kRunW};
    MaskChain      chain  = buildConstMaskChain(kRun5, {maskShape});
    Graph         &g      = chain.g;
    const TensorId source = chain.masks[0];

    // A Reduce over the mask itself: not pointwise-fusable, so it stays outside the region and
    // holds the full-rank constant live.
    TensorDesc ro;
    ro.name     = "mask_reduced";
    ro.isOutput = true;
    TensorId r  = g.addTensor(ro);
    Node     red;
    red.type    = OpType::Reduce;
    red.name    = "mask_consumer";
    red.subOp   = (int) ReduceType::Sum;
    red.inputs  = {source};
    red.outputs = {r};
    Attr axes;
    axes.kind            = Attr::Ints;
    axes.ints            = {(int64_t) maskShape.size() - 1};
    red.attr.map["axes"] = axes;
    Attr keep;
    keep.kind                = Attr::Int;
    keep.i                   = 0;
    red.attr.map["keepdims"] = keep;
    g.nodes.push_back(red);
    g.outputs.push_back(r);

    inferShapes(g, 1);
    fusePointwiseChains(g, /*strictFuse*/ false);

    ASSERT_TRUE(g.isInitializer(source)) << "a constant a second consumer still reads keeps its payload";
    const std::vector<float> payload = maskPayload(maskShape);
    const float             *bytes   = g.initializers.at(source).f32();
    for (size_t i = 0; i < payload.size(); ++i)
    {
        EXPECT_FLOAT_EQ(bytes[i], payload[i]) << "element " << i;
    }
    const Shape groupedMask {kRunN * kRunV, kBcast, kRunW};
    bool        haveGrouped = false;
    for (const TensorDesc &td: g.tensors)
    {
        haveGrouped = haveGrouped || (td.shape == groupedMask && td.name.find("#pwrc") != std::string::npos);
    }
    EXPECT_TRUE(haveGrouped) << "the region still reads the mask at the grouped shape";
}

// Byte-stability control: a rank-4 graph is below the collapse threshold, so the pass must not
// touch it -- no "#pwrc" view, copy, or node may appear anywhere.
TEST(PwRankCollapse, RankFourGraphGainsNoCollapseViews) {
    Graph g = buildConstMaskChainGraph({kRunN, kRunC, kRunH, kRunW}, {{kRunN, kRunC, kBcast, kBcast}});
    inferShapes(g, 1);
    fusePointwiseChains(g, /*strictFuse*/ false);
    for (const Node &nd: g.nodes)
    {
        EXPECT_EQ(nd.name.find("#pwrc"), std::string::npos) << "rank-4 runs are outside the collapse threshold";
    }
    for (const TensorDesc &td: g.tensors)
    {
        EXPECT_EQ(td.name.find("#pwrc"), std::string::npos) << "rank-4 runs must gain no collapse tensors";
    }
}
