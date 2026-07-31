// Rank collapse of pointwise runs beyond the fused plan's rank budget (collapsePwRunRanks).
//
// PwPlanner has no encoding for a run of rank above kPwMaxRank -- neither the flat plan (bounded
// broadcast rank) nor the NC4HW4 plan (rank 4 exactly) covers it -- so without the collapse a
// rank-5 elementwise chain never fuses and decays to standalone per-op dispatches. The pre-pass
// regroups adjacent run axes: two axes merge only when every data input of the region carries
// them BOTH full or BOTH broadcast (a NumPy-legal merge), run axes of extent 1 constrain nothing,
// and the cut lands immediately left of the conflicting axis, so the grouping is a pure function
// of the shapes. Regions rewire through "#pwrc" views: members connect DIRECTLY at the grouped
// shape, boundary runtime inputs collapse behind Reshape views (initializers become byte-sharing
// grouped copies), and expand Reshapes restore the full-rank tensors only at the region's true
// boundary. These tests pin the merge rule, the no-legal-grouping refusal, rank<=4 byte-stability
// (no "#pwrc" views appear), and exact CPU execution of a collapsed chain against plain
// right-aligned broadcast loops.
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

    // x[run] -> Abs -> chain of Binary ops, one per mask shape in `maskShapes`. Masks are
    // INITIALIZERS consumed directly by the Binary steps. Ops alternate Mul/Add so no two steps
    // merge. Rank-generic: the run may be the rank-5 shape under test or a rank-4 control.
    Graph buildConstMaskChainGraph(const Shape &run, const std::vector<Shape> &maskShapes) {
        Graph      g;
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

            TensorId y = g.addTensor(namedDesc(("y" + std::to_string(k)).c_str()));
            Node     bin;
            bin.type    = OpType::Binary;
            bin.name    = "mask_step_" + std::to_string(k);
            bin.subOp   = (int) (k % 2 == 0 ? BinaryType::Mul : BinaryType::Add);
            bin.inputs  = {cur, m};
            bin.outputs = {y};
            g.nodes.push_back(bin);
            cur = y;
        }
        g.outputs = {cur};
        return g;
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
