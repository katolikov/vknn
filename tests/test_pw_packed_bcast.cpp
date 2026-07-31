// The generic packed broadcast class and the right-align pre-pass for pw-unit operands.
//
// Any operand of a rank-4 pointwise run whose right-aligned NCHW axes are each 1 or the run's
// extent classifies as kPwBcastPacked: buildPwPlan derives per-axis vec4-space strides for it, so
// the NC4HW4 kernel addresses it in closed form instead of flat-forcing the unit. The named
// classes (Same/Channel/Scalar/Spatial/Row/Col/RowSplat/ColSplat) are tested first, so every
// previously classified shape keeps its code; only the leftovers land in the packed class.
// rightAlignPwOperands rewires each rank<4 RUNTIME operand of such a run through an explicit
// "#pwr4" Reshape to its right-aligned rank-4 shape -- the device packing then follows the same
// reading the classifier judges by, and a mask that is a named class after alignment gets that
// named class, not kPwBcastGeneral. The CPU FusedPointwise kernel reads every operand through
// right-aligned NumPy strides regardless of class, so it is the value reference for all masks.
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/nchw.h"
#include "vknn/session.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    // Run extents. The batch is 2 so no mask axis degenerates: an n=1 mask is a real batch
    // broadcast and the single-batch-only named classes (Spatial/*Splat) cannot engage.
    constexpr int64_t kRunN = 2;
    constexpr int64_t kRunC = 6;
    constexpr int64_t kRunH = 4;
    constexpr int64_t kRunW = 5;
    // A broadcast (size-1) mask axis / the single-batch run extent.
    constexpr int64_t kBcast       = 1;
    constexpr int64_t kSingleBatch = 1;
    // A channel extent that is neither 1 nor the run's: the malformed negative control.
    constexpr int64_t kOffFullC = 3;

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

    // x[N,C,H,W] -> Abs -> chain of Binary ops, one per mask shape in `maskShapes`. Masks are
    // INITIALIZERS consumed directly by the Binary steps, so the classifier judges them exactly
    // as the constant-operand packer (pwOperandBuf) reads them: by their right-aligned shape.
    // Ops alternate Mul/Add so no two steps merge.
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
            hb.resizeElems(mv.size(), DType::Float32);
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

    // Same chain, but each mask is a graph INPUT routed through its own Abs so it reaches the
    // unit as a runtime activation -- the population rightAlignPwOperands exists for.
    Graph buildRuntimeMaskChainGraph(const Shape &run, const std::vector<Shape> &maskShapes) {
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

            TensorId y = g.addTensor(namedDesc(("y" + std::to_string(k)).c_str()));
            Node     bin;
            bin.type    = OpType::Binary;
            bin.name    = "mask_step_" + std::to_string(k);
            bin.subOp   = (int) (k % 2 == 0 ? BinaryType::Mul : BinaryType::Add);
            bin.inputs  = {cur, mm};
            bin.outputs = {y};
            g.nodes.push_back(bin);
            cur = y;
        }
        g.outputs = {cur};
        return g;
    }

    // The classes the pass assigned across every fused unit's steps.
    std::vector<int64_t> fusedBcastClasses(Graph &g) {
        inferShapes(g, 1);
        fusePointwiseChains(g, /*strictFuse*/ false);
        std::vector<int64_t> classes;
        for (const Node &nd: g.nodes)
        {
            if (!nd.attr.has("pw_steps"))
            {
                continue;
            }
            const auto &steps = nd.attr.getints("pw_steps");
            for (size_t s = 0; s + kPwStepBcastField < steps.size(); s += kPwStepInts)
            {
                classes.push_back(steps[s + kPwStepBcastField]);
            }
        }
        return classes;
    }

    bool contains(const std::vector<int64_t> &v, int64_t x) {
        return std::find(v.begin(), v.end(), x) != v.end();
    }

    int64_t countOf(const std::vector<int64_t> &v, int64_t x) {
        return (int64_t) std::count(v.begin(), v.end(), x);
    }

    // Builds the single-mask runtime chain, runs the fusion, and pins the right-align contract:
    // the "#pwr4" Reshape exists with the right-aligned rank-4 shape, the fused unit reads the
    // reshaped view (not the original rank<4 tensor), and the view classifies as `expectedClass`.
    void expectRightAlignedFusion(const Shape &run, const Shape &maskShape, const Shape &alignedShape, int64_t expectedClass) {
        Graph g = buildRuntimeMaskChainGraph(run, {maskShape});
        inferShapes(g, 1);
        fusePointwiseChains(g, /*strictFuse*/ false);

        TensorId view = kNoTensor;
        for (const Node &nd: g.nodes)
        {
            if (nd.type == OpType::Reshape && nd.name.find("#pwr4") != std::string::npos)
            {
                EXPECT_EQ(view, kNoTensor) << "one rank<4 operand must insert exactly one right-align Reshape";
                view = nd.outputs[0];
                EXPECT_EQ(g.desc(view).shape, alignedShape) << "the #pwr4 view must carry the right-aligned rank-4 shape";
            }
        }
        ASSERT_NE(view, kNoTensor) << "a rank<4 runtime operand must be rewired through a #pwr4 Reshape";

        const Node *unit = nullptr;
        for (const Node &nd: g.nodes)
        {
            if (nd.attr.has("pw_steps") && std::find(nd.inputs.begin(), nd.inputs.end(), view) != nd.inputs.end())
            {
                unit = &nd;
            }
        }
        ASSERT_NE(unit, nullptr) << "the fused unit must consume the reshaped view as its operand";

        std::vector<int64_t> classes;
        const auto          &steps = unit->attr.getints("pw_steps");
        for (size_t s = 0; s + kPwStepBcastField < steps.size(); s += kPwStepInts)
        {
            classes.push_back(steps[s + kPwStepBcastField]);
        }
        EXPECT_TRUE(contains(classes, expectedClass)) << "the aligned view must classify as class " << expectedClass;
        EXPECT_FALSE(contains(classes, kPwBcastGeneral)) << "a right-aligned operand must not flat-force its unit";
    }

    // Right-aligned rank-4 view of a mask shape: kNchwRank axes, missing leading axes are 1.
    Shape rightAligned(const Shape &s) {
        Shape rs = s;
        rs.insert(rs.begin(), kNchwRank - rs.size(), kBcast);
        return rs;
    }

    // Flat offset of the mask element a right-aligned NumPy read yields at output (n, c, h, w):
    // a broadcast axis reads index 0.
    size_t maskOffset(const Shape &aligned, int64_t n, int64_t c, int64_t h, int64_t w) {
        const int64_t mn = aligned[0] == kBcast ? 0 : n;
        const int64_t mc = aligned[1] == kBcast ? 0 : c;
        const int64_t mh = aligned[2] == kBcast ? 0 : h;
        const int64_t mw = aligned[3] == kBcast ? 0 : w;
        return (size_t) (((mn * aligned[1] + mc) * aligned[2] + mh) * aligned[3] + mw);
    }

    // Compiles the constant-mask chain for the CPU backend and checks the output against plain
    // right-aligned broadcast loops, element for element. The CPU fused kernel chains fp32 steps
    // in the same order as the loops, so equality is exact.
    void expectCpuMatchesReference(const Shape &run, const std::vector<Shape> &maskShapes) {
        Graph  g = buildConstMaskChainGraph(run, maskShapes);
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        ASSERT_NE(sess, nullptr);

        const size_t       total = (size_t) numElements(run);
        std::vector<float> src(total);
        for (size_t i = 0; i < total; ++i)
        {
            src[i] = 0.5f + (float) (i % 11) * 0.125f;
        }
        std::vector<IOTensor> in, out;
        in.resize(1);
        in[0].name  = "x";
        in[0].shape = run;
        in[0].dtype = DType::Float32;
        in[0].data.resize(src.size() * sizeof(float));
        std::memcpy(in[0].data.data(), src.data(), src.size() * sizeof(float));
        ASSERT_EQ(sess->run(in, out), Status::Ok);
        ASSERT_EQ(out.size(), 1u);

        std::vector<std::vector<float>> maskValues;
        for (const Shape &s: maskShapes)
        {
            maskValues.push_back(maskPayload(s));
        }

        const float *got = out[0].f32();
        for (int64_t n = 0; n < run[0]; ++n)
        {
            for (int64_t c = 0; c < run[1]; ++c)
            {
                for (int64_t h = 0; h < run[2]; ++h)
                {
                    for (int64_t w = 0; w < run[3]; ++w)
                    {
                        const size_t i   = (size_t) ((((n * run[1] + c) * run[2] + h) * run[3]) + w);
                        float        ref = std::abs(src[i]);
                        for (size_t k = 0; k < maskShapes.size(); ++k)
                        {
                            const float mv = maskValues[k][maskOffset(rightAligned(maskShapes[k]), n, c, h, w)];
                            ref            = k % 2 == 0 ? ref * mv : ref + mv;
                        }
                        EXPECT_FLOAT_EQ(got[i], ref) << "batch " << n << " channel " << c << " row " << h << " col " << w;
                    }
                }
            }
        }
    }
} // namespace

// Every mask here is 1-or-full on each axis but matches no named class (the batched run keeps
// Spatial and the *Splat classes out; a broadcast n or c keeps Channel/Row/Col out), so each must
// land in the generic packed class -- one packed step per mask, and never the general class.
TEST(PwPackedBcast, BatchedOneOrFullPlaneMasksClassifyPacked) {
    const Shape run = {kRunN, kRunC, kRunH, kRunW};
    const std::vector<Shape> masks {{kRunN, kBcast, kRunH, kRunW}, {kBcast, kRunC, kRunH, kRunW}, {kRunN, kBcast, kBcast, kRunW}, {kBcast, kBcast, kRunH, kRunW}};
    Graph g       = buildConstMaskChainGraph(run, masks);
    auto  classes = fusedBcastClasses(g);
    EXPECT_EQ(countOf(classes, kPwBcastPacked), (int64_t) masks.size()) << "each 1-or-full mask must take the packed class";
    EXPECT_FALSE(contains(classes, kPwBcastGeneral)) << "no 1-or-full mask may stay general: one general step forces the whole unit off NC4HW4";
}

TEST(PwPackedBcast, BatchedOneOrFullVectorMasksClassifyPacked) {
    const Shape run = {kRunN, kRunC, kRunH, kRunW};
    const std::vector<Shape> masks {{kRunN, kBcast, kRunH, kBcast}, {kBcast, kRunC, kBcast, kBcast}, {kRunN, kBcast, kBcast, kBcast}, {kBcast, kBcast, kBcast, kRunW}, {kBcast, kBcast, kRunH, kBcast}};
    Graph g       = buildConstMaskChainGraph(run, masks);
    auto  classes = fusedBcastClasses(g);
    EXPECT_EQ(countOf(classes, kPwBcastPacked), (int64_t) masks.size()) << "each 1-or-full mask must take the packed class";
    EXPECT_FALSE(contains(classes, kPwBcastGeneral)) << "no 1-or-full mask may stay general: one general step forces the whole unit off NC4HW4";
}

// The named classes are tested before the packed predicate, so every shape that classified before
// keeps its code -- existing encodings stay byte-stable.
TEST(PwPackedBcast, NamedClassControlsKeepTheirCodes) {
    const Shape run = {kRunN, kRunC, kRunH, kRunW};
    Graph       g = buildConstMaskChainGraph(run, {{kRunN, kRunC, kBcast, kBcast}, {kBcast}, {kRunN, kRunC, kRunH, kBcast}, {kBcast, kBcast, kBcast, kBcast}});
    auto        classes = fusedBcastClasses(g);
    EXPECT_TRUE(contains(classes, kPwBcastChannel)) << "a full [N,C,1,1] mask must keep the per-channel class";
    EXPECT_TRUE(contains(classes, kPwBcastRow)) << "a full [N,C,H,1] mask must keep the per-row class";
    EXPECT_EQ(countOf(classes, kPwBcastScalar), 2) << "a [1] and a [1,1,1,1] mask must both keep the scalar class";
    EXPECT_FALSE(contains(classes, kPwBcastPacked)) << "a named class must win before the generic packed predicate";
}

// Right-align pre-pass: a rank-3 runtime [C,1,1] mask is a left-aligned kPwBcastGeneral without
// the Reshape; behind its #pwr4 view it is the named per-channel class.
TEST(PwPackedBcast, RuntimeRank3ChannelMaskRightAlignsToChannel) {
    expectRightAlignedFusion({kSingleBatch, kRunC, kRunH, kRunW}, {kRunC, kBcast, kBcast}, {kSingleBatch, kRunC, kBcast, kBcast}, kPwBcastChannel);
}

TEST(PwPackedBcast, RuntimeRank2SpatialMaskRightAlignsToSpatial) {
    expectRightAlignedFusion({kSingleBatch, kRunC, kRunH, kRunW}, {kRunH, kRunW}, {kSingleBatch, kBcast, kRunH, kRunW}, kPwBcastSpatial);
}

TEST(PwPackedBcast, RuntimeRank3SpatialMaskRightAlignsToSpatial) {
    expectRightAlignedFusion({kSingleBatch, kRunC, kRunH, kRunW}, {kBcast, kRunH, kRunW}, {kSingleBatch, kBcast, kRunH, kRunW}, kPwBcastSpatial);
}

// A rank-3 [C,H,1] mask against a batched run right-aligns to [1,C,H,1] -- an n-broadcast no named
// class covers (Row needs the full batch), so the view classifies packed.
TEST(PwPackedBcast, RuntimeRank3BatchBroadcastMaskRightAlignsToPacked) {
    expectRightAlignedFusion({kRunN, kRunC, kRunH, kRunW}, {kRunC, kRunH, kBcast}, {kBcast, kRunC, kRunH, kBcast}, kPwBcastPacked);
}

// CPU correctness of packed-class chains: the fused unit must select the same element the plain
// right-aligned NumPy walk selects, at every position.
TEST(PwPackedBcast, CpuComputesChannelBroadcastPackedChain) {
    expectCpuMatchesReference({kRunN, kRunC, kRunH, kRunW}, {{kRunN, kBcast, kRunH, kRunW}});
}

TEST(PwPackedBcast, CpuComputesBatchBroadcastPackedChain) {
    expectCpuMatchesReference({kRunN, kRunC, kRunH, kRunW}, {{kBcast, kRunC, kRunH, kRunW}});
}

TEST(PwPackedBcast, CpuComputesMixedPackedChain) {
    expectCpuMatchesReference({kRunN, kRunC, kRunH, kRunW}, {{kRunN, kBcast, kBcast, kRunW}, {kBcast, kRunC, kBcast, kBcast}});
}

TEST(PwPackedBcast, CpuComputesRowLikePackedChain) {
    expectCpuMatchesReference({kRunN, kRunC, kRunH, kRunW}, {{kBcast, kBcast, kRunH, kBcast}, {kRunN, kBcast, kRunH, kBcast}});
}

// The two channel-carrying single-batch plane masks: against a BATCHED run their n axis is a real
// broadcast, so Row/Col cannot claim them and they fall to the packed class.
TEST(PwPackedBcast, BatchedSingleBatchRowAndColumnMasksClassifyPacked) {
    const Shape              run = {kRunN, kRunC, kRunH, kRunW};
    const std::vector<Shape> masks {{kBcast, kRunC, kRunH, kBcast}, {kBcast, kRunC, kBcast, kRunW}};
    Graph                    g       = buildConstMaskChainGraph(run, masks);
    auto                     classes = fusedBcastClasses(g);
    EXPECT_EQ(countOf(classes, kPwBcastPacked), (int64_t) masks.size()) << "a batch-broadcast row/column plane is a packed mask";
    EXPECT_FALSE(contains(classes, kPwBcastRow)) << "the per-row class needs the run's full batch";
    EXPECT_FALSE(contains(classes, kPwBcastCol)) << "the per-column class needs the run's full batch";
    EXPECT_FALSE(contains(classes, kPwBcastGeneral));
}

TEST(PwPackedBcast, CpuComputesSingleBatchRowAndColumnPackedChain) {
    expectCpuMatchesReference({kRunN, kRunC, kRunH, kRunW}, {{kBcast, kRunC, kRunH, kBcast}, {kBcast, kRunC, kBcast, kRunW}});
}

// A rank<4 CONSTANT is judged by its right-aligned rank-4 reading -- the shape pwOperandBuf packs
// it at -- so it reaches the same named classes a written-out rank-4 constant would, with no
// Reshape in the graph (only runtime operands need one).
TEST(PwPackedBcast, ConstantMasksBelowRankFourClassifyByTheirRightAlignedShape) {
    const Shape run = {kSingleBatch, kRunC, kRunH, kRunW};
    // [C,1,1] -> [1,C,1,1] per-channel; [H,W] -> [1,1,H,W] per-pixel; [C,H,1] -> [1,C,H,1] per-row;
    // [W] -> [1,1,1,W] column-splat.
    const std::vector<Shape> masks {{kRunC, kBcast, kBcast}, {kRunH, kRunW}, {kRunC, kRunH, kBcast}, {kRunW}};
    Graph                    g       = buildConstMaskChainGraph(run, masks);
    auto                     classes = fusedBcastClasses(g);
    EXPECT_TRUE(contains(classes, kPwBcastChannel)) << "a [C,1,1] constant right-aligns to the per-channel class";
    EXPECT_TRUE(contains(classes, kPwBcastSpatial)) << "an [H,W] constant right-aligns to the per-pixel class";
    EXPECT_TRUE(contains(classes, kPwBcastRow)) << "a [C,H,1] constant right-aligns to the per-row class";
    EXPECT_TRUE(contains(classes, kPwBcastColSplat)) << "a [W] constant right-aligns to the column-splat class";
    EXPECT_FALSE(contains(classes, kPwBcastGeneral)) << "a right-aligned constant must not flat-force its unit";
    for (const TensorDesc &td: g.tensors)
    {
        EXPECT_EQ(td.name.find("#pwr4"), std::string::npos) << "a constant is packed at its right-aligned shape: no Reshape is inserted";
    }
}

// The same rule reaching the generic packed class: a rank-3 [C,H,1] constant against a BATCHED run
// right-aligns to [1,C,H,1], which no named class covers.
TEST(PwPackedBcast, ConstantMaskBelowRankFourReachesThePackedClass) {
    const Shape run     = {kRunN, kRunC, kRunH, kRunW};
    Graph       g       = buildConstMaskChainGraph(run, {{kRunC, kRunH, kBcast}});
    auto        classes = fusedBcastClasses(g);
    EXPECT_TRUE(contains(classes, kPwBcastPacked)) << "the right-aligned [1,C,H,1] reading is a packed mask";
    EXPECT_FALSE(contains(classes, kPwBcastGeneral));
}

TEST(PwPackedBcast, CpuComputesConstantSubRankMaskChain) {
    expectCpuMatchesReference({kSingleBatch, kRunC, kRunH, kRunW}, {{kRunH, kRunW}, {kRunC, kBcast, kBcast}, {kRunW}});
}

// One rank<4 runtime operand read by two separate units takes ONE #pwr4 view between them, while a
// non-pointwise consumer keeps reading the original tensor at its own rank.
TEST(PwPackedBcast, OneRightAlignViewIsSharedAndTheOriginalSurvivesForOtherReaders) {
    const Shape run {kSingleBatch, kRunC, kRunH, kRunW};
    const Shape maskShape {kRunH, kRunW};

    Graph      g;
    TensorDesc mi;
    mi.name    = "m";
    mi.shape   = maskShape;
    mi.isInput = true;
    TensorId m = g.addTensor(mi);
    g.inputs   = {m};

    // Two independent chains, each x -> Abs -> Mul(m): separate regions, one shared operand.
    for (int chain = 0; chain < 2; ++chain)
    {
        const std::string suffix = std::to_string(chain);
        TensorDesc        xi;
        xi.name    = "x" + suffix;
        xi.shape   = run;
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs.push_back(x);

        TensorId t = g.addTensor(namedDesc(("t" + suffix).c_str()));
        Node     act;
        act.type    = OpType::Unary;
        act.name    = "producer" + suffix;
        act.subOp   = (int) UnaryType::Abs;
        act.inputs  = {x};
        act.outputs = {t};
        g.nodes.push_back(act);

        TensorDesc yo;
        yo.name     = "y" + suffix;
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     bin;
        bin.type    = OpType::Binary;
        bin.name    = "mask_step_" + suffix;
        bin.subOp   = (int) BinaryType::Mul;
        bin.inputs  = {t, m};
        bin.outputs = {y};
        g.nodes.push_back(bin);
        g.outputs.push_back(y);
    }

    // A Reduce is not pointwise-fusable, so it stays outside every unit and reads the original.
    TensorDesc ro;
    ro.name     = "mask_reduced";
    ro.isOutput = true;
    TensorId r  = g.addTensor(ro);
    Node     red;
    red.type    = OpType::Reduce;
    red.name    = "mask_consumer";
    red.subOp   = (int) ReduceType::Sum;
    red.inputs  = {m};
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

    TensorId view  = kNoTensor;
    int      views = 0;
    for (const Node &nd: g.nodes)
    {
        if (nd.type == OpType::Reshape && nd.name.find("#pwr4") != std::string::npos)
        {
            ++views;
            view = nd.outputs[0];
        }
    }
    EXPECT_EQ(views, 1) << "both units must share one right-align view of the same operand";
    ASSERT_NE(view, kNoTensor);
    EXPECT_EQ(g.desc(view).shape, (Shape {kSingleBatch, kBcast, kRunH, kRunW}));

    int readers = 0;
    for (const Node &nd: g.nodes)
    {
        if (nd.attr.has("pw_steps") && std::find(nd.inputs.begin(), nd.inputs.end(), view) != nd.inputs.end())
        {
            ++readers;
        }
    }
    EXPECT_EQ(readers, 2) << "both fused units must read the shared view";

    const Node *outside = nullptr;
    for (const Node &nd: g.nodes)
    {
        if (nd.type == OpType::Reduce)
        {
            outside = &nd;
        }
    }
    ASSERT_NE(outside, nullptr);
    ASSERT_FALSE(outside->inputs.empty());
    EXPECT_EQ(outside->inputs[0], m) << "a non-pointwise consumer keeps the original operand";
    EXPECT_EQ(g.desc(m).shape, maskShape) << "the original tensor keeps its own rank";
}

// Negative control: an axis extent that is neither 1 nor the run's is no packed mask. The Binary
// output still infers to the run shape (per-dim max), so the unit's run is the full one and only
// the operand is off -- it must classify general, exactly as before the packed class existed.
TEST(PwPackedBcast, NonOneOrFullMaskStaysGeneral) {
    const Shape run     = {kRunN, kRunC, kRunH, kRunW};
    Graph       g       = buildConstMaskChainGraph(run, {{kRunN, kOffFullC, kRunH, kRunW}});
    auto        classes = fusedBcastClasses(g);
    EXPECT_TRUE(contains(classes, kPwBcastGeneral)) << "a non-1-or-full mask has no closed-form packed index";
    EXPECT_FALSE(contains(classes, kPwBcastPacked)) << "the packed predicate must refuse an axis that is neither 1 nor full";
}
