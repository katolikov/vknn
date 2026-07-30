// Row/column mask operands in a fused pointwise unit.
//
// A [N,C,H,1] row, [N,C,1,W] column, [1,1,H,1] row-splat or [1,1,1,W] column-splat operand used to
// classify as kPwBcastGeneral, which is addressable only by the flat kernel's per-axis div/mod
// walk. One of them therefore forced its whole unit off NC4HW4 (see `nc4Ok` in
// fuse_pointwise_chains.cpp) and excluded it from the layout re-vote, so the unit ran scalar and
// picked up a ConvertLayout on each full-size edge. They now classify as kPwBcastRow / kPwBcastCol
// / kPwBcastRowSplat / kPwBcastColSplat, which both kernels index in closed form.
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {
    constexpr int kPwStepInts       = 8;
    constexpr int kPwStepBcastField = 6;

    // MSVC at C++17 rejects designated initializers; a tiny helper names the tensor.
    TensorDesc namedDesc(const char *name) {
        TensorDesc d;
        d.name = name;
        return d;
    }

    // x[N,C,H,W] -> Abs -> chain of Binary ops, one per mask shape in `maskShapes`. Masks are graph
    // inputs routed through their own Abs so they reach the unit as runtime activations (the packer
    // path for constants is separate); ops alternate Mul/Add so no two steps merge.
    Graph buildMaskChainGraph(const Shape &run, const std::vector<Shape> &maskShapes) {
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

    // Distinct-per-position mask payload so a wrong row/column read is a gross mismatch.
    std::vector<float> maskPayload(const Shape &s) {
        std::vector<float> v((size_t) numElements(s));
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = 0.25f + 0.125f * (float) (i % 13);
        }
        return v;
    }

    void bindInputs(std::vector<IOTensor> &in, const Shape &run, const std::vector<Shape> &maskShapes, const std::vector<float> &src) {
        in.resize(1 + maskShapes.size());
        in[0].name  = "x";
        in[0].shape = run;
        in[0].dtype = DType::Float32;
        in[0].data.resize(src.size() * sizeof(float));
        std::memcpy(in[0].data.data(), src.data(), src.size() * sizeof(float));
        for (size_t k = 0; k < maskShapes.size(); ++k)
        {
            std::vector<float> mv = maskPayload(maskShapes[k]);
            in[1 + k].name        = "m" + std::to_string(k);
            in[1 + k].shape       = maskShapes[k];
            in[1 + k].dtype       = DType::Float32;
            in[1 + k].data.resize(mv.size() * sizeof(float));
            std::memcpy(in[1 + k].data.data(), mv.data(), mv.size() * sizeof(float));
        }
    }

    // Run the graph on both backends (GPU pinned to the blocked world so the NC4HW4 appliers are
    // what executes) and compare element for element. Skips without a Vulkan device.
    void expectGpuMatchesCpu(const Shape &run, const std::vector<Shape> &maskShapes) {
        const size_t       n = (size_t) numElements(run);
        std::vector<float> src(n);
        for (size_t i = 0; i < n; ++i)
        {
            src[i] = 0.5f + (float) (i % 11) * 0.125f;
        }

        Config gpu;
        gpu.backend          = BackendKind::Vulkan;
        gpu.allowCpuFallback = false;
        gpu.setHint(Hint::FlatLayout, (int) Mode::Off); // pin the blocked world: the NC4HW4 applier is what is under test
        Graph                    gg = buildMaskChainGraph(run, maskShapes);
        std::unique_ptr<Session> gsess;
        try
        { gsess = Session::create(std::move(gg), gpu); } catch (const std::exception &)
        { gsess.reset(); }
        if (!gsess)
        {
            GTEST_SKIP() << "no Vulkan device";
        }
        std::vector<IOTensor> gin, gout;
        bindInputs(gin, run, maskShapes, src);
        ASSERT_EQ(gsess->run(gin, gout), Status::Ok);
        ASSERT_EQ(gout.size(), 1u);

        Config cpu;
        cpu.backend = BackendKind::Cpu;
        Graph cg    = buildMaskChainGraph(run, maskShapes);
        auto  csess = Session::create(std::move(cg), cpu);
        ASSERT_NE(csess, nullptr);
        std::vector<IOTensor> cin, cout;
        bindInputs(cin, run, maskShapes, src);
        ASSERT_EQ(csess->run(cin, cout), Status::Ok);

        const float *gpuOut   = gout[0].f32();
        const float *cpuOut   = cout[0].f32();
        const size_t plane    = (size_t) (run[2] * run[3]);
        size_t       reported = 0;
        for (size_t i = 0; i < n && reported < 8; ++i)
        {
            if (std::abs(gpuOut[i] - cpuOut[i]) > 1e-2f)
            {
                ++reported;
                ADD_FAILURE() << "element " << i << " channel " << (i / plane) % (size_t) run[1] << " pixel " << i % plane << ": gpu=" << gpuOut[i] << " cpu=" << cpuOut[i];
            }
        }
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
} // namespace

TEST(PwRowColBcast, SingleBatchMasksClassifyClosedForm) {
    Graph g       = buildMaskChainGraph({1, 8, 4, 5}, {{1, 8, 4, 1}, {1, 8, 1, 5}, {1, 1, 4, 1}, {1, 1, 1, 5}});
    auto  classes = fusedBcastClasses(g);
    EXPECT_TRUE(contains(classes, kPwBcastRow)) << "a [N,C,H,1] operand must classify as the per-row class";
    EXPECT_TRUE(contains(classes, kPwBcastCol)) << "a [N,C,1,W] operand must classify as the per-column class";
    EXPECT_TRUE(contains(classes, kPwBcastRowSplat)) << "a [1,1,H,1] operand must classify as the row-splat class";
    EXPECT_TRUE(contains(classes, kPwBcastColSplat)) << "a [1,1,1,W] operand must classify as the column-splat class";
    EXPECT_FALSE(contains(classes, kPwBcastGeneral)) << "no step may stay general: one general step forces the whole unit off NC4HW4";
}

TEST(PwRowColBcast, BatchedRowColClassifyClosedFormButSplatsStayGeneral) {
    // The Row/Col classes carry the batch in the channel-block index, so N=2 is legal for them;
    // the *Splat classes derive the pixel from vecIdx % HW, which drops the batch, so they must
    // NOT engage on a batched run.
    Graph gRowCol = buildMaskChainGraph({2, 8, 4, 5}, {{2, 8, 4, 1}, {2, 8, 1, 5}});
    auto  rowCol  = fusedBcastClasses(gRowCol);
    EXPECT_TRUE(contains(rowCol, kPwBcastRow));
    EXPECT_TRUE(contains(rowCol, kPwBcastCol));
    EXPECT_FALSE(contains(rowCol, kPwBcastGeneral));

    Graph gSplat = buildMaskChainGraph({2, 8, 4, 5}, {{1, 1, 4, 1}, {1, 1, 1, 5}});
    auto  splat  = fusedBcastClasses(gSplat);
    EXPECT_FALSE(contains(splat, kPwBcastRowSplat)) << "a batched run must not use the batch-dropping row-splat index";
    EXPECT_FALSE(contains(splat, kPwBcastColSplat)) << "a batched run must not use the batch-dropping column-splat index";
}

// The closed-form index must select the SAME element the general strided walk would have. The CPU
// backend reads operands by NumPy-style strides regardless of the class fields, so it is the
// reference for the value at every position.
TEST(PwRowColBcast, CpuComputesTheMaskedChain) {
    const Shape        run = {1, 8, 4, 5};
    std::vector<Shape> maskShapes {{1, 8, 4, 1}, {1, 8, 1, 5}, {1, 1, 4, 1}, {1, 1, 1, 5}};
    Graph              g = buildMaskChainGraph(run, maskShapes);

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_NE(sess, nullptr);

    const size_t       n = (size_t) numElements(run);
    std::vector<float> src(n);
    for (size_t i = 0; i < n; ++i)
    {
        src[i] = 1.0f + (float) (i % 7);
    }
    std::vector<IOTensor> in, out;
    bindInputs(in, run, maskShapes, src);
    ASSERT_EQ(sess->run(in, out), Status::Ok);
    ASSERT_EQ(out.size(), 1u);

    auto          rm = maskPayload(maskShapes[0]), cm = maskPayload(maskShapes[1]);
    auto          rs = maskPayload(maskShapes[2]), cs = maskPayload(maskShapes[3]);
    const float  *got = out[0].f32();
    const int64_t C = run[1], H = run[2], W = run[3];
    for (int64_t c = 0; c < C; ++c)
    {
        for (int64_t h = 0; h < H; ++h)
        {
            for (int64_t w = 0; w < W; ++w)
            {
                const size_t i   = (size_t) ((c * H + h) * W + w);
                const float  ref = ((src[i] * rm[(size_t) (c * H + h)] + cm[(size_t) (c * W + w)]) * rs[(size_t) h]) + cs[(size_t) w];
                EXPECT_FLOAT_EQ(got[i], ref) << "channel " << c << " row " << h << " col " << w;
            }
        }
    }
}

// The GPU must agree with the CPU reference element for element, on the blocked (NC4HW4) path the
// classes exist for. Unaligned C exercises the padded tail block; H=16/W=24 keeps hw/W and hw%W
// away from trivial values.
TEST(PwRowColBcast, BlockedLayoutRowColOperandsMatchCpu) {
    expectGpuMatchesCpu({1, 6, 16, 24}, {{1, 6, 16, 1}, {1, 6, 1, 24}});
}

TEST(PwRowColBcast, BlockedLayoutSplatOperandsMatchCpu) {
    expectGpuMatchesCpu({1, 6, 16, 24}, {{1, 1, 16, 1}, {1, 1, 1, 24}});
}

TEST(PwRowColBcast, BlockedLayoutBatchedRowColMatchCpu) {
    expectGpuMatchesCpu({2, 6, 16, 24}, {{2, 6, 16, 1}, {2, 6, 1, 24}});
}

TEST(PwRowColBcast, BlockedLayoutAllFourMaskShapesMatchCpu) {
    expectGpuMatchesCpu({1, 8, 16, 24}, {{1, 8, 16, 1}, {1, 8, 1, 24}, {1, 1, 16, 1}, {1, 1, 1, 24}});
}
