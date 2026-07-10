// fuseRope (src/import/fuse_rope.cpp + the Rope op): the rotate-half chain a lowered contrib
// RotaryEmbedding expands to — raw Binary form and the FusedPointwise form the pointwise fusion
// builds from it — collapses into ONE Rope node whose CPU kernel reproduces the decomposed values
// exactly; a chain node carrying fused work, a non-half slice, an fp32-pinned internal tensor, or a
// CNN graph refuses the fusion; the load-time hint gate leaves the decomposed graph running
// unchanged. GPU parity is verified on device against this same CPU path as the oracle.
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    TensorId addInput(Graph &g, const std::string &name, Shape shape, DType dtype = DType::Float32) {
        TensorDesc td;
        td.name    = name;
        td.shape   = std::move(shape);
        td.dtype   = dtype;
        td.isInput = true;
        TensorId t = g.addTensor(td);
        g.inputs.push_back(t);
        return t;
    }

    TensorId addTemp(Graph &g, const std::string &name) {
        TensorDesc td;
        td.name = name;
        return g.addTensor(td);
    }

    TensorId addI64Init(Graph &g, const std::string &name, const std::vector<int64_t> &vals) {
        TensorDesc td;
        td.name          = name;
        td.shape         = {(int64_t) vals.size()};
        td.dtype         = DType::Int64;
        td.isInitializer = true;
        TensorId   t     = g.addTensor(td);
        HostBuffer hb;
        hb.resizeElems((int64_t) vals.size(), DType::Int64);
        std::memcpy(hb.i64(), vals.data(), vals.size() * sizeof(int64_t));
        g.initializers[t] = hb;
        return t;
    }

    TensorId addF32Init(Graph &g, const std::string &name, Shape shape, const std::vector<float> &vals) {
        TensorDesc td;
        td.name          = name;
        td.shape         = std::move(shape);
        td.isInitializer = true;
        TensorId   t     = g.addTensor(td);
        HostBuffer hb;
        hb.resizeElems((int64_t) vals.size(), DType::Float32);
        std::memcpy(hb.f32(), vals.data(), vals.size() * sizeof(float));
        g.initializers[t] = hb;
        return t;
    }

    Node &addNode(Graph &g, OpType type, const std::string &name, std::vector<TensorId> ins, TensorId out) {
        Node n;
        n.type    = type;
        n.name    = name;
        n.inputs  = std::move(ins);
        n.outputs = {out};
        g.nodes.push_back(n);
        return g.nodes.back();
    }

    void setInts(Node &n, const char *k, std::vector<int64_t> v) {
        Attr a;
        a.kind        = Attr::Ints;
        a.ints        = std::move(v);
        n.attr.map[k] = a;
    }
    void setInt(Node &n, const char *k, int64_t v) {
        Attr a;
        a.kind        = Attr::Int;
        a.i           = v;
        n.attr.map[k] = a;
    }

    // Test geometry: x [1,S,H*head], positions [1,S], tables [P, half].
    constexpr int64_t kS = 3, kH = 4, kHead = 8, kHalf = kHead / 2, kP = 16;

    std::vector<float> tableVals(bool sine) {
        std::vector<float> v((size_t) (kP * kHalf));
        for (int64_t p = 0; p < kP; ++p)
        {
            for (int64_t j = 0; j < kHalf; ++j)
            {
                double a               = 0.13 * (double) p * (double) (j + 1);
                v[(size_t) (p * kHalf + j)] = (float) (sine ? std::sin(a) : std::cos(a));
            }
        }
        return v;
    }

    // The primitive expansion expandRotary (lower_ort_contrib.cpp) emits: Reshape to [1,S,H,head],
    // Slice the last-axis halves, Gather+Unsqueeze a table row per position, the four Muls and the
    // Sub/Add rotate combines, Concat, Reshape back to [1,S,H*head].
    Graph ropeGraph() {
        Graph    g;
        TensorId x   = addInput(g, "x", {1, kS, kH * kHead});
        TensorId pos = addInput(g, "pos", {1, kS}, DType::Int64);
        TensorId cosT = addF32Init(g, "cos_table", {kP, kHalf}, tableVals(false));
        TensorId sinT = addF32Init(g, "sin_table", {kP, kHalf}, tableVals(true));

        TensorId r4 = addTemp(g, "r4");
        addNode(g, OpType::Reshape, "reshape4", {x, addI64Init(g, "s4", {1, kS, kH, kHead})}, r4);
        TensorId x1 = addTemp(g, "x1");
        TensorId x2 = addTemp(g, "x2");
        {
            Node &s1 = addNode(g, OpType::Slice, "slice_x1", {r4}, x1);
            setInts(s1, "starts", {0});
            setInts(s1, "ends", {kHalf});
            setInts(s1, "axes", {3});
            Node &s2 = addNode(g, OpType::Slice, "slice_x2", {r4}, x2);
            setInts(s2, "starts", {kHalf});
            setInts(s2, "ends", {kHead});
            setInts(s2, "axes", {3});
        }
        auto gatherRow = [&](TensorId table, const char *tag) {
            TensorId rows = addTemp(g, std::string(tag) + "_rows");
            Node    &gn   = addNode(g, OpType::Gather, std::string("gather_") + tag, {table, pos}, rows);
            setInt(gn, "axis", 0);
            TensorId un = addTemp(g, std::string(tag) + "_bcast");
            Node    &uq = addNode(g, OpType::Unsqueeze, std::string("unsq_") + tag, {rows}, un);
            setInts(uq, "axes", {2});
            return un;
        };
        TensorId cosB = gatherRow(cosT, "cos");
        TensorId sinB = gatherRow(sinT, "sin");

        auto mul = [&](const char *name, TensorId a, TensorId b) {
            TensorId t = addTemp(g, name);
            Node    &n = addNode(g, OpType::Binary, name, {a, b}, t);
            n.subOp    = (int32_t) BinaryType::Mul;
            return t;
        };
        TensorId x1c = mul("mul_x1c", x1, cosB), x2s = mul("mul_x2s", x2, sinB);
        TensorId x1s = mul("mul_x1s", x1, sinB), x2c = mul("mul_x2c", x2, cosB);
        TensorId o1 = addTemp(g, "o1");
        {
            Node &sub = addNode(g, OpType::Binary, "sub", {x1c, x2s}, o1);
            sub.subOp = (int32_t) BinaryType::Sub;
        }
        TensorId o2 = addTemp(g, "o2");
        addNode(g, OpType::Add, "add", {x1s, x2c}, o2);
        TensorId cat = addTemp(g, "cat");
        Node    &cc  = addNode(g, OpType::Concat, "concat", {o1, o2}, cat);
        setInt(cc, "axis", 3);
        TensorDesc yd;
        yd.name     = "y";
        yd.isOutput = true;
        TensorId y  = g.addTensor(yd);
        addNode(g, OpType::Reshape, "reshape3", {cat, addI64Init(g, "s3", {1, kS, kH * kHead})}, y);
        g.outputs = {y};
        return g;
    }

    const Node *findType(const Graph &g, OpType t) {
        for (const Node &n: g.nodes)
        {
            if (n.type == t)
            {
                return &n;
            }
        }
        return nullptr;
    }

    int countType(const Graph &g, OpType t) {
        int c = 0;
        for (const Node &n: g.nodes)
        {
            c += n.type == t ? 1 : 0;
        }
        return c;
    }

    std::vector<float> rampX() {
        std::vector<float> v((size_t) (kS * kH * kHead));
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = std::sin(0.21f * (float) i) + 0.02f * (float) (i % 13);
        }
        return v;
    }

    const std::vector<int64_t> kPositions = {5, 0, 11};

    std::vector<IOTensor> ropeInputs() {
        std::vector<IOTensor> ins(2);
        ins[0].name  = "x";
        ins[0].shape = {1, kS, kH * kHead};
        ins[0].dtype = DType::Float32;
        std::vector<float> xv = rampX();
        ins[0].data.resize(xv.size() * 4);
        std::memcpy(ins[0].data.data(), xv.data(), xv.size() * 4);
        ins[1].name  = "pos";
        ins[1].shape = {1, kS};
        ins[1].dtype = DType::Int64;
        ins[1].data.resize(kPositions.size() * 8);
        std::memcpy(ins[1].data.data(), kPositions.data(), kPositions.size() * 8);
        return ins;
    }

    // Ground-truth rotate-half straight off the raw inputs.
    std::vector<float> ropeReference() {
        std::vector<float> x = rampX(), cosT = tableVals(false), sinT = tableVals(true);
        std::vector<float> y(x.size());
        for (int64_t s = 0; s < kS; ++s)
        {
            const float *cr = cosT.data() + kPositions[(size_t) s] * kHalf;
            const float *sr = sinT.data() + kPositions[(size_t) s] * kHalf;
            for (int64_t h = 0; h < kH; ++h)
            {
                const float *xr = x.data() + (s * kH + h) * kHead;
                float       *yr = y.data() + (s * kH + h) * kHead;
                for (int64_t j = 0; j < kHalf; ++j)
                {
                    yr[j]         = xr[j] * cr[j] - xr[j + kHalf] * sr[j];
                    yr[j + kHalf] = xr[j] * sr[j] + xr[j + kHalf] * cr[j];
                }
            }
        }
        return y;
    }

    std::vector<float> runGraph(Graph g, bool ropeFusion, const std::vector<IOTensor> &ins) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        if (!ropeFusion)
        {
            cfg.setHint(Hint::RopeFusion, (int) Mode::Off);
        }
        auto sess = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run(ins, outs), Status::Ok);
        for (IOTensor &o: outs)
        {
            if (o.name == "y")
            {
                return std::vector<float>(o.f32(), o.f32() + numElements(o.shape));
            }
        }
        ADD_FAILURE() << "no output y";
        return {};
    }

} // namespace

// The raw Binary form (Mul x4 + Sub + Add) fuses: one Rope node with (x=r4, pos, cos, sin) inputs
// and the half attribute; the slices, gathers, unsqueezes, rotate products and concat leave the
// graph, while the outer reshapes (planner-aliased metadata) stay.
TEST(RopeFusion, FusesRawBinaryChain) {
    Graph g = ropeGraph();
    inferShapes(g);
    EXPECT_EQ(fuseRope(g), 1);

    const Node *rope = findType(g, OpType::Rope);
    ASSERT_NE(rope, nullptr);
    ASSERT_EQ(rope->inputs.size(), 4u);
    EXPECT_EQ(g.desc(rope->inputs[0]).name, "r4");
    EXPECT_EQ(g.desc(rope->inputs[1]).name, "pos");
    EXPECT_EQ(g.desc(rope->inputs[2]).name, "cos_table");
    EXPECT_EQ(g.desc(rope->inputs[3]).name, "sin_table");
    EXPECT_EQ(rope->attr.geti("half", -1), kHalf);
    EXPECT_EQ(g.desc(rope->outputs[0]).name, "cat");
    EXPECT_EQ(countType(g, OpType::Slice), 0);
    EXPECT_EQ(countType(g, OpType::Gather), 0);
    EXPECT_EQ(countType(g, OpType::Unsqueeze), 0);
    EXPECT_EQ(countType(g, OpType::Concat), 0);
    EXPECT_EQ(countType(g, OpType::Binary), 0);
    EXPECT_EQ(countType(g, OpType::Reshape), 2);

    // Idempotent: a second run finds nothing.
    EXPECT_EQ(fuseRope(g), 0);
}

// The FusedPointwise form the standard passes build (the shape every compiled .vxm carries) fuses
// the same way.
TEST(RopeFusion, FusesPointwiseUnitChain) {
    Graph g = ropeGraph();
    runStandardPasses(g, PassOptions {});
    ASSERT_EQ(countType(g, OpType::FusedPointwise), 2) << "precondition: the rotate products fused into two units";
    EXPECT_EQ(fuseRope(g), 1);
    const Node *rope = findType(g, OpType::Rope);
    ASSERT_NE(rope, nullptr);
    EXPECT_EQ(rope->attr.geti("half", -1), kHalf);
    EXPECT_EQ(countType(g, OpType::FusedPointwise), 0);
    EXPECT_EQ(countType(g, OpType::Gather), 0);
    EXPECT_EQ(countType(g, OpType::Concat), 0);
}

// The fused CPU kernel reproduces the decomposed chain and the ground-truth formula exactly (same
// fp32 expression per element, contraction pinned off on both sides).
TEST(RopeFusion, CpuFusedMatchesDecomposedAndReference) {
    std::vector<IOTensor> ins    = ropeInputs();
    std::vector<float>    fused  = runGraph(ropeGraph(), true, ins);
    std::vector<float>    plain  = runGraph(ropeGraph(), false, ins);
    std::vector<float>    ref    = ropeReference();
    ASSERT_EQ(fused.size(), ref.size());
    ASSERT_EQ(plain.size(), ref.size());
    for (size_t i = 0; i < ref.size(); ++i)
    {
        EXPECT_NEAR(fused[i], ref[i], 1e-6f) << "fused vs reference at " << i;
        EXPECT_NEAR(fused[i], plain[i], 1e-6f) << "fused vs decomposed at " << i;
    }
}

// Hint off: the session plans the decomposed chain and still computes the same values.
TEST(RopeFusion, HintOffKeepsDecomposedValues) {
    std::vector<IOTensor> ins   = ropeInputs();
    std::vector<float>    plain = runGraph(ropeGraph(), false, ins);
    std::vector<float>    ref   = ropeReference();
    ASSERT_EQ(plain.size(), ref.size());
    for (size_t i = 0; i < ref.size(); ++i)
    {
        EXPECT_NEAR(plain[i], ref[i], 1e-6f) << i;
    }
}

// A chain node carrying fused work is not pure rotate arithmetic — the fusion would drop that
// computation, so the site refuses.
TEST(RopeFusion, EpilogueCarryingNodeRefuses) {
    Graph g = ropeGraph();
    inferShapes(g);
    for (Node &n: g.nodes)
    {
        if (n.name == "slice_x1")
        {
            Attr marker;
            marker.kind            = Attr::Int;
            marker.i               = 1;
            n.attr.map["pw_steps"] = marker;
        }
    }
    EXPECT_EQ(fuseRope(g), 0);
    EXPECT_EQ(findType(g, OpType::Rope), nullptr);
}

// Slices that are not the exact [0, half) / [half, head) split are not rotate-half; the site
// refuses (both slices reading the LOW half here).
TEST(RopeFusion, NonHalfSliceRefuses) {
    Graph g = ropeGraph();
    for (Node &n: g.nodes)
    {
        if (n.name == "slice_x2")
        {
            setInts(n, "starts", {0});
            setInts(n, "ends", {kHalf});
        }
    }
    inferShapes(g);
    EXPECT_EQ(fuseRope(g), 0);
}

// A rotate expression with the wrong combine sign (o2 built with Sub) is not RoPE; the site refuses.
TEST(RopeFusion, WrongCombineSignRefuses) {
    Graph g = ropeGraph();
    for (Node &n: g.nodes)
    {
        if (n.name == "add")
        {
            n.type  = OpType::Binary;
            n.subOp = (int32_t) BinaryType::Sub;
        }
    }
    inferShapes(g);
    EXPECT_EQ(fuseRope(g), 0);
}

// An fp32-pinned internal tensor keeps its decomposed form: the fusion would remove the fp32 store
// the pin exists for (mirrors the foldMatMulViews refusal).
TEST(RopeFusion, Fp32PinnedInternalTensorRefuses) {
    {
        Graph g = ropeGraph();
        inferShapes(g);
        EXPECT_EQ(fuseRope(g, "x1"), 0);
    }
    {
        Graph g = ropeGraph();
        inferShapes(g);
        EXPECT_EQ(fuseRope(g, "unrelated_name"), 1);
    }
}

// A CNN-style graph (conv + relu + pool: no Concat-of-halves anywhere) is completely untouched.
TEST(RopeFusion, CnnGraphUntouched) {
    Graph    g;
    TensorId x = addInput(g, "img", {1, 3, 8, 8});
    TensorId w = addF32Init(g, "w", {4, 3, 3, 3}, std::vector<float>(4 * 3 * 3 * 3, 0.1f));
    TensorId c = addTemp(g, "conv_out");
    {
        Node &conv = addNode(g, OpType::Conv, "conv", {x, w}, c);
        setInts(conv, "kernel_shape", {3, 3});
        setInts(conv, "pads", {1, 1, 1, 1});
        setInts(conv, "strides", {1, 1});
    }
    TensorId r = addTemp(g, "relu_out");
    addNode(g, OpType::Relu, "relu", {c}, r);
    TensorDesc yd;
    yd.name     = "y";
    yd.isOutput = true;
    TensorId y  = g.addTensor(yd);
    addNode(g, OpType::GlobalAvgPool, "gap", {r}, y);
    g.outputs = {y};
    inferShapes(g);
    const size_t nodes = g.nodes.size();
    EXPECT_EQ(fuseRope(g), 0);
    EXPECT_EQ(g.nodes.size(), nodes);
    EXPECT_EQ(findType(g, OpType::Rope), nullptr);
}
