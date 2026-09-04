// fuseDfl (import/fuse_dfl.cpp): the detection head's Reshape -> Transpose -> Softmax ->
// Transpose -> Conv1x1 -> Reshape decode becomes one FusedDfl node, only when the chain is closed
// and its shapes agree with the [N,S,B,L] view; the CPU FusedDfl op is checked against the unfused
// CPU chain on the same map and weights.
#include "core/dfl.h"
#include "import/passes.h"
#include "vknn/op_descriptor.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {
    constexpr int64_t kN = 1, kSides = 2, kBins = 4, kL = 5;

    TensorId addAct(Graph &g, const std::string &name, Shape shape, bool isInput = false) {
        TensorDesc d;
        d.name    = name;
        d.shape   = std::move(shape);
        d.isInput = isInput;
        return g.addTensor(d);
    }
    TensorId addFloatInit(Graph &g, const std::string &name, const Shape &shape, const std::vector<float> &values) {
        TensorDesc d;
        d.name          = name;
        d.shape         = shape;
        d.isInitializer = true;
        TensorId   t    = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems(numElements(shape), DType::Float32);
        for (int64_t i = 0; i < numElements(shape); ++i)
        {
            hb.f32()[i] = values[(size_t) i % values.size()];
        }
        g.initializers[t] = hb;
        return t;
    }
    TensorId addI64Init(Graph &g, const std::string &name, const std::vector<int64_t> &values) {
        TensorDesc d;
        d.name          = name;
        d.shape         = {(int64_t) values.size()};
        d.dtype         = DType::Int64;
        d.isInitializer = true;
        TensorId   t    = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) values.size(), DType::Int64);
        for (size_t i = 0; i < values.size(); ++i)
        {
            hb.i64()[i] = values[i];
        }
        g.initializers[t] = hb;
        return t;
    }
    Attr ints(std::vector<int64_t> v) {
        Attr a;
        a.kind = Attr::Ints;
        a.ints = std::move(v);
        return a;
    }
    Attr integer(int64_t v) {
        Attr a;
        a.kind = Attr::Int;
        a.i    = v;
        return a;
    }
    Node node(OpType type, const char *name, std::vector<TensorId> in, TensorId out) {
        Node n;
        n.type    = type;
        n.name    = name;
        n.inputs  = std::move(in);
        n.outputs = {out};
        return n;
    }

    struct Variant {
        bool    wrongSoftmaxAxis = false; // softmax over the sides instead of the bins
        bool    withBias         = false; // the bin conv carries a bias
        bool    extraReader      = false; // the softmax output has a second reader
        int64_t reshapeSides     = kSides; // the closing reshape's side count
    };
    // x [N,S*B,L] -> Reshape [N,S,B,L] -> Transpose(0,3,1,2) -> Softmax(axis 3) -> Transpose(0,3,2,1)
    // -> Conv1x1(B->1, w = bin values) -> Reshape [N,S,L] -> y
    Graph dflGraph(const Variant &v = {}) {
        Graph    g;
        TensorId x = addAct(g, "x", {kN, kSides * kBins, kL}, true);
        g.inputs   = {x};
        TensorId view = addAct(g, "view", {kN, kSides, kBins, kL});
        TensorId t0   = addAct(g, "t0", {kN, kL, kSides, kBins});
        TensorId sm   = addAct(g, "sm", {kN, kL, kSides, kBins});
        TensorId t1   = addAct(g, "t1", {kN, kBins, kSides, kL});
        TensorId c    = addAct(g, "c", {kN, 1, kSides, kL});
        TensorId y    = addAct(g, "y", {kN, v.reshapeSides, kL});
        g.nodes.push_back(node(OpType::Reshape, "reshape", {x, addI64Init(g, "shape0", {kN, kSides, kBins, kL})}, view));
        Node tr0             = node(OpType::Transpose, "transpose0", {view}, t0);
        tr0.attr.map["perm"] = ints({0, 3, 1, 2});
        g.nodes.push_back(tr0);
        Node smx             = node(OpType::Softmax, "softmax", {t0}, sm);
        smx.attr.map["axis"] = integer(v.wrongSoftmaxAxis ? 2 : 3);
        g.nodes.push_back(smx);
        Node tr1             = node(OpType::Transpose, "transpose1", {sm}, t1);
        tr1.attr.map["perm"] = ints({0, 3, 2, 1});
        g.nodes.push_back(tr1);
        std::vector<float> bins;
        for (int64_t b = 0; b < kBins; ++b)
        {
            bins.push_back((float) b * 1.5f);
        }
        std::vector<TensorId> convIn = {t1, addFloatInit(g, "w", {1, kBins, 1, 1}, bins)};
        if (v.withBias)
        {
            convIn.push_back(addFloatInit(g, "b", {1}, {0.25f}));
        }
        Node conv                    = node(OpType::Conv, "binconv", convIn, c);
        conv.attr.map["kernel_shape"] = ints({1, 1});
        conv.attr.map["strides"]      = ints({1, 1});
        conv.attr.map["pads"]         = ints({0, 0, 0, 0});
        conv.attr.map["dilations"]    = ints({1, 1});
        conv.attr.map["group"]        = integer(1);
        g.nodes.push_back(conv);
        g.nodes.push_back(node(OpType::Reshape, "reshape1", {c, addI64Init(g, "shape1", {kN, v.reshapeSides, kL})}, y));
        g.outputs = {y};
        if (v.extraReader)
        {
            TensorId z = addAct(g, "z", {kN, kL, kSides, kBins});
            Node     relu;
            relu.type    = OpType::Relu;
            relu.name    = "extra";
            relu.inputs  = {sm};
            relu.outputs = {z};
            g.nodes.push_back(relu);
            g.outputs.push_back(z);
        }
        return g;
    }
    int countOfType(const Graph &g, OpType t) {
        int n = 0;
        for (const Node &nd: g.nodes)
        {
            n += nd.type == t;
        }
        return n;
    }
    std::vector<float> runCpu(Graph g, const std::vector<float> &src) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_NE(sess, nullptr);
        std::vector<IOTensor> in(1), out;
        in[0].name  = "x";
        in[0].shape = {kN, kSides * kBins, kL};
        in[0].dtype = DType::Float32;
        in[0].data.resize(src.size() * sizeof(float));
        std::memcpy(in[0].data.data(), src.data(), src.size() * sizeof(float));
        EXPECT_EQ(sess->run(in, out), Status::Ok);
        EXPECT_GE(out.size(), 1u);
        const float *f = out[0].f32();
        return std::vector<float>(f, f + (size_t) numElements(out[0].shape));
    }
    std::vector<float> boxMap() {
        std::vector<float> x((size_t) (kN * kSides * kBins * kL));
        for (size_t i = 0; i < x.size(); ++i)
        {
            x[i] = std::sin((float) i * 0.61f) * 3.0f + (float) (i % 3);
        }
        return x;
    }
} // namespace

TEST(FuseDfl, ChainBecomesOneNodeOverTheBoxMap) {
    Graph g = dflGraph();
    fuseDfl(g);
    ASSERT_EQ(countOfType(g, OpType::FusedDfl), 1);
    EXPECT_EQ(countOfType(g, OpType::Reshape), 0);
    EXPECT_EQ(countOfType(g, OpType::Transpose), 0);
    EXPECT_EQ(countOfType(g, OpType::Softmax), 0);
    EXPECT_EQ(countOfType(g, OpType::Conv), 0);
    ASSERT_EQ(g.nodes.size(), 1u);
    const Node &dfl = g.nodes[0];
    ASSERT_EQ(dfl.inputs.size(), 2u);
    EXPECT_EQ(g.desc(dfl.inputs[0]).name, "x") << "the fused node reads the [N,S*B,L] map itself";
    EXPECT_EQ(g.desc(dfl.inputs[1]).name, "w");
    EXPECT_EQ(g.desc(dfl.outputs[0]).name, "y") << "the closing Reshape's output id survives";
    EXPECT_EQ(dfl.attr.geti("bins", 0), kBins);
}

TEST(FuseDfl, RefusesAnOpenOrMismatchedChain) {
    Variant wrongAxis;
    wrongAxis.wrongSoftmaxAxis = true;
    Graph g1                   = dflGraph(wrongAxis);
    fuseDfl(g1);
    EXPECT_EQ(countOfType(g1, OpType::FusedDfl), 0) << "a softmax over the sides is not the decode";
    Variant biased;
    biased.withBias = true;
    Graph g2        = dflGraph(biased);
    fuseDfl(g2);
    EXPECT_EQ(countOfType(g2, OpType::FusedDfl), 0) << "a biased bin conv is left alone";
    Variant open;
    open.extraReader = true;
    Graph g3         = dflGraph(open);
    fuseDfl(g3);
    EXPECT_EQ(countOfType(g3, OpType::FusedDfl), 0) << "an intermediate with a second reader cannot be folded away";
    Variant wrongClose;
    wrongClose.reshapeSides = kSides * kL; // the closing reshape does not produce [N,S,L]
    Graph g4                = dflGraph(wrongClose);
    g4.tensors[g4.nodes.back().outputs[0]].shape = {kN, kSides * kL, 1};
    fuseDfl(g4);
    EXPECT_EQ(countOfType(g4, OpType::FusedDfl), 0);
}

TEST(FuseDfl, PartOfTheDefaultLevelAndShapedByInference) {
    Graph       g   = dflGraph();
    PassOptions opt = PassOptions::forOptLevel(1);
    EXPECT_TRUE(opt.fuseDfl);
    runStandardPasses(g, opt);
    ASSERT_EQ(countOfType(g, OpType::FusedDfl), 1);
    for (const Node &nd: g.nodes)
    {
        if (nd.type == OpType::FusedDfl)
        {
            EXPECT_EQ(g.desc(nd.outputs[0]).shape, (Shape {kN, kSides, kL}));
        }
    }
    Graph       g0   = dflGraph();
    PassOptions opt0 = PassOptions::forOptLevel(0);
    runStandardPasses(g0, opt0);
    EXPECT_EQ(countOfType(g0, OpType::FusedDfl), 0) << "-O0 keeps one kernel per op";
}

// The CPU FusedDfl op must agree with the unfused CPU chain (the fused op is the oracle the GPU
// kernel is gated on).
TEST(FuseDfl, FusedCpuOpMatchesTheUnfusedChain) {
    const std::vector<float> x       = boxMap();
    std::vector<float>       unfused = runCpu(dflGraph(), x);
    Graph                    fusedGraph = dflGraph();
    fuseDfl(fusedGraph);
    ASSERT_EQ(countOfType(fusedGraph, OpType::FusedDfl), 1);
    std::vector<float> fused = runCpu(std::move(fusedGraph), x);
    ASSERT_EQ(fused.size(), (size_t) (kN * kSides * kL));
    ASSERT_EQ(fused.size(), unfused.size());
    for (size_t i = 0; i < fused.size(); ++i)
    {
        EXPECT_NEAR(fused[i], unfused[i], 1e-4f) << "element " << i;
    }
    // Hand check of one element: softmax over the bins, expectation against the bin values.
    const int64_t l = 2;
    float         m = x[(size_t) l];
    for (int64_t b = 1; b < kBins; ++b)
    {
        m = std::max(m, x[(size_t) (b * kL + l)]);
    }
    double sum = 0, acc = 0;
    for (int64_t b = 0; b < kBins; ++b)
    {
        const double e = std::exp((double) x[(size_t) (b * kL + l)] - m);
        sum += e;
        acc += e * (double) b * 1.5;
    }
    EXPECT_NEAR(fused[(size_t) l], (float) (acc / sum), 1e-5f);
}

// The op runs on the blocked layout (its map comes from the head's blocked spatial concat), and the
// descriptor table must cover an op type appended at the end of the enum (the table is sized from
// the OpTypeEnd sentinel; a table sized from an older last member served an appended op a default
// row, and a kernel then read a buffer of the other layout).
TEST(FuseDfl, DescriptorRowIsBlockedAndTheTableCoversTheWholeEnum) {
    EXPECT_EQ(opDescriptor(OpType::FusedDfl).layout, LayoutClass::Nc4);
    EXPECT_FALSE(opDescriptor(OpType::FusedDfl).pwMember);
    EXPECT_GT((int) OpType::OpTypeEnd, (int) OpType::FusedDfl);
}
