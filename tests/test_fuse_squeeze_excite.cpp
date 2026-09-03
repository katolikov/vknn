// fuseSqueezeExcite (import/fuse_squeeze_excite.cpp): the GlobalAvgPool -> Conv1x1 -> activation ->
// Conv1x1 -> gate chain becomes one FusedSE node that pools the feature map itself, in every form
// the activation survives at pass time, and only when the chain is closed and fits the fused
// kernel. The CPU FusedSE op is checked against the unfused CPU ops on the same weights.
#include "core/squeeze_excite.h"
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {
    constexpr int64_t kN = 1, kC = 8, kCr = 4, kH = 3, kW = 5;

    TensorId addAct(Graph &g, const std::string &name, Shape shape, bool isInput = false) {
        TensorDesc d;
        d.name    = name;
        d.shape   = std::move(shape);
        d.isInput = isInput;
        return g.addTensor(d);
    }
    // A deterministic, non-uniform initializer: value(i) = base + (i mod 7) * step.
    TensorId addPatternInit(Graph &g, const std::string &name, const Shape &shape, float base, float step) {
        TensorDesc d;
        d.name          = name;
        d.shape         = shape;
        d.isInitializer = true;
        TensorId   t    = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems(numElements(shape), DType::Float32);
        for (int64_t i = 0; i < numElements(shape); ++i)
        {
            hb.f32()[i] = base + (float) (i % 7) * step;
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
    Node conv1x1(TensorId x, TensorId w, TensorId b, TensorId out, const char *name) {
        Node conv;
        conv.type    = OpType::Conv;
        conv.name    = name;
        conv.inputs  = {x, w, b};
        conv.outputs = {out};
        conv.attr.map["kernel_shape"] = ints({1, 1});
        conv.attr.map["strides"]      = ints({1, 1});
        conv.attr.map["pads"]         = ints({0, 0, 0, 0});
        conv.attr.map["dilations"]    = ints({1, 1});
        Attr group;
        group.kind             = Attr::Int;
        group.i                = 1;
        conv.attr.map["group"] = group;
        return conv;
    }
    Node unary(const char *name, UnaryType u, TensorId in, TensorId out, float lo = 0.f, float hi = 0.f) {
        Node n;
        n.type    = OpType::Unary;
        n.subOp   = (int) u;
        n.name    = name;
        n.inputs  = {in};
        n.outputs = {out};
        n.actLo   = lo;
        n.actHi   = hi;
        return n;
    }
    Node mul(const char *name, TensorId a, TensorId b, TensorId out) {
        Node n;
        n.type    = OpType::Binary;
        n.subOp   = (int) BinaryType::Mul;
        n.name    = name;
        n.inputs  = {a, b};
        n.outputs = {out};
        return n;
    }

    enum class ActForm { ReluNode, ConvEpilogueRelu, SiluDiamond, HardSwishDiamond, HardSwishNode };
    enum class GateForm { HardSigmoid, Sigmoid };

    // x -> GAP -> conv1 -> activation -> conv2 -> gate -> Mul(x, scale) -> y. `poolExtraReader`
    // adds a second consumer of the pooled tensor (an open chain).
    Graph seGraph(ActForm act, GateForm gate, bool poolExtraReader = false) {
        Graph    g;
        TensorId x     = addAct(g, "x", {kN, kC, kH, kW}, true);
        g.inputs       = {x};
        TensorId avg   = addAct(g, "avg", {kN, kC, 1, 1});
        TensorId w1    = addPatternInit(g, "w1", {kCr, kC, 1, 1}, -0.3f, 0.11f);
        TensorId b1    = addPatternInit(g, "b1", {kCr}, 0.05f, 0.02f);
        TensorId w2    = addPatternInit(g, "w2", {kC, kCr, 1, 1}, 0.2f, -0.07f);
        TensorId b2    = addPatternInit(g, "b2", {kC}, -0.1f, 0.03f);
        TensorId c1    = addAct(g, "c1", {kN, kCr, 1, 1});
        TensorId a1    = addAct(g, "a1", {kN, kCr, 1, 1});
        TensorId c2    = addAct(g, "c2", {kN, kC, 1, 1});
        TensorId scale = addAct(g, "scale", {kN, kC, 1, 1});
        TensorId y     = addAct(g, "y", {kN, kC, kH, kW});
        Node     pool;
        pool.type    = OpType::GlobalAvgPool;
        pool.name    = "pool";
        pool.inputs  = {x};
        pool.outputs = {avg};
        g.nodes.push_back(pool);
        Node conv1 = conv1x1(avg, w1, b1, act == ActForm::ConvEpilogueRelu ? a1 : c1, "conv1");
        if (act == ActForm::ConvEpilogueRelu)
        {
            conv1.fusedAct = ActType::Relu;
        }
        g.nodes.push_back(conv1);
        switch (act)
        {
            case ActForm::ReluNode: {
                Node relu;
                relu.type    = OpType::Relu;
                relu.name    = "relu";
                relu.inputs  = {c1};
                relu.outputs = {a1};
                g.nodes.push_back(relu);
                break;
            }
            case ActForm::HardSwishNode:
                g.nodes.push_back(unary("hswish", UnaryType::HardSwish, c1, a1));
                break;
            case ActForm::SiluDiamond: {
                TensorId sig = addAct(g, "sig", {kN, kCr, 1, 1});
                g.nodes.push_back(unary("sigmoid", UnaryType::Sigmoid, c1, sig));
                g.nodes.push_back(mul("silu", c1, sig, a1));
                break;
            }
            case ActForm::HardSwishDiamond: {
                TensorId hs = addAct(g, "hsig", {kN, kCr, 1, 1});
                g.nodes.push_back(unary("hardsigmoid_gate", UnaryType::HardSigmoid, c1, hs, 1.0f / 6.0f, 0.5f));
                g.nodes.push_back(mul("hswish", hs, c1, a1)); // operand order must not matter
                break;
            }
            case ActForm::ConvEpilogueRelu:
                break;
        }
        g.nodes.push_back(conv1x1(a1, w2, b2, c2, "conv2"));
        if (gate == GateForm::HardSigmoid)
        {
            g.nodes.push_back(unary("gate", UnaryType::HardSigmoid, c2, scale, 0.2f, 0.5f));
        } else
        {
            g.nodes.push_back(unary("gate", UnaryType::Sigmoid, c2, scale));
        }
        g.nodes.push_back(mul("apply", x, scale, y));
        if (poolExtraReader)
        {
            TensorId z = addAct(g, "z", {kN, kC, 1, 1});
            g.nodes.push_back(unary("extra", UnaryType::Sigmoid, avg, z));
            g.outputs = {y, z};
        } else
        {
            g.outputs = {y};
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
    const Node *findByType(const Graph &g, OpType t) {
        for (const Node &nd: g.nodes)
        {
            if (nd.type == t)
            {
                return &nd;
            }
        }
        return nullptr;
    }

    std::vector<float> runCpu(Graph g, const std::vector<float> &src) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_NE(sess, nullptr);
        std::vector<IOTensor> in(1), out;
        in[0].name  = "x";
        in[0].shape = {kN, kC, kH, kW};
        in[0].dtype = DType::Float32;
        in[0].data.resize(src.size() * sizeof(float));
        std::memcpy(in[0].data.data(), src.data(), src.size() * sizeof(float));
        EXPECT_EQ(sess->run(in, out), Status::Ok);
        EXPECT_GE(out.size(), 1u);
        const float *f = out[0].f32();
        return std::vector<float>(f, f + (size_t) numElements(out[0].shape));
    }
    std::vector<float> featureMap() {
        std::vector<float> x((size_t) (kN * kC * kH * kW));
        for (size_t i = 0; i < x.size(); ++i)
        {
            x[i] = std::sin((float) i * 0.37f) * 2.0f + (float) (i % 5) * 0.1f;
        }
        return x;
    }
    void expectFused(const Graph &g, ActType act, UnaryType gate) {
        ASSERT_EQ(countOfType(g, OpType::FusedSE), 1);
        EXPECT_EQ(countOfType(g, OpType::GlobalAvgPool), 0);
        EXPECT_EQ(countOfType(g, OpType::Conv), 0);
        EXPECT_EQ(countOfType(g, OpType::Relu), 0);
        EXPECT_EQ(countOfType(g, OpType::Unary), 0);
        EXPECT_EQ(countOfType(g, OpType::Binary), 1) << "the broadcast Mul that applies the scale stays";
        const Node *se = findByType(g, OpType::FusedSE);
        ASSERT_NE(se, nullptr);
        ASSERT_EQ(se->inputs.size(), 5u);
        EXPECT_EQ(g.desc(se->inputs[0]).name, "x") << "the fused node pools the feature map itself";
        EXPECT_EQ(g.desc(se->inputs[1]).name, "w1");
        EXPECT_EQ(g.desc(se->inputs[2]).name, "b1");
        EXPECT_EQ(g.desc(se->inputs[3]).name, "w2");
        EXPECT_EQ(g.desc(se->inputs[4]).name, "b2");
        ASSERT_EQ(se->outputs.size(), 1u);
        EXPECT_EQ(g.desc(se->outputs[0]).name, "scale") << "the gate's output id survives for the Mul";
        EXPECT_EQ(se->fusedAct, act);
        EXPECT_EQ((UnaryType) se->subOp, gate);
        const Node *apply = findByType(g, OpType::Binary);
        ASSERT_NE(apply, nullptr);
        EXPECT_EQ(apply->inputs[1], se->outputs[0]);
    }
} // namespace

TEST(FuseSqueezeExcite, ReluNodeWithHardSigmoidGate) {
    Graph g = seGraph(ActForm::ReluNode, GateForm::HardSigmoid);
    fuseSqueezeExcite(g);
    expectFused(g, ActType::Relu, UnaryType::HardSigmoid);
    const Node *se = findByType(g, OpType::FusedSE);
    EXPECT_FLOAT_EQ(se->actLo, 0.2f) << "HardSigmoid alpha";
    EXPECT_FLOAT_EQ(se->actHi, 0.5f) << "HardSigmoid beta";
}

TEST(FuseSqueezeExcite, ConvEpilogueReluIsTheSameChain) {
    Graph g = seGraph(ActForm::ConvEpilogueRelu, GateForm::HardSigmoid);
    fuseSqueezeExcite(g);
    expectFused(g, ActType::Relu, UnaryType::HardSigmoid);
}

TEST(FuseSqueezeExcite, SiluDiamondWithSigmoidGate) {
    Graph g = seGraph(ActForm::SiluDiamond, GateForm::Sigmoid);
    fuseSqueezeExcite(g);
    expectFused(g, ActType::SiLU, UnaryType::Sigmoid);
}

TEST(FuseSqueezeExcite, HardSwishDiamondAndHardSwishNode) {
    Graph g = seGraph(ActForm::HardSwishDiamond, GateForm::HardSigmoid);
    fuseSqueezeExcite(g);
    expectFused(g, ActType::HardSwish, UnaryType::HardSigmoid);
    Graph h = seGraph(ActForm::HardSwishNode, GateForm::Sigmoid);
    fuseSqueezeExcite(h);
    expectFused(h, ActType::HardSwish, UnaryType::Sigmoid);
}

TEST(FuseSqueezeExcite, OpenChainStaysUnfused) {
    Graph g = seGraph(ActForm::ReluNode, GateForm::HardSigmoid, /*poolExtraReader=*/true);
    fuseSqueezeExcite(g);
    EXPECT_EQ(countOfType(g, OpType::FusedSE), 0) << "a pooled tensor with a second reader cannot be folded away";
    EXPECT_EQ(countOfType(g, OpType::GlobalAvgPool), 1);
    EXPECT_EQ(countOfType(g, OpType::Conv), 2);
}

TEST(FuseSqueezeExcite, ChainWiderThanTheKernelStaysUnfused) {
    Graph g = seGraph(ActForm::ReluNode, GateForm::HardSigmoid);
    // Re-declare the weights wider than the kernel's shared arrays; only the shapes matter here.
    for (Node &n: g.nodes)
    {
        if (n.type == OpType::Conv && n.name == "conv1")
        {
            g.tensors[n.inputs[1]].shape = {kSeMaxSqueeze + 1, kC, 1, 1};
        }
        if (n.type == OpType::Conv && n.name == "conv2")
        {
            g.tensors[n.inputs[1]].shape = {kC, kSeMaxSqueeze + 1, 1, 1};
        }
    }
    fuseSqueezeExcite(g);
    EXPECT_EQ(countOfType(g, OpType::FusedSE), 0);
}

// The fusion is opt-in (-O2 / --fuse-se): it needs no activation prefold, and the default level
// keeps the chain unfused (measured faster on the release device).
TEST(FuseSqueezeExcite, OptInAtLevelTwoWithoutAPrefold) {
    Graph       g   = seGraph(ActForm::SiluDiamond, GateForm::Sigmoid);
    PassOptions opt = PassOptions::forOptLevel(2);
    EXPECT_TRUE(opt.fuseSqueezeExcite);
    opt.fuseDwPw = false; // no prefold runs; the pass matches the Sigmoid-Mul diamond itself
    runStandardPasses(g, opt);
    EXPECT_EQ(countOfType(g, OpType::FusedSE), 1);
    EXPECT_EQ(countOfType(g, OpType::GlobalAvgPool), 0);
    Graph       g1   = seGraph(ActForm::SiluDiamond, GateForm::Sigmoid);
    PassOptions opt1 = PassOptions::forOptLevel(1);
    EXPECT_FALSE(opt1.fuseSqueezeExcite);
    runStandardPasses(g1, opt1);
    EXPECT_EQ(countOfType(g1, OpType::FusedSE), 0) << "-O1 keeps the chain unfused";
}

// The CPU FusedSE op must agree with the unfused CPU chain on the same weights, for every
// activation and gate the fusion accepts (the fused op is the oracle the GPU kernel is gated on).
TEST(FuseSqueezeExcite, FusedCpuOpMatchesTheUnfusedChain) {
    const std::vector<float> x = featureMap();
    const struct {
        ActForm  act;
        GateForm gate;
    } cases[] = {{ActForm::ReluNode, GateForm::HardSigmoid}, {ActForm::SiluDiamond, GateForm::Sigmoid}, {ActForm::HardSwishDiamond, GateForm::Sigmoid}, {ActForm::HardSwishNode, GateForm::HardSigmoid}};
    for (const auto &c: cases)
    {
        std::vector<float> unfused = runCpu(seGraph(c.act, c.gate), x);
        Graph              fusedGraph = seGraph(c.act, c.gate);
        fuseSqueezeExcite(fusedGraph);
        ASSERT_EQ(countOfType(fusedGraph, OpType::FusedSE), 1);
        std::vector<float> fused = runCpu(std::move(fusedGraph), x);
        ASSERT_EQ(fused.size(), unfused.size());
        for (size_t i = 0; i < fused.size(); ++i)
        {
            EXPECT_NEAR(fused[i], unfused[i], 1e-4f) << "act " << (int) c.act << " gate " << (int) c.gate << " element " << i;
        }
    }
}
