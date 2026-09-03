// The conv input-affine prologue (core/input_affine.h): a per-channel affine plus activation whose
// producer cannot host it is applied by its single consumer Conv at input load instead of running
// as a standalone FusedPointwise node. DenseNet's BatchNorm-ReLU-Conv blocks after a Concat view
// and a squeeze-excite scale before a projection conv are the two shapes this serves.
#include "core/input_affine.h"
#include "import/passes.h"
#include "vknn/graph.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    TensorId addAct(Graph &g, const std::string &name, Shape shape, bool isInput = false) {
        TensorDesc d;
        d.name    = name;
        d.shape   = std::move(shape);
        d.isInput = isInput;
        return g.addTensor(d);
    }

    TensorId addFloatInit(Graph &g, const std::string &name, const Shape &shape, float value) {
        TensorDesc d;
        d.name          = name;
        d.shape         = shape;
        d.isInitializer = true;
        TensorId   t    = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems(numElements(shape), DType::Float32);
        for (int64_t i = 0; i < numElements(shape); ++i)
        {
            hb.f32()[i] = value;
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

    Node makeConv(Graph &g, TensorId x, TensorId out, int64_t k, const char *name) {
        Node conv;
        conv.type    = OpType::Conv;
        conv.name    = name;
        conv.inputs  = {x, addFloatInit(g, std::string(name) + "_w", {8, 8, k, k}, 0.125f), addFloatInit(g, std::string(name) + "_b", {8}, 0.f)};
        conv.outputs = {out};
        conv.attr.map["kernel_shape"] = ints({k, k});
        conv.attr.map["strides"]      = ints({1, 1});
        conv.attr.map["pads"]         = ints({k / 2, k / 2, k / 2, k / 2});
        conv.attr.map["dilations"]    = ints({1, 1});
        Attr group;
        group.kind             = Attr::Int;
        group.i                = 1;
        conv.attr.map["group"] = group;
        return conv;
    }

    Node binary(const char *name, BinaryType kind, TensorId a, TensorId b, TensorId out) {
        Node n;
        n.type    = kind == BinaryType::Add ? OpType::Add : OpType::Binary;
        n.subOp   = kind == BinaryType::Add ? 0 : (int) kind;
        n.name    = name;
        n.inputs  = {a, b};
        n.outputs = {out};
        return n;
    }

    // x (graph input, nothing can host a unit on it) -> Mul(scale) -> Add(shift) -> Relu -> Conv kxk
    // [-> a second reader of the unit output when `twoReaders`].
    Graph bnReluConvGraph(int64_t k, bool twoReaders) {
        Graph    g;
        TensorId x      = addAct(g, "x", {1, 8, 6, 6}, true);
        g.inputs        = {x};
        TensorId s      = addFloatInit(g, "bn_scale", {1, 8, 1, 1}, 0.5f);
        TensorId b      = addFloatInit(g, "bn_shift", {1, 8, 1, 1}, 0.25f);
        TensorId mulOut = addAct(g, "mul_out", {1, 8, 6, 6});
        TensorId addOut = addAct(g, "add_out", {1, 8, 6, 6});
        TensorId relOut = addAct(g, "relu_out", {1, 8, 6, 6});
        TensorId y      = addAct(g, "y", {1, 8, 6, 6});
        g.nodes.push_back(binary("mul", BinaryType::Mul, x, s, mulOut));
        g.nodes.push_back(binary("add", BinaryType::Add, mulOut, b, addOut));
        Node relu;
        relu.type    = OpType::Relu;
        relu.name    = "relu";
        relu.inputs  = {addOut};
        relu.outputs = {relOut};
        g.nodes.push_back(relu);
        g.nodes.push_back(makeConv(g, relOut, y, k, "conv"));
        g.desc(y).isOutput = true;
        g.outputs          = {y};
        if (twoReaders)
        {
            TensorId y2 = addAct(g, "y2", {1, 8, 6, 6});
            g.nodes.push_back(makeConv(g, relOut, y2, 1, "conv2"));
            g.desc(y2).isOutput = true;
            g.outputs.push_back(y2);
        }
        return g;
    }

    // x -> Mul(s, an activation [1,8,1,1] graph input: the squeeze-excite scale) -> Conv 1x1.
    Graph seScaleGraph(int64_t k) {
        Graph    g;
        TensorId x      = addAct(g, "x", {1, 8, 6, 6}, true);
        TensorId s      = addAct(g, "se_scale", {1, 8, 1, 1}, true);
        g.inputs        = {x, s};
        TensorId mulOut = addAct(g, "mul_out", {1, 8, 6, 6});
        TensorId y      = addAct(g, "y", {1, 8, 6, 6});
        g.nodes.push_back(binary("se_mul", BinaryType::Mul, x, s, mulOut));
        g.nodes.push_back(makeConv(g, mulOut, y, k, "project"));
        g.desc(y).isOutput = true;
        g.outputs          = {y};
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

    const Node *findByName(const Graph &g, const char *name) {
        for (const Node &nd: g.nodes)
        {
            if (nd.name == name)
            {
                return &nd;
            }
        }
        return nullptr;
    }

} // namespace

TEST(InputAffine, BatchNormReluBeforeAConvHostsOnTheConvInput) {
    for (int64_t k: {3, 5})
    {
        Graph g = bnReluConvGraph(k, /*twoReaders=*/false);
        runStandardPasses(g, PassOptions::forOptLevel(1));
        EXPECT_EQ(countOfType(g, OpType::FusedPointwise), 0) << "k=" << k;
        EXPECT_EQ(countOfType(g, OpType::Binary), 0);
        EXPECT_EQ(countOfType(g, OpType::Add), 0);
        EXPECT_EQ(countOfType(g, OpType::Relu), 0);
        const Node *conv = findByName(g, "conv");
        ASSERT_NE(conv, nullptr);
        ASSERT_TRUE(inputAffineActive(*conv));
        EXPECT_EQ(conv->inputs[0], g.inputs[0]) << "the conv reads the unit's entry directly";
        const int64_t scaleAt = conv->attr.geti("pro_scale", -1), shiftAt = conv->attr.geti("pro_shift", -1);
        ASSERT_GE(scaleAt, 3);
        ASSERT_GE(shiftAt, 3);
        EXPECT_EQ(g.desc(conv->inputs[(size_t) scaleAt]).name, "bn_scale");
        EXPECT_EQ(g.desc(conv->inputs[(size_t) shiftAt]).name, "bn_shift");
        EXPECT_EQ(conv->attr.geti("pro_act", 0), (int64_t) ActType::Relu);
        EXPECT_EQ(conv->attr.geti("pro_opbase", -1), 3) << "the prologue operands follow the conv's own inputs";
        EXPECT_EQ(pwCoreInputs(*conv), 3u) << "positional reads of bias stop before the prologue operands";
        EXPECT_FALSE(conv->attr.has("pw_steps"));
        EXPECT_EQ(conv->outputs[0], g.outputs[0]);
    }
}

// A pointwise consumer keeps the standalone unit: a 1x1 kernel reuses each loaded input over too
// few multiplies to absorb the prologue (kInputAffineMinTaps).
TEST(InputAffine, APointwiseConsumerKeepsTheStandaloneUnit) {
    Graph g = bnReluConvGraph(1, /*twoReaders=*/false);
    runStandardPasses(g, PassOptions::forOptLevel(1));
    EXPECT_EQ(countOfType(g, OpType::FusedPointwise), 1);
    const Node *conv = findByName(g, "conv");
    ASSERT_NE(conv, nullptr);
    EXPECT_FALSE(inputAffineActive(*conv));
    EXPECT_EQ(pwCoreInputs(*conv), 3u);
    // The standalone unit is marked as a channel affine for the vec4-per-thread kernel.
    const Node *unitNode = nullptr;
    for (const Node &nd: g.nodes)
    {
        if (nd.type == OpType::FusedPointwise)
        {
            unitNode = &nd;
        }
    }
    ASSERT_NE(unitNode, nullptr);
    EXPECT_TRUE(unitNode->attr.has("pw_affine_act"));
    EXPECT_EQ(unitNode->attr.geti("pw_affine_act", -1), (int64_t) ActType::Relu);
    const int64_t scaleAt = unitNode->attr.geti("pw_affine_scale", -1), shiftAt = unitNode->attr.geti("pw_affine_shift", -1);
    ASSERT_GE(scaleAt, 1);
    ASSERT_GE(shiftAt, 1);
    EXPECT_EQ(g.desc(unitNode->inputs[(size_t) scaleAt]).name, "bn_scale");
    EXPECT_EQ(g.desc(unitNode->inputs[(size_t) shiftAt]).name, "bn_shift");
}

// A squeeze-excite scale (a runtime [1,C,1,1] tensor) before a 3x3 conv hosts as a scale-only
// prologue; before a 1x1 projection it stays a Binary node (the usual squeeze-excite shape).
TEST(InputAffine, SqueezeExciteScaleHostsAsAScaleOnlyPrologueOnAKxKConsumer) {
    Graph g = seScaleGraph(3);
    runStandardPasses(g, PassOptions::forOptLevel(1));
    EXPECT_EQ(countOfType(g, OpType::FusedPointwise), 0);
    EXPECT_EQ(countOfType(g, OpType::Binary), 0);
    const Node *conv = findByName(g, "project");
    ASSERT_NE(conv, nullptr);
    ASSERT_TRUE(inputAffineActive(*conv));
    EXPECT_EQ(conv->inputs[0], g.inputs[0]);
    const int64_t scaleAt = conv->attr.geti("pro_scale", -1);
    ASSERT_GE(scaleAt, 3);
    EXPECT_EQ(conv->inputs[(size_t) scaleAt], g.inputs[1]) << "the scale is the runtime SE tensor";
    EXPECT_EQ(conv->attr.geti("pro_shift", -1), -1);
    EXPECT_EQ(conv->attr.geti("pro_act", 0), (int64_t) ActType::None);
    Graph g1 = seScaleGraph(1);
    runStandardPasses(g1, PassOptions::forOptLevel(1));
    EXPECT_EQ(countOfType(g1, OpType::Binary), 1);
    EXPECT_FALSE(inputAffineActive(*findByName(g1, "project")));
}

TEST(InputAffine, AUnitWithTwoReadersStaysStandalone) {
    Graph g = bnReluConvGraph(3, /*twoReaders=*/true);
    runStandardPasses(g, PassOptions::forOptLevel(1));
    EXPECT_EQ(countOfType(g, OpType::FusedPointwise), 1) << "two readers: the unit must materialize once";
    for (const Node &nd: g.nodes)
    {
        EXPECT_FALSE(inputAffineActive(nd)) << nd.name;
    }
}

TEST(InputAffine, CoreInputsHonourThePrologueBase) {
    Node n;
    n.inputs = {0, 1, 2, 3, 4};
    EXPECT_EQ(pwCoreInputs(n), 5u);
    Attr base;
    base.kind                = Attr::Int;
    base.i                   = 3;
    n.attr.map["pro_opbase"] = base;
    EXPECT_EQ(pwCoreInputs(n), 3u);
    Attr steps;
    steps.kind             = Attr::Ints;
    steps.ints             = std::vector<int64_t>(8, 0);
    n.attr.map["pw_steps"] = steps;
    Attr pwBase;
    pwBase.kind             = Attr::Int;
    pwBase.i                = 4;
    n.attr.map["pw_opbase"] = pwBase;
    EXPECT_EQ(pwCoreInputs(n), 3u) << "the earlier of the two operand bases bounds the core inputs";
}
