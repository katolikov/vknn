// A swish diamond on a Conv - Sigmoid + Mul (SiLU) or HardSigmoid(1/6, 1/2) + Mul (HardSwish) -
// hosts on the kernel's own activation code, like a lone Relu does, instead of becoming a one-step
// pointwise-VM epilogue. The kernels carry both codes with the VM step's formula, so the bytes are
// the same and the per-element VM interpretation (measured at more than the conv it decorated on a
// large map) is gone. Strict fusion keeps the two rounded steps and therefore the VM unit; an entry
// that is also consumed outside the diamond cannot host, since the pre-activation value must stay
// materialized.
#include "import/passes.h"
#include "vknn/graph.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {
    TensorId addAct(Graph &g, const std::string &name, Shape shape) {
        TensorDesc d;
        d.name  = name;
        d.shape = std::move(shape);
        return g.addTensor(d);
    }
    TensorId addFloatInit(Graph &g, const std::string &name, const Shape &shape) {
        TensorDesc d;
        d.name          = name;
        d.shape         = shape;
        d.isInitializer = true;
        TensorId   t    = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems(numElements(shape), DType::Float32);
        for (int64_t i = 0; i < numElements(shape); ++i)
        {
            hb.f32()[i] = 0.5f;
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
    Node makeConv(Graph &g, TensorId x, TensorId out) {
        Node conv;
        conv.type                     = OpType::Conv;
        conv.name                     = "conv";
        conv.inputs                   = {x, addFloatInit(g, "w", {4, 4, 3, 3}), addFloatInit(g, "b", {4})};
        conv.outputs                  = {out};
        conv.attr.map["kernel_shape"] = ints({3, 3});
        conv.attr.map["strides"]      = ints({1, 1});
        conv.attr.map["pads"]         = ints({1, 1, 1, 1});
        conv.attr.map["dilations"]    = ints({1, 1});
        Attr group;
        group.kind             = Attr::Int;
        group.i                = 1;
        conv.attr.map["group"] = group;
        return conv;
    }
    // Conv -> gate(conv_out) -> Mul(conv_out, gate) -> graph output. `hardSigmoid` selects the
    // HardSwish diamond (HardSigmoid with ONNX's alpha 1/6, beta 1/2 in actLo/actHi).
    Graph swishGraph(bool hardSigmoid, bool entryAlsoOutput) {
        Graph      g;
        TensorDesc in;
        in.name          = "x";
        in.shape         = {1, 4, 8, 8};
        in.isInput       = true;
        TensorId x       = g.addTensor(in);
        g.inputs         = {x};
        TensorId convOut = addAct(g, "conv_out", {1, 4, 8, 8});
        TensorId gate    = addAct(g, "gate", {1, 4, 8, 8});
        TensorId y       = addAct(g, "y", {1, 4, 8, 8});
        g.nodes.push_back(makeConv(g, x, convOut));
        Node sig;
        sig.type    = OpType::Unary;
        sig.name    = "gate";
        sig.subOp   = (int) (hardSigmoid ? UnaryType::HardSigmoid : UnaryType::Sigmoid);
        sig.actLo   = hardSigmoid ? 1.0f / 6.0f : 0.f;
        sig.actHi   = hardSigmoid ? 0.5f : 0.f;
        sig.inputs  = {convOut};
        sig.outputs = {gate};
        g.nodes.push_back(sig);
        Node mul;
        mul.type    = OpType::Binary;
        mul.name    = "swish";
        mul.subOp   = (int) BinaryType::Mul;
        mul.inputs  = {convOut, gate};
        mul.outputs = {y};
        g.nodes.push_back(mul);
        g.desc(y).isOutput = true;
        g.outputs          = {y};
        if (entryAlsoOutput)
        {
            g.desc(convOut).isOutput = true;
            g.outputs.push_back(convOut);
        }
        return g;
    }
    int countOfType(const Graph &g, OpType t) {
        int n = 0;
        for (const Node &nd: g.nodes)
        {
            n += nd.type == t ? 1 : 0;
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
} // namespace

TEST(SwishHosting, SiluDiamondHostsOnTheConvActivationCode) {
    Graph g = swishGraph(/*hardSigmoid=*/false, /*entryAlsoOutput=*/false);
    runStandardPasses(g, PassOptions::forOptLevel(1));
    EXPECT_EQ(countOfType(g, OpType::Unary), 0);
    EXPECT_EQ(countOfType(g, OpType::Binary), 0);
    const Node *conv = findByType(g, OpType::Conv);
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->fusedAct, ActType::SiLU) << "the diamond must land as the kernel's plain SiLU, not a VM unit";
    EXPECT_FALSE(conv->attr.has("pw_steps"));
    ASSERT_EQ(conv->outputs.size(), 1u);
    EXPECT_EQ(conv->outputs[0], g.outputs[0]) << "the Conv writes the diamond's output tensor directly";
}

TEST(SwishHosting, HardSwishDiamondHostsOnTheConvActivationCode) {
    Graph g = swishGraph(/*hardSigmoid=*/true, /*entryAlsoOutput=*/false);
    runStandardPasses(g, PassOptions::forOptLevel(1));
    EXPECT_EQ(countOfType(g, OpType::Unary), 0);
    EXPECT_EQ(countOfType(g, OpType::Binary), 0);
    const Node *conv = findByType(g, OpType::Conv);
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->fusedAct, ActType::HardSwish);
    EXPECT_FALSE(conv->attr.has("pw_steps"));
}

// The pre-activation value is also a graph output, so the entry stays materialized: the diamond
// must not collapse onto the Conv's activation code (the hosting rule requires an unexported entry).
// The pass may still fuse it as a VM unit that exports the entry; either way the Conv keeps
// producing the pre-activation tensor.
TEST(SwishHosting, ExportedEntryKeepsTheConvActivationFree) {
    Graph g = swishGraph(/*hardSigmoid=*/false, /*entryAlsoOutput=*/true);
    runStandardPasses(g, PassOptions::forOptLevel(1));
    const Node *conv = findByType(g, OpType::Conv);
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->fusedAct, ActType::None);
    ASSERT_GE(g.outputs.size(), 2u);
    const TensorId preAct = g.outputs[1];
    bool           listed = false;
    for (TensorId o: conv->outputs)
    {
        listed = listed || o == preAct;
    }
    EXPECT_TRUE(listed) << "the Conv still produces the pre-activation tensor (unfused, or as a VM-unit export)";
}

// -O0 is the reference level: nothing fuses, the diamond stays three nodes.
TEST(SwishHosting, OptLevelZeroLeavesTheDiamond) {
    Graph g = swishGraph(/*hardSigmoid=*/false, /*entryAlsoOutput=*/false);
    runStandardPasses(g, PassOptions::forOptLevel(0));
    EXPECT_EQ(countOfType(g, OpType::Unary), 1);
    EXPECT_EQ(countOfType(g, OpType::Binary), 1);
    const Node *conv = findByType(g, OpType::Conv);
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->fusedAct, ActType::None);
}
