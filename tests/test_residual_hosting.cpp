// A pointwise Conv followed by a residual Add (and optionally one Relu / Clip) hosts the residual on
// the kernel's own fusedResidual input and the activation on fusedAct, instead of becoming a
// pointwise-VM unit with an operand. The pointwise kernels add the residual in the fp32 accumulator
// and activate before the store, the VM unit's exact order, so the bytes match while the per-element
// interpretation (measured at as much as the conv itself on a large map) is gone. A KxK conv keeps the
// VM unit: only the pointwise kernels carry a native residual input.
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
            hb.f32()[i] = 0.25f;
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
    Node makeConv(Graph &g, TensorId x, TensorId out, int64_t k) {
        Node conv;
        conv.type                     = OpType::Conv;
        conv.name                     = "conv";
        conv.inputs                   = {x, addFloatInit(g, "w", {4, 4, k, k}), addFloatInit(g, "b", {4})};
        conv.outputs                  = {out};
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
    // x -> Conv(k) -> Add(conv_out, r) [-> Relu] -> graph output, with r a second graph input.
    Graph residualGraph(int64_t k, bool relu) {
        Graph    g;
        TensorId x       = addAct(g, "x", {1, 4, 8, 8}, true);
        TensorId r       = addAct(g, "r", {1, 4, 8, 8}, true);
        g.inputs         = {x, r};
        TensorId convOut = addAct(g, "conv_out", {1, 4, 8, 8});
        TensorId addOut  = addAct(g, "add_out", {1, 4, 8, 8});
        g.nodes.push_back(makeConv(g, x, convOut, k));
        Node add;
        add.type    = OpType::Add;
        add.name    = "residual";
        add.inputs  = {convOut, r};
        add.outputs = {addOut};
        g.nodes.push_back(add);
        TensorId out = addOut;
        if (relu)
        {
            out = addAct(g, "y", {1, 4, 8, 8});
            Node act;
            act.type    = OpType::Relu;
            act.name    = "relu";
            act.inputs  = {addOut};
            act.outputs = {out};
            g.nodes.push_back(act);
        }
        g.desc(out).isOutput = true;
        g.outputs            = {out};
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

TEST(ResidualHosting, PointwiseAddReluHostsOnTheKernel) {
    Graph g = residualGraph(1, /*relu=*/true);
    runStandardPasses(g, PassOptions::forOptLevel(1));
    EXPECT_EQ(countOfType(g, OpType::Add), 0);
    EXPECT_EQ(countOfType(g, OpType::Relu), 0);
    const Node *conv = findByType(g, OpType::Conv);
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->fusedResidual, g.inputs[1]) << "the residual rides the kernel's native input";
    EXPECT_EQ(conv->fusedAct, ActType::Relu);
    EXPECT_FALSE(conv->attr.has("pw_steps")) << "no VM unit may remain";
    ASSERT_EQ(conv->outputs.size(), 1u);
    EXPECT_EQ(conv->outputs[0], g.outputs[0]);
}

TEST(ResidualHosting, PointwiseAddAloneHostsWithoutAnActivation) {
    Graph g = residualGraph(1, /*relu=*/false);
    runStandardPasses(g, PassOptions::forOptLevel(1));
    EXPECT_EQ(countOfType(g, OpType::Add), 0);
    const Node *conv = findByType(g, OpType::Conv);
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->fusedResidual, g.inputs[1]);
    EXPECT_EQ(conv->fusedAct, ActType::None);
    EXPECT_FALSE(conv->attr.has("pw_steps"));
}

// A 3x3 conv has no native residual input on the Vulkan side, so the chain stays a VM unit (or its
// own nodes) and the conv's fusedResidual stays clear.
TEST(ResidualHosting, KxKConvKeepsTheVmUnit) {
    Graph g = residualGraph(3, /*relu=*/true);
    runStandardPasses(g, PassOptions::forOptLevel(1));
    const Node *conv = findByType(g, OpType::Conv);
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->fusedResidual, kNoTensor);
    EXPECT_TRUE(conv->attr.has("pw_steps") || countOfType(g, OpType::Add) == 1) << "the residual is a VM unit or an unfused Add";
}

TEST(ResidualHosting, OptLevelZeroLeavesTheChain) {
    Graph g = residualGraph(1, /*relu=*/true);
    runStandardPasses(g, PassOptions::forOptLevel(0));
    EXPECT_EQ(countOfType(g, OpType::Add), 1);
    EXPECT_EQ(countOfType(g, OpType::Relu), 1);
    const Node *conv = findByType(g, OpType::Conv);
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->fusedResidual, kNoTensor);
}
