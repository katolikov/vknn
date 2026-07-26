// Who owns activation folding, and what the experimental block-fusion flags may change.
//
// fuseActivations exists as the PREREQUISITE of fuseSqueezeExcite / fuseDwPw: both match on
// conv.fusedAct and both run before the general pointwise fusion. It is not a second, independent
// activation optimization -- fusePointwiseChains' inline-act path folds the same Relu/Clip into the
// same conv.fusedAct, and measured over the classifier/depthwise/detector suite, running
// fuseActivations first as well yields a graph isomorphic to the pointwise pass's own result. That
// is why it stays out of the -O1 set: it would buy no fusion and change compiled bytes.
//
// The tests below pin the two halves of that decision, so a future change cannot quietly invalidate
// it: activation folding at -O1 does NOT depend on the experimental flags (if the pointwise pass
// ever stops folding, this fails and says fuseActivations is needed again), and -O0 still folds
// nothing (the reference-output contract, which an "always run fuseActivations" fix would break).
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

    // A 1x1 Conv from `x` into `out`, with the attributes shape inference needs.
    Node makeConv(Graph &g, TensorId x, TensorId out) {
        Node conv;
        conv.type    = OpType::Conv;
        conv.name    = "conv";
        conv.inputs  = {x, addFloatInit(g, "w", {4, 4, 1, 1}), addFloatInit(g, "b", {4})};
        conv.outputs = {out};
        Attr kernel;
        kernel.kind                   = Attr::Ints;
        kernel.ints                   = {1, 1};
        conv.attr.map["kernel_shape"] = kernel;
        Attr strides;
        strides.kind             = Attr::Ints;
        strides.ints             = {1, 1};
        conv.attr.map["strides"] = strides;
        Attr pads;
        pads.kind             = Attr::Ints;
        pads.ints             = {0, 0, 0, 0};
        conv.attr.map["pads"] = pads;
        Attr group;
        group.kind             = Attr::Int;
        group.i                = 1;
        conv.attr.map["group"] = group;
        return conv;
    }

    // Conv -> Relu -> graph output. A lone activation on a Conv is the shape the inline-act fast
    // path claims, so the fold lands as conv.fusedAct -- exactly what fuseActivations would set.
    Graph convReluGraph() {
        Graph      g;
        TensorDesc in;
        in.name          = "x";
        in.shape         = {1, 4, 8, 8};
        in.isInput       = true;
        TensorId x       = g.addTensor(in);
        g.inputs         = {x};
        TensorId convOut = addAct(g, "conv_out", {1, 4, 8, 8});
        TensorId reluOut = addAct(g, "relu_out", {1, 4, 8, 8});
        g.nodes.push_back(makeConv(g, x, convOut));
        Node act;
        act.type    = OpType::Relu;
        act.name    = "relu";
        act.inputs  = {convOut};
        act.outputs = {reluOut};
        g.nodes.push_back(act);
        g.desc(reluOut).isOutput = true;
        g.outputs                = {reluOut};
        return g;
    }

    // Conv -> Relu -> residual Add -> Relu: a multi-member pointwise chain. The activations still
    // vanish as nodes, but the epilogue is encoded as the producer's pw_steps unit rather than as
    // fusedAct, so "was it folded" is the node count plus epilogue presence, not fusedAct alone.
    Graph convReluAndResidualReluGraph() {
        Graph      g;
        TensorDesc in;
        in.name    = "x";
        in.shape   = {1, 4, 8, 8};
        in.isInput = true;
        TensorId x = g.addTensor(in);
        g.inputs   = {x};

        TensorId convOut = addAct(g, "conv_out", {1, 4, 8, 8});
        TensorId relu1   = addAct(g, "relu1", {1, 4, 8, 8});
        TensorId addOut  = addAct(g, "add_out", {1, 4, 8, 8});
        TensorId relu2   = addAct(g, "relu2", {1, 4, 8, 8});

        g.nodes.push_back(makeConv(g, x, convOut));

        Node act1;
        act1.type    = OpType::Relu;
        act1.name    = "relu1";
        act1.inputs  = {convOut};
        act1.outputs = {relu1};
        g.nodes.push_back(act1);

        Node add;
        add.type    = OpType::Add;
        add.name    = "residual_add";
        add.inputs  = {relu1, x};
        add.outputs = {addOut};
        g.nodes.push_back(add);

        Node act2;
        act2.type    = OpType::Relu;
        act2.name    = "relu2";
        act2.inputs  = {addOut};
        act2.outputs = {relu2};
        g.nodes.push_back(act2);

        g.desc(relu2).isOutput = true;
        g.outputs              = {relu2};
        return g;
    }

    int countOfType(const Graph &g, OpType t) {
        int n = 0;
        for (const Node &nd: g.nodes)
        {
            if (nd.type == t)
            {
                ++n;
            }
        }
        return n;
    }

    // A folded activation reaches the producer either as its fusedAct epilogue (the inline-act fast
    // path, and what fuseActivations sets) or as a pw_steps unit (a multi-step chain).
    bool carriesEpilogue(const Node &n) {
        return n.fusedAct != ActType::None || n.attr.has("pw_steps");
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

// The load-bearing fact behind keeping fuseActivations out of -O1: with both experimental flags
// off, the general pointwise fusion sets the very conv.fusedAct that fuseActivations would.
TEST(ActivationFoldOwnership, PointwiseFusionSetsFusedActWithoutTheExperimentalFlags) {
    Graph       g   = convReluGraph();
    PassOptions opt = PassOptions::forOptLevel(1);
    ASSERT_FALSE(opt.fuseSqueezeExcite);
    ASSERT_FALSE(opt.fuseDwPw);
    runStandardPasses(g, opt);

    EXPECT_EQ(countOfType(g, OpType::Relu), 0) << "no activation may survive as its own node at -O1";
    const Node *conv = findByType(g, OpType::Conv);
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->fusedAct, ActType::Relu) << "the Conv must carry the folded activation epilogue";
}

// The residual form fuseActivations also claims (its Add producer case): the pointwise pass folds
// the whole Relu -> Add -> Relu chain onto the Conv, so no activation survives here either.
TEST(ActivationFoldOwnership, PointwiseFusionClaimsTheResidualActivationChain) {
    Graph g = convReluAndResidualReluGraph();
    runStandardPasses(g, PassOptions::forOptLevel(1));

    EXPECT_EQ(countOfType(g, OpType::Relu), 0) << "no activation may survive as its own node at -O1";
    EXPECT_EQ(countOfType(g, OpType::Add), 0) << "the residual Add joins the same fused unit";
    const Node *conv = findByType(g, OpType::Conv);
    ASSERT_NE(conv, nullptr);
    EXPECT_TRUE(carriesEpilogue(*conv)) << "the Conv must host the chain as its epilogue";
}

// -O0 is the reference-output level: one kernel per op, no optional fusion. An "always run
// fuseActivations" fix to the flag coupling would silently break this.
TEST(ActivationFoldOwnership, OptLevelZeroFoldsNothing) {
    Graph       g   = convReluAndResidualReluGraph();
    PassOptions opt = PassOptions::forOptLevel(0);
    runStandardPasses(g, opt);

    EXPECT_EQ(countOfType(g, OpType::Relu), 2) << "-O0 must leave both activations as their own nodes";
    const Node *conv = findByType(g, OpType::Conv);
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->fusedAct, ActType::None);
}

// The trap this file documents: turning one experimental block fusion off must not take activation
// folding away from the other, or from the pipeline at large. Each pass now runs its own
// prerequisite fold, so every flag combination that reaches a fusion level still folds.
TEST(ActivationFoldOwnership, NeitherExperimentalFlagGatesActivationFolding) {
    struct Combination {
        const char *label;
        bool        squeezeExcite, dwPw;
    };
    const Combination combinations[] = {
        {"-O2", true, true},
        {"-O2 --no-fuse-dwpw", true, false},
        {"-O2 --no-fuse-se", false, true},
        {"-O2 --no-fuse-se --no-fuse-dwpw", false, false},
    };
    for (const Combination &c: combinations)
    {
        Graph       g         = convReluGraph();
        PassOptions opt       = PassOptions::forOptLevel(2);
        opt.fuseSqueezeExcite = c.squeezeExcite;
        opt.fuseDwPw          = c.dwPw;
        runStandardPasses(g, opt);

        EXPECT_EQ(countOfType(g, OpType::Relu), 0) << c.label << ": an activation survived as its own node";
        const Node *conv = findByType(g, OpType::Conv);
        ASSERT_NE(conv, nullptr) << c.label;
        EXPECT_EQ(conv->fusedAct, ActType::Relu) << c.label << ": the Conv lost its activation epilogue";
    }
}
