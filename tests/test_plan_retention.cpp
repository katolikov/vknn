// Host tests for the plan-retention rules (src/core/plan_retention.h): which initializer payloads a
// built plan bucket may drop, and whether a session's pristine imported graph can still yield a
// bucket the default one does not cover.
//
// Both rules are pure functions of the graph, so they are exercised here directly. The session-side
// consumers are the freeWeightsAfterUpload reclaim at the end of Session::buildBucket and the
// pristine-graph retention in Session::create; the end-to-end behaviour those drive (prepareShapes
// stays correct on a static-input model) is pinned at the bottom of this file through the public API.
#include "core/plan_retention.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <gtest/gtest.h>
#include <vector>

using namespace vknn;

namespace {

    Attr ints(std::vector<int64_t> v) {
        Attr a;
        a.kind = Attr::Ints;
        a.ints = std::move(v);
        return a;
    }

    // Register a named initializer of `count` fp32 elements and give it a distinct constant payload.
    TensorId addInitializer(Graph &g, const std::string &name, int64_t count, float fill) {
        TensorDesc d;
        d.name          = name;
        d.shape         = {count};
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems(count, DType::Float32);
        for (int64_t i = 0; i < count; ++i)
        {
            hb.f32()[i] = fill;
        }
        g.initializers[id] = hb;
        return id;
    }

    TensorId addActivation(Graph &g, const std::string &name) {
        TensorDesc d;
        d.name = name;
        return g.addTensor(d);
    }

    // Total bytes of the initializer payloads still held by the graph.
    size_t initializerBytes(const Graph &g) {
        size_t total = 0;
        for (const auto &kv: g.initializers)
        {
            total += kv.second.bytes.size();
        }
        return total;
    }

    // Backend assignment for a single-node graph: the node runs on the GPU.
    const std::vector<bool> kSoleNodeOnGpu {false};
    // Backend assignment for a single-node graph: the node fell back to the CPU.
    const std::vector<bool> kSoleNodeOnCpu {true};
    // No assignment at all, as an incompletely planned bucket would carry.
    const std::vector<bool> kNoNodeAssignment {};

    // Erase every reclaimable initializer, mirroring what Session::buildBucket does once the bucket's
    // segments are built.
    void applyReclaim(Graph &g, const std::vector<bool> &nodeRunsOnCpu) {
        for (TensorId id: reclaimableInitializers(g, nodeRunsOnCpu))
        {
            g.initializers.erase(id);
        }
    }

    // A fused-depthwise-plus-pointwise node, the shape a mobile CNN backbone fuses into: the depthwise
    // weight/bias and the pointwise weight/bias are all initializers of ONE node whose type is neither
    // Conv nor MatMul nor Gemm nor ConvGemm.
    Graph makeFusedDwPwGraph(int64_t depthwiseElems, int64_t pointwiseElems) {
        Graph    g;
        TensorId x        = addActivation(g, "x");
        g.desc(x).isInput = true;
        g.inputs.push_back(x);
        TensorId dwWeight  = addInitializer(g, "dw.weight", depthwiseElems, 0.5f);
        TensorId dwBias    = addInitializer(g, "dw.bias", 8, 0.25f);
        TensorId pwWeight  = addInitializer(g, "pw.weight", pointwiseElems, 1.5f);
        TensorId pwBias    = addInitializer(g, "pw.bias", 8, 0.125f);
        TensorId y         = addActivation(g, "y");
        g.desc(y).isOutput = true;

        Node n;
        n.type    = OpType::FusedDwPw;
        n.name    = "backbone.block0";
        n.inputs  = {x, dwWeight, dwBias, pwWeight, pwBias};
        n.outputs = {y};
        g.nodes.push_back(n);
        g.outputs = {y};
        return g;
    }

} // namespace

// The reclaim covers a fused node's weights. FusedDwPw carries the bulk of a fused mobile CNN's
// weight bytes and its type is not one of the four the reclaim used to enumerate, so an op-type
// allowlist leaves every one of those payloads resident for the session lifetime.
TEST(PlanRetention, FusedDepthwisePointwiseWeightsAreReclaimed) {
    Graph        g           = makeFusedDwPwGraph(/*depthwiseElems=*/8 * 3 * 3, /*pointwiseElems=*/8 * 16);
    const size_t beforeBytes = initializerBytes(g);
    ASSERT_GT(beforeBytes, 0u);

    applyReclaim(g, kSoleNodeOnGpu);

    EXPECT_EQ(initializerBytes(g), 0u) << "a GPU-assigned fused node's weights must not stay resident";
    EXPECT_TRUE(g.initializers.empty());
}

// Same for the other weighted op types an allowlist has to remember: FusedSE and ConvTranspose. The
// rule is derived from what is still NEEDED, so no op type has to be listed for its weights to be
// reclaimed.
TEST(PlanRetention, FusedSeAndConvTransposeWeightsAreReclaimed) {
    for (OpType type: {OpType::FusedSE, OpType::ConvTranspose})
    {
        Graph g         = makeFusedDwPwGraph(/*depthwiseElems=*/4 * 3 * 3, /*pointwiseElems=*/4 * 8);
        g.nodes[0].type = type;
        applyReclaim(g, kSoleNodeOnGpu);
        EXPECT_TRUE(g.initializers.empty()) << "op type " << opTypeName(type);
    }
}

// A CPU-assigned node's operands stay resolvable: the CPU op reads its payload from the runtime pool,
// and the graph entry keeps isInitializer() true for the boundary and dump paths that key off it.
TEST(PlanRetention, CpuAssignedNodeKeepsItsOperands) {
    Graph        g           = makeFusedDwPwGraph(/*depthwiseElems=*/8 * 3 * 3, /*pointwiseElems=*/8 * 16);
    const size_t beforeBytes = initializerBytes(g);

    applyReclaim(g, kSoleNodeOnCpu);

    EXPECT_EQ(initializerBytes(g), beforeBytes);
    EXPECT_EQ(g.initializers.size(), 4u);
}

// A node vector shorter than the graph's node list reads as CPU-assigned for the missing tail, so an
// incomplete assignment can only ever keep too much, never free something still in use.
TEST(PlanRetention, MissingBackendAssignmentKeepsOperands) {
    Graph        g           = makeFusedDwPwGraph(/*depthwiseElems=*/4, /*pointwiseElems=*/4);
    const size_t beforeBytes = initializerBytes(g);

    applyReclaim(g, kNoNodeAssignment);

    EXPECT_EQ(initializerBytes(g), beforeBytes);
}

// A pointwise-chain epilogue operand (an input at or past pwCoreInputs()) uploads lazily while
// recording, so it stays resident even on a GPU-assigned node; the node's core weights still go.
TEST(PlanRetention, PointwiseEpilogueOperandStaysResident) {
    Graph    g     = makeFusedDwPwGraph(/*depthwiseElems=*/8 * 3 * 3, /*pointwiseElems=*/8 * 16);
    TensorId slope = addInitializer(g, "prelu.slope", 8, 0.01f);
    Node    &n     = g.nodes[0];
    n.inputs.push_back(slope);
    Attr steps;
    steps.kind             = Attr::Ints;
    steps.ints             = {1};
    n.attr.map["pw_steps"] = steps;
    Attr opbase;
    opbase.kind             = Attr::Int;
    opbase.i                = (int64_t) n.inputs.size() - 1;
    n.attr.map["pw_opbase"] = opbase;

    applyReclaim(g, kSoleNodeOnGpu);

    ASSERT_EQ(g.initializers.size(), 1u);
    EXPECT_EQ(g.initializers.begin()->first, slope);
}

// A fused residual/bias edge is referenced outside node.inputs, so it is kept for every node.
TEST(PlanRetention, FusedResidualAndBiasEdgesStayResident) {
    Graph    g           = makeFusedDwPwGraph(/*depthwiseElems=*/4, /*pointwiseElems=*/4);
    TensorId bias        = addInitializer(g, "fused.bias", 8, 3.f);
    g.nodes[0].fusedBias = bias;

    applyReclaim(g, kSoleNodeOnGpu);

    ASSERT_EQ(g.initializers.size(), 1u);
    EXPECT_EQ(g.initializers.begin()->first, bias);
}

// A constant that is itself a graph output is never reclaimed: run() reads it back.
TEST(PlanRetention, ConstantGraphOutputStaysResident) {
    Graph    g        = makeFusedDwPwGraph(/*depthwiseElems=*/4, /*pointwiseElems=*/4);
    TensorId constant = addInitializer(g, "const.out", 4, 7.f);
    g.outputs.push_back(constant);

    applyReclaim(g, kSoleNodeOnGpu);

    ASSERT_EQ(g.initializers.size(), 1u);
    EXPECT_EQ(g.initializers.begin()->first, constant);
}

// A model whose inputs are all statically shaped can never plan a second bucket: shape resolution
// fills only dimensions the model left dynamic, so its pristine imported graph is dead weight once
// the default bucket is built.
TEST(PlanRetention, StaticInputGraphCannotPlanNewShapes) {
    Graph g                   = makeFusedDwPwGraph(/*depthwiseElems=*/4, /*pointwiseElems=*/4);
    g.desc(g.inputs[0]).shape = {1, 8, 16, 16};
    EXPECT_FALSE(importedGraphCanPlanNewShapes(g));
}

// One dynamic axis anywhere in the inputs is enough: a declared shape resolves it, so the pristine
// graph must stay.
TEST(PlanRetention, DynamicInputAxisKeepsPristineGraphUseful) {
    Graph g                   = makeFusedDwPwGraph(/*depthwiseElems=*/4, /*pointwiseElems=*/4);
    g.desc(g.inputs[0]).shape = {-1, 8, 16, 16};
    EXPECT_TRUE(importedGraphCanPlanNewShapes(g));

    g.desc(g.inputs[0]).shape = {1, 8, -1, 16};
    EXPECT_TRUE(importedGraphCanPlanNewShapes(g));
}

namespace {

    // A fully static 1x1-Conv graph: input "data" [1,C,H,W] -> Conv -> "y". Nothing about it is
    // dynamic, so its plan is fixed at one bucket.
    Graph makeStaticConv(int64_t C, int64_t H, int64_t W) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "data";
        xi.shape   = {1, C, H, W};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs.push_back(x);

        TensorDesc wi;
        wi.name          = "w";
        wi.shape         = {C, C, 1, 1};
        wi.isInitializer = true;
        TensorId   w     = g.addTensor(wi);
        HostBuffer hb;
        hb.resizeElems(C * C, DType::Float32);
        for (int64_t oc = 0; oc < C; ++oc)
        {
            hb.f32()[oc * C + oc] = 2.f;
        }
        g.initializers[w] = hb;

        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);

        Node n;
        n.type                     = OpType::Conv;
        n.name                     = "conv";
        n.inputs                   = {x, w};
        n.outputs                  = {y};
        n.attr.map["strides"]      = ints({1, 1});
        n.attr.map["pads"]         = ints({0, 0, 0, 0});
        n.attr.map["dilations"]    = ints({1, 1});
        n.attr.map["kernel_shape"] = ints({1, 1});
        g.nodes.push_back(n);
        g.outputs = {y};
        return g;
    }

} // namespace

// Releasing the pristine graph on a static-input model keeps prepareShapes() correct: the declared
// shape resolves to the bucket that already exists, so the call is the same idempotent no-op it is
// when the pristine graph is retained.
TEST(PlanRetention, PrepareShapesStaysIdempotentWithoutPristineGraph) {
    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto s      = Session::create(makeStaticConv(4, 3, 3), cfg);
    ASSERT_TRUE(s);
    ASSERT_EQ(s->bucketCount(), 1u);

    EXPECT_EQ(s->prepareShapes({{"data", {1, 4, 3, 3}}}), Status::Ok);
    EXPECT_EQ(s->bucketCount(), 1u);

    // A rank the model does not declare is still rejected, exactly as the pass pipeline rejects it.
    EXPECT_EQ(s->prepareShapes({{"data", {1, 4, 3}}}), Status::InvalidArgument);
    EXPECT_EQ(s->bucketCount(), 1u);

    // The model still runs after the release, and the constant weight it needed is the one the
    // reclaim kept: y = 2*x on the channel diagonal.
    IOTensor in;
    in.name  = "data";
    in.shape = {1, 4, 3, 3};
    in.dtype = DType::Float32;
    in.data.assign(4 * 3 * 3 * sizeof(float), 0);
    for (size_t i = 0; i < 4 * 3 * 3; ++i)
    {
        in.f32()[i] = (float) i;
    }
    std::vector<IOTensor> outputs;
    ASSERT_EQ(s->run({in}, outputs), Status::Ok);
    ASSERT_EQ(outputs.size(), 1u);
    const std::vector<float> y = outputs[0].toFloat32();
    ASSERT_EQ(y.size(), 4u * 3 * 3);
    for (size_t i = 0; i < y.size(); ++i)
    {
        EXPECT_FLOAT_EQ(y[i], 2.f * (float) i) << "element " << i;
    }
}

// A standalone FusedPointwise node names its operands in the plan's step words, not as a
// contiguous tail past pw_opbase: buildPwPlan reads each step's srcA/srcB/srcC field, where
// operand i is encoded as kPwRefOp0 - i indexing node.inputs. Reclaiming such an operand leaves
// pwOperandBuf with no host payload AND no activation buffer on the first record, which is a null
// dereference rather than a diagnosable error.
TEST(PlanRetention, StandalonePointwiseOperandNamedByAStepStaysResident) {
    Graph    g       = makeFusedDwPwGraph(/*depthwiseElems=*/8 * 3 * 3, /*pointwiseElems=*/8 * 16);
    TensorId operand = addInitializer(g, "pw.mul.operand", 8, 0.5f);
    Node    &n       = g.nodes[0];
    n.inputs.push_back(operand);
    const int64_t operandIndex = (int64_t) n.inputs.size() - 1;

    // One step: kind, code, srcA = the accumulator, srcB = our operand, srcC unused, dst, bcast,
    // bcastSrc. The operand is named ONLY here -- pw_opbase deliberately points past every input,
    // which is what a standalone unit whose plan references an early input looks like.
    Attr steps;
    steps.kind             = Attr::Ints;
    steps.ints             = {0, 0, kPwRefAcc, kPwRefOp0 - operandIndex, kPwRefNone, kPwRefNone, 0, 0};
    n.attr.map["pw_steps"] = steps;
    Attr opbase;
    opbase.kind             = Attr::Int;
    opbase.i                = (int64_t) n.inputs.size();
    n.attr.map["pw_opbase"] = opbase;

    applyReclaim(g, kSoleNodeOnGpu);

    ASSERT_EQ(g.initializers.count(operand), 1u) << "an operand the plan names by step reference must keep its host payload";
}

// A step field that names the accumulator, the entry value, a register or nothing must not be
// mistaken for an operand index and drag an unrelated input into the kept set.
TEST(PlanRetention, NonOperandStepReferencesKeepNothing) {
    Graph g = makeFusedDwPwGraph(/*depthwiseElems=*/8 * 3 * 3, /*pointwiseElems=*/8 * 16);
    Node &n = g.nodes[0];
    Attr  steps;
    steps.kind             = Attr::Ints;
    steps.ints             = {0, 0, kPwRefAcc, kPwRefEntry, kPwRefReg0, kPwRefNone, 0, 0};
    n.attr.map["pw_steps"] = steps;
    Attr opbase;
    opbase.kind             = Attr::Int;
    opbase.i                = (int64_t) n.inputs.size();
    n.attr.map["pw_opbase"] = opbase;

    applyReclaim(g, kSoleNodeOnGpu);

    EXPECT_TRUE(g.initializers.empty()) << "no step field named an operand, so both conv weights stay reclaimable";
}
