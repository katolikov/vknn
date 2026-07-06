// Locks the support-report oracle to the engine: vkSupportSurvey (the code path behind
// `vknn_compile --support-report` and VulkanBackend::supportsNode) must assign a known graph's
// nodes to the expected backends with the expected refusal reasons. A change in gate behavior or
// reason vocabulary fails here before it silently shifts the scratchpad oracle snapshots.
#include "core/vk_gates.h"
#include "vknn/graph.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    Attr intAttr(int64_t v) {
        Attr a;
        a.kind = Attr::Int;
        a.i    = v;
        return a;
    }
    Attr strAttr(std::string s) {
        Attr a;
        a.kind = Attr::String;
        a.str  = std::move(s);
        return a;
    }

    TensorId tensor(Graph &g, const std::string &name, Shape shape, DType dt = DType::Float32) {
        TensorDesc d;
        d.name  = name;
        d.shape = std::move(shape);
        d.dtype = dt;
        return g.addTensor(d);
    }

    TensorId initializer(Graph &g, const std::string &name, Shape shape) {
        TensorId   id = tensor(g, name, std::move(shape));
        HostBuffer hb;
        hb.resizeElems((size_t) numElements(g.desc(id).shape), DType::Float32);
        g.initializers[id] = hb;
        return id;
    }

    void addNode(Graph &g, OpType type, const std::string &name, std::vector<TensorId> in, std::vector<TensorId> out, Attributes attr = {}) {
        Node n;
        n.type    = type;
        n.name    = name;
        n.inputs  = std::move(in);
        n.outputs = std::move(out);
        n.attr    = std::move(attr);
        g.nodes.push_back(std::move(n));
    }

} // namespace

TEST(SupportReport, SurveyMatchesExpectedAssignment) {
    Graph g;

    // dense conv: GPU (group == 1 path)
    TensorId x  = tensor(g, "x", {1, 8, 8, 8});
    TensorId w0 = initializer(g, "w0", {8, 8, 3, 3});
    TensorId c0 = tensor(g, "conv_out", {1, 8, 8, 8});
    addNode(g, OpType::Conv, "conv_dense", {x, w0}, {c0});

    // grouped (non-depthwise) conv: CPU with the grouped-conv gate reason
    TensorId w1 = initializer(g, "w1", {8, 2, 3, 3});
    TensorId c1 = tensor(g, "gconv_out", {1, 8, 8, 8});
    Attributes gattr;
    gattr.map["group"] = intAttr(4);
    addNode(g, OpType::Conv, "conv_grouped", {c0, w1}, {c1}, gattr);

    // cubic GridSample: GPU (the mode list includes cubic/bicubic)
    TensorId grid = tensor(g, "grid", {1, 8, 8, 2});
    TensorId gs   = tensor(g, "gs_out", {1, 8, 8, 8});
    Attributes gsattr;
    gsattr.map["mode"] = strAttr("cubic");
    addNode(g, OpType::GridSample, "gridsample_cubic", {c1, grid}, {gs}, gsattr);

    // cast whose input is an int64 shape/index tensor to float: GPU (the int64 lanes decode to
    // compute-precision float at the pack boundary; `to`=1 is FLOAT).
    TensorId   idx = tensor(g, "idx", {4}, DType::Int64);
    TensorId   cast = tensor(g, "cast_out", {4});
    Attributes castattr;
    castattr.map["to"] = intAttr(1);
    addNode(g, OpType::Cast, "cast_i64", {idx}, {cast}, castattr);

    // TopK: CPU-only kernel
    TensorId tv = tensor(g, "topk_vals", {1, 8, 8, 4});
    TensorId ti = tensor(g, "topk_idx", {1, 8, 8, 4}, DType::Int64);
    addNode(g, OpType::TopK, "topk", {gs}, {tv, ti});

    // Relu: GPU (registered, ungated)
    TensorId r = tensor(g, "relu_out", {1, 8, 8, 8});
    addNode(g, OpType::Relu, "relu", {gs}, {r});

    // Dropout: erased at import; a survivor has no kernel in either backend
    TensorId d = tensor(g, "drop_out", {1, 8, 8, 8});
    addNode(g, OpType::Dropout, "dropout_kept", {r}, {d});

    std::vector<NodeSupport> rows = vkSupportSurvey(g);
    ASSERT_EQ(rows.size(), 7u);

    EXPECT_EQ(rows[0].node, "conv_dense");
    EXPECT_EQ(rows[0].backend, "vulkan");
    EXPECT_TRUE(rows[0].reason.empty());

    EXPECT_EQ(rows[1].node, "conv_grouped");
    EXPECT_EQ(rows[1].backend, "cpu");
    EXPECT_EQ(rows[1].reason, "Conv: grouped conv is GPU-supported only as pure depthwise");

    EXPECT_EQ(rows[2].node, "gridsample_cubic");
    EXPECT_EQ(rows[2].backend, "vulkan");
    EXPECT_TRUE(rows[2].reason.empty());

    EXPECT_EQ(rows[3].node, "cast_i64");
    EXPECT_EQ(rows[3].backend, "vulkan");
    EXPECT_TRUE(rows[3].reason.empty());

    EXPECT_EQ(rows[4].node, "topk");
    EXPECT_EQ(rows[4].backend, "cpu");
    EXPECT_EQ(rows[4].reason, "no vulkan kernel registered");

    EXPECT_EQ(rows[5].node, "relu");
    EXPECT_EQ(rows[5].backend, "vulkan");
    EXPECT_TRUE(rows[5].reason.empty());

    EXPECT_EQ(rows[6].node, "dropout_kept");
    EXPECT_EQ(rows[6].backend, "none");
    EXPECT_EQ(rows[6].reason, "no kernel in any backend");
}

TEST(SupportReport, CastFromInt64TargetGate) {
    // An int64 input Cast runs on the GPU for the shape-arithmetic targets (FLOAT/FLOAT16/DOUBLE,
    // INT32, INT64): the int64 lanes decode to compute-precision float at the pack boundary, so
    // cast.comp reads them like any float operand. A narrow integer target (INT8 here) keeps the CPU
    // op, where the wider int range and saturation are exact. `to` is the ONNX TensorProto dtype.
    auto castNode = [&](Graph &g, const char *name, int64_t to, DType inDt) {
        TensorId   in  = tensor(g, std::string(name) + "_in", {4}, inDt);
        TensorId   out = tensor(g, std::string(name) + "_out", {4});
        Attributes a;
        a.map["to"] = intAttr(to);
        addNode(g, OpType::Cast, name, {in}, {out}, a);
    };
    Graph g;
    castNode(g, "cast_i64_to_float", 1, DType::Int64); // FLOAT  -> GPU
    castNode(g, "cast_i64_to_int32", 6, DType::Int64); // INT32  -> GPU
    castNode(g, "cast_i64_to_int64", 7, DType::Int64); // INT64  -> GPU
    castNode(g, "cast_i64_to_int8", 3, DType::Int64);  // INT8   -> CPU (narrow target)

    std::vector<NodeSupport> rows = vkSupportSurvey(g);
    ASSERT_EQ(rows.size(), 4u);
    EXPECT_EQ(rows[0].backend, "vulkan");
    EXPECT_TRUE(rows[0].reason.empty());
    EXPECT_EQ(rows[1].backend, "vulkan");
    EXPECT_EQ(rows[2].backend, "vulkan");
    EXPECT_EQ(rows[3].backend, "cpu");
    EXPECT_EQ(rows[3].reason, "Cast: int64 input to a narrow integer target");
}

TEST(SupportReport, QuantizeDequantizeRunOnGpuWithConstScale) {
    // Quantize/DequantizeLinear are the graph-boundary quant hops the import-time dequantize pass
    // keeps as nodes (a genuine int-graph-boundary dequant, e.g. a quantized embedding-table lookup).
    // With a constant scale/zero_point (uploaded flat in prepare) the flat GPU kernel runs them; the
    // survey assigns the Vulkan backend and no refusal reason.
    Graph    g;
    TensorId x  = tensor(g, "x", {1, 8, 8, 8});
    TensorId xs = initializer(g, "x_s", {1});
    TensorId zp = initializer(g, "x_zp", {1});
    TensorId q  = tensor(g, "q_out", {1, 8, 8, 8}, DType::UInt8);
    addNode(g, OpType::QuantizeLinear, "quant", {x, xs, zp}, {q});
    TensorId dq = tensor(g, "dq_out", {1, 8, 8, 8});
    addNode(g, OpType::DequantizeLinear, "dequant", {q, xs, zp}, {dq});

    std::vector<NodeSupport> rows = vkSupportSurvey(g);
    ASSERT_EQ(rows.size(), 2u);
    for (const NodeSupport &row: rows)
    {
        EXPECT_EQ(row.backend, "vulkan") << row.node;
        EXPECT_TRUE(row.reason.empty()) << row.node;
    }
}

TEST(SupportReport, DequantizeLinearRuntimeScaleFallsBackToCpu) {
    // A runtime (non-initializer) scale has no constant to bake in prepare, so the gate declines the
    // GPU and the exact CPU op runs the dequant. The refusal reason is stable and op-named.
    Graph    g;
    TensorId x  = tensor(g, "x", {1, 8, 8, 8}, DType::UInt8);
    TensorId xs = tensor(g, "x_s_runtime", {1}); // runtime tensor, not an initializer
    TensorId dq = tensor(g, "dq_out", {1, 8, 8, 8});
    addNode(g, OpType::DequantizeLinear, "dequant_runtime_scale", {x, xs}, {dq});

    std::vector<NodeSupport> rows = vkSupportSurvey(g);
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].backend, "cpu");
    EXPECT_EQ(rows[0].reason, "DequantizeLinear: runtime scale input");
}

TEST(SupportReport, FusedQLinearOpsReportNoKernel) {
    // The fused QLinear-family ops (QLinearConv/QLinearMatMul/QGemm/QLinearAdd/
    // QLinearGlobalAveragePool) are recognized at import but have no kernel in either backend: the
    // dequantize pass lowers them to plain float ops, so any survivor reports backend "none",
    // exactly like an Unknown op -- never a claimed Vulkan assignment.
    Graph    g;
    TensorId x = tensor(g, "x", {1, 8, 8, 8});
    TensorId w = initializer(g, "w", {8, 8, 1, 1});
    for (OpType t: {OpType::QLinearConv, OpType::QLinearMatMul, OpType::QGemm, OpType::QLinearAdd, OpType::QLinearGlobalAveragePool})
    {
        TensorId y = tensor(g, std::string("y_") + opTypeName(t), {1, 8, 8, 8});
        addNode(g, t, std::string("q_") + opTypeName(t), {x, w}, {y});
    }
    std::vector<NodeSupport> rows = vkSupportSurvey(g);
    ASSERT_EQ(rows.size(), 5u);
    for (const NodeSupport &row: rows)
    {
        EXPECT_EQ(row.backend, "none") << row.node;
        EXPECT_EQ(row.reason, "no kernel in any backend") << row.node;
    }
}

TEST(SupportReport, GateReasonIsNullSafeAndStable) {
    Graph g;
    // A runtime-bounded Clip refuses with a stable reason string; a null whyNot is legal.
    TensorId x   = tensor(g, "x", {1, 4, 4, 4});
    TensorId lo  = tensor(g, "lo", {1}); // runtime tensor, not an initializer
    TensorId out = tensor(g, "clip_out", {1, 4, 4, 4});
    addNode(g, OpType::Clip, "clip_runtime", {x, lo}, {out});

    EXPECT_FALSE(vkNodeGate(g, g.nodes[0], nullptr));
    std::string why;
    EXPECT_FALSE(vkNodeGate(g, g.nodes[0], &why));
    EXPECT_EQ(why, "Clip: runtime min/max input");

    // An accepting gate leaves whyNot untouched.
    Node relu;
    relu.type    = OpType::Relu;
    relu.inputs  = {x};
    relu.outputs = {out};
    why          = "sentinel";
    EXPECT_TRUE(vkNodeGate(g, relu, &why));
    EXPECT_EQ(why, "sentinel");
}
