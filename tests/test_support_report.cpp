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

    // cast whose input is an int64 shape/index tensor: CPU (no int64 GPU buffers)
    TensorId idx  = tensor(g, "idx", {4}, DType::Int64);
    TensorId cast = tensor(g, "cast_out", {4});
    addNode(g, OpType::Cast, "cast_i64", {idx}, {cast});

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
    EXPECT_EQ(rows[3].backend, "cpu");
    EXPECT_EQ(rows[3].reason, "Cast: int64 input");

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
