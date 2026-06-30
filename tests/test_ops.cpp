// vknn operator unit tests (host, CPU backend). One self-contained graph per op, checked against a
// reference (hand-computed, or an onnxruntime golden noted inline). The CPU op is the correctness
// oracle the Vulkan path is diffed against on-device, so a regression here surfaces before the GPU.
//
// ConvTranspose covers the explicit-pad, auto_pad SAME, and output_shape paths -- SAME and
// output_shape yield the same output size here but different values, so a regression in either
// attribute is caught.
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    Attr ints(std::vector<int64_t> v) {
        Attr a;
        a.kind = Attr::Ints;
        a.ints = std::move(v);
        return a;
    }
    Attr str(std::string s) {
        Attr a;
        a.kind = Attr::String;
        a.str  = std::move(s);
        return a;
    }

    struct Init {
        std::vector<int64_t> shape;
        std::vector<float>   data;
    };
    struct OpOut {
        std::vector<float>   data;
        std::vector<int64_t> shape;
    };

    // Build a single-op graph: one float input "x" + N constant initializers, run on CPU, return
    // the output values and shape. subOp carries the UnaryType/BinaryType code for kUnary/kBinary.
    OpOut runOp(OpType type, int subOp, const Attributes &attr, const std::vector<int64_t> &xshape, const std::vector<float> &xdata, const std::vector<Init> &inits) {
        Graph      g;
        TensorDesc xi;
        xi.name                 = "x";
        xi.shape                = xshape;
        xi.isInput              = true;
        TensorId              x = g.addTensor(xi);
        std::vector<TensorId> ids {x};
        g.inputs.push_back(x);
        for (size_t k = 0; k < inits.size(); ++k)
        {
            TensorDesc d;
            d.name          = "i" + std::to_string(k);
            d.shape         = inits[k].shape;
            d.isInitializer = true;
            TensorId   id   = g.addTensor(d);
            HostBuffer hb;
            hb.resizeElems(inits[k].data.size(), DType::Float32);
            for (size_t i = 0; i < inits[k].data.size(); ++i)
            {
                hb.f32()[i] = inits[k].data[i];
            }
            g.initializers[id] = hb;
            ids.push_back(id);
        }
        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     n;
        n.type    = type;
        n.name    = "op";
        n.subOp   = subOp;
        n.inputs  = ids;
        n.outputs = {y};
        n.attr    = attr;
        g.nodes.push_back(n);
        g.outputs = {y};

        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = "x";
        in.shape = xshape;
        in.dtype = DType::Float32;
        in.data.resize(xdata.size() * 4);
        for (size_t i = 0; i < xdata.size(); ++i)
        {
            reinterpret_cast<float *>(in.data.data())[i] = xdata[i];
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        EXPECT_FALSE(outs.empty());
        if (outs.empty())
        {
            return {};
        }
        const float *o = outs[0].f32();
        return {std::vector<float>(o, o + numElements(outs[0].shape)), outs[0].shape};
    }

    void expectNear(const std::vector<float> &got, const std::vector<float> &ref, float tol = 1e-4f) {
        ASSERT_EQ(got.size(), ref.size());
        for (size_t i = 0; i < ref.size(); ++i)
        {
            EXPECT_NEAR(got[i], ref[i], tol) << "i=" << i;
        }
    }

} // namespace

// --- Conv (1x1, identity*2 weight + bias) feeding Relu: a two-op chain. ---
TEST(Ops, Conv1x1Relu) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 2, 1, 1};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorDesc wi;
    wi.name          = "w";
    wi.shape         = {2, 2, 1, 1};
    wi.isInitializer = true;
    TensorId   w     = g.addTensor(wi);
    HostBuffer wb;
    wb.resizeElems(4, DType::Float32);
    wb.f32()[0]       = 2;
    wb.f32()[1]       = 0;
    wb.f32()[2]       = 0;
    wb.f32()[3]       = 2;
    g.initializers[w] = wb;
    TensorDesc bi;
    bi.name          = "b";
    bi.shape         = {2};
    bi.isInitializer = true;
    TensorId   b     = g.addTensor(bi);
    HostBuffer bb;
    bb.resizeElems(2, DType::Float32);
    bb.f32()[0]       = -3;
    bb.f32()[1]       = 0;
    g.initializers[b] = bb;
    TensorDesc yo;
    yo.name    = "y";
    TensorId y = g.addTensor(yo);
    Node     conv;
    conv.type    = OpType::Conv;
    conv.name    = "conv";
    conv.inputs  = {x, w, b};
    conv.outputs = {y};
    g.nodes.push_back(conv);
    TensorDesc y2o;
    y2o.name     = "y2";
    y2o.isOutput = true;
    TensorId y2  = g.addTensor(y2o);
    Node     relu;
    relu.type    = OpType::Relu;
    relu.name    = "relu";
    relu.inputs  = {y};
    relu.outputs = {y2};
    g.nodes.push_back(relu);
    g.outputs = {y2};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);
    IOTensor in;
    in.name  = "x";
    in.shape = {1, 2, 1, 1};
    in.dtype = DType::Float32;
    in.data.resize(2 * 4);
    reinterpret_cast<float *>(in.data.data())[0] = 1.0f; // 2*1-3 = -1 -> relu 0
    reinterpret_cast<float *>(in.data.data())[1] = 5.0f; // 2*5+0 = 10 -> relu 10
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_FALSE(outs.empty());
    EXPECT_NEAR(outs[0].f32()[0], 0.0f, 1e-5);
    EXPECT_NEAR(outs[0].f32()[1], 10.0f, 1e-5);
}

// --- Unary family: Sigmoid + HardSwish. ---
TEST(Ops, UnarySigmoidHardSwish) {
    float vals[4] = {-2.f, -0.5f, 0.5f, 3.f};
    for (int sub: {(int) UnaryType::Sigmoid, (int) UnaryType::HardSwish})
    {
        auto               out = runOp(OpType::Unary, sub, {}, {1, 4}, {vals[0], vals[1], vals[2], vals[3]}, {});
        std::vector<float> ref(4);
        for (int i = 0; i < 4; ++i)
        {
            ref[i] = sub == (int) UnaryType::Sigmoid ? 1.f / (1.f + std::exp(-vals[i])) : vals[i] * std::min(std::max(vals[i] + 3.f, 0.f), 6.f) / 6.f;
        }
        expectNear(out.data, ref, 1e-5f);
    }
}

// --- Binary Mul with a broadcast scalar. ---
TEST(Ops, BinaryMulBroadcast) {
    auto out = runOp(OpType::Binary, (int) BinaryType::Mul, {}, {1, 3}, {1, 2, 3}, {{{1}, {3.f}}});
    expectNear(out.data, {3, 6, 9}, 1e-5f);
}

// --- Add with a per-channel broadcast bias. ---
TEST(Ops, AddBroadcast) {
    auto out = runOp(OpType::Add, 0, {}, {1, 3}, {0, 1, 2}, {{{1, 3}, {10, 20, 30}}});
    expectNear(out.data, {10, 21, 32}, 1e-5f);
}

// --- GlobalAveragePool over HxW. ---
TEST(Ops, GlobalAveragePool) {
    auto out = runOp(OpType::GlobalAvgPool, 0, {}, {1, 1, 2, 2}, {1, 2, 3, 4}, {});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 1, 1, 1}));
    expectNear(out.data, {2.5f}, 1e-5f);
}

// --- MatMul A[2,3] @ B[3,2] (B constant). ---
TEST(Ops, MatMul) {
    auto out = runOp(OpType::MatMul, 0, {}, {2, 3}, {1, 2, 3, 4, 5, 6}, {{{3, 2}, {1, 0, 0, 1, 1, 1}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {2, 2}));
    expectNear(out.data, {4, 5, 10, 11}, 1e-4f);
}

// --- ConvTranspose stride 2, kernel 2, all-ones weight: each pixel tiles a 2x2 block. ---
TEST(Ops, ConvTransposeBasicStride2) {
    Attributes attr;
    attr.map["strides"] = ints({2, 2});
    auto out            = runOp(OpType::ConvTranspose, 0, attr, {1, 1, 2, 2}, {1, 2, 3, 4}, {{{1, 1, 2, 2}, {1, 1, 1, 1}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 1, 4, 4}));
    expectNear(out.data, {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4});
}

// --- ConvTranspose auto_pad SAME_UPPER, stride 2, kernel 3: output size is in*stride = 4 (not the
// explicit-pad formula's 5); pad_begin = 0. ORT golden. ---
TEST(Ops, ConvTransposeAutoPadSameUpper) {
    Attributes attr;
    attr.map["strides"]  = ints({2, 2});
    attr.map["auto_pad"] = str("SAME_UPPER");
    auto out             = runOp(OpType::ConvTranspose, 0, attr, {1, 1, 2, 2}, {1, 2, 3, 4}, {{{1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 1, 4, 4}));
    expectNear(out.data, {1, 2, 5, 4, 4, 5, 14, 10, 10, 14, 36, 24, 12, 15, 34, 20});
}

// --- ConvTranspose output_shape [4,4], stride 2, kernel 3: same size as SAME_UPPER but pad_begin = 1
// (default split puts the larger half at the begin), so the values differ. ORT golden. ---
TEST(Ops, ConvTransposeOutputShape) {
    Attributes attr;
    attr.map["strides"]      = ints({2, 2});
    attr.map["output_shape"] = ints({4, 4});
    auto out                 = runOp(OpType::ConvTranspose, 0, attr, {1, 1, 2, 2}, {1, 2, 3, 4}, {{{1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 1, 4, 4}));
    expectNear(out.data, {5, 14, 10, 12, 14, 36, 24, 30, 15, 34, 20, 24, 24, 55, 32, 36});
}

// --- ConvTranspose with a per-output-channel bias (Cin=1, Cout=2). ORT golden. ---
TEST(Ops, ConvTransposeBias) {
    Attributes attr;
    attr.map["strides"] = ints({2, 2});
    // weight [1,2,2,2]: channel 0 = [[1,0],[0,1]], channel 1 = [[1,1],[1,1]]; bias [2] = {10, -5}.
    auto out = runOp(OpType::ConvTranspose, 0, attr, {1, 1, 2, 2}, {1, 2, 3, 4}, {{{1, 2, 2, 2}, {1, 0, 0, 1, 1, 1, 1, 1}}, {{2}, {10, -5}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 2, 4, 4}));
    expectNear(out.data, {11, 10, 12, 10, 10, 11, 10, 12, 13, 10, 14, 10, 10, 13, 10, 14, -4, -4, -3, -3, -4, -4, -3, -3, -2, -2, -1, -1, -2, -2, -1, -1});
}

// --- Greater vs a scalar: strict >, ties are 0. ---
TEST(Ops, GreaterScalar) {
    auto out = runOp(OpType::Greater, 0, {}, {2, 3}, {1, 2, 3, 4, 5, 6}, {{{1}, {3.f}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {2, 3}));
    EXPECT_EQ(out.data, (std::vector<float> {0, 0, 0, 1, 1, 1}));
}

// --- GreaterOrEqual vs a scalar: ties are 1 (the only difference from Greater on this input). ---
TEST(Ops, GreaterEqualScalarTies) {
    auto out = runOp(OpType::GreaterEqual, 0, {}, {2, 3}, {1, 2, 3, 4, 5, 6}, {{{1}, {3.f}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {2, 3}));
    EXPECT_EQ(out.data, (std::vector<float> {0, 0, 1, 1, 1, 1}));
}

// --- Greater with NumPy broadcasting: [2,3] vs a [3] row. ---
TEST(Ops, GreaterBroadcastRow) {
    auto out = runOp(OpType::Greater, 0, {}, {2, 3}, {1, 5, 0, 3, 4, 2}, {{{3}, {2, 4, 1}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {2, 3}));
    // row0 {1,5,0} vs {2,4,1} -> {0,1,0}; row1 {3,4,2} vs {2,4,1} -> {1,0,1}
    EXPECT_EQ(out.data, (std::vector<float> {0, 1, 0, 1, 0, 1}));
}

// --- Greater with rank-4 broadcasting: [1,2,2,2] vs a per-channel [1,2,1,1] threshold. ---
TEST(Ops, GreaterBroadcastPerChannel) {
    auto out = runOp(OpType::Greater, 0, {}, {1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {{{1, 2, 1, 1}, {2.5f, 6.5f}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 2, 2, 2}));
    // ch0 {1,2,3,4} > 2.5 -> {0,0,1,1}; ch1 {5,6,7,8} > 6.5 -> {0,0,1,1}
    EXPECT_EQ(out.data, (std::vector<float> {0, 0, 1, 1, 0, 0, 1, 1}));
}
