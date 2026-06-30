// vknn unit tests (host): dtype/fp16, config JSON, graph passes, layout packing math, and the
// ergonomic Session API. Operator correctness lives in test_ops.cpp; Vulkan correctness is
// validated on-device (see scripts).
#include "vknn/config.h"
#include "vknn/dtype.h"
#include "vknn/graph.h"
#include "vknn/model.h"
#include "vknn/session.h"
#include "vknn/tensor_format.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace vknn;

TEST(DType, HalfRoundTrip) {
    for (float v: {0.f, 1.f, -1.f, 0.5f, 6.f, 3.14159f, -2.71828f, 65504.f})
    {
        float r = halfToFloat(floatToHalf(v));
        EXPECT_NEAR(r, v, std::fabs(v) * 0.01f + 1e-3f) << "v=" << v;
    }
    EXPECT_EQ(dtypeSize(DType::Float32), 4u);
    EXPECT_EQ(dtypeSize(DType::Float16), 2u);
}

TEST(Config, JsonRoundTrip) {
    Config c;
    c.backend        = BackendKind::Vulkan;
    c.precision      = Precision::Low;
    c.maxSubmitNodes = 250;
    c.profile        = true;
    c.cacheMode      = CacheMode::Tune;
    c.setHint(Hint::Winograd, (int) Mode::Off);
    std::string js = c.toJson();
    Config      d  = Config::fromJsonString(js);
    EXPECT_EQ(d.backend, BackendKind::Vulkan);
    EXPECT_EQ(d.precision, Precision::Low);
    EXPECT_EQ(d.maxSubmitNodes, 250);
    EXPECT_TRUE(d.profile);
    EXPECT_EQ(d.cacheMode, CacheMode::Tune);
    EXPECT_FALSE(d.cachesWeights());
    EXPECT_TRUE(d.cachesTuning());
    EXPECT_EQ(precisionFromStr("normal"), Precision::Normal);
    EXPECT_EQ(precisionFromStr("high"), Precision::High);
    EXPECT_EQ(precisionFromStr("low"), Precision::Low);
    EXPECT_EQ(d.hint(Hint::Winograd, 0), (int) Mode::Off);
}

TEST(Config, ParseExplicit) {
    Config c = Config::fromJsonString(R"({"backend":"CPU","precision":"fp32","fallback":["VULKAN","CPU"],"cacheDir":"/tmp/x"})");
    EXPECT_EQ(c.backend, BackendKind::Cpu);
    EXPECT_EQ(c.precision, Precision::High);
    EXPECT_EQ(c.cacheDir, "/tmp/x");
    ASSERT_EQ(c.fallback.size(), 2u);
    EXPECT_EQ(c.fallback[0], BackendKind::Vulkan);
}

TEST(Layout, PackMath) {
    EXPECT_EQ(cBlocks(3), 1);
    EXPECT_EQ(cBlocks(4), 1);
    EXPECT_EQ(cBlocks(5), 2);
    EXPECT_EQ(cBlocks(1280), 320);
    NCHW s = NCHW::from({1, 32, 7, 7});
    EXPECT_EQ(s.n, 1);
    EXPECT_EQ(s.c, 32);
    EXPECT_EQ(s.h, 7);
}

// Humane Tensor API: construct, shape/size accessors, argmax.
TEST(Api, TensorHelpers) {
    Tensor t({1.f, 5.f, 2.f, 9.f, 3.f, 0.f}, {1, 6});
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.size(), 6);
    EXPECT_EQ(t.dim(1), 6);
    EXPECT_EQ(t.shapeString(), "1x6");
    EXPECT_EQ(t.argmax(), 3);
    EXPECT_NEAR(t.max(), 9.f, 1e-6);
    Tensor flat(std::vector<float> {1.f, 2.f, 3.f});
    EXPECT_EQ(flat.rank(), 1);
    EXPECT_EQ(flat.shapeString(), "3");
}

// Ergonomic API: infer()/inputInfo() — caller passes only data, metadata comes from the model.
TEST(Api, AutoShapesFromModel) {
    // input[1,2,1,1] -> Conv 1x1 (weight 2*I, bias {-3,0}) -> y
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
    yo.name     = "y";
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     conv;
    conv.type    = OpType::Conv;
    conv.name    = "conv";
    conv.inputs  = {x, w, b};
    conv.outputs = {y};
    g.nodes.push_back(conv);
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    // query metadata instead of hand-specifying it
    auto in = sess->inputInfo();
    ASSERT_EQ(in.size(), 1u);
    EXPECT_EQ(in[0].name, "x");
    EXPECT_EQ(in[0].elems, 2);
    EXPECT_EQ(sess->outputInfo()[0].elems, 2);
    // infer() with just data
    std::vector<float> out = sess->infer({1.0f, 5.0f}); // -> {2*1-3, 2*5} = {-1, 10}
    ASSERT_EQ(out.size(), 2u);
    EXPECT_NEAR(out[0], -1.0f, 1e-5);
    EXPECT_NEAR(out[1], 10.0f, 1e-5);
}
