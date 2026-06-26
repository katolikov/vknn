// vxrt unit tests (host): dtype/fp16, config JSON, ONNX import, graph passes, CPU ops vs
// reference, layout packing math. Vulkan correctness is validated on-device (see scripts).
#include <gtest/gtest.h>
#include <cmath>
#include "vx/config.h"
#include "vx/dtype.h"
#include "vx/graph.h"
#include "vx/model.h"
#include "vx/session.h"
#include "vx/tensor_format.h"

using namespace vx;

TEST(DType, HalfRoundTrip) {
  for (float v : {0.f, 1.f, -1.f, 0.5f, 6.f, 3.14159f, -2.71828f, 65504.f}) {
    float r = halfToFloat(floatToHalf(v));
    EXPECT_NEAR(r, v, std::fabs(v) * 0.01f + 1e-3f) << "v=" << v;
  }
  EXPECT_EQ(dtypeSize(DType::kFloat32), 4u);
  EXPECT_EQ(dtypeSize(DType::kFloat16), 2u);
}

TEST(Config, JsonRoundTrip) {
  Config c;
  c.backend = BackendKind::kVulkan;
  c.precision = Precision::kFp16;
  c.cpuThreads = 8;
  c.profile = true;
  std::string js = c.toJson();
  Config d = Config::fromJsonString(js);
  EXPECT_EQ(d.backend, BackendKind::kVulkan);
  EXPECT_EQ(d.precision, Precision::kFp16);
  EXPECT_EQ(d.cpuThreads, 8);
  EXPECT_TRUE(d.profile);
}

TEST(Config, ParseExplicit) {
  Config c = Config::fromJsonString(
      R"({"backend":"CPU","precision":"fp32","fallback":["VULKAN","CPU"],"cacheDir":"/tmp/x"})");
  EXPECT_EQ(c.backend, BackendKind::kCpu);
  EXPECT_EQ(c.precision, Precision::kFp32);
  EXPECT_EQ(c.cacheDir, "/tmp/x");
  ASSERT_EQ(c.fallback.size(), 2u);
  EXPECT_EQ(c.fallback[0], BackendKind::kVulkan);
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

// Build a tiny graph: Conv 1x1 (identity weight) + Relu, run on CPU, check values.
TEST(CpuOps, Conv1x1ReluReference) {
  Graph g;
  // input [1,2,1,1]
  TensorDesc xi;
  xi.name = "x";
  xi.shape = {1, 2, 1, 1};
  xi.isInput = true;
  TensorId x = g.addTensor(xi);
  g.inputs.push_back(x);
  // weight [2,2,1,1] = identity*2 ; bias [2] = {-3, 0}
  TensorDesc wi;
  wi.name = "w";
  wi.shape = {2, 2, 1, 1};
  wi.isInitializer = true;
  TensorId w = g.addTensor(wi);
  HostBuffer wb;
  wb.resizeElems(4, DType::kFloat32);
  wb.f32()[0] = 2;
  wb.f32()[1] = 0;
  wb.f32()[2] = 0;
  wb.f32()[3] = 2;
  g.initializers[w] = wb;
  TensorDesc bi;
  bi.name = "b";
  bi.shape = {2};
  bi.isInitializer = true;
  TensorId b = g.addTensor(bi);
  HostBuffer bb;
  bb.resizeElems(2, DType::kFloat32);
  bb.f32()[0] = -3;
  bb.f32()[1] = 0;
  g.initializers[b] = bb;
  TensorDesc yo;
  yo.name = "y";
  yo.isOutput = true;
  TensorId y = g.addTensor(yo);

  Node conv;
  conv.type = OpType::kConv;
  conv.name = "conv";
  conv.inputs = {x, w, b};
  conv.outputs = {y};
  g.nodes.push_back(conv);
  Node relu;
  relu.type = OpType::kRelu;
  relu.name = "relu";
  // relu reads y -> y2
  TensorDesc y2o;
  y2o.name = "y2";
  TensorId y2 = g.addTensor(y2o);
  relu.inputs = {y};
  relu.outputs = {y2};
  g.nodes.push_back(relu);
  g.outputs = {y2};

  Config cfg;
  cfg.backend = BackendKind::kCpu;
  auto sess = Session::create(std::move(g), cfg);
  ASSERT_TRUE(sess);
  IOTensor in;
  in.name = "x";
  in.shape = {1, 2, 1, 1};
  in.dtype = DType::kFloat32;
  in.data.resize(2 * 4);
  reinterpret_cast<float*>(in.data.data())[0] = 1.0f;  // -> 2*1-3 = -1 -> relu 0
  reinterpret_cast<float*>(in.data.data())[1] = 5.0f;  // -> 2*5+0 = 10 -> relu 10
  std::vector<IOTensor> outs;
  ASSERT_EQ(sess->run({in}, outs), Status::kOk);
  ASSERT_FALSE(outs.empty());
  const float* o = outs[0].f32();
  EXPECT_NEAR(o[0], 0.0f, 1e-5);
  EXPECT_NEAR(o[1], 10.0f, 1e-5);
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
  Tensor flat(std::vector<float>{1.f, 2.f, 3.f});
  EXPECT_EQ(flat.rank(), 1);
  EXPECT_EQ(flat.shapeString(), "3");
}

// Ergonomic API: infer()/inputInfo() — caller passes only data, metadata comes from the model.
TEST(Api, AutoShapesFromModel) {
  // input[1,2,1,1] -> Conv 1x1 (weight 2*I, bias {-3,0}) -> y
  Graph g;
  TensorDesc xi; xi.name = "x"; xi.shape = {1, 2, 1, 1}; xi.isInput = true;
  TensorId x = g.addTensor(xi); g.inputs.push_back(x);
  TensorDesc wi; wi.name = "w"; wi.shape = {2, 2, 1, 1}; wi.isInitializer = true;
  TensorId w = g.addTensor(wi);
  HostBuffer wb; wb.resizeElems(4, DType::kFloat32);
  wb.f32()[0] = 2; wb.f32()[1] = 0; wb.f32()[2] = 0; wb.f32()[3] = 2; g.initializers[w] = wb;
  TensorDesc bi; bi.name = "b"; bi.shape = {2}; bi.isInitializer = true;
  TensorId b = g.addTensor(bi);
  HostBuffer bb; bb.resizeElems(2, DType::kFloat32); bb.f32()[0] = -3; bb.f32()[1] = 0;
  g.initializers[b] = bb;
  TensorDesc yo; yo.name = "y"; yo.isOutput = true; TensorId y = g.addTensor(yo);
  Node conv; conv.type = OpType::kConv; conv.name = "conv"; conv.inputs = {x, w, b};
  conv.outputs = {y}; g.nodes.push_back(conv); g.outputs = {y};

  Config cfg; cfg.backend = BackendKind::kCpu;
  auto sess = Session::create(std::move(g), cfg);
  // query metadata instead of hand-specifying it
  auto in = sess->inputInfo();
  ASSERT_EQ(in.size(), 1u);
  EXPECT_EQ(in[0].name, "x");
  EXPECT_EQ(in[0].elems, 2);
  EXPECT_EQ(sess->outputInfo()[0].elems, 2);
  // infer() with just data
  std::vector<float> out = sess->infer({1.0f, 5.0f});  // -> {2*1-3, 2*5} = {-1, 10}
  ASSERT_EQ(out.size(), 2u);
  EXPECT_NEAR(out[0], -1.0f, 1e-5);
  EXPECT_NEAR(out[1], 10.0f, 1e-5);
}

// Unary family: Sigmoid + HardSwish on CPU.
TEST(CpuOps, UnarySigmoidHardSwish) {
  for (int sub : {(int)UnaryType::kSigmoid, (int)UnaryType::kHardSwish}) {
    Graph g;
    TensorDesc xi; xi.name = "x"; xi.shape = {1, 4}; xi.isInput = true;
    TensorId x = g.addTensor(xi); g.inputs.push_back(x);
    TensorDesc yo; yo.name = "y"; yo.isOutput = true; TensorId y = g.addTensor(yo);
    Node u; u.type = OpType::kUnary; u.name = "u"; u.subOp = sub; u.inputs = {x}; u.outputs = {y};
    g.nodes.push_back(u); g.outputs = {y};
    Config cfg; cfg.backend = BackendKind::kCpu;
    auto sess = Session::create(std::move(g), cfg);
    IOTensor in; in.name = "x"; in.shape = {1, 4}; in.data.resize(4 * 4);
    float vals[4] = {-2.f, -0.5f, 0.5f, 3.f};
    for (int i = 0; i < 4; ++i) reinterpret_cast<float*>(in.data.data())[i] = vals[i];
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::kOk);
    const float* o = outs[0].f32();
    for (int i = 0; i < 4; ++i) {
      float e = sub == (int)UnaryType::kSigmoid
                    ? 1.f / (1.f + std::exp(-vals[i]))
                    : vals[i] * std::min(std::max(vals[i] + 3.f, 0.f), 6.f) / 6.f;
      EXPECT_NEAR(o[i], e, 1e-5) << "sub=" << sub << " i=" << i;
    }
  }
}

// Binary family: Mul (with broadcast) on CPU.
TEST(CpuOps, BinaryMul) {
  Graph g;
  TensorDesc ai; ai.name = "a"; ai.shape = {1, 3}; ai.isInput = true;
  TensorId a = g.addTensor(ai); g.inputs.push_back(a);
  TensorDesc bi; bi.name = "b"; bi.shape = {1}; bi.isInitializer = true;  // broadcast scalar
  TensorId b = g.addTensor(bi);
  HostBuffer bb; bb.resizeElems(1, DType::kFloat32); bb.f32()[0] = 3.f; g.initializers[b] = bb;
  TensorDesc co; co.name = "c"; co.isOutput = true; TensorId c = g.addTensor(co);
  Node m; m.type = OpType::kBinary; m.name = "mul"; m.subOp = (int)BinaryType::kMul; m.inputs = {a, b}; m.outputs = {c};
  g.nodes.push_back(m); g.outputs = {c};
  Config cfg; cfg.backend = BackendKind::kCpu;
  auto sess = Session::create(std::move(g), cfg);
  IOTensor in; in.name = "a"; in.shape = {1, 3}; in.data.resize(3 * 4);
  for (int i = 0; i < 3; ++i) reinterpret_cast<float*>(in.data.data())[i] = (float)(i + 1);
  std::vector<IOTensor> outs;
  ASSERT_EQ(sess->run({in}, outs), Status::kOk);
  const float* o = outs[0].f32();
  EXPECT_NEAR(o[0], 3.f, 1e-5);
  EXPECT_NEAR(o[1], 6.f, 1e-5);
  EXPECT_NEAR(o[2], 9.f, 1e-5);
}

// Add with broadcasting (bias-style) on CPU.
TEST(CpuOps, AddBroadcast) {
  Graph g;
  TensorDesc ai;
  ai.name = "a";
  ai.shape = {1, 3};
  ai.isInput = true;
  TensorId a = g.addTensor(ai);
  g.inputs.push_back(a);
  TensorDesc bi;
  bi.name = "b";
  bi.shape = {1, 3};
  bi.isInitializer = true;
  TensorId b = g.addTensor(bi);
  HostBuffer bb;
  bb.resizeElems(3, DType::kFloat32);
  bb.f32()[0] = 10;
  bb.f32()[1] = 20;
  bb.f32()[2] = 30;
  g.initializers[b] = bb;
  TensorDesc co;
  co.name = "c";
  co.isOutput = true;
  TensorId c = g.addTensor(co);
  Node add;
  add.type = OpType::kAdd;
  add.name = "add";
  add.inputs = {a, b};
  add.outputs = {c};
  g.nodes.push_back(add);
  g.outputs = {c};
  Config cfg;
  cfg.backend = BackendKind::kCpu;
  auto sess = Session::create(std::move(g), cfg);
  IOTensor in;
  in.name = "a";
  in.shape = {1, 3};
  in.data.resize(3 * 4);
  for (int i = 0; i < 3; ++i) reinterpret_cast<float*>(in.data.data())[i] = (float)i;
  std::vector<IOTensor> outs;
  ASSERT_EQ(sess->run({in}, outs), Status::kOk);
  const float* o = outs[0].f32();
  EXPECT_NEAR(o[0], 10, 1e-5);
  EXPECT_NEAR(o[1], 21, 1e-5);
  EXPECT_NEAR(o[2], 32, 1e-5);
}
