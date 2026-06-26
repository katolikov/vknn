// vknn integration test: full MobileNetV2 on the CPU backend vs the onnxruntime golden.
// Skips gracefully if assets are not present (run scripts/get_golden.py first).
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>

#include "vknn/session.h"

using namespace vknn;

static std::vector<uint8_t> readFile(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return f ? std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>())
           : std::vector<uint8_t>();
}

TEST(Integration, MobileNetV2_CPU_vs_Golden) {
  const std::string model = "assets/mobilenetv2.onnx";
  auto inData = readFile("assets/input.bin");
  auto gold = readFile("assets/golden.bin");
  if (inData.empty() || gold.empty() || readFile(model).empty()) {
    GTEST_SKIP() << "assets missing (run scripts/get_golden.py)";
  }
  Config cfg;
  cfg.backend = BackendKind::kCpu;
  auto sess = Runtime::load(model, cfg);
  ASSERT_TRUE(sess);
  IOTensor in;
  in.name = "input";
  in.shape = {1, 3, 224, 224};
  in.dtype = DType::kFloat32;
  in.data = inData;
  std::vector<IOTensor> outs;
  ASSERT_EQ(sess->run({in}, outs), Status::kOk);
  ASSERT_FALSE(outs.empty());

  const float* y = outs[0].f32();
  const float* g = reinterpret_cast<const float*>(gold.data());
  int64_t n = numElements(outs[0].shape);
  ASSERT_EQ(n, (int64_t)(gold.size() / 4));
  double dot = 0, na = 0, nb = 0;
  int vTop = 0, gTop = 0;
  for (int64_t i = 0; i < n; ++i) {
    dot += (double)y[i] * g[i];
    na += (double)y[i] * y[i];
    nb += (double)g[i] * g[i];
    if (y[i] > y[vTop])
      vTop = (int)i;
    if (g[i] > g[gTop])
      gTop = (int)i;
  }
  double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
  EXPECT_EQ(vTop, gTop) << "top-1 mismatch";
  EXPECT_GE(cos, 0.999) << "cosine too low: " << cos;
}
