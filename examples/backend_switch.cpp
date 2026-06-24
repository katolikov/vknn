// vx_backend_switch — select the backend via config (VULKAN | CPU | ENN) with no other change.
// Demonstrates the pluggable-backend path: ENN is consulted first then falls back (stub).
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include "vx/session.h"

using namespace vx;
static std::vector<uint8_t> readFile(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return f ? std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>())
           : std::vector<uint8_t>();
}
static const char* argval(int c, char** v, const char* k, const char* d) {
  for (int i = 1; i < c - 1; ++i) if (!strcmp(v[i], k)) return v[i + 1];
  return d;
}

static void runWith(const std::string& model, const std::vector<uint8_t>& inData, BackendKind be) {
  Config cfg;
  cfg.backend = be;
  cfg.fallback = {BackendKind::kVulkan, BackendKind::kCpu};  // ENN/others fall back through these
  cfg.precision = Precision::kFp16;
  printf("\n=== config.backend = %s ===\n", backendName(be));
  auto sess = Runtime::load(model, cfg);
  if (!sess) { printf("  load failed\n"); return; }
  IOTensor in; in.name = "input"; in.shape = {1, 3, 224, 224}; in.dtype = DType::kFloat32;
  in.data = inData;
  std::vector<IOTensor> outs;
  if (sess->run({in}, outs) != Status::kOk) { printf("  run failed\n"); return; }
  // backend usage histogram
  std::map<BackendKind, int> hist;
  for (BackendKind k : sess->nodeBackends()) hist[k]++;
  printf("  node routing:");
  for (auto& kv : hist) printf(" %s=%d", backendName(kv.first), kv.second);
  printf("\n");
  const float* y = outs[0].f32();
  int64_t n = numElements(outs[0].shape), top = 0;
  for (int64_t i = 1; i < n; ++i) if (y[i] > y[top]) top = i;
  printf("  top-1 = class %lld (score %.4f)\n", (long long)top, y[top]);
}

int main(int argc, char** argv) {
  std::string model = argval(argc, argv, "--model", "assets/mobilenetv2.onnx");
  auto in = readFile(argval(argc, argv, "--input", "assets/input.bin"));
  if (in.empty()) in.assign(1 * 3 * 224 * 224 * 4, 0);
  for (BackendKind be : {BackendKind::kVulkan, BackendKind::kCpu, BackendKind::kEnn})
    runWith(model, in, be);
  return 0;
}
