// vx_convert — pre-convert an ONNX model to an optimized ".vxm" so the runtime skips ONNX parsing
// AND all graph passes (fusion/fold/shape-infer). With --fp16, weights are stored as fp16, which
// halves the file and, more importantly, the runtime host memory (a 965M-param model: 3.85GB
// -> 1.9GB) — the difference between fitting an 8GB device and OOM. The Vulkan layout pass
// (NC4HW4<->flat) is applied per-target at load, so a .vxm is backend-agnostic: convert once on the
// host, run on device.
//
//   vx_convert model.onnx out.vxm [--fp16]
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "import/passes.h"
#include "vx/dtype.h"
#include "vx/graph.h"

using namespace vx;

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage: %s model.onnx out.vxm [--fp16]\n", argv[0]);
    return 1;
  }
  std::string onnx = argv[1], out = argv[2];
  bool fp16 = false;
  for (int i = 3; i < argc; ++i)
    if (!strcmp(argv[i], "--fp16"))
      fp16 = true;

  printf("[convert] importing %s ...\n", onnx.c_str());
  Graph g = importOnnx(onnx);
  printf("[convert] %zu nodes, %zu weights. running standard passes ...\n", g.nodes.size(),
         g.initializers.size());
  runStandardPasses(g, 1);
  printf("[convert] post-passes: %zu nodes, %zu weights\n", g.nodes.size(), g.initializers.size());

  if (fp16) {
    int64_t before = 0, after = 0, conv = 0, kept = 0;
    for (auto& kv : g.initializers) {
      TensorDesc& d = g.tensors[kv.first];
      before += (int64_t)kv.second.bytes.size();
      if (d.dtype != DType::kFloat32) {  // int64 shape tensors, etc. — leave as-is
        after += (int64_t)kv.second.bytes.size();
        ++kept;
        continue;
      }
      int64_t n = numElements(d.shape);
      const float* src = kv.second.f32();
      std::vector<uint8_t> half((size_t)n * 2);
      fp16_t* h = reinterpret_cast<fp16_t*>(half.data());
      for (int64_t i = 0; i < n; ++i)
        h[i] = floatToHalf(src[i]);
      kv.second.bytes = std::move(half);
      d.dtype = DType::kFloat16;
      after += (int64_t)kv.second.bytes.size();
      ++conv;
    }
    printf("[convert] fp16: converted %lld weights (%lld kept non-fp32), %.0f MB -> %.0f MB\n",
           (long long)conv, (long long)kept, before / 1e6, after / 1e6);
  }

  if (!saveGraphBin(g, out)) {
    printf("[convert] save failed\n");
    return 2;
  }
  printf("[convert] wrote %s\n", out.c_str());
  return 0;
}
