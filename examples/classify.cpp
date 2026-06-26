// vknn_classify - load ONNX, run an input, print top-5; optional golden compare.
//
// Flags:
//   --model PATH      ONNX model (default assets/mobilenetv2.onnx)
//   --input PATH      raw float32 NCHW input (default input.bin)
//   --shape N,C,H,W   input shape (default 1,3,224,224)
//   --backend NAME    vulkan|cpu (default vulkan)
//   --precision P     fp32|fp16 (default fp16)
//   --config PATH     JSON config (overrides flags it sets)
//   --golden PATH     raw float32 golden output for cosine/top-1 check
//   --profile         enable per-op profiler + print table
//   --layer-dump DIR  dump every layer output to DIR
//   --cache DIR       cache directory
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "vknn/session.h"

using namespace vknn;

static std::vector<uint8_t> readFile(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f)
    return {};
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}
static const char* argval(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i < argc - 1; ++i)
    if (!strcmp(argv[i], key))
      return argv[i + 1];
  return def;
}
static bool hasflag(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], key))
      return true;
  return false;
}

int main(int argc, char** argv) {
  std::string model = argval(argc, argv, "--model", "assets/mobilenetv2.onnx");
  std::string inpath = argval(argc, argv, "--input", "assets/input.bin");
  // Empty by default: the input shape comes from the model (inputInfo); --shape only OVERRIDES it.
  // (A non-empty default like "1,3,224,224" silently forces every model to 224x224 -> wrong input
  // crop for 299x299 Inception, 640x640 YOLO, and any non-image model.)
  std::string shapeStr = argval(argc, argv, "--shape", "");
  std::string backend = argval(argc, argv, "--backend", "vulkan");
  std::string precision = argval(argc, argv, "--precision", "fp16");
  std::string goldpath = argval(argc, argv, "--golden", "");
  std::string cfgpath = argval(argc, argv, "--config", "");

  Config cfg;
  if (!cfgpath.empty())
    cfg = Config::fromJsonFile(cfgpath);
  cfg.backend = backendFromStr(backend);
  cfg.precision = (precision == "fp32") ? Precision::kFp32 : Precision::kFp16;
  cfg.cacheDir = argval(argc, argv, "--cache", cfg.cacheDir.c_str());
  if (hasflag(argc, argv, "--profile"))
    cfg.profile = true;
  if (hasflag(argc, argv, "--layer-dump")) {
    cfg.layerDump = true;
    cfg.layerDumpDir = argval(argc, argv, "--layer-dump", cfg.layerDumpDir.c_str());
  }

  Shape shape;
  {
    std::string s = shapeStr;
    size_t pos;
    while ((pos = s.find(',')) != std::string::npos || !s.empty()) {
      std::string tok = s.substr(0, pos);
      shape.push_back(std::stoll(tok));
      if (pos == std::string::npos)
        break;
      s = s.substr(pos + 1);
    }
  }

  printf("model=%s backend=%s precision=%s input=%s shape=%s\n", model.c_str(),
         backendName(cfg.backend), precision.c_str(), inpath.c_str(), shapeStr.c_str());

  auto sess = Runtime::load(model, cfg);
  if (!sess) {
    fprintf(stderr, "failed to load model\n");
    return 1;
  }
  if (hasflag(argc, argv, "--show-graph")) {
    const Graph& gg = sess->graph();
    int i = 0;
    for (const auto& nd : gg.nodes) {
      TensorId o = nd.outputs.empty() ? kNoTensor : nd.outputs[0];
      printf("  [%3d] %-18s out=%-8s %s act=%d\n", i++, opTypeName(nd.type),
             o == kNoTensor ? "?" : gg.tensors[o].name.c_str(),
             o == kNoTensor ? "" : ::vknn::shapeStr(gg.tensors[o].shape).c_str(), (int)nd.fusedAct);
    }
  }

  // Name + shape come from the model (inputInfo); --shape only needed if you want to override.
  IOTensor in;
  auto inInfo = sess->inputInfo();
  if (!inInfo.empty()) {
    in.name = inInfo[0].name;
    in.shape = shape.empty() ? inInfo[0].shape : shape;
  } else {
    in.name = "input";
    in.shape = shape;
  }
  in.dtype = DType::kFloat32;
  in.data = readFile(inpath);
  int64_t need = numElements(in.shape) * 4;
  if ((int64_t)in.data.size() < need) {
    fprintf(stderr, "input file too small (%zu < %lld); using zeros\n", in.data.size(),
            (long long)need);
    in.data.assign(need, 0);
  }

  std::vector<IOTensor> outs;
  Status st = sess->run({in}, outs);
  if (st != Status::kOk || outs.empty()) {
    fprintf(stderr, "run failed\n");
    return 2;
  }

  const float* y = outs[0].f32();
  int64_t n = numElements(outs[0].shape);
  std::vector<int> idx(n);
  for (int64_t i = 0; i < n; ++i)
    idx[i] = (int)i;
  std::partial_sort(idx.begin(), idx.begin() + std::min<int64_t>(5, n), idx.end(),
                    [&](int a, int b) { return y[a] > y[b]; });
  printf("top-5:\n");
  for (int k = 0; k < std::min<int64_t>(5, n); ++k)
    printf("  #%d  class %4d  score %.5f\n", k + 1, idx[k], y[idx[k]]);

  if (!goldpath.empty()) {
    auto gb = readFile(goldpath);
    const float* g = reinterpret_cast<const float*>(gb.data());
    int64_t gn = (int64_t)gb.size() / 4;
    int64_t m = std::min(n, gn);
    double dot = 0, na = 0, nb = 0, maxErr = 0;
    for (int64_t i = 0; i < m; ++i) {
      dot += (double)y[i] * g[i];
      na += (double)y[i] * y[i];
      nb += (double)g[i] * g[i];
      maxErr = std::max(maxErr, (double)std::fabs(y[i] - g[i]));
    }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    int goldTop = 0;
    for (int64_t i = 1; i < m; ++i)
      if (g[i] > g[goldTop])
        goldTop = (int)i;
    printf("golden compare: cosine=%.6f maxAbsErr=%.4e  top1 vknn=%d golden=%d  => %s\n", cos,
           maxErr, idx[0], goldTop, (idx[0] == goldTop && cos >= 0.99) ? "PASS" : "CHECK");
  }

  if (cfg.profile)
    sess->profiler().printTable();

  // Optional latency benchmark: --bench N  (warmup 5, then N timed runs).
  int benchN = atoi(argval(argc, argv, "--bench", "0"));
  if (benchN > 0) {
    // Bind the input vector ONCE and reuse it; reconstructing {in} each call re-copies the whole
    // input tensor (large mmap/free per run) which is benchmark-harness noise, not engine cost.
    std::vector<IOTensor> invec{in};
    {  // cold latency: very first run after session create (matches MNN's no-warmup protocol)
      auto c0 = std::chrono::high_resolution_clock::now();
      sess->run(invec, outs);
      auto c1 = std::chrono::high_resolution_clock::now();
      printf("cold (first run): %.2f ms\n",
             std::chrono::duration<double, std::milli>(c1 - c0).count());
    }
    for (int i = 0; i < 5; ++i)
      sess->run(invec, outs);  // warmup
    std::vector<double> ms;
    for (int i = 0; i < benchN; ++i) {
      auto t0 = std::chrono::high_resolution_clock::now();
      sess->run(invec, outs);
      auto t1 = std::chrono::high_resolution_clock::now();
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    double mean = 0;
    for (double v : ms)
      mean += v;
    mean /= ms.size();
    double med = ms[ms.size() / 2];
    double p90 = ms[(size_t)(ms.size() * 0.9)];
    printf(
        "\nbench (%d runs): min=%.2f ms  median=%.2f ms  p90=%.2f ms  => %.1f fps (min) / %.1f fps "
        "(median)\n",
        benchN, ms.front(), med, p90, 1000.0 / ms.front(), 1000.0 / med);
    (void)mean;
  }
  return 0;
}
