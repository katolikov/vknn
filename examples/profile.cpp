// vknn_profile - run a model with profiling: per-op timing table + JSON + Chrome trace.
//
// Runs the model twice (a warmup pass that fills kernel/weight caches, then a profiled pass) and
// emits the profiler's per-op timing table to stdout, the records as JSON, and a chrome://tracing
// file. Assumes a single fp32 NCHW input named "input" of shape 1x3x224x224.
//
// Flags:
//   --model PATH      ONNX model to load (default assets/mobilenetv2.onnx)
//   --input PATH      raw float32 NCHW input blob; missing or short -> zero-filled (default assets/input.bin)
//   --backend NAME    vulkan|cpu (default vulkan)
//   --precision P     low|normal|high (default low)
//   --trace PATH      chrome://tracing output (default /data/local/tmp/vxrt/trace.json)
//   --json PATH       per-op records JSON output (default /data/local/tmp/vxrt/profile.json)
#include "vknn/session.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

using namespace vknn;
// Return the argument following flag `k` in argv, or default `d` if the flag is absent.
static const char *argval(int c, char **v, const char *k, const char *d) noexcept {
    for (int i = 1; i < c - 1; ++i)
    {
        if (!strcmp(v[i], k))
        {
            return v[i + 1];
        }
    }
    return d;
}
// Read `p` wholesale into a byte vector; returns an empty vector if the file cannot be opened.
static std::vector<uint8_t> readFile(const std::string &p) {
    std::ifstream f(p, std::ios::binary);
    return f ? std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()) : std::vector<uint8_t>();
}

int main(int argc, char **argv) {
    std::string model  = argval(argc, argv, "--model", "assets/mobilenetv2.onnx");
    std::string inpath = argval(argc, argv, "--input", "assets/input.bin");
    std::string trace  = argval(argc, argv, "--trace", "/data/local/tmp/vxrt/trace.json");
    std::string jsonp  = argval(argc, argv, "--json", "/data/local/tmp/vxrt/profile.json");

    Config cfg;
    cfg.backend   = backendFromStr(argval(argc, argv, "--backend", "vulkan"));
    cfg.precision = precisionFromStr(argval(argc, argv, "--precision", "low"));
    cfg.profile   = true;

    auto sess = Runtime::load(model, cfg);
    if (!sess)
    {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    IOTensor in;
    in.name  = "input";
    in.shape = {1, 3, 224, 224};
    in.dtype = DType::Float32;
    in.data  = readFile(inpath);
    // Fall back to a zero-filled input when the file is missing or too short (4 bytes per fp32 element).
    if (in.data.size() < numElements(in.shape) * 4)
    {
        in.data.assign(numElements(in.shape) * 4, 0);
    }

    std::vector<IOTensor> outs;
    sess->run({in}, outs); // warmup (fills caches)
    sess->run({in}, outs); // profiled run

    sess->profiler().printTable();

    std::ofstream(jsonp) << sess->profiler().toJson();
    sess->profiler().writeChromeTrace(trace);
    printf("\nGPU total: %.3f ms   CPU total: %.3f ms\n", sess->profiler().totalGpuMs(), sess->profiler().totalCpuMs());
    printf("wrote profile JSON -> %s\nwrote Chrome trace -> %s (load in chrome://tracing)\n", jsonp.c_str(), trace.c_str());
    return 0;
}
