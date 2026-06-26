// vknn_profile - run a model with profiling: per-op timing table + JSON + Chrome trace.
//
// Flags: --model PATH --input PATH --backend NAME --precision P --trace PATH --json PATH
#include "vknn/session.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

using namespace vknn;
static const char *argval(int c, char **v, const char *k, const char *d) {
    for (int i = 1; i < c - 1; ++i)
    {
        if (!strcmp(v[i], k))
        {
            return v[i + 1];
        }
    }
    return d;
}
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
    cfg.precision = strcmp(argval(argc, argv, "--precision", "fp16"), "fp32") == 0 ? Precision::kFp32 : Precision::kFp16;
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
    in.dtype = DType::kFloat32;
    in.data  = readFile(inpath);
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
