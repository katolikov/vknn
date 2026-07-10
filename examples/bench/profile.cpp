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
//   --bucket N        profile plan bucket N of a multi-bucket model (default 0); the run binds
//                     bucket N's declared input shapes, so run() dispatches to that bucket
#include "vknn/session.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
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
    // A kernel-isolation probe (a bare op chain) is exactly the shape the island fold reassigns to
    // the CPU; profiling wants the GPU kernels, so keep the islands on the GPU like run_io's
    // --no-fold-islands.
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--no-fold-islands"))
        {
            cfg.setHint(Hint::GpuIslandFold, (int) Mode::Off);
        }
        if (!strcmp(argv[i], "--no-matmul-view-fold"))
        {
            cfg.setHint(Hint::MatMulViewFold, (int) Mode::Off);
        }
        if (!strcmp(argv[i], "--no-fused-attention"))
        {
            cfg.setHint(Hint::FusedAttention, (int) Mode::Off);
        }
    }

    auto sess = Runtime::load(model, cfg);
    if (!sess)
    {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    // Inputs come from the model's own declaration: every input gets its declared name/shape/dtype;
    // --input fills the FIRST input from a raw file (missing or short -> zero-filled). Extra inputs
    // (a decoder's KV set) are zero-filled, which profiles fine. --bucket selects which plan
    // bucket's shapes are bound (an LLM .vxm stores prefill and decode as separate buckets).
    const size_t bucket = (size_t) atoi(argval(argc, argv, "--bucket", "0"));
    const auto   infos  = sess->inputInfo(bucket);
    if (infos.empty())
    {
        fprintf(stderr, "bucket %zu has no inputs (out of range?)\n", bucket);
        return 1;
    }
    std::vector<IOTensor> ins;
    for (const IOInfo &info: infos)
    {
        IOTensor in;
        in.name           = info.name;
        in.shape          = info.shape;
        in.dtype          = info.dtype == DType::Int64 || info.dtype == DType::Int32 ? info.dtype : DType::Float32;
        const size_t need = (size_t) std::max<int64_t>(numElements(in.shape), 1) * dtypeSize(in.dtype);
        if (ins.empty())
        {
            in.data = readFile(inpath);
        }
        if (in.data.size() < need)
        {
            in.data.assign(need, 0);
        }
        ins.push_back(std::move(in));
    }

    std::vector<IOTensor> outs;
    sess->run(ins, outs); // warmup (fills caches)
    sess->run(ins, outs); // profiled run

    sess->profiler().printTable();

    std::ofstream(jsonp) << sess->profiler().toJson();
    sess->profiler().writeChromeTrace(trace);
    printf("\nGPU total: %.3f ms   CPU total: %.3f ms\n", sess->profiler().totalGpuMs(), sess->profiler().totalCpuMs());
    printf("wrote profile JSON -> %s\nwrote Chrome trace -> %s (load in chrome://tracing)\n", jsonp.c_str(), trace.c_str());
    return 0;
}
