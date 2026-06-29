// Generic multi-input / multi-output runner. Loads a model (.onnx or .vxm), feeds raw fp32 .bin
// files (in model input order), runs, dumps each output to <outdir>/<name>.bin.
//   vknn_run_io model outdir [flags] in0.bin in1.bin ...
// Flags:
//   --backend cpu|vulkan   (default vulkan)   --precision fp16|fp32 (default fp16)
//   --no-weight-cache      don't cache prepacked weights in RAM (saves memory for big models)
//   --keep-weights         keep host weights after upload (default: free them)
//   --no-flat              disable the flat-layout GPU pass
//   --timing               print pack/submit/unpack timing
//   --cache DIR            cache directory
//   --winograd auto|on|off force the 3x3-conv kernel (on/off skip autotuning -> deterministic choice)
//   --tuning off|fast|thorough  conv autotune effort (off = no per-shape kernel measurement)
#include "vknn/session.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace vknn;

static bool flag(int c, char **v, const char *k) {
    for (int i = 3; i < c; ++i)
    {
        if (!strcmp(v[i], k))
        {
            return true;
        }
    }
    return false;
}
static const char *opt(int c, char **v, const char *k, const char *d) {
    for (int i = 3; i < c - 1; ++i)
    {
        if (!strcmp(v[i], k))
        {
            return v[i + 1];
        }
    }
    return d;
}

int main(int argc, char **argv) {
    if (argc < 3)
    {
        printf("usage: %s model outdir [--backend cpu|vulkan] [--precision fp16|fp32] [--no-weight-cache]"
               " [--no-flat] [--timing] [--cache DIR] [--winograd auto|on|off]"
               " [--tuning off|fast|thorough] in0.bin in1.bin ...\n",
               argv[0]);
        return 1;
    }
    std::string model = argv[1], outdir = argv[2];

    Config cfg;
    cfg.backend                = backendFromStr(opt(argc, argv, "--backend", "vulkan"));
    cfg.precision              = !strcmp(opt(argc, argv, "--precision", "fp16"), "fp32") ? Precision::Fp32 : Precision::Fp16;
    cfg.cacheWeights           = !flag(argc, argv, "--no-weight-cache");
    cfg.freeWeightsAfterUpload = !flag(argc, argv, "--keep-weights");
    cfg.noFlatOps              = flag(argc, argv, "--no-flat");
    cfg.timing                 = flag(argc, argv, "--timing");
    cfg.cacheDir               = opt(argc, argv, "--cache", cfg.cacheDir.c_str());
    cfg.dumpTensors            = opt(argc, argv, "--dump", "");
    cfg.profile                = flag(argc, argv, "--profile");
    cfg.setHint(Hint::Winograd, (int) winogradFromStr(opt(argc, argv, "--winograd", "auto")));
    cfg.setHint(Hint::Tuning, (int) tuningFromStr(opt(argc, argv, "--tuning", "fast")));

    auto sess = Runtime::load(model, cfg);
    if (!sess)
    {
        fprintf(stderr, "failed to load %s\n", model.c_str());
        return 1;
    }
    auto infos = sess->inputInfo();
    // positional input files = args after argv[2] that aren't a flag (or a flag's value).
    std::vector<std::string> inFiles;
    for (int i = 3; i < argc; ++i)
    {
        if (argv[i][0] == '-')
        {
            if (!strcmp(argv[i], "--backend") || !strcmp(argv[i], "--precision") || !strcmp(argv[i], "--cache") || !strcmp(argv[i], "--dump") ||
                !strcmp(argv[i], "--winograd") || !strcmp(argv[i], "--tuning"))
            {
                ++i; // skip the flag's value
            }
            continue;
        }
        inFiles.push_back(argv[i]);
    }

    std::vector<IOTensor> ins;
    for (size_t i = 0; i < infos.size(); ++i)
    {
        IOTensor in;
        in.name      = infos[i].name;
        in.shape     = infos[i].shape;
        in.dtype     = DType::Float32;
        int64_t need = numElements(in.shape) * 4;
        in.data.assign(need, 0);
        if (i < inFiles.size())
        {
            std::ifstream f(inFiles[i], std::ios::binary);
            if (f)
            {
                f.read(reinterpret_cast<char *>(in.data.data()), need);
            }
        }
        printf("input  '%s'  %s\n", in.name.c_str(), shapeStr(in.shape).c_str());
        ins.push_back(std::move(in));
    }

    std::vector<IOTensor> outs;
    Status                st = sess->run(ins, outs);
    if (st != Status::Ok)
    {
        fprintf(stderr, "run failed (status %d)\n", (int) st);
        return 2;
    }
    for (auto &o: outs)
    {
        // sanitize: tensor names can contain '/' (e.g. "/enc/backbone/..."); flatten to one filename.
        std::string safe = o.name;
        for (char &ch: safe)
        {
            if (ch == '/' || ch == ':')
            {
                ch = '_';
            }
        }
        std::string   fn = outdir + "/" + safe + ".bin";
        std::ofstream f(fn, std::ios::binary);
        f.write(reinterpret_cast<const char *>(o.data.data()), o.data.size());
        if (!f)
        {
            fprintf(stderr, "WARN: failed to write %s\n", fn.c_str());
        }
        printf("output '%s'  %s  -> %s\n", o.name.c_str(), shapeStr(o.shape).c_str(), fn.c_str());
    }
    if (cfg.profile)
    {
        sess->profiler().printTable();
        printf("GPU total: %.1f ms\n", sess->profiler().totalGpuMs());
    }
    return 0;
}
