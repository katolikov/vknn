// Generic multi-input / multi-output runner. Loads a model (.onnx or .vxm), feeds raw fp32 .bin
// files (in model input order), runs, dumps each output to <outdir>/<name>.bin.
//   vknn_run_io model outdir [flags] in0.bin in1.bin ...
// Flags:
//   --backend cpu|vulkan   (default vulkan)   --precision low|normal|high (default low; normal = fp16 + selective fp32)
//   --priority low|normal|high  GPU queue scheduling priority (default normal; Vulkan queue global priority)
//   --tuning none|fast|heavy    load-time conv autotune effort (none = no per-shape measurement) (default fast)
//   --keep-weights         keep host weights after upload (default: free them)
//   --no-flat              disable the flat-layout GPU pass (advanced)
//   --no-fold-islands      keep tiny GPU op-islands on the GPU instead of folding to CPU (advanced)
//   --no-cache             skip cache read/write (cold compile every load)
//   --timing               print pack/submit/unpack timing
//   --cache DIR            cache directory
//   --winograd auto|on|off force the 3x3-conv kernel (on/off skip autotuning -> deterministic choice)
//   --max-submit-nodes N   split the GPU command buffer every N nodes (watchdog/TDR mitigation)
//   --max-submit-bindings N  split the command buffer once it accumulates N descriptor bindings
//   --repeat N             re-run the same inputs N times; only the last run's outputs are written
//   --profile              print the per-op GPU profile table and the GPU total
//   --dump NAMES           dump the named intermediate tensors (comma-separated) as fp32
//   --fp32-tensors NAMES   force the named tensors to fp32 compute (comma-separated)
//   --disable-vk-ops NAMES force the named ops onto the CPU backend (comma-separated)
//   --layer-dump           dump every layer's output    --layer-dump-dir DIR  where to write them
//   --debug-segments       print the segment (CPU/GPU island) partition
#include "vknn/session.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace vknn;

// True if the boolean flag `k` appears anywhere in argv. Scanning starts at index 3 so the
// program name, model, and outdir positional args are never mistaken for flags.
static bool flag(int c, char **v, const char *k) noexcept {
    for (int i = 3; i < c; ++i)
    {
        if (!strcmp(v[i], k))
        {
            return true;
        }
    }
    return false;
}
// The value following the option `k` (e.g. "--cache DIR"), or the default `d` if `k` is absent.
// Stops one short of the last arg so a trailing bare option name cannot read past argv.
static const char *opt(int c, char **v, const char *k, const char *d) noexcept {
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
        printf("usage: %s model outdir [--backend cpu|vulkan] [--precision low|normal|high] [--priority low|normal|high]"
               " [--tuning none|fast|heavy] [--no-cache] [--no-flat] [--no-fold-islands] [--timing] [--cache DIR]"
               " [--winograd auto|on|off] [--max-submit-nodes N] in0.bin in1.bin ...\n",
               argv[0]);
        return 1;
    }
    std::string model = argv[1], outdir = argv[2];
    ::mkdir(outdir.c_str(), 0755); // create the output dir if missing

    Config cfg;
    cfg.backend                = backendFromStr(opt(argc, argv, "--backend", "vulkan"));
    cfg.precision              = precisionFromStr(opt(argc, argv, "--precision", "low"));
    cfg.priority               = priorityFromStr(opt(argc, argv, "--priority", "normal"));
    cfg.tuning                 = tuningFromStr(opt(argc, argv, "--tuning", "fast"));
    cfg.noCache                = flag(argc, argv, "--no-cache");
    cfg.freeWeightsAfterUpload = !flag(argc, argv, "--keep-weights");
    if (flag(argc, argv, "--no-flat"))
    {
        cfg.setHint(Hint::FlatLayout, (int) Mode::Off);
    }
    if (flag(argc, argv, "--no-fold-islands"))
    {
        cfg.setHint(Hint::GpuIslandFold, (int) Mode::Off);
    }
    cfg.layerDump              = flag(argc, argv, "--layer-dump");
    cfg.debugSegments          = flag(argc, argv, "--debug-segments");
    cfg.layerDumpDir           = opt(argc, argv, "--layer-dump-dir", cfg.layerDumpDir.c_str());
    cfg.timing                 = flag(argc, argv, "--timing");
    cfg.cacheDir               = opt(argc, argv, "--cache", cfg.cacheDir.c_str());
    cfg.dumpTensors            = opt(argc, argv, "--dump", "");
    cfg.fp32Tensors            = opt(argc, argv, "--fp32-tensors", "");
    cfg.profile                = flag(argc, argv, "--profile");
    cfg.setHint(Hint::Winograd, winogradFromStr(opt(argc, argv, "--winograd", "auto")));
    cfg.maxSubmitNodes    = atoi(opt(argc, argv, "--max-submit-nodes", std::to_string(cfg.maxSubmitNodes).c_str()));
    cfg.maxSubmitBindings = atoi(opt(argc, argv, "--max-submit-bindings", std::to_string(cfg.maxSubmitBindings).c_str()));
    cfg.disableVkOps      = opt(argc, argv, "--disable-vk-ops", "");

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
            if (!strcmp(argv[i], "--backend") || !strcmp(argv[i], "--precision") || !strcmp(argv[i], "--priority") || !strcmp(argv[i], "--cache") || !strcmp(argv[i], "--dump") ||
                !strcmp(argv[i], "--winograd") || !strcmp(argv[i], "--tuning") || !strcmp(argv[i], "--fp32-tensors") || !strcmp(argv[i], "--layer-dump-dir") || !strcmp(argv[i], "--max-submit-nodes") || !strcmp(argv[i], "--max-submit-bindings") || !strcmp(argv[i], "--disable-vk-ops") || !strcmp(argv[i], "--repeat"))
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
        in.name = infos[i].name;
        in.shape = infos[i].shape;
        // Feed the model's DECLARED input dtype: a UINT8/FLOAT16 .bin is read as native bytes and the
        // Session converts at the boundary. The Session also accepts fp32 (it converts either way).
        in.dtype     = infos[i].dtype;
        int64_t need = numElements(in.shape) * (int64_t) dtypeSize(in.dtype);
        in.data.assign(need, 0);
        if (i < inFiles.size())
        {
            std::ifstream f(inFiles[i], std::ios::binary);
            if (!f)
            {
                fprintf(stderr, "cannot open input file '%s' for '%s'\n", inFiles[i].c_str(), in.name.c_str());
                return 1; // silently feeding zeros would fake a successful run on wrong data
            }
            f.read(reinterpret_cast<char *>(in.data.data()), need);
        }
        printf("input  '%s'  %s  %s\n", in.name.c_str(), shapeStr(in.shape).c_str(), dtypeStr(in.dtype));
        ins.push_back(std::move(in));
    }

    // --repeat N re-runs the same inputs N times (default 1). The first run pays one-time costs (command
    // buffer record, pipeline build, first-run autotune); later runs show steady-state I/O timing.
    int                   repeat = atoi(opt(argc, argv, "--repeat", "1"));
    std::vector<IOTensor> outs;
    Status                st = Status::Ok;
    for (int r = 0; r < (repeat < 1 ? 1 : repeat); ++r)
    {
        outs.clear();
        st = sess->run(ins, outs);
        if (st != Status::Ok)
        {
            fprintf(stderr, "run failed (status %d)\n", (int) st);
            return 2;
        }
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
