// vknn_run_io - the "run a whole model straight from files" example, and the first stop for anyone
// meeting VKNN. It loads a model, feeds it one raw .bin file per input (in the model's own input
// order), runs it, and writes each output tensor to <outdir>/<name>.bin. Every engine knob is a
// command-line flag, so this file doubles as the reference runner the test scripts drive.
//
// The VKNN API used here (everything else in this file is plumbing so the demo can stand alone):
//   Config                        - the plain struct of engine knobs; each CLI flag sets one field/hint.
//   Runtime::load(path, cfg)      - load a model (.onnx or .vxm) and build a ready-to-run Session.
//   session->bucketCount()        - how many compiled plan buckets the model has (an LLM ships >1).
//   session->inputInfo(bucket)    - what the model expects: each input's name, shape and dtype.
//   session->run(inputs, outputs) - run once; fills `outputs` with every result tensor.
//   session->profiler()           - per-op GPU timing, printed when --profile is set.
// The values crossing the boundary are IOTensor (name + shape + dtype + a raw byte payload); the
// matching IOInfo carries the same fields, read straight from the model so the caller never guesses.
//
//   vknn_run_io model outdir [flags] in0.bin in1.bin ...
// Flags:
//   --backend cpu|vulkan   (default vulkan)   --precision low|normal|high (default low; normal = fp16 + selective fp32)
//   --priority low|normal|high  GPU queue scheduling priority (default normal; Vulkan queue global priority)
//   --tuning none|fast|heavy    load-time conv autotune effort (none = no per-shape measurement) (default fast)
//   --cpu-threads N        CPU-backend worker threads (default 4; 1 = serial). Effort only: output is bit-identical.
//   --keep-weights         keep host weights after upload (default: free them)
//   --no-flat              disable the flat-layout GPU pass (advanced)
//   --no-fold-islands      keep tiny GPU op-islands on the GPU instead of folding to CPU (advanced)
//   --no-cache             skip cache read/write (cold compile every load)
//   --timing               print pack/submit/unpack timing
//   --timing-summary       print per-segment averages: submit call, fence wait, GPU busy, GPU gap
//   --cache DIR            directory to hold the model's cache file (default: beside the model)
//   --winograd auto|on|off force the 3x3-conv kernel (on/off skip autotuning -> deterministic choice)
//   --max-submit-nodes N   split the GPU command buffer every N nodes (watchdog/TDR mitigation)
//   --max-submit-bindings N  split the command buffer once it accumulates N descriptor bindings
//   --repeat N             re-run the same inputs N times; only the last run's outputs are written
//   --bucket N             run plan bucket N of a multi-bucket model (default 0); the positional
//                          inputs bind bucket N's declared inputs, so run() dispatches to that
//                          bucket and the written outputs (and --dump tensors) are bucket N's
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
#ifdef _WIN32
#include <direct.h> // _mkdir (one-argument; no mode bits on Windows)
#endif

using namespace vknn;

// ---------------------------------------------------------------------------------------------------
// Not VKNN - just tiny argv parsing so the demo takes its knobs from the command line. Both scans
// start at index 3 so the program name, model, and outdir positional args are never read as flags.
// ---------------------------------------------------------------------------------------------------

// True when the boolean flag `name` appears anywhere in argv (e.g. "--no-cache").
static bool hasFlag(int argc, char **argv, const char *name) noexcept {
    for (int i = 3; i < argc; ++i)
    {
        if (!strcmp(argv[i], name))
        {
            return true;
        }
    }
    return false;
}

// The value following the option `name` (e.g. "--cache DIR"), or `dflt` when `name` is absent. Stops
// one short of the last arg so a trailing bare option name cannot read past the end of argv.
static const char *optValue(int argc, char **argv, const char *name, const char *dflt) noexcept {
    for (int i = 3; i < argc - 1; ++i)
    {
        if (!strcmp(argv[i], name))
        {
            return argv[i + 1];
        }
    }
    return dflt;
}

int main(int argc, char **argv) {
    if (argc < 3)
    {
        printf("usage: %s model outdir [--backend cpu|vulkan] [--precision low|normal|high] [--priority low|normal|high]"
               " [--tuning none|fast|heavy] [--no-cache] [--no-flat] [--no-fold-islands] [--no-matmul-view-fold] [--no-rope-fusion] [--no-fused-attention] "
               "[--timing] [--cache DIR]"
               " [--winograd auto|on|off] [--max-submit-nodes N] [--bucket N] in0.bin in1.bin ...\n",
               argv[0]);
        return 1;
    }

    // Not VKNN: the two positional args are the model path and the output directory. Make the output
    // directory up front so the writes at the end always land.
    std::string model = argv[1], outdir = argv[2];
#ifdef _WIN32
    ::_mkdir(outdir.c_str()); // create the output dir if missing
#else
    ::mkdir(outdir.c_str(), 0755); // create the output dir if missing
#endif

    // Step 1 - translate the command line into a Config.
    // The Config is the one struct of engine knobs; every flag below sets a single field or hint. The
    // defaults (vulkan / low precision / fast tuning) are the ordinary release path, so a bare
    // `vknn_run_io model outdir input.bin` just runs the model on the GPU.
    Config cfg;
    cfg.backend                = backendFromStr(optValue(argc, argv, "--backend", "vulkan"));
    cfg.precision              = precisionFromStr(optValue(argc, argv, "--precision", "low"));
    cfg.priority               = priorityFromStr(optValue(argc, argv, "--priority", "normal"));
    cfg.tuning                 = tuningFromStr(optValue(argc, argv, "--tuning", "fast"));
    cfg.noCache                = hasFlag(argc, argv, "--no-cache");
    cfg.verbosity              = atoi(optValue(argc, argv, "--verbosity", "1"));
    cfg.freeWeightsAfterUpload = !hasFlag(argc, argv, "--keep-weights");
    // Advanced GPU-pass off switches: each --no-* flag turns its optimization hint Off (default On).
    if (hasFlag(argc, argv, "--no-flat"))
    {
        cfg.setHint(Hint::FlatLayout, (int) Mode::Off);
    }
    if (hasFlag(argc, argv, "--no-fold-islands"))
    {
        cfg.setHint(Hint::GpuIslandFold, (int) Mode::Off);
    }
    if (hasFlag(argc, argv, "--no-matmul-view-fold"))
    {
        cfg.setHint(Hint::MatMulViewFold, (int) Mode::Off);
    }
    if (hasFlag(argc, argv, "--no-rope-fusion"))
    {
        cfg.setHint(Hint::RopeFusion, (int) Mode::Off);
    }
    if (hasFlag(argc, argv, "--no-fused-attention"))
    {
        cfg.setHint(Hint::FusedAttention, (int) Mode::Off);
    }
    if (hasFlag(argc, argv, "--no-kv-concat-fold"))
    {
        cfg.setHint(Hint::KvConcatFold, (int) Mode::Off);
    }
    cfg.layerDump     = hasFlag(argc, argv, "--layer-dump");
    cfg.debugSegments = hasFlag(argc, argv, "--debug-segments");
    cfg.layerDumpDir  = optValue(argc, argv, "--layer-dump-dir", cfg.layerDumpDir.c_str());
    cfg.timing        = hasFlag(argc, argv, "--timing");
    cfg.timingSummary = hasFlag(argc, argv, "--timing-summary");
    // --cache names a directory to hold every model's cache file; without it the cache lands beside the
    // model (Runtime::load's default). The directory is created on the first write.
    if (const char *cacheDir = optValue(argc, argv, "--cache", ""); cacheDir[0])
    {
        cfg.cacheFile = Runtime::cacheFileIn(cacheDir, model);
    }
    cfg.dumpTensors = optValue(argc, argv, "--dump", "");
    cfg.fp32Tensors = optValue(argc, argv, "--fp32-tensors", "");
    cfg.profile     = hasFlag(argc, argv, "--profile");
    cfg.setHint(Hint::Winograd, winogradFromStr(optValue(argc, argv, "--winograd", "auto")));
    cfg.maxSubmitNodes    = atoi(optValue(argc, argv, "--max-submit-nodes", std::to_string(cfg.maxSubmitNodes).c_str()));
    cfg.maxSubmitBindings = atoi(optValue(argc, argv, "--max-submit-bindings", std::to_string(cfg.maxSubmitBindings).c_str()));
    cfg.disableVkOps      = optValue(argc, argv, "--disable-vk-ops", "");
    cfg.cpuThreads        = atoi(optValue(argc, argv, "--cpu-threads", std::to_string(cfg.cpuThreads).c_str()));

    // Step 2 - load the model and build a Session.
    // Runtime::load picks the loader from the file extension (.vxm = pre-optimized, anything else =
    // ONNX), resolves the per-model cache next to the file, runs the graph passes, and hands back a
    // Session ready to run(). A null result means the file could not be loaded.
    std::unique_ptr<Session> session = Runtime::load(model, cfg);
    if (!session)
    {
        fprintf(stderr, "failed to load %s\n", model.c_str());
        return 1;
    }

    // Step 3 - choose the plan bucket and ask the model what it expects.
    // --bucket selects which plan bucket the positional inputs describe (an LLM .vxm stores prefill
    // and decode as separate buckets). Binding bucket N's declared input names/shapes makes run()
    // dispatch to that bucket, so the written outputs (and any --dump tensors) are bucket N's.
    const size_t bucket = (size_t) atoi(optValue(argc, argv, "--bucket", "0"));
    if (bucket >= session->bucketCount())
    {
        fprintf(stderr, "bucket %zu is out of range: %s has %zu bucket(s)\n", bucket, model.c_str(), session->bucketCount());
        return 1;
    }
    // inputInfo(bucket) reports each input's name, concrete shape and declared dtype - everything
    // needed to size and type the buffers, read straight from the model instead of hard-coded here.
    std::vector<IOInfo> modelInputs = session->inputInfo(bucket);

    // Step 4 - not VKNN: collect the positional input files from argv (the args after outdir that are
    // not a flag or a flag's value), one per model input in order.
    std::vector<std::string> inputFiles;
    for (int i = 3; i < argc; ++i)
    {
        if (argv[i][0] == '-')
        {
            if (!strcmp(argv[i], "--backend") || !strcmp(argv[i], "--precision") || !strcmp(argv[i], "--priority") || !strcmp(argv[i], "--cache") || !strcmp(argv[i], "--dump") || !strcmp(argv[i], "--winograd") || !strcmp(argv[i], "--tuning") || !strcmp(argv[i], "--fp32-tensors") || !strcmp(argv[i], "--layer-dump-dir") || !strcmp(argv[i], "--max-submit-nodes") || !strcmp(argv[i], "--max-submit-bindings") || !strcmp(argv[i], "--disable-vk-ops") || !strcmp(argv[i], "--repeat") || !strcmp(argv[i], "--cpu-threads") || !strcmp(argv[i], "--bucket") || !strcmp(argv[i], "--verbosity"))
            {
                ++i; // skip the flag's value
            }
            continue;
        }
        inputFiles.push_back(argv[i]);
    }

    // Step 5 - build one IOTensor per model input, filled from its .bin file.
    // An IOTensor is the boundary value the engine reads: a name + shape + dtype + a raw byte payload.
    // Each tensor is created at the model's DECLARED dtype, so a UINT8/FLOAT16/FLOAT32 .bin is read as
    // native bytes and the Session converts at the boundary either way.
    std::vector<IOTensor> inputs;
    for (size_t i = 0; i < modelInputs.size(); ++i)
    {
        IOTensor input;
        input.name  = modelInputs[i].name;
        input.shape = modelInputs[i].shape;
        input.dtype = modelInputs[i].dtype;
        // Size the byte payload to the declared shape x dtype, zero-filled (a missing file runs on
        // zeros for that input).
        int64_t neededBytes = numElements(input.shape) * (int64_t) dtypeSize(input.dtype);
        input.data.assign(neededBytes, 0);
        if (i < inputFiles.size())
        {
            std::ifstream inputFile(inputFiles[i], std::ios::binary);
            if (!inputFile)
            {
                fprintf(stderr, "cannot open input file '%s' for '%s'\n", inputFiles[i].c_str(), input.name.c_str());
                return 1; // silently feeding zeros would fake a successful run on wrong data
            }
            // The file must hold exactly the declared payload: a short file (wrong shape or dtype on
            // the producing side) would zero-fill the tail and an oversized one would be silently
            // truncated - both run "successfully" on garbage and surface as an inexplicable
            // near-zero-cosine output instead of an error here.
            inputFile.seekg(0, std::ios::end);
            int64_t fileBytes = (int64_t) inputFile.tellg();
            inputFile.seekg(0, std::ios::beg);
            if (fileBytes != neededBytes)
            {
                fprintf(stderr, "input file '%s' for '%s' holds %lld bytes but the declared %s %s input needs %lld\n", inputFiles[i].c_str(), input.name.c_str(), (long long) fileBytes,
                        shapeStr(input.shape).c_str(), dtypeStr(input.dtype), (long long) neededBytes);
                return 1;
            }
            inputFile.read(reinterpret_cast<char *>(input.data.data()), neededBytes);
        }
        printf("input  '%s'  %s  %s\n", input.name.c_str(), shapeStr(input.shape).c_str(), dtypeStr(input.dtype));
        inputs.push_back(std::move(input));
    }

    // Step 6 - run the model, once (or --repeat N times).
    // run() dispatches the bucket whose declared shapes match the bound inputs and fills `outputs`
    // with every result. --repeat re-runs the same inputs (default 1); the first run pays one-time
    // costs (command-buffer record, pipeline build, first-run autotune) and later runs show
    // steady-state timing, so only the last run's outputs are kept.
    int                   repeatCount = atoi(optValue(argc, argv, "--repeat", "1"));
    std::vector<IOTensor> outputs;
    Status                status = Status::Ok;
    // `outputs` keeps the previous run's buffers on purpose: run() reclaims that storage for the
    // tensors it is about to write, then clears the vector itself. Clearing it here would free the
    // buffers first, so every run would re-allocate and zero-fill the output before overwriting it.
    for (int runIndex = 0; runIndex < (repeatCount < 1 ? 1 : repeatCount); ++runIndex)
    {
        status = session->run(inputs, outputs);
        if (status != Status::Ok)
        {
            fprintf(stderr, "run failed (status %d)\n", (int) status);
            return 2;
        }
    }

    // Step 7 - write each output tensor to <outdir>/<name>.bin.
    // Each IOTensor in `outputs` is fully described (name, shape, dtype); its `data` is raw bytes at
    // the output's dtype, which a reader pairs with the printed shape+dtype to interpret.
    for (IOTensor &output: outputs)
    {
        // Not VKNN: tensor names can contain '/' or ':' (e.g. "/enc/backbone/..."); flatten to one
        // filename.
        std::string safeName = output.name;
        for (char &ch: safeName)
        {
            if (ch == '/' || ch == ':')
            {
                ch = '_';
            }
        }
        std::string   outPath = outdir + "/" + safeName + ".bin";
        std::ofstream outputFile(outPath, std::ios::binary);
        outputFile.write(reinterpret_cast<const char *>(output.data.data()), output.data.size());
        if (!outputFile)
        {
            fprintf(stderr, "WARN: failed to write %s\n", outPath.c_str());
        }
        printf("output '%s'  %s  -> %s\n", output.name.c_str(), shapeStr(output.shape).c_str(), outPath.c_str());
    }

    // Step 8 - when --profile is set, print the per-op GPU timing table and the GPU total.
    if (cfg.profile)
    {
        session->profiler().printTable();
        printf("GPU total: %.1f ms\n", session->profiler().totalGpuMs());
    }
    return 0;
}
