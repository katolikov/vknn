// vknn_backend_switch - run the same classifier twice, once per backend, changing only one config field.
//
// The Vulkan (GPU) run and the CPU run are built from the exact same model and fed the exact same
// input; the ONLY difference between them is Config::backend. For each run this prints how the graph's
// nodes were routed across backends (an op the preferred backend cannot run falls through
// Config::fallback to the CPU) and the top-1 class, so the two backends line up side by side.
//
// The VKNN API used here:
//   - Runtime::load(path, cfg)       Load a model file (".onnx", or a pre-optimized ".vxm") and build a
//                                    ready-to-run Session. cfg.backend picks the preferred backend.
//   - IOTensor                       A named tensor at the API boundary: name + shape + dtype + host
//                                    bytes. Fill one per model input; get one back per model output.
//   - session->run(inputs, outputs)  Execute the model, filling `outputs` with the results.
//   - session->nodeBackends()        The backend that actually ran each node, in node order (an op the
//                                    preferred backend could not run appears here as a CPU fallback).
//   - backendName(kind)              The readable name of a BackendKind ("VULKAN" / "CPU").
//   - numElements(shape)             Element count of a tensor shape (the product of its dims).
//
//   vknn_backend_switch [--model model.onnx] [--input input.bin]
#include "vknn/session.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

using namespace vknn;

// --- Not VKNN - just plumbing so the demo runs standalone ------------------------------------------
// readFile() and argval() are ordinary file reading and command-line parsing. They have nothing to do
// with the engine; they only produce the model path and the raw input bytes that the VKNN calls below
// consume. A real integration gets those from wherever its data already lives.

/// Read an entire file into a byte vector; returns an empty vector if the file cannot be opened.
static std::vector<uint8_t> readFile(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    return file ? std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()) : std::vector<uint8_t>();
}

/// Return the argument following flag `flag` on the command line, or `fallback` when the flag is absent.
static const char *argval(int argc, char **argv, const char *flag, const char *fallback) noexcept
{
    for (int i = 1; i < argc - 1; ++i)
    {
        if (!strcmp(argv[i], flag))
        {
            return argv[i + 1];
        }
    }
    return fallback;
}

// --- VKNN - one full inference run on a single backend ---------------------------------------------
// One inference run, end to end: load the model, describe the input, run it, then inspect the routing
// and the prediction. The two calls in main() differ ONLY in the `backend` argument.
static void runOnBackend(const std::string &modelPath, const std::vector<uint8_t> &inputBytes, BackendKind backend)
{
    // Step 1 - load the model onto the chosen backend. `backend` is the one field that changes between
    // the Vulkan and CPU runs; `fallback` is the ordered chain tried for any op the backend refuses.
    Config config;
    config.backend   = backend;
    config.fallback  = {BackendKind::Vulkan, BackendKind::Cpu}; // others fall back through these
    config.precision = Precision::Low;
    printf("\n=== config.backend = %s ===\n", backendName(backend));
    std::unique_ptr<Session> session = Runtime::load(modelPath, config);
    if (!session)
    {
        printf("  load failed\n");
        return;
    }

    // Step 2 - describe the input the model expects. This classifier takes a single fp32 NCHW image
    // (1x3x224x224); the raw pixel bytes come straight from inputBytes.
    IOTensor input;
    input.name  = "input";
    input.shape = {1, 3, 224, 224};
    input.dtype = DType::Float32;
    input.data  = inputBytes;

    // Step 3 - run inference. `outputs` comes back holding one fully described IOTensor per model output.
    std::vector<IOTensor> outputs;
    if (session->run({input}, outputs) != Status::Ok)
    {
        printf("  run failed\n");
        return;
    }

    // Step 4 - report where the work actually ran. nodeBackends() gives the backend for each node;
    // counting them into a histogram makes any CPU fallback under a GPU run visible at a glance.
    std::map<BackendKind, int> nodeCountByBackend;
    for (BackendKind kind: session->nodeBackends())
    {
        nodeCountByBackend[kind]++;
    }
    printf("  node routing:");
    for (const std::pair<const BackendKind, int> &entry: nodeCountByBackend)
    {
        printf(" %s=%d", backendName(entry.first), entry.second);
    }
    printf("\n");

    // Step 5 - report the prediction. The output is a vector of class scores; the top-1 class is the
    // index of the largest score (argmax over the logits/scores).
    const float *scores     = outputs[0].f32();
    int64_t      scoreCount = numElements(outputs[0].shape);
    int64_t      topClass   = 0;
    for (int64_t i = 1; i < scoreCount; ++i)
    {
        if (scores[i] > scores[topClass])
        {
            topClass = i;
        }
    }
    printf("  top-1 = class %lld (score %.4f)\n", (long long) topClass, scores[topClass]);
}

int main(int argc, char **argv)
{
    // Step 1 - decide which model and input to use (plumbing: CLI parsing + file reading, above).
    std::string          modelPath  = argval(argc, argv, "--model", "assets/mobilenetv2.onnx");
    std::vector<uint8_t> inputBytes = readFile(argval(argc, argv, "--input", "assets/input.bin"));
    if (inputBytes.empty())
    {
        // No input file given or readable: fall back to a zeroed 1x3x224x224 fp32 image so the demo
        // still runs and both backends get the same bytes.
        inputBytes.assign(1 * 3 * 224 * 224 * 4, 0);
    }

    // Step 2 - run the same model on each backend in turn. Every other input is held fixed, so the two
    // blocks of output isolate exactly what the backend choice changes.
    for (BackendKind backend: {BackendKind::Vulkan, BackendKind::Cpu})
    {
        runOnBackend(modelPath, inputBytes, backend);
    }
    return 0;
}
