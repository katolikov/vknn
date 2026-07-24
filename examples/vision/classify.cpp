// vknn_classify - your first VKNN program: load a model, run one input, print the top-5 classes.
//
// This example walks the everyday VKNN call path end to end. It reads a raw fp32 tensor from a file,
// runs it through an image classifier (MobileNetV2 by default), and prints the five highest-scoring
// class indices. It can also check the result against a golden output file and benchmark inference
// latency. I/O here is plain host memory (bytes live in IOTensor::data) - the simplest binding mode;
// see examples/io/dmabuf_fd_io.cpp for the zero-copy fd variant.
//
// The VKNN API used here (every call this file makes into the engine):
//   Config                        - a plain struct of run knobs (backend, precision, cache, autotuning).
//   Config::fromJsonFile(path)    - load those knobs from a JSON file (CLI flags then override).
//   cfg.setHint(Hint::X, value)   - set an advanced kernel-selection knob (e.g. Winograd on/off).
//   backendFromStr / precisionFromStr / winogradFromStr / tuningFromStr
//                                 - parse a string flag into the matching engine enum.
//   Runtime::load(path, cfg)      - load an ONNX (or ".vxm") model and build a ready-to-run Session.
//   session->inputInfo()          - ask the model what inputs it expects (name, shape, dtype).
//   session->run(inputs, outputs) - execute the model; results come back as IOTensor values.
//   IOTensor::f32()               - view a tensor's raw bytes as fp32.
//   session->graph()              - the optimized graph (node/tensor listing), for --show-graph.
//   session->profiler()           - per-op timing table, for --profile.
//
// Flags:
//   --model PATH      ONNX model (default assets/mobilenetv2.onnx)
//   --input PATH      raw float32 NCHW input (default assets/input.bin)
//   --shape N,C,H,W   override the input shape (default: taken from the model)
//   --backend NAME    vulkan|cpu (default vulkan)
//   --precision P     low|normal|high (default low)
//   --winograd MODE   auto|on|off  3x3 Winograd selection (default auto = best+fast per-shape pick)
//   --tuning LEVEL    off|fast|thorough  kernel autotuning (default fast)
//   --wino-unit N     0=auto (default), 4=force F(4,3), 6=force F(6,3) Winograd (research)
//   --wino-variant N  0=tiled-GEMM (default), 1/2/3 = experimental fused variants
//   --timing          print pack/submit/unpack + per-stage timing
//   --debug-seg       trace per-segment execution
//   --show-graph      list the optimized graph's nodes
//   --config PATH     JSON config (overrides flags it sets)
//   --golden PATH     raw float32 golden output for cosine/top-1 check
//   --profile         enable per-op profiler + print table
//   --layer-dump DIR  dump every layer output to DIR
//   --cache DIR       cache directory
//   --bench N         after inference, run a warmup then N timed runs and print latency stats
#include "vknn/session.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace vknn;

// ---------------------------------------------------------------------------------------------------
// Plumbing helpers (NOT VKNN - just enough file/argument handling to make the demo self-contained).
// A newcomer can skip straight to main(); none of this touches the engine API.
// ---------------------------------------------------------------------------------------------------

// Slurp an entire file into a byte buffer; returns an empty vector if the path cannot be opened.
static std::vector<uint8_t> readFile(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        return {};
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// Return the argument following `flag` in argv, or `fallback` if the flag is absent (--flag VALUE form).
static const char *argval(int argc, char **argv, const char *flag, const char *fallback) noexcept {
    for (int i = 1; i < argc - 1; ++i)
    {
        if (!strcmp(argv[i], flag))
        {
            return argv[i + 1];
        }
    }
    return fallback;
}

// Return true if the bare flag `flag` appears anywhere in argv (valueless --flag form).
static bool hasflag(int argc, char **argv, const char *flag) noexcept {
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], flag))
        {
            return true;
        }
    }
    return false;
}

int main(int argc, char **argv) {
    // Step 1 - read the command line (plumbing). Everything has a sensible default, so the demo runs
    // with no arguments at all.
    std::string model       = argval(argc, argv, "--model", "assets/mobilenetv2.onnx");
    std::string inputPath   = argval(argc, argv, "--input", "assets/input.bin");
    // Empty by default: the input shape comes from the model (inputInfo); --shape only OVERRIDES it.
    // (A non-empty default like "1,3,224,224" silently forces every model to 224x224 -> wrong input
    // crop for 299x299 Inception, 640x640 YOLO, and any non-image model.)
    std::string shapeArg     = argval(argc, argv, "--shape", "");
    std::string backendArg   = argval(argc, argv, "--backend", "vulkan");
    std::string precisionArg = argval(argc, argv, "--precision", "low");
    std::string goldenPath   = argval(argc, argv, "--golden", "");
    std::string configPath   = argval(argc, argv, "--config", "");

    // Step 2 - assemble the run configuration. Config is a plain struct of knobs; a JSON file (if
    // given) seeds it, then the individual flags layer on top so a command-line flag always wins.
    Config cfg;
    if (!configPath.empty())
    {
        cfg = Config::fromJsonFile(configPath);
    }
    cfg.backend   = backendFromStr(backendArg);     // vulkan|cpu
    cfg.precision = precisionFromStr(precisionArg); // low|normal|high
    cfg.cacheDir  = argval(argc, argv, "--cache", cfg.cacheDir.c_str());
    {
        // Kernel-selection hints are passed through setHint(Hint, value) rather than plain fields.
        std::string winogradMode = argval(argc, argv, "--winograd", "auto"); // auto|on|off
        cfg.setHint(Hint::Winograd, winogradFromStr(winogradMode));
        std::string tuningLevel = argval(argc, argv, "--tuning", "fast"); // none|fast|heavy
        cfg.tuning              = tuningFromStr(tuningLevel);
        // Advanced hints: force Winograd unit / variant for research.
        int winogradUnit = atoi(argval(argc, argv, "--wino-unit", "0")); // 0=auto, 4=force F(4,3), 6=force F(6,3)
        if (winogradUnit)
        {
            cfg.setHint(Hint::WinogradUnit, winogradUnit);
        }
        int winogradVariant = atoi(argval(argc, argv, "--wino-variant", "0")); // 0=tiled-GEMM,1=fused,2=split,3=full
        if (winogradVariant)
        {
            cfg.setHint(Hint::WinogradVariant, winogradVariant);
        }
    }
    // Debug / diagnostics knobs, off unless their flag is present.
    if (hasflag(argc, argv, "--timing"))
    {
        cfg.timing = true;
    }
    if (hasflag(argc, argv, "--debug-seg"))
    {
        cfg.debugSegments = true;
    }
    if (hasflag(argc, argv, "--profile"))
    {
        cfg.profile = true;
    }
    if (hasflag(argc, argv, "--layer-dump"))
    {
        cfg.layerDump    = true;
        cfg.layerDumpDir = argval(argc, argv, "--layer-dump", cfg.layerDumpDir.c_str());
    }

    // Step 3 - parse the optional --shape override into a Shape (plumbing). An empty string leaves
    // `shape` empty, which later means "use the shape the model declares".
    Shape shape;
    {
        std::string remaining = shapeArg;
        size_t      comma;
        while ((comma = remaining.find(',')) != std::string::npos || !remaining.empty())
        {
            std::string token = remaining.substr(0, comma);
            shape.push_back(std::stoll(token));
            if (comma == std::string::npos)
            {
                break;
            }
            remaining = remaining.substr(comma + 1);
        }
    }

    // Echo what we are about to run so the output is self-describing.
    printf("model=%s backend=%s precision=%s input=%s shape=%s\n", model.c_str(), backendName(cfg.backend), precisionArg.c_str(), inputPath.c_str(), shapeArg.c_str());

    // Step 4 - load the model and build a ready-to-run Session. Runtime::load picks the loader from the
    // file extension (".vxm" = pre-optimized, anything else = ONNX) and resolves the per-model cache.
    std::unique_ptr<Session> session = Runtime::load(model, cfg);
    if (!session)
    {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    // Step 5 (optional) - list the optimized graph. session->graph() exposes the engine's internal IR
    // after its passes have run, so --show-graph reveals what actually executes (fused ops included).
    if (hasflag(argc, argv, "--show-graph"))
    {
        const Graph &graph     = session->graph();
        int          nodeIndex = 0;
        for (const Node &node: graph.nodes)
        {
            TensorId outputId = node.outputs.empty() ? kNoTensor : node.outputs[0];
            printf("  [%3d] %-18s out=%-8s %s act=%d\n", nodeIndex++, opTypeName(node.type), outputId == kNoTensor ? "?" : graph.tensors[outputId].name.c_str(),
                   outputId == kNoTensor ? "" : ::vknn::shapeStr(graph.tensors[outputId].shape).c_str(), (int) node.fusedAct);
        }
    }

    // Step 6 - ask the model what input it expects, then fill the input tensor. The name, shape, and
    // dtype come from the model itself (inputInfo); --shape only needs to be passed to override them.
    IOTensor            input;
    std::vector<IOInfo> modelInputs = session->inputInfo();
    if (!modelInputs.empty())
    {
        input.name  = modelInputs[0].name;
        input.shape = shape.empty() ? modelInputs[0].shape : shape;
    } else
    {
        input.name  = "input";
        input.shape = shape;
    }
    input.dtype = DType::Float32;
    input.data  = readFile(inputPath);
    // The input file must hold at least one fp32 element per shape slot (4 bytes each); a short file is
    // zero-padded so the demo still runs (useful for a smoke test with no real input).
    int64_t neededBytes = numElements(input.shape) * 4;
    if ((int64_t) input.data.size() < neededBytes)
    {
        fprintf(stderr, "input file too small (%zu < %lld); using zeros\n", input.data.size(), (long long) neededBytes);
        input.data.assign(neededBytes, 0);
    }

    // Step 7 - run the model. run() takes the bound inputs and fills `outputs` with every result; a
    // non-Ok status (or no outputs) means the run failed.
    std::vector<IOTensor> outputs;
    Status                status = session->run({input}, outputs);
    if (status != Status::Ok || outputs.empty())
    {
        fprintf(stderr, "run failed\n");
        return 2;
    }

    // Step 8 - print the top-5 classes. The first output is a vector of per-class scores; f32() views
    // its bytes as float. Rank the indices by descending score and print the leaders.
    const float     *scores     = outputs[0].f32();
    int64_t          classCount = numElements(outputs[0].shape);
    std::vector<int> ranking(classCount);
    for (int64_t i = 0; i < classCount; ++i)
    {
        ranking[i] = (int) i;
    }
    // Number of highest-scoring classes to select and print.
    constexpr int64_t kTopK = 5;
    std::partial_sort(ranking.begin(), ranking.begin() + std::min<int64_t>(kTopK, classCount), ranking.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });
    printf("top-5:\n");
    for (int k = 0; k < std::min<int64_t>(kTopK, classCount); ++k)
    {
        printf("  #%d  class %4d  score %.5f\n", k + 1, ranking[k], scores[ranking[k]]);
    }

    // Step 9 (optional) - compare against a golden output. Reads a reference fp32 vector and reports
    // cosine similarity, max absolute error, and whether the top-1 class agrees (a correctness gate).
    if (!goldenPath.empty())
    {
        std::vector<uint8_t> goldenBytes = readFile(goldenPath);
        const float         *golden      = reinterpret_cast<const float *>(goldenBytes.data());
        int64_t              goldenCount = (int64_t) goldenBytes.size() / 4;
        int64_t              count       = std::min(classCount, goldenCount);
        double               dot = 0, normA = 0, normB = 0, maxAbsErr = 0;
        for (int64_t i = 0; i < count; ++i)
        {
            dot += (double) scores[i] * golden[i];
            normA += (double) scores[i] * scores[i];
            normB += (double) golden[i] * golden[i];
            maxAbsErr = std::max(maxAbsErr, (double) std::fabs(scores[i] - golden[i]));
        }
        double cosine    = dot / (std::sqrt(normA) * std::sqrt(normB) + 1e-12);
        int    goldenTop = 0;
        for (int64_t i = 1; i < count; ++i)
        {
            if (golden[i] > golden[goldenTop])
            {
                goldenTop = (int) i;
            }
        }
        printf("golden compare: cosine=%.6f maxAbsErr=%.4e  top1 vknn=%d golden=%d  => %s\n", cosine, maxAbsErr, ranking[0], goldenTop, (ranking[0] == goldenTop && cosine >= 0.99) ? "PASS" : "CHECK");
    }

    // Step 10 (optional) - print the per-op profiler table gathered during the run.
    if (cfg.profile)
    {
        session->profiler().printTable();
    }

    // Step 11 (optional) - latency benchmark: --bench N (warmup 5, then N timed runs).
    int benchN = atoi(argval(argc, argv, "--bench", "0"));
    if (benchN > 0)
    {
        // Bind the input vector ONCE and reuse it; reconstructing {input} each call re-copies the whole
        // input tensor (large mmap/free per run) which is benchmark-harness noise, not engine cost.
        std::vector<IOTensor> inputBatch {input};
        { // cold latency: very first run after session create (matches MNN's no-warmup protocol)
            auto coldStart = std::chrono::high_resolution_clock::now();
            session->run(inputBatch, outputs);
            auto coldEnd = std::chrono::high_resolution_clock::now();
            printf("cold (first run): %.2f ms\n", std::chrono::duration<double, std::milli>(coldEnd - coldStart).count());
        }
        for (int i = 0; i < 5; ++i)
        {
            session->run(inputBatch, outputs); // warmup
        }
        std::vector<double> samples;
        for (int i = 0; i < benchN; ++i)
        {
            auto runStart = std::chrono::high_resolution_clock::now();
            session->run(inputBatch, outputs);
            auto runEnd = std::chrono::high_resolution_clock::now();
            samples.push_back(std::chrono::duration<double, std::milli>(runEnd - runStart).count());
        }
        std::sort(samples.begin(), samples.end());
        double mean = 0;
        for (double v: samples)
        {
            mean += v;
        }
        mean /= samples.size();
        double median = samples[samples.size() / 2];
        double p90    = samples[(size_t) (samples.size() * 0.9)];
        printf("\nbench (%d runs): min=%.2f ms  median=%.2f ms  p90=%.2f ms  => %.1f fps (min) / %.1f fps "
               "(median)\n",
               benchN, samples.front(), median, p90, 1000.0 / samples.front(), 1000.0 / median);
        (void) mean;
    }
    return 0;
}
