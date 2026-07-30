// vknn_readme_quickstart - your first VKNN program: load a model, run it once, read the result.
//
// This is the minimal load-set-run-read loop the README links to. It loads a model, fills the first
// input with data, runs the model on the GPU, and prints a summary of the first output. Everything the
// engine needs — tensor names, shapes, and dtypes — is read from the model itself, so the only thing
// this program supplies is the input data.
//
// The VKNN API used here:
//   Config                 - picks the backend + precision (defaults are already Vulkan + Low).
//   Model::load(path, cfg) - loads a compiled .vxm plan OR imports+optimizes a raw .onnx (auto-detected).
//   model.inputs()         - what the model expects: name, shape, dtype, element count per input.
//   Tensor(data, shape, name) - wraps host float data as an input tensor for a named model input.
//   model.run({input})     - runs once and returns every output tensor, each carrying its name + shape.
//   out.argmax() / out.max() - convenience accessors on the returned Tensor (e.g. a classifier's top class).
//
//   vknn_readme_quickstart model.vxm          # ramp input, prints the argmax + first values
//   vknn_readme_quickstart model.onnx in.bin  # feed a raw fp32 NCHW .bin as the input
#include "vknn/model.h"
#include <cstdio>
#include <fstream>
#include <vector>

using namespace vknn;

// --- not VKNN, just plumbing so the demo runs -------------------------------------------------------
// Read a raw little-endian fp32 file into a float vector (row-major NCHW, the layout the API expects).
// This is ordinary file I/O so the example can accept a real input from disk; it is not part of the
// engine. Returns empty on any error.
static std::vector<float> readFloat32Bin(const char *path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        return {};
    }
    std::vector<float> values((size_t) file.tellg() / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(values.data()), (std::streamsize)(values.size() * sizeof(float)));
    return values;
}
// --- end plumbing -----------------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 2)
    {
        printf("usage: %s <model.vxm|model.onnx> [input.bin]\n", argv[0]);
        return 1;
    }

    // Step 1 - choose backend + precision, then load the model.
    // Config selects where and how the model runs: Vulkan runs on the GPU with a CPU fallback for any
    // op the GPU declines, and Precision::Low is fp16 storage with fp32 accumulation. These are already
    // the defaults (an empty Config would behave the same), so they are spelled out here only to show
    // the two knobs a newcomer reaches for first.
    Config config;
    config.backend   = BackendKind::Vulkan;
    config.precision = Precision::Low;

    // Model::load auto-detects the file: a .vxm loads the compiled plan directly, a .onnx is imported
    // and optimized on the spot. The result is a ready-to-run handle.
    Model model = Model::load(argv[1], config);
    if (!model)
    {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    // Step 2 - ask the model what it expects.
    // The model reports its own inputs (name, shape, dtype, element count), so the caller never
    // hand-wires any of that. This example drives the first input.
    std::vector<TensorInfo> modelInputs = model.inputs();
    if (modelInputs.empty())
    {
        fprintf(stderr, "model has no inputs\n");
        return 1;
    }
    const TensorInfo &firstInput = modelInputs.front();

    // Step 3 - build the input data sized exactly to that first input.
    // Either load a caller-supplied .bin from disk, or synthesize a deterministic ramp. Either way the
    // buffer is trimmed/checked against firstInput.count so it matches what the model needs.
    std::vector<float> inputData;
    if (argc >= 3)
    {
        inputData = readFloat32Bin(argv[2]);
        if ((int64_t) inputData.size() < firstInput.count)
        {
            fprintf(stderr, "input %s has %zu floats, model needs %lld\n", argv[2], inputData.size(), (long long) firstInput.count);
            return 1;
        }
        inputData.resize((size_t) firstInput.count);
    } else
    {
        inputData.resize((size_t) firstInput.count);
        for (size_t i = 0; i < inputData.size(); ++i)
        {
            inputData[i] = (float) (i % 255) / 255.0f - 0.5f;
        }
    }

    // Step 4 - wrap the data as a Tensor and run.
    // A Tensor pairs the raw float data with the shape and name of the model input it feeds; run()
    // takes the inputs and returns every output tensor, each already carrying its own name and shape.
    Tensor              input(std::move(inputData), firstInput.shape, firstInput.name);
    std::vector<Tensor> outputs = model.run({input});
    if (outputs.empty())
    {
        fprintf(stderr, "run produced no outputs\n");
        return 1;
    }

    // Step 5 - read the first output.
    // The returned Tensor exposes ready-made accessors: argmax() (the top index, e.g. a predicted
    // class) and max() (its value), plus its own name and shape string.
    const Tensor &output = outputs.front();
    printf("output '%s' %s  argmax=%lld  max=%.4f\n", output.name().c_str(), output.shapeString().c_str(), (long long) output.argmax(), output.max());
    return 0;
}
