// vknn_readme_quickstart - the minimal load-set-run-read program the README links to.
//
// Loads a model (a compiled .vxm or a raw .onnx — Model::load auto-detects), fills the first input
// with data, runs, and reads the first output. Names, shapes, and dtypes all come from the model, so
// the only thing the caller supplies is the input data.
//
//   vknn_readme_quickstart model.vxm          # ramp input, prints the argmax + first values
//   vknn_readme_quickstart model.onnx in.bin  # feed a raw fp32 NCHW .bin as the input
#include "vknn/model.h"
#include <cstdio>
#include <fstream>
#include <vector>

using namespace vknn;

// Read a raw little-endian fp32 file into a float vector (row-major NCHW, the layout the API expects).
// Returns empty on any error.
static std::vector<float> readF32Bin(const char *path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
    {
        return {};
    }
    std::vector<float> v((size_t) f.tellg() / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(v.data()), (std::streamsize) (v.size() * sizeof(float)));
    return v;
}

int main(int argc, char **argv) {
    if (argc < 2)
    {
        printf("usage: %s <model.vxm|model.onnx> [input.bin]\n", argv[0]);
        return 1;
    }

    // Config selects the backend + precision. Vulkan runs on the GPU with a CPU fallback for any op the
    // GPU declines; Precision::Low is fp16 storage with fp32 accumulation. The defaults already are
    // Vulkan + Low, so an empty Config would do — they are spelled out here for clarity.
    Config cfg;
    cfg.backend   = BackendKind::Vulkan;
    cfg.precision = Precision::Low;

    Model net = Model::load(argv[1], cfg); // .vxm loads the compiled plan directly; .onnx imports + optimizes
    if (!net)
    {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    // The model reports its own inputs/outputs; the caller never hand-wires names or shapes.
    std::vector<TensorInfo> in = net.inputs();
    if (in.empty())
    {
        fprintf(stderr, "model has no inputs\n");
        return 1;
    }

    // Fill the first input: either a supplied .bin, or a deterministic ramp sized to the input.
    std::vector<float> data;
    if (argc >= 3)
    {
        data = readF32Bin(argv[2]);
        if ((int64_t) data.size() < in[0].count)
        {
            fprintf(stderr, "input %s has %zu floats, model needs %lld\n", argv[2], data.size(), (long long) in[0].count);
            return 1;
        }
        data.resize((size_t) in[0].count);
    } else
    {
        data.resize((size_t) in[0].count);
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] = (float) (i % 255) / 255.0f - 0.5f;
        }
    }

    Tensor input(std::move(data), in[0].shape, in[0].name);
    std::vector<Tensor> outputs = net.run({input}); // one input in, every output back (named + shaped)
    if (outputs.empty())
    {
        fprintf(stderr, "run produced no outputs\n");
        return 1;
    }

    const Tensor &out = outputs.front();
    printf("output '%s' %s  argmax=%lld  max=%.4f\n", out.name().c_str(), out.shapeString().c_str(), (long long) out.argmax(), out.max());
    return 0;
}
