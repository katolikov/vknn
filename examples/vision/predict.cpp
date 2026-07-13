// vknn_predict - the "hello world" of VKNN: load a model, run it once, print the top-1 result.
//
// This is the shortest end-to-end path through the high-level Model API. Hand it an ONNX graph (or a
// pre-optimized .vxm), and it reports what the model expects, feeds it one input, and prints the
// winning class. There are no tensor names, shapes, or dtypes to wire up by hand — the Model reads
// all of that from the graph, so you can run an unfamiliar model without knowing anything about it.
//
// The VKNN API used here:
//   vknn::Model::load(path)   - compile/load a model; picks the Vulkan backend when available, else CPU.
//   model.inputs()            - the model's declared input tensors (name, shape, element count).
//   model.outputs()           - the model's declared output tensors (name, shape).
//   TensorInfo::shapeString() - a tensor's shape as a readable "1x3x224x224" string.
//   model.run(values)         - run a single-input model from a flat fp32 vector; returns the first
//                               output as a Tensor.
//   Tensor::argmax() / max()  - top-1 class index and its score, read straight off the output tensor.
//   model.save(path)          - serialize the optimized model to .vxm for a faster reload next time.
//
// Usage:  vknn_predict <model.onnx|model.vxm> [input.bin] [out.vxm]
//   model      ONNX graph to load (or a .vxm produced by an earlier save() for a faster reload).
//   input.bin  Optional raw little-endian fp32 blob for the first input; missing or short -> zeros.
//   out.vxm    Optional path to save the optimized model for subsequent fast reloads.
#include "vknn/model.h"
#include <cstdio>
#include <fstream>
#include <vector>

// --- not VKNN, just plumbing so the demo runs ---------------------------------------------------
// Fill `values` from a raw little-endian fp32 file. A missing or short file simply leaves the
// remaining elements at their initial value (zero), which is a valid input for a quick smoke test.
// This is ordinary file I/O; nothing here touches the VKNN API.
static void loadRawFloats(const char *path, std::vector<float> &values)
{
    std::ifstream inputFile(path, std::ios::binary);
    if (inputFile)
    {
        inputFile.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(float));
    }
}

int main(int argc, char **argv)
{
    // --- not VKNN, just plumbing: read the command-line arguments -------------------------------
    if (argc < 2)
    {
        printf("usage: %s model.onnx [input.bin]\n", argv[0]);
        return 1;
    }
    const char *modelPath = argv[1];
    const char *inputPath = (argc >= 3) ? argv[2] : nullptr; // optional raw fp32 input blob
    const char *savePath  = (argc >= 4) ? argv[3] : nullptr; // optional destination for the .vxm save

    // Step 1 - load the model. VKNN compiles the ONNX graph (or reloads a .vxm) and selects the
    // Vulkan backend when the device has one, falling back to CPU otherwise. The returned handle is
    // falsy when the load fails, so guard on it before using the model.
    vknn::Model model = vknn::Model::load(modelPath);
    if (!model)
    {
        printf("failed to load %s\n", modelPath);
        return 1;
    }

    // Step 2 - see what the model expects and produces. This is purely informational: the Model
    // already knows every name/shape/dtype, so you never have to set them. Printing them here just
    // shows a newcomer the shape of the graph they just loaded.
    for (const vknn::TensorInfo &modelInput : model.inputs())
    {
        printf("input  '%s'  %s  (%lld values)\n", modelInput.name.c_str(), modelInput.shapeString().c_str(), (long long) modelInput.count);
    }
    for (const vknn::TensorInfo &modelOutput : model.outputs())
    {
        printf("output '%s'  %s\n", modelOutput.name.c_str(), modelOutput.shapeString().c_str());
    }

    // Step 3 - build the input buffer. The Model reported the first input's element count, so size a
    // flat fp32 vector to match and start it at zeros. When the caller passes an input.bin, fill as
    // many values as the file provides (the reading itself is plain I/O, not VKNN — see loadRawFloats).
    int64_t            inputElementCount = model.inputs().empty() ? 0 : model.inputs()[0].count;
    std::vector<float> inputData(inputElementCount, 0.f);
    if (inputPath)
    {
        loadRawFloats(inputPath, inputData);
    }

    // Step 4 - run the model. The vector-in overload feeds a single-input model and hands back its
    // first output tensor; argmax()/max() read the top-1 class index and its score right off it.
    vknn::Tensor output = model.run(inputData);
    printf("result: shape=%s  top1=%lld  max=%.4f\n", output.shapeString().c_str(), (long long) output.argmax(), output.max());

    // Step 5 - optionally save the optimized model. Writing a .vxm lets a later Model::load() on that
    // path skip ONNX parsing and the graph passes, for a faster warm start.
    if (savePath)
    {
        if (model.save(savePath))
        {
            printf("saved optimized model -> %s\n", savePath);
        }
    }
    return 0;
}
