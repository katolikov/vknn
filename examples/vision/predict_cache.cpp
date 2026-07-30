// vknn_predict_cache - the plain "load a model, run it, read the answer" example.
//
// This is the everyday inference path: vknn owns the I/O buffers. You hand it ordinary host data
// (row-major fp32, NCHW) and read ordinary host results back — no DMA-BUF, no zero-copy, nothing to
// wire up by hand. It also turns on the per-model cache: on teardown vknn writes a cache file (compiled
// pipelines + prepacked weights + the autotune table) next to the model, so the NEXT load is a fast warm
// start that skips shader compilation and tuning. (For the zero-copy fd variant, see vknn_dmabuf_fd_io.)
//
// The VKNN API used here:
//   - Config                    the knobs a load is compiled with (here: fp16 precision + a cache file).
//   - Model::load(path, cfg)     load + compile the .vxm/.onnx into a ready-to-run Model handle.
//   - model.session()            drop to the lower-level Session for the describe/run calls below.
//   - session->inputInfo()       what the model expects: each input's name, shape, dtype, element count.
//   - Tensor(data, shape, name)  wrap a host fp32 buffer as a named input tensor.
//   - model.run(inputs)          run it — host tensors in, host tensors out; vknn owns the buffers.
//   - Tensor::argmax() / [] etc. read the result (predicted class + the first few raw values).
//   - session->config()          read back the config the session was built with (e.g. the cache path).
//
//   vknn_predict_cache model.vxm
#include "vknn/model.h"
#include "vknn/session.h"
#include <cstdio>
#include <utility>
#include <vector>

using namespace vknn;

int main(int argc, char **argv) {
    // Step 0 - read the single CLI argument. (Not VKNN, just plumbing: the model path to load.)
    if (argc < 2)
    {
        printf("usage: %s model.vxm\n", argv[0]);
        return 1;
    }

    // Step 1 - describe how to load the model. fp16 storage is the fastest / lowest-memory tier, and
    // pointing cacheFile at "<model>.cache" makes vknn persist its warm-start artifacts there so the
    // next load of this model is fast.
    Config config;
    config.precision = Precision::Low;                  // fp16 GPU inference
    config.cacheFile = std::string(argv[1]) + ".cache"; // reused next load for a warm start

    // Step 2 - load + compile the model. The returned handle is falsy if the load failed.
    Model model = Model::load(argv[1], config);
    if (!model)
    {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    // Step 3 - reach the low-level Session, which exposes what the model expects and produces so you
    // never hand-specify tensor names or shapes.
    Session *session = model.session();

    // Step 4 - build one host input tensor per model input, sized straight from the model's own
    // description. The 0.5f fill is NOT VKNN — it is placeholder data so the demo has something to run;
    // a real caller drops its actual row-major fp32 (NCHW) pixels/features in here instead.
    std::vector<Tensor> inputs;
    for (const IOInfo &inputSpec: session->inputInfo())
    {
        std::vector<float> inputData((size_t) inputSpec.elems, 0.5f);
        inputs.push_back(Tensor(std::move(inputData), inputSpec.shape, inputSpec.name));
    }

    // Step 5 - run. Host tensors go in and host tensors come back; vknn allocates and owns the buffers
    // on both sides (the classic path, no DMA-BUF).
    std::vector<Tensor> outputs = model.run(inputs);
    if (outputs.empty())
    {
        fprintf(stderr, "run failed\n");
        return 2;
    }

    // Step 6 - read the first output: its name, element count, first few raw values, and argmax (the
    // predicted class for a classifier). Then report where the warm-start cache was written.
    const Tensor &output = outputs[0];
    printf("output '%s' [%lld values]: %.4f %.4f %.4f %.4f ...  argmax=%lld\n", output.name().c_str(), (long long) output.size(), output[0], output[1], output[2], output[3],
           (long long) output.argmax());
    printf("done (cache: %s)\n", session->config().cacheFile.c_str());
    return 0;
}
