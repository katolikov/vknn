// vknn_predict_cache - normal (non-zero-copy) inference: fp16 on the GPU, with a warm per-model cache.
//
// The counterpart to vknn_zerocopy_simple. Here vknn owns the I/O buffers: the caller hands ordinary
// host data (row-major fp32, NCHW) and reads ordinary host outputs — the classic path, no DMA-BUF. The
// unified per-model cache file (pipeline + prepacked weights + autotune) is written on teardown and
// reused on the next load for a fast warm start.
//
//   vknn_predict_cache model.vxm
#include "vknn/model.h"
#include "vknn/session.h"
#include <cstdio>
#include <utility>
#include <vector>

using namespace vknn;

int main(int argc, char **argv) {
    if (argc < 2)
    {
        printf("usage: %s model.vxm\n", argv[0]);
        return 1;
    }
    Config cfg;
    cfg.precision = Precision::Fp16;                 // fp16 GPU inference
    cfg.cacheFile = std::string(argv[1]) + ".cache"; // reused next load for a warm start
    Model net     = Model::load(argv[1], cfg);
    if (!net)
    {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    Session *sess = net.session();

    // one host input tensor per model input — plain row-major fp32 (NCHW); fill with your real data.
    std::vector<Tensor> inputs;
    for (const auto &info: sess->inputInfo())
    {
        std::vector<float> data((size_t) info.elems, 0.5f);
        inputs.push_back(Tensor(std::move(data), info.shape, info.name));
    }

    std::vector<Tensor> outs = net.run(inputs); // host in, host out — vknn owns the buffers
    if (outs.empty())
    {
        fprintf(stderr, "run failed\n");
        return 2;
    }
    const Tensor &y = outs[0];
    printf("output '%s' [%lld values]: %.4f %.4f %.4f %.4f ...  argmax=%lld\n", y.name().c_str(), (long long) y.size(), y[0], y[1], y[2], y[3], (long long) y.argmax());
    printf("done (cache: %s)\n", sess->config().cacheFile.c_str());
    return 0;
}
