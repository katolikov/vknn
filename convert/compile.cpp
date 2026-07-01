// vknn_compile — compile an ONNX model into VKNN's optimized ".vxm" format so the runtime can skip
// ONNX parsing AND all graph passes (fusion / constant-fold / shape inference) at load time. With
// --fp16, weights are stored as fp16, which halves the file and, more importantly, the runtime host
// memory (a 965M-param model: 3.85GB -> 1.9GB) — the difference between fitting an 8GB device and
// OOM. The Vulkan layout pass (NC4HW4<->flat) is applied per-target at load, so a .vxm is
// backend-agnostic: compile once on the host, run anywhere.
//
//   vknn_compile <model.onnx> <out.vxm> [flags]
//     --fp16            store weights as fp16 (default: fp32)
//     --no-fuse-swish   disable HardSwish/SiLU -> conv-epilogue fusion (default: on)
//     --fuse-se         fuse the Squeeze-Excite chain (experimental, default: off)
//     --fuse-dwpw       fuse depthwise-3x3 + 1x1-project (experimental, default: off)
//     --no-fuse-pointwise  disable pointwise-chain fusion into one kernel (default: on)
//     --dump-big        log tensors > 50M elements after shape inference (debug)
#include "import/passes.h"
#include "vknn/dtype.h"
#include "vknn/graph.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vknn;

static bool has(int c, char **v, const char *flag) {
    for (int i = 3; i < c; ++i)
    {
        if (!strcmp(v[i], flag))
        {
            return true;
        }
    }
    return false;
}

int main(int argc, char **argv) {
    if (argc < 3)
    {
        printf("usage: %s <model.onnx> <out.vxm> [--fp16] [--no-fuse-swish] [--fuse-se] [--fuse-dwpw] "
               "[--no-fuse-pointwise] [--dump-big]\n",
               argv[0]);
        return 1;
    }
    std::string onnx = argv[1], out = argv[2];
    bool        fp16 = has(argc, argv, "--fp16");

    PassOptions opt;
    opt.fuseSwish           = !has(argc, argv, "--no-fuse-swish");
    opt.fuseSqueezeExcite   = has(argc, argv, "--fuse-se");
    opt.fuseDwPw            = has(argc, argv, "--fuse-dwpw");
    opt.fusePointwiseChains = !has(argc, argv, "--no-fuse-pointwise");
    opt.dumpBig             = has(argc, argv, "--dump-big");

    printf("[compile] importing %s ...\n", onnx.c_str());
    Graph g = importOnnx(onnx);
    printf("[compile] %zu nodes, %zu weights. running passes (fuse-swish=%d fuse-se=%d fuse-dwpw=%d fuse-pointwise=%d)\n", g.nodes.size(), g.initializers.size(), opt.fuseSwish, opt.fuseSqueezeExcite, opt.fuseDwPw, opt.fusePointwiseChains);
    runStandardPasses(g, opt);
    printf("[compile] post-passes: %zu nodes, %zu weights\n", g.nodes.size(), g.initializers.size());

    if (fp16)
    {
        int64_t before = 0, after = 0, conv = 0, kept = 0;
        for (auto &kv: g.initializers)
        {
            TensorDesc &d = g.tensors[kv.first];
            before += (int64_t) kv.second.bytes.size();
            if (d.dtype != DType::Float32)
            { // int64 shape tensors, etc. — leave as-is
                after += (int64_t) kv.second.bytes.size();
                ++kept;
                continue;
            }
            int64_t              n   = numElements(d.shape);
            const float         *src = kv.second.f32();
            std::vector<uint8_t> half((size_t) n * 2);
            fp16_t              *h = reinterpret_cast<fp16_t *>(half.data());
            for (int64_t i = 0; i < n; ++i)
            {
                h[i] = floatToHalf(src[i]);
            }
            kv.second.bytes = std::move(half);
            d.dtype         = DType::Float16;
            after += (int64_t) kv.second.bytes.size();
            ++conv;
        }
        printf("[compile] fp16: converted %lld weights (%lld kept non-fp32), %.0f MB -> %.0f MB\n", (long long) conv, (long long) kept, before / 1e6, after / 1e6);
    }

    if (!saveGraphBin(g, out))
    {
        printf("[compile] save failed\n");
        return 2;
    }
    printf("[compile] wrote %s\n", out.c_str());
    return 0;
}
