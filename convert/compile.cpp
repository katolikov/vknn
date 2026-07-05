// vknn_compile — compile an ONNX model into VKNN's optimized ".vxm" format so the runtime can skip
// ONNX parsing AND all graph passes (fusion / constant-fold / shape inference) at load time. With
// --fp16, weights are stored as fp16, which halves the file and, more importantly, the runtime host
// memory (a 965M-param model: 3.85GB -> 1.9GB) — the difference between fitting an 8GB device and
// OOM. The Vulkan layout pass (NC4HW4<->flat) is applied per-target at load, so a .vxm is
// backend-agnostic: compile once on the host, run anywhere.
//
//   vknn_compile <model.onnx> <out.vxm> [flags]
//     --fp16            store weights as fp16 (default: fp32)
//     -O0..-O3 / --opt N  optimization level (default -O1):
//                         O0 = no optional fusion (reference), O1 = swish + pointwise chains
//                         (bit-exact production set), O2/O3 = + experimental SE and dwpw fusions
//     --[no-]fuse-swish / --[no-]fuse-se / --[no-]fuse-dwpw / --[no-]fuse-pointwise
//                       advanced per-fusion overrides applied on top of the level
//     --dump-big        log tensors > 50M elements after shape inference (debug)
#include "import/passes.h"
#include "vknn/dtype.h"
#include "vknn/graph.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace vknn;

/// True iff \p flag appears among the argument vector. The scan starts at index 3 so the program name
/// (argv[0]) and the two required positional args (argv[1]=model.onnx, argv[2]=out.vxm) are never
/// mistaken for flags.
static bool has(int c, char **v, const char *flag) noexcept {
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
        printf("usage: %s <model.onnx> <out.vxm> [--fp16] [-O0..-O3 | --opt N] "
               "[--[no-]fuse-swish] [--[no-]fuse-se] [--[no-]fuse-dwpw] [--[no-]fuse-pointwise] [--dump-big]\n",
               argv[0]);
        return 1;
    }
    std::string onnx = argv[1], out = argv[2];
    bool        fp16 = has(argc, argv, "--fp16");

    int optLevel = 1;
    for (int i = 3; i < argc; ++i)
    {
        if (argv[i][0] == '-' && argv[i][1] == 'O' && argv[i][2] >= '0' && argv[i][2] <= '3' && argv[i][3] == 0)
        {
            optLevel = argv[i][2] - '0';
        } else if (!strcmp(argv[i], "--opt") && i + 1 < argc)
        {
            optLevel = atoi(argv[i + 1]);
        }
    }
    PassOptions opt = PassOptions::forOptLevel(optLevel);
    // per-fusion overrides on top of the level
    auto over = [&](const char *on, const char *off, bool &v) {
        if (has(argc, argv, on))
        {
            v = true;
        }
        if (has(argc, argv, off))
        {
            v = false;
        }
    };
    over("--fuse-swish", "--no-fuse-swish", opt.fuseSwish);
    over("--fuse-se", "--no-fuse-se", opt.fuseSqueezeExcite);
    over("--fuse-dwpw", "--no-fuse-dwpw", opt.fuseDwPw);
    over("--fuse-pointwise", "--no-fuse-pointwise", opt.fusePointwiseChains);
    opt.dumpBig = has(argc, argv, "--dump-big");

    printf("[compile] importing %s ...\n", onnx.c_str());
    Graph g = importOnnx(onnx);
    printf("[compile] %zu nodes, %zu weights. running passes (-O%d: fuse-swish=%d fuse-se=%d fuse-dwpw=%d fuse-pointwise=%d)\n", g.nodes.size(), g.initializers.size(), optLevel, opt.fuseSwish, opt.fuseSqueezeExcite, opt.fuseDwPw,
           opt.fusePointwiseChains);
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
