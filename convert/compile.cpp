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
//                         O0 = no optional fusion (reference), O1 = the general pointwise
//                         fusion (bit-exact production set), O2/O3 = + experimental SE and dwpw
//     --[no-]fuse-se / --[no-]fuse-dwpw / --[no-]fuse-pointwise / --[no-]lower-conv
//     --strict-fuse     rounded fusion steps everywhere: fused == unfused byte-identical (the byte
//                       gate); the default fast mode fp32-chains each fused unit and rounds once
//                       per stored stream — faster, and at least as accurate as the unfused graph
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
               "[--[no-]fuse-se] [--[no-]fuse-dwpw] [--[no-]fuse-pointwise] [--[no-]strict-fuse] [--[no-]lower-conv] [--dump-big]\n",
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
    over("--fuse-se", "--no-fuse-se", opt.fuseSqueezeExcite);
    over("--fuse-dwpw", "--no-fuse-dwpw", opt.fuseDwPw);
    over("--fuse-pointwise", "--no-fuse-pointwise", opt.fusePointwiseChains);
    over("--strict-fuse", "--no-strict-fuse", opt.strictFuse);
    over("--lower-conv", "--no-lower-conv", opt.lowerConv);
    opt.dumpBig = has(argc, argv, "--dump-big");

    printf("[compile] importing %s ...\n", onnx.c_str());
    Graph g = importOnnx(onnx);
    printf("[compile] %zu nodes, %zu weights. running passes (-O%d: fuse-se=%d fuse-dwpw=%d fuse-pointwise=%d lower-conv=%d)\n", g.nodes.size(), g.initializers.size(), optLevel, opt.fuseSqueezeExcite, opt.fuseDwPw,
           opt.fusePointwiseChains, opt.lowerConv);
    runStandardPasses(g, opt);
    printf("[compile] post-passes: %zu nodes, %zu weights\n", g.nodes.size(), g.initializers.size());

    if (fp16)
    {
        Fp16ConvertStats st = convertInitializersFp16(g);
        printf("[compile] fp16: converted %lld weights (%lld kept non-fp32), %.0f MB -> %.0f MB\n", (long long) st.converted, (long long) st.kept, st.bytesBefore / 1e6, st.bytesAfter / 1e6);
    }

    if (!saveGraphBin(g, out))
    {
        printf("[compile] save failed\n");
        return 2;
    }
    printf("[compile] wrote %s\n", out.c_str());
    return 0;
}
