// vknn_compile — compile an ONNX model into VKNN's optimized ".vxm" format so the runtime can skip
// ONNX parsing AND all graph passes (fusion / constant-fold / shape inference) at load time. With
// --fp16, weights are stored as fp16, which halves the file and, more importantly, the runtime host
// memory (a 965M-param model: 3.85GB -> 1.9GB) — the difference between fitting an 8GB device and
// OOM. The Vulkan layout pass (NC4HW4<->flat) is applied per-target at load, so a .vxm is
// backend-agnostic: compile once on the host, run anywhere.
//
//   vknn_compile <model.onnx|model.vxm> <out.vxm> [flags]
//     --fp16            store weights as fp16 (default: fp32)
//     -O0..-O3 / --opt N  optimization level (default -O1):
//                         O0 = no optional fusion (reference), O1 = the general pointwise
//                         fusion (bit-exact production set), O2/O3 = + experimental SE and dwpw
//     --[no-]fuse-se / --[no-]fuse-dwpw / --[no-]fuse-pointwise / --[no-]lower-conv
//     --strict-fuse     rounded fusion steps everywhere: fused == unfused byte-identical (the byte
//                       gate); the default fast mode fp32-chains each fused unit and rounds once
//                       per stored stream — faster, and at least as accurate as the unfused graph
//     --support-report <out.json>  write the per-node backend assignment (node, op, backend, and
//                       the refusal reason for every CPU node) computed by the engine's own
//                       capability model (vkSupportSurvey — the exact gate code the device runs).
//                       Consumed by tools/check_model_support.py --engine-report
//     --dump-big        log tensors > 50M elements after shape inference (debug)
//
// A ".vxm" input skips import + passes (they ran at its compile time) and serves the remaining
// stages — the support report and the save — from the stored graph.
#include "core/vk_gates.h"
#include "import/passes.h"
#include "vknn/dtype.h"
#include "vknn/graph.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

/// JSON string escaping for the support report (quotes, backslashes, control characters).
static std::string jsonEscape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c: s)
    {
        if (c == '"' || c == '\\')
        {
            out += '\\';
            out += c;
        } else if ((unsigned char) c < 0x20)
        {
            char buf[8];
            snprintf(buf, sizeof buf, "\\u%04x", (unsigned) (unsigned char) c);
            out += buf;
        } else
        {
            out += c;
        }
    }
    return out;
}

/// Writes the per-node backend assignment of `g` as JSON: the model path, one row per node
/// ({name, op, backend, reason when not on the GPU}), and summary counts (per backend + per
/// reason). The assignment comes from vkSupportSurvey — the same gate code the device engine
/// evaluates in supportsNode — so the report cannot drift from the engine.
static bool writeSupportReport(const Graph &g, const std::string &modelPath, const std::string &outPath) {
    std::vector<NodeSupport> rows = vkSupportSurvey(g);
    FILE                    *f    = fopen(outPath.c_str(), "w");
    if (!f)
    {
        return false;
    }
    fprintf(f, "{\n  \"model\": \"%s\",\n  \"nodes\": [\n", jsonEscape(modelPath).c_str());
    std::map<std::string, size_t> byBackend, byReason;
    for (size_t i = 0; i < rows.size(); ++i)
    {
        const NodeSupport &r = rows[i];
        ++byBackend[r.backend];
        fprintf(f, "    {\"name\": \"%s\", \"op\": \"%s\", \"backend\": \"%s\"", jsonEscape(r.node).c_str(), jsonEscape(r.op).c_str(), jsonEscape(r.backend).c_str());
        if (!r.reason.empty())
        {
            ++byReason[r.reason];
            fprintf(f, ", \"reason\": \"%s\"", jsonEscape(r.reason).c_str());
        }
        fprintf(f, "}%s\n", i + 1 < rows.size() ? "," : "");
    }
    fprintf(f, "  ],\n  \"summary\": {\n    \"total\": %zu", rows.size());
    for (const auto &kv: byBackend)
    {
        fprintf(f, ",\n    \"%s\": %zu", jsonEscape(kv.first).c_str(), kv.second);
    }
    fprintf(f, ",\n    \"reasons\": {");
    bool first = true;
    for (const auto &kv: byReason)
    {
        fprintf(f, "%s\n      \"%s\": %zu", first ? "" : ",", jsonEscape(kv.first).c_str(), kv.second);
        first = false;
    }
    fprintf(f, "%s}\n  }\n}\n", byReason.empty() ? "" : "\n    ");
    return fclose(f) == 0;
}

int main(int argc, char **argv) {
    if (argc < 3)
    {
        printf("usage: %s <model.onnx|model.vxm> <out.vxm> [--fp16] [-O0..-O3 | --opt N] "
               "[--[no-]fuse-se] [--[no-]fuse-dwpw] [--[no-]fuse-pointwise] [--[no-]strict-fuse] [--[no-]lower-conv] [--support-report <out.json>] [--dump-big]\n",
               argv[0]);
        return 1;
    }
    std::string onnx = argv[1], out = argv[2];
    bool        fp16 = has(argc, argv, "--fp16");

    int         optLevel = 1;
    std::string supportReport;
    for (int i = 3; i < argc; ++i)
    {
        if (argv[i][0] == '-' && argv[i][1] == 'O' && argv[i][2] >= '0' && argv[i][2] <= '3' && argv[i][3] == 0)
        {
            optLevel = argv[i][2] - '0';
        } else if (!strcmp(argv[i], "--opt") && i + 1 < argc)
        {
            optLevel = atoi(argv[i + 1]);
        } else if (!strcmp(argv[i], "--support-report") && i + 1 < argc)
        {
            supportReport = argv[i + 1];
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

    Graph      g;
    const bool vxmInput = onnx.size() > 4 && onnx.compare(onnx.size() - 4, 4, ".vxm") == 0;
    if (vxmInput)
    {
        printf("[compile] loading %s (.vxm input: passes already applied at its compile time)\n", onnx.c_str());
        if (!loadGraphBin(g, onnx))
        {
            printf("[compile] load failed\n");
            return 2;
        }
        printf("[compile] %zu nodes, %zu weights\n", g.nodes.size(), g.initializers.size());
    } else
    {
        printf("[compile] importing %s ...\n", onnx.c_str());
        g = importOnnx(onnx);
        printf("[compile] %zu nodes, %zu weights. running passes (-O%d: fuse-se=%d fuse-dwpw=%d fuse-pointwise=%d lower-conv=%d)\n", g.nodes.size(), g.initializers.size(), optLevel, opt.fuseSqueezeExcite, opt.fuseDwPw,
               opt.fusePointwiseChains, opt.lowerConv);
        runStandardPasses(g, opt);
        printf("[compile] post-passes: %zu nodes, %zu weights\n", g.nodes.size(), g.initializers.size());
    }

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

    if (!supportReport.empty())
    {
        if (!writeSupportReport(g, onnx, supportReport))
        {
            printf("[compile] support report write failed\n");
            return 2;
        }
        printf("[compile] wrote support report %s\n", supportReport.c_str());
    }

    if (!saveGraphBin(g, out))
    {
        printf("[compile] save failed\n");
        return 2;
    }
    printf("[compile] wrote %s\n", out.c_str());
    return 0;
}
