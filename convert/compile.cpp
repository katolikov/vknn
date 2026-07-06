// vknn_compile — compile an ONNX model into VKNN's optimized ".vxm" format so the runtime can skip
// ONNX parsing AND all graph passes (fusion / constant-fold / shape inference) at load time. With
// --fp16, weights are stored as fp16, which halves the file and, more importantly, the runtime host
// memory (a 965M-param model: 3.85GB -> 1.9GB) — the difference between fitting an 8GB device and
// OOM. The Vulkan layout pass (NC4HW4<->flat) is applied per-target at load, so a .vxm is
// backend-agnostic: compile once on the host, run anywhere.
//
//   vknn_compile <model.onnx|model.vxm> <out.vxm> [flags]
//     --fp16            store weights as fp16 (default: fp32)
//     --batch N         resolve a dynamic LEADING (batch) axis to N (default 1)
//     --shape NAME=D0xD1x...  declare a graph input's full concrete shape (repeatable), resolving
//                       every dynamic axis of that input, e.g. --shape pixel_values=1x3x224x224. An
//                       undeclared dynamic non-batch axis is a hard error (never a silent 1x1 plan).
//     --bucket "NAME=D0x...;NAME2=..."  declare ONE shape bucket per occurrence (repeatable): the
//                       model is compiled once per bucket over a fresh import and the buckets share
//                       one initializer pool in a multi-bucket .vxm. --batch/--shape are the shared
//                       fallback. With no --bucket, a single bucket is written (legacy .vxm bytes).
//     -O0..-O3 / --opt N  optimization level (default -O1):
//                         O0 = no optional fusion (reference), O1 = the general pointwise
//                         fusion (bit-exact production set), O2/O3 = + experimental SE and dwpw
//     --[no-]fuse-se / --[no-]fuse-dwpw / --[no-]fuse-pointwise / --[no-]lower-conv
//     --no-dequantize   keep QuantizeLinear/DequantizeLinear ops instead of folding DQ weights and
//                       collapsing matching QDQ sandwiches (default: quantized checkpoints compile
//                       to plain float graphs and run dequantized)
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

/// Parse a `--shape NAME=D0xD1x...` value (e.g. "pixel_values=1x3x224x224") into the input name and
/// its concrete dims. Dims are 'x'-separated non-negative integers. Returns false (and leaves *name /
/// *shape unspecified) on a malformed spec: no '=', an empty name, an empty/garbage dim, or a negative
/// dim -- a declared shape must be fully concrete.
static bool parseShapeSpec(const char *spec, std::string *name, Shape *shape) {
    const char *eq = strchr(spec, '=');
    if (!eq || eq == spec)
    {
        return false;
    }
    name->assign(spec, eq);
    shape->clear();
    const char *p = eq + 1;
    if (*p == 0)
    {
        return false;
    }
    while (*p)
    {
        char *end = nullptr;
        long  v   = strtol(p, &end, 10);
        if (end == p || v < 0)
        {
            return false; // no digits, or a negative/dynamic dim
        }
        shape->push_back((int64_t) v);
        p = end;
        if (*p == 'x' || *p == 'X')
        {
            ++p;
            if (*p == 0)
            {
                return false; // trailing separator
            }
        } else if (*p != 0)
        {
            return false; // unexpected char between dims
        }
    }
    return true;
}

/// Parse a `--bucket` value -- a ';'-separated list of `NAME=D0xD1x...` input-shape specs, e.g.
/// "pixel_values=1x3x224x224;mask=1x224x224" -- into a per-input shape map. Every segment must be a
/// valid shape spec (see parseShapeSpec); an empty value or any malformed segment fails. Returns
/// false and leaves *shapes unspecified on error.
static bool parseBucketSpec(const char *spec, std::map<std::string, Shape> *shapes) {
    shapes->clear();
    std::string s(spec);
    if (s.empty())
    {
        return false;
    }
    size_t start = 0;
    while (start <= s.size())
    {
        size_t      semi = s.find(';', start);
        std::string seg  = s.substr(start, semi == std::string::npos ? std::string::npos : semi - start);
        if (seg.empty())
        {
            return false; // empty segment (leading/trailing/double ';')
        }
        std::string name;
        Shape       shape;
        if (!parseShapeSpec(seg.c_str(), &name, &shape))
        {
            return false;
        }
        (*shapes)[name] = shape;
        if (semi == std::string::npos)
        {
            break;
        }
        start = semi + 1;
    }
    return true;
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
        printf("usage: %s <model.onnx|model.vxm> <out.vxm> [--fp16] [--batch N] [--shape NAME=D0xD1x...] [--bucket \"NAME=...;NAME2=...\"] [-O0..-O3 | --opt N] "
               "[--[no-]fuse-se] [--[no-]fuse-dwpw] [--[no-]fuse-pointwise] [--[no-]strict-fuse] [--[no-]lower-conv] [--no-dequantize] [--support-report <out.json>] [--dump-big]\n",
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
    over("--dequantize", "--no-dequantize", opt.dequantize);
    opt.dumpBig = has(argc, argv, "--dump-big");

    // Resolve dynamic input dims. --batch N sets the fallback substituted into a dynamic LEADING
    // (batch) axis; --shape NAME=D0xD1x... (repeatable) declares an input's full concrete shape and
    // resolves EVERY dynamic axis of that input. An undeclared dynamic non-batch axis is a hard error
    // in the passes (inferShapes), never a silent 1x1 plan.
    //
    // --bucket "NAME=D0xD1x...;NAME2=..." (repeatable) declares ONE shape bucket per occurrence: the
    // whole model is compiled once per bucket over a fresh import (the passes mutate the graph
    // irreversibly), and the buckets share one initializer pool in a multi-bucket .vxm. With no
    // --bucket, exactly one bucket is produced from --shape/--batch (a legacy single-graph .vxm).
    // --batch/--shape act as the shared fallback under every bucket.
    std::vector<std::map<std::string, Shape>> bucketShapes; // one entry per --bucket occurrence
    std::vector<std::string>                  bucketLabels; // the raw --bucket value, as the bucket's label
    for (int i = 3; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--batch") && i + 1 < argc)
        {
            opt.batch = atoll(argv[i + 1]);
        } else if (!strcmp(argv[i], "--shape") && i + 1 < argc)
        {
            std::string name;
            Shape       shape;
            if (!parseShapeSpec(argv[i + 1], &name, &shape))
            {
                printf("[compile] bad --shape '%s' (expected NAME=D0xD1x..., non-negative dims)\n", argv[i + 1]);
                return 1;
            }
            opt.inputShapes[name] = shape;
        } else if (!strcmp(argv[i], "--bucket") && i + 1 < argc)
        {
            std::map<std::string, Shape> shapes;
            if (!parseBucketSpec(argv[i + 1], &shapes))
            {
                printf("[compile] bad --bucket '%s' (expected NAME=D0xD1x...[;NAME2=...], non-negative dims)\n", argv[i + 1]);
                return 1;
            }
            bucketShapes.push_back(std::move(shapes));
            bucketLabels.emplace_back(argv[i + 1]);
        }
    }

    const bool vxmInput = onnx.size() > 4 && onnx.compare(onnx.size() - 4, 4, ".vxm") == 0;

    // Multi-bucket compile: one full import+passes product per --bucket occurrence, sharing one
    // initializer pool. Requires an ONNX input (a .vxm has its passes already baked at one shape).
    if (!bucketShapes.empty())
    {
        if (vxmInput)
        {
            printf("[compile] --bucket needs an ONNX input (a .vxm is already compiled at one shape)\n");
            return 2;
        }
        std::vector<Graph>       buckets;
        std::vector<std::string> names;
        for (size_t b = 0; b < bucketShapes.size(); ++b)
        {
            // Fresh import per bucket: runStandardPasses mutates the graph irreversibly, so each
            // bucket gets its own graph. bopt starts from the shared options (--batch and any base
            // --shape declarations); the bucket's own declarations override per input.
            PassOptions bopt = opt;
            for (const auto &kv: bucketShapes[b])
            {
                bopt.inputShapes[kv.first] = kv.second;
            }
            printf("[compile] bucket %zu '%s': importing %s ...\n", b, bucketLabels[b].c_str(), onnx.c_str());
            Graph gb;
            try
            {
                gb = importOnnx(onnx);
                runStandardPasses(gb, bopt);
            } catch (const Error &e)
            {
                printf("[compile] bucket %zu '%s': %s\n", b, bucketLabels[b].c_str(), e.what());
                return 2;
            }
            if (fp16)
            {
                convertInitializersFp16(gb);
            }
            printf("[compile] bucket %zu '%s': post-passes %zu nodes, %zu weights\n", b, bucketLabels[b].c_str(), gb.nodes.size(), gb.initializers.size());
            buckets.push_back(std::move(gb));
            names.push_back(bucketLabels[b]);
        }
        if (!supportReport.empty())
        {
            // The support report describes one graph; emit bucket 0's assignment.
            if (!writeSupportReport(buckets.front(), onnx, supportReport))
            {
                printf("[compile] support report write failed\n");
                return 2;
            }
            printf("[compile] wrote support report %s (bucket 0)\n", supportReport.c_str());
        }
        if (!saveGraphBinBuckets(buckets, names, out))
        {
            printf("[compile] save failed\n");
            return 2;
        }
        printf("[compile] wrote %s (%zu buckets)\n", out.c_str(), buckets.size());
        return 0;
    }

    Graph g;
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
        try
        {
            g = importOnnx(onnx);
            printf("[compile] %zu nodes, %zu weights. running passes (-O%d: fuse-se=%d fuse-dwpw=%d fuse-pointwise=%d lower-conv=%d)\n", g.nodes.size(), g.initializers.size(), optLevel, opt.fuseSqueezeExcite, opt.fuseDwPw,
                   opt.fusePointwiseChains, opt.lowerConv);
            runStandardPasses(g, opt);
        } catch (const Error &e)
        {
            // A dynamic non-batch input axis with no --shape declaration lands here (as does any other
            // import/pass failure). Report it and exit nonzero instead of aborting on the exception.
            printf("[compile] %s\n", e.what());
            return 2;
        }
        printf("[compile] post-passes: %zu nodes, %zu weights\n", g.nodes.size(), g.initializers.size());
    }

    if (fp16)
    {
        Fp16ConvertStats st = convertInitializersFp16(g);
        printf("[compile] fp16: converted %lld weights (%lld kept non-fp32), %.0f MB -> %.0f MB\n", (long long) st.converted, (long long) st.kept, st.bytesBefore / 1e6, st.bytesAfter / 1e6);
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
