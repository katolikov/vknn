// vknn_compile — compile an ONNX model into VKNN's optimized ".vxm" format so the runtime can skip
// ONNX parsing AND all graph passes (fusion / constant-fold / shape inference) at load time. With
// --fp16, weights are stored as fp16, which halves the file and, more importantly, the runtime host
// memory (a 965M-param model: 3.85GB -> 1.9GB) — the difference between fitting an 8GB device and
// OOM. The Vulkan layout pass (NC4HW4<->flat) is applied per-target at load, so a .vxm is
// backend-agnostic: compile once on the host, run anywhere.
//
//   vknn_compile <model.onnx|model.vxm> <out.vxm> [flags]
//     --fp16            store weights as fp16 (default: fp32)
//     --batch N         resolve a dynamic LEADING batch axis to N (default 1). Applies to a leading
//                       axis that is unnamed or batch-named ("N"/"B"/*batch*); a leading axis with any
//                       other dim_param name (num_frames, views...) must be bound with --dim — an
//                       unbound one is a hard error, never a silent freeze to 1
//     --dim NAME=VALUE  bind a symbolic input dimension by its ONNX dim_param name (repeatable), e.g.
//                       --dim sequence_length=1 --dim past_sequence_length=256. Every dynamic input axis
//                       whose dim_param resolves from the bindings — a bare symbol, an integer literal, or
//                       a compound like "past_sequence_length + sequence_length" — is filled automatically,
//                       so a many-input decoder needs a couple of --dim instead of one --shape per tensor.
//     --list-dims       import the model and print its free symbolic input dimensions (the names to bind
//                       with --dim), then exit without compiling.
//     --shape NAME=D0xD1x...  declare a graph input's full concrete shape (repeatable), resolving
//                       every dynamic axis of that input, e.g. --shape pixel_values=1x3x224x224. Overrides
//                       --dim for that tensor. An undeclared, unbound dynamic non-batch axis is a hard
//                       error (never a silent 1x1 plan).
//     --bucket "NAME=D0x...;dim:NAME2=VALUE;..."  declare ONE shape bucket per occurrence (repeatable):
//                       the model is compiled once per bucket over a fresh import and the buckets share
//                       one initializer pool in a multi-bucket .vxm. A segment is either a per-tensor
//                       "NAME=D0xD1x..." shape or a "dim:NAME=VALUE" symbolic binding. --batch/--shape/--dim
//                       are the shared fallback. With no --bucket, a single bucket is written (legacy bytes).
//
//   vknn_compile <out.vxm> --graph "FILE.onnx[;NAME=D0x...;dim:NAME2=VALUE;...]" [--graph ...] [flags]
//     Multi-graph form (no positional input model): each --graph occurrence compiles ONE bucket from
//     its own source file with its own shape/dim segments (same segment syntax as --bucket), layered
//     over the shared --batch/--shape/--dim fallback. Occurrences may repeat a file at different
//     shapes (a decoder's prefill + decode plans) or name different files (a vision tower + its
//     decoder); all buckets land in ONE multi-bucket .vxm over a content-deduped initializer pool,
//     and the runtime dispatches each run() to the bucket matching its bound input names+shapes.
//     -O0..-O3 / --opt N  optimization level (default -O1):
//                         O0 = no optional fusion (reference), O1 = the general pointwise
//                         fusion (bit-exact production set), O2/O3 = + experimental SE and dwpw
//     --[no-]fuse-se / --[no-]fuse-dwpw / --[no-]fuse-pointwise / --[no-]fuse-gridsample-warp / --[no-]lower-conv
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
#include "import/dim_expr.h"
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

/// True iff \p flag appears among the argument vector at or past index \p start. The start skips the
/// positional args so they are never mistaken for flags: 3 for the legacy `<model> <out>` form
/// (argv[1]=model.onnx, argv[2]=out.vxm), 2 for the `--graph` form (argv[1]=out.vxm only).
static bool has(int c, char **v, const char *flag, int start) noexcept {
    for (int i = start; i < c; ++i)
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

/// Parse a `--dim NAME=VALUE` spec (e.g. "past_sequence_length=256") into the symbol name and its
/// non-negative integer value. Returns false on a malformed spec: no '=', an empty name, or a
/// non-numeric / negative / trailing-garbage value -- a bound dim must be a single concrete extent.
static bool parseDimSpec(const char *spec, std::string *name, int64_t *value) {
    const char *eq = strchr(spec, '=');
    if (!eq || eq == spec)
    {
        return false;
    }
    name->assign(spec, eq);
    const char *p = eq + 1;
    if (*p == 0)
    {
        return false;
    }
    char     *end = nullptr;
    long long v   = strtoll(p, &end, 10);
    if (end == p || *end != 0 || v < 0)
    {
        return false;
    }
    *value = (int64_t) v;
    return true;
}

/// One --bucket occurrence: per-tensor concrete shapes plus symbolic-dim bindings, either of which may
/// be empty. A bucket segment is a "NAME=D0xD1x..." shape or a "dim:NAME=VALUE" symbol binding.
struct BucketSpec {
    std::map<std::string, Shape>   shapes;
    std::map<std::string, int64_t> dims;
};

/// Parse a `--bucket` value -- a ';'-separated list of segments, each either a `NAME=D0xD1x...` per-tensor
/// shape (see parseShapeSpec) or a `dim:NAME=VALUE` symbolic binding (see parseDimSpec), e.g.
/// "input_ids=1x1;dim:past_sequence_length=256". An empty value or any malformed/empty segment fails.
/// Returns false and leaves *out unspecified on error.
static bool parseBucketSpec(const char *spec, BucketSpec *out) {
    out->shapes.clear();
    out->dims.clear();
    std::string s(spec);
    if (s.empty())
    {
        return false;
    }
    static const char    kDimPrefix[]  = "dim:";
    constexpr size_t     kDimPrefixLen = 4; // strlen("dim:")
    size_t               start         = 0;
    while (start <= s.size())
    {
        size_t      semi = s.find(';', start);
        std::string seg  = s.substr(start, semi == std::string::npos ? std::string::npos : semi - start);
        if (seg.empty())
        {
            return false; // empty segment (leading/trailing/double ';')
        }
        if (seg.compare(0, kDimPrefixLen, kDimPrefix) == 0)
        {
            std::string name;
            int64_t     value = 0;
            if (!parseDimSpec(seg.c_str() + kDimPrefixLen, &name, &value))
            {
                return false;
            }
            out->dims[name] = value;
        } else
        {
            std::string name;
            Shape       shape;
            if (!parseShapeSpec(seg.c_str(), &name, &shape))
            {
                return false;
            }
            out->shapes[name] = shape;
        }
        if (semi == std::string::npos)
        {
            break;
        }
        start = semi + 1;
    }
    return true;
}

/// Print each graph input's shape -- a concrete extent, or `$symbol` for a dynamic axis -- and the
/// distinct set of dim_param symbols the model leaves free (the names to bind with --dim). A
/// batch-NAMED symbol ("N"/"B"/*batch*) on a leading axis is omitted: that axis honors --batch. Any
/// other leading symbol is must-bind, exactly like shape resolution treats it. Used by --list-dims
/// to answer "what do I need to bind?" without compiling.
static void listDims(const Graph &g) {
    const std::map<std::string, int64_t> noBind;
    std::vector<std::string>             freeSyms; // distinct, first-seen order
    auto                                 addSym = [&](const std::string &sym) {
        for (const auto &f: freeSyms)
        {
            if (f == sym)
            {
                return;
            }
        }
        freeSyms.push_back(sym);
    };
    printf("[compile] %zu graph input(s):\n", g.inputs.size());
    for (TensorId in: g.inputs)
    {
        const TensorDesc &d = g.desc(in);
        std::string       line;
        for (size_t ax = 0; ax < d.shape.size(); ++ax)
        {
            if (ax)
            {
                line += "x";
            }
            const std::string sym = ax < d.dimParams.size() ? d.dimParams[ax] : std::string();
            if (d.shape[ax] >= 0)
            {
                line += std::to_string(d.shape[ax]);
            } else if (!sym.empty())
            {
                line += "$" + sym;
                // A batch-NAMED leading symbol has the --batch fallback, so it is not "must-bind";
                // every other symbol (leading or not) must be bound or the compile hard-errors.
                if (ax != 0 || !batchLikeDimSymbol(sym))
                {
                    DimEval e = evalDimExpr(sym, noBind);
                    for (const std::string &fs: e.freeSymbols)
                    {
                        addSym(fs);
                    }
                }
            } else
            {
                line += "?";
            }
        }
        printf("    %-32s [%s]\n", d.name.c_str(), line.c_str());
    }
    if (freeSyms.empty())
    {
        printf("[compile] no free symbolic dims -- this model compiles with no --dim/--shape.\n");
        return;
    }
    printf("[compile] free symbolic dim(s) to bind with --dim NAME=VALUE:\n      ");
    for (size_t i = 0; i < freeSyms.size(); ++i)
    {
        printf("%s%s", i ? ", " : "", freeSyms[i].c_str());
    }
    printf("\n      (a leading batch axis also honors --batch N.)\n");
}

/// JSON string escaping for the support report (quotes, backslashes, control characters).
static std::string jsonEscape(const std::string &s) {
    // JSON requires the C0 control characters (everything below the first printable
    // ASCII codepoint, U+0020 space) to be emitted as \uXXXX escapes.
    constexpr unsigned char kFirstPrintableAscii = 0x20;
    std::string             out;
    out.reserve(s.size() + 8);
    for (char c: s)
    {
        if (c == '"' || c == '\\')
        {
            out += '\\';
            out += c;
        } else if ((unsigned char) c < kFirstPrintableAscii)
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

/// Writes one support report per bucket: bucket 0 at `outPath` exactly (the single-graph schema and
/// path, so existing consumers are unchanged), bucket N at `<outPath minus .json>.bucketN.json`. The
/// per-node backend assignment is shape- and graph-dependent, so a multi-bucket compile needs every
/// bucket checked — a decoder bucket can fall back where the prefill bucket does not.
static bool writeSupportReports(const std::vector<Graph> &buckets, const std::vector<std::string> &labels, const std::string &outPath) {
    for (size_t b = 0; b < buckets.size(); ++b)
    {
        std::string p = outPath;
        if (b > 0)
        {
            const std::string suffix = ".bucket" + std::to_string(b);
            constexpr size_t  extLen = 5; // strlen(".json")
            if (p.size() > extLen && p.compare(p.size() - extLen, extLen, ".json") == 0)
            {
                p.insert(p.size() - extLen, suffix);
            } else
            {
                p += suffix;
            }
        }
        if (!writeSupportReport(buckets[b], b < labels.size() ? labels[b] : std::string(), p))
        {
            printf("[compile] support report write failed (%s)\n", p.c_str());
            return false;
        }
        printf("[compile] wrote support report %s (bucket %zu)\n", p.c_str(), b);
    }
    return true;
}

int main(int argc, char **argv) {
    // `--graph` selects the multi-graph form: no positional input model, one source graph per
    // occurrence, all compiled into one multi-bucket .vxm over a shared initializer pool.
    bool graphMode = false;
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--graph"))
        {
            graphMode = true;
            break;
        }
    }
    if (argc < (graphMode ? 2 : 3))
    {
        printf("usage: %s <model.onnx|model.vxm> <out.vxm> [--fp16] [--batch N] [--dim NAME=VALUE] [--list-dims] [--shape NAME=D0xD1x...] [--bucket \"NAME=...;dim:NAME2=VALUE;...\"] [-O0..-O3 | --opt N] "
               "[--[no-]fuse-se] [--[no-]fuse-dwpw] [--[no-]fuse-pointwise] [--[no-]strict-fuse] [--[no-]lower-conv] [--no-dequantize] [--support-report <out.json>] [--dump-big]\n"
               "   or: %s <out.vxm> --graph \"FILE.onnx[;NAME=D0xD1x...;dim:NAME2=VALUE;...]\" [--graph ...] [shared flags as above]\n"
               "       each --graph occurrence compiles ONE bucket from its file (with its own shape/dim segments);\n"
               "       all buckets share one initializer pool in a single multi-graph .vxm\n",
               argv[0], argv[0]);
        return 1;
    }
    const int   flagStart = graphMode ? 2 : 3;
    std::string onnx = graphMode ? std::string() : argv[1];
    std::string out  = graphMode ? argv[1] : argv[2];
    if (graphMode && argc > 2 && argv[2][0] != '-')
    {
        // Mixed forms: `vknn_compile model.onnx out.vxm --graph ...` would make argv[1] the OUTPUT
        // and silently overwrite the source model. In graph mode the only positional is the output.
        printf("[compile] --graph form takes ONE positional (the output .vxm); '%s' would be overwritten as the output and '%s' ignored. Use: %s <out.vxm> --graph \"FILE[;segments]\" ...\n", argv[1], argv[2], argv[0]);
        return 1;
    }
    if (graphMode && !strcmp(argv[argc - 1], "--graph"))
    {
        printf("[compile] trailing --graph with no value\n");
        return 1;
    }
    bool fp16 = has(argc, argv, "--fp16", flagStart);

    int         optLevel = 1;
    std::string supportReport;
    for (int i = flagStart; i < argc; ++i)
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
    // Per-fusion overrides layered on top of the -O level: --<flag> forces the pass on,
    // --no-<flag> forces it off; absent flags keep the level's default.
    auto applyFlagOverride = [&](const char *enableFlag, const char *disableFlag, bool &option) {
        if (has(argc, argv, enableFlag, flagStart))
        {
            option = true;
        }
        if (has(argc, argv, disableFlag, flagStart))
        {
            option = false;
        }
    };
    applyFlagOverride("--fuse-se", "--no-fuse-se", opt.fuseSqueezeExcite);
    applyFlagOverride("--fuse-dwpw", "--no-fuse-dwpw", opt.fuseDwPw);
    applyFlagOverride("--fuse-pointwise", "--no-fuse-pointwise", opt.fusePointwiseChains);
    applyFlagOverride("--fuse-gridsample-warp", "--no-fuse-gridsample-warp", opt.fuseGridSampleWarp);
    applyFlagOverride("--strict-fuse", "--no-strict-fuse", opt.strictFuse);
    applyFlagOverride("--lower-conv", "--no-lower-conv", opt.lowerConv);
    applyFlagOverride("--dequantize", "--no-dequantize", opt.dequantize);
    opt.dumpBig = has(argc, argv, "--dump-big", flagStart);

    // Resolve dynamic input dims. --batch N sets the fallback substituted into a dynamic LEADING (batch)
    // axis; --dim NAME=VALUE (repeatable) binds a symbolic dim_param by name (every input axis whose
    // dim_param resolves from the bindings is filled automatically); --shape NAME=D0xD1x... (repeatable)
    // declares an input's full concrete shape and overrides --dim for that tensor. An undeclared, unbound
    // dynamic non-batch axis is a hard error in the passes (inferShapes), never a silent 1x1 plan.
    //
    // --bucket "..." (repeatable) declares ONE shape bucket per occurrence, each a mix of per-tensor
    // "NAME=D0xD1x..." shapes and "dim:NAME=VALUE" symbolic bindings: the whole model is compiled once per
    // bucket over a fresh import (the passes mutate the graph irreversibly), and the buckets share one
    // initializer pool in a multi-bucket .vxm. With no --bucket, exactly one bucket is produced from the
    // global --shape/--dim/--batch (a legacy single-graph .vxm). --batch/--shape/--dim act as the shared
    // fallback under every bucket.
    std::vector<BucketSpec>  bucketShapes; // one entry per --bucket occurrence
    std::vector<std::string> bucketLabels; // the raw --bucket value, as the bucket's label
    std::vector<std::string> graphFiles;   // one entry per --graph occurrence: the source model file
    std::vector<BucketSpec>  graphSpecs;   // that occurrence's own shape/dim segments
    std::vector<std::string> graphLabels;  // the raw --graph value, as the bucket's label
    for (int i = flagStart; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--batch") && i + 1 < argc)
        {
            opt.batch = atoll(argv[i + 1]);
        } else if (!strcmp(argv[i], "--dim") && i + 1 < argc)
        {
            std::string name;
            int64_t     value = 0;
            if (!parseDimSpec(argv[i + 1], &name, &value))
            {
                printf("[compile] bad --dim '%s' (expected NAME=VALUE, a non-negative integer)\n", argv[i + 1]);
                return 1;
            }
            opt.dimBindings[name] = value;
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
            BucketSpec bs;
            if (!parseBucketSpec(argv[i + 1], &bs))
            {
                printf("[compile] bad --bucket '%s' (expected NAME=D0xD1x... or dim:NAME=VALUE segments, ';'-separated)\n", argv[i + 1]);
                return 1;
            }
            bucketShapes.push_back(std::move(bs));
            bucketLabels.emplace_back(argv[i + 1]);
        } else if (!strcmp(argv[i], "--graph") && i + 1 < argc)
        {
            // "FILE[;NAME=D0xD1x...;dim:NAME2=VALUE;...]" -- the file, then optional bucket segments.
            std::string spec(argv[i + 1]);
            size_t      semi = spec.find(';');
            std::string file = spec.substr(0, semi);
            BucketSpec  bs;
            if (file.empty())
            {
                printf("[compile] bad --graph '%s' (expected FILE.onnx[;NAME=D0xD1x...;dim:NAME2=VALUE;...])\n", argv[i + 1]);
                return 1;
            }
            if (semi != std::string::npos && !parseBucketSpec(spec.c_str() + semi + 1, &bs))
            {
                printf("[compile] bad --graph segments in '%s' (expected NAME=D0xD1x... or dim:NAME=VALUE, ';'-separated)\n", argv[i + 1]);
                return 1;
            }
            graphFiles.push_back(std::move(file));
            graphSpecs.push_back(std::move(bs));
            graphLabels.emplace_back(argv[i + 1]);
        }
    }
    if (graphMode && !bucketShapes.empty())
    {
        printf("[compile] --graph and --bucket are mutually exclusive: give each --graph occurrence its own segments instead\n");
        return 1;
    }

    // A ".vxm" path is a pre-compiled graph; anything else is treated as ONNX to import.
    constexpr const char *kVxmExt   = ".vxm";
    const size_t          vxmExtLen = 4; // strlen(".vxm")
    const bool            vxmInput  = onnx.size() > vxmExtLen && onnx.compare(onnx.size() - vxmExtLen, vxmExtLen, kVxmExt) == 0;

    // --list-dims: import the model(s) and report the free symbolic input dimensions, then exit
    // without compiling. Needs ONNX input (a .vxm already has concrete shapes). In graph mode every
    // --graph file is listed in turn.
    if (has(argc, argv, "--list-dims", flagStart))
    {
        if (vxmInput)
        {
            printf("[compile] --list-dims needs an ONNX input (a .vxm already has concrete shapes)\n");
            return 2;
        }
        std::vector<std::string> files = graphMode ? graphFiles : std::vector<std::string> {onnx};
        for (const std::string &file: files)
        {
            printf("[compile] importing %s (--list-dims) ...\n", file.c_str());
            Graph g;
            try
            {
                g = importOnnx(file);
            } catch (const Error &e)
            {
                printf("[compile] %s\n", e.what());
                return 2;
            }
            listDims(g);
        }
        return 0;
    }

    // Multi-graph compile: one bucket per --graph occurrence, each imported fresh from its own source
    // file with its own shape/dim segments layered over the shared --batch/--shape/--dim fallback.
    // Different occurrences may name the same file (a decoder compiled at prefill and decode shapes)
    // or different files (a vision tower and its decoder): every bucket lands in ONE .vxm whose
    // initializer pool is content-deduped, so N shape buckets of one graph cost one copy of its
    // weights on disk, and the runtime serves each caller by its bound input names+shapes.
    if (graphMode)
    {
        if (graphFiles.empty())
        {
            printf("[compile] --graph given no value\n");
            return 1;
        }
        constexpr const char *kVxmExtG = ".vxm";
        std::vector<Graph>    buckets;
        for (size_t b = 0; b < graphFiles.size(); ++b)
        {
            const std::string &file = graphFiles[b];
            if (file.size() > 4 && file.compare(file.size() - 4, 4, kVxmExtG) == 0)
            {
                printf("[compile] --graph '%s': a .vxm cannot be a graph source (its passes are baked at one shape) -- give the .onnx\n", file.c_str());
                return 2;
            }
            PassOptions bopt = opt;
            for (const auto &kv: graphSpecs[b].shapes)
            {
                bopt.inputShapes[kv.first] = kv.second;
            }
            for (const auto &kv: graphSpecs[b].dims)
            {
                bopt.dimBindings[kv.first] = kv.second;
            }
            printf("[compile] graph %zu '%s': importing %s ...\n", b, graphLabels[b].c_str(), file.c_str());
            Graph gb;
            try
            {
                gb = importOnnx(file);
                runStandardPasses(gb, bopt);
            } catch (const Error &e)
            {
                printf("[compile] graph %zu '%s': %s\n", b, graphLabels[b].c_str(), e.what());
                return 2;
            }
            if (fp16)
            {
                convertInitializersFp16(gb);
            }
            printf("[compile] graph %zu '%s': post-passes %zu nodes, %zu weights\n", b, graphLabels[b].c_str(), gb.nodes.size(), gb.initializers.size());
            buckets.push_back(std::move(gb));
        }
        if (!supportReport.empty() && !writeSupportReports(buckets, graphFiles, supportReport))
        {
            return 2;
        }
        if (!saveGraphBinBuckets(buckets, graphLabels, out))
        {
            printf("[compile] save failed\n");
            return 2;
        }
        printf("[compile] wrote %s (%zu graph bucket(s))\n", out.c_str(), buckets.size());
        return 0;
    }

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
            // --shape/--dim declarations); the bucket's own shapes and dim bindings override per entry.
            PassOptions bopt = opt;
            for (const auto &kv: bucketShapes[b].shapes)
            {
                bopt.inputShapes[kv.first] = kv.second;
            }
            for (const auto &kv: bucketShapes[b].dims)
            {
                bopt.dimBindings[kv.first] = kv.second;
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
            // One report per bucket: the backend assignment is shape-dependent, so every bucket is
            // checked, not just bucket 0.
            std::vector<std::string> models(buckets.size(), onnx);
            if (!writeSupportReports(buckets, models, supportReport))
            {
                return 2;
            }
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
