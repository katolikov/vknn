#include "passes_internal.h"

namespace vknn {

    // The Config::fp32Tensors matcher: a comma list of substrings; a leading '-' marks an EXCLUDE
    // (a name with an excluded substring is never marked even if it matches an include), so a
    // fragile sub-region can be carved out. Shared with the fusion pass's compile-time prediction
    // (pwTensorIsFp32) — the two MUST agree or a fused unit can span a tensor markFp32 pins.
    bool fp32NameMatch(const std::string &nm, const std::string &substrs) {
        if (nm.empty() || substrs.empty())
        {
            return false;
        }
        std::vector<std::string> incl, excl;
        for (size_t p = 0, c;; p = c + 1)
        {
            c             = substrs.find(',', p);
            std::string s = substrs.substr(p, c == std::string::npos ? c : c - p);
            if (!s.empty())
            {
                (s[0] == '-' ? excl : incl).push_back(s[0] == '-' ? s.substr(1) : s);
            }
            if (c == std::string::npos)
            {
                break;
            }
        }
        for (const auto &s: excl)
        {
            if (nm.find(s) != std::string::npos)
            {
                return false;
            }
        }
        for (const auto &s: incl)
        {
            if (nm.find(s) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }

    // Selective fp32: mark every activation tensor whose name contains one of the comma-separated
    // substrings (Config::fp32Tensors) so its buffer stays fp32 under fp16 compute, then bridge the
    // fp16/fp32 frontier with ConvertDtype nodes — for each node, any activation input whose storage
    // dtype differs from the node's (its output[0]) gets a convert, exactly mirroring insertLayoutConverts.
    // Initializers are skipped: ops upload them at the node's precision (env.useFp16). Runs at load, after
    // insertLayoutConverts, so it operates on the final flat names.
    void markFp32(Graph &g, const std::string &substrs) {
        // Substring marks from Config::fp32Tensors are additive on top of any storage an earlier pass
        // already pinned to fp32 (pinGatherIndexFp32's index chains). Only flat tensors are eligible for a
        // substring mark: the flat transformer/geometry kernels all #include precision.glsl so an fp32
        // SPIR-V variant exists, whereas the NC4HW4 conv family (conv/wino/dwconv/fc/pool) is hand-written
        // fp16-only. Marking an NC4HW4 tensor would request a non-existent fp32 kernel.
        int marked = 0;
        if (!substrs.empty())
        {
            auto matches = [&](const std::string &nm) {
                return fp32NameMatch(nm, substrs);
            };
            for (auto &t: g.tensors)
            {
                if (!t.isInitializer && t.gpuFlat && matches(t.name))
                {
                    t.storeFp32 = true;
                    ++marked;
                }
            }
            if (!marked)
            {
                VKNN_INFO << "markFp32: no tensor matched fp32Tensors=\"" << substrs << "\"";
            }
        }
        // The frontier walk below runs whether or not a substring matched, so a chain pinned by an earlier
        // pass still gets its ConvertDtype bridges. With nothing anywhere in fp32 it is a pure no-op.
        // A kernel writes every output in ONE storage precision (its outputs[0]'s), so all outputs
        // of a multi-output node must share a mark — a fused unit's exported stream (pw_outs) pinned
        // differently from the main output would be written half-empty (fp16 stores into an fp32
        // buffer) or overrun (the reverse). Align to outputs[0]; consumers needing the other
        // precision get their ConvertDtype from the frontier walk below.
        for (auto &nd: g.nodes)
        {
            if (nd.outputs.size() < 2 || nd.outputs[0] == kNoTensor)
            {
                continue;
            }
            bool nodeFp32 = g.desc(nd.outputs[0]).storeFp32;
            for (size_t k = 1; k < nd.outputs.size(); ++k)
            {
                if (nd.outputs[k] != kNoTensor)
                {
                    g.desc(nd.outputs[k]).storeFp32 = nodeFp32;
                }
            }
        }
        // (source tensor, wantFp32) -> already-converted tensor, so one frontier tensor consumed at a
        // given precision by several nodes is converted once and the result shared.
        std::map<std::pair<TensorId, bool>, TensorId> cache;
        // New ConvertDtype nodes are buffered rather than appended in-place: mutating g.nodes while the
        // range-for below iterates it would invalidate that loop. They are spliced in after the walk.
        std::vector<Node>                             converts;
        int                                           n = 0;
        for (auto &nd: g.nodes)
        {
            if (nd.outputs.empty() || nd.outputs[0] == kNoTensor)
            {
                continue;
            }
            bool nodeFp32 = g.desc(nd.outputs[0]).storeFp32; // the precision this node's kernel runs in
            for (size_t inIdx = 0; inIdx < nd.inputs.size(); ++inIdx)
            {
                TensorId &in = nd.inputs[inIdx];
                if (in == kNoTensor || g.isInitializer(in))
                {
                    continue; // initializers upload at the node's precision (env.useFp16)
                }
                // A Gather reads its index (input 1) as fp32 no matter the kernel's compute precision
                // (gather.comp binding 1 is float), so a pinned fp32 index must not be narrowed back to
                // fp16 by a frontier convert -- that would re-overflow a large token id to +inf.
                if (nd.type == OpType::Gather && inIdx == 1)
                {
                    continue;
                }
                // An fp16 GridSample decodes its grid (input 1) at the grid's OWN storage precision
                // (gridsample_fp16.comp reads raw words per the GRID_FP32 spec constant), so a grid
                // pinned fp32 by pinGridSampleGridFp32 must not be narrowed back to fp16 -- that would
                // re-quantize the sampling coordinates the pin exists to protect. An fp32 GridSample
                // (nodeFp32) keeps the bridge: gridsample.comp reads the grid as float only.
                if (nd.type == OpType::GridSample && inIdx == 1 && !nodeFp32)
                {
                    continue;
                }
                if (g.desc(in).storeFp32 == nodeFp32)
                {
                    continue;
                }
                auto key = std::make_pair(in, nodeFp32);
                auto it  = cache.find(key);
                if (it == cache.end())
                {
                    TensorDesc d    = g.desc(in);
                    d.name          = g.desc(in).name + (nodeFp32 ? "#f32" : "#f16") + std::to_string(n);
                    d.isInitializer = d.isInput = d.isOutput = false;
                    d.storeFp32                              = nodeFp32;
                    d.gpuFlat                                = g.desc(in).gpuFlat; // dtype change only, same layout
                    TensorId t2                              = g.addTensor(d);
                    Node     cv;
                    cv.type    = OpType::ConvertDtype;
                    cv.name    = "cvtdt" + std::to_string(n++);
                    cv.inputs  = {in};
                    cv.outputs = {t2};
                    converts.push_back(cv);
                    it = cache.emplace(key, t2).first;
                }
                in = it->second;
            }
        }
        if (!converts.empty())
        {
            // The converts are appended at the tail (out of dependency order), then topoSort restores a
            // valid execution order so each ConvertDtype runs before the node that reads its output.
            for (auto &c: converts)
            {
                g.nodes.push_back(std::move(c));
            }
            g.topoSort();
        }
        VKNN_INFO << "markFp32: marked " << marked << " tensor(s) fp32, inserted " << converts.size() << " convert(s)";
    }

    void pinGatherIndexFp32(Graph &g) {
        // Last writer of each tensor, so the index can be traced back to its boundary source.
        std::vector<int> producer(g.tensors.size(), -1);
        for (int ni = 0; ni < (int) g.nodes.size(); ++ni)
        {
            for (TensorId o: g.nodes[ni].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[(size_t) o] = ni;
                }
            }
        }
        int pinned = 0;
        for (const auto &nd: g.nodes)
        {
            if (nd.type != OpType::Gather || nd.inputs.size() < 2 || nd.inputs[1] == kNoTensor)
            {
                continue;
            }
            // A constant index is uploaded fp32 by the op itself, so only a runtime index needs pinning.
            // Walk it up through pure passthrough producers (the ConvertLayout the flat pass splices in) to
            // the boundary input, pinning every hop to fp32 so an integer index never rounds to fp16 (a
            // token id above 65504 would otherwise store as +inf and the lookup would read the wrong row).
            TensorId t = nd.inputs[1];
            for (int hop = 0; t != kNoTensor && !g.isInitializer(t) && hop < 64; ++hop)
            {
                if (g.desc(t).storeFp32)
                {
                    break; // already pinned (a shared index, or a prior hop)
                }
                g.desc(t).storeFp32 = true;
                ++pinned;
                int p = producer[(size_t) t];
                if (p < 0)
                {
                    break; // graph-input boundary: pinned, nothing upstream to follow
                }
                const Node &pn = g.nodes[(size_t) p];
                if (pn.type == OpType::ConvertLayout && pn.inputs.size() == 1)
                {
                    t = pn.inputs[0]; // same values, different layout -- keep pinning toward the source
                } else
                {
                    break; // a real op computes the index; its (now-pinned) output runs fp32 via nodeFp32
                }
            }
        }
        if (pinned)
        {
            VKNN_INFO << "pinGatherIndexFp32: pinned " << pinned << " index tensor(s) to fp32";
        }
    }

    void pinGridSampleGridFp32(Graph &g) {
        // Last writer of each tensor, so the grid can be traced back toward its source.
        std::vector<int> producer(g.tensors.size(), -1);
        for (int ni = 0; ni < (int) g.nodes.size(); ++ni)
        {
            for (TensorId o: g.nodes[ni].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[(size_t) o] = ni;
                }
            }
        }
        int pinned = 0;
        for (const auto &nd: g.nodes)
        {
            if (nd.type != OpType::GridSample || nd.inputs.size() < 2 || nd.inputs[1] == kNoTensor)
            {
                continue;
            }
            // A warp-mode GridSample (fuseGridSampleWarp) has no runtime grid to pin: input 1 is the
            // NCHW flow (read fp16 in the NC4HW4 activation layout) and the base grid (input 2, uploaded
            // fp32 by the op) carries the coordinates' full precision. The fp16 sampler reproduces the
            // split Mul's fp16-rounded product, so flow must stay fp16 — pinning it would diverge.
            if (nd.attr.has("warp"))
            {
                continue;
            }
            // The grid holds normalized sampling COORDINATES: fp16 storage quantizes them at ~4.9e-4
            // near |g|=1, which drifts the sample point by up to ~0.5 px at 1920-wide inputs — a direct
            // warp-quality (UV) loss. A constant grid is uploaded fp32 by the op itself, so only a
            // runtime grid (optical flow) needs pinning. Walk it up through pure passthrough producers
            // (the ConvertLayout the flat pass splices in) pinning every hop, but only while the hop is
            // FLAT: the NC4HW4 conv family is hand-written fp16-only, so pinning a conv output would
            // request a non-existent fp32 kernel (the frontier walk in markFp32 bridges the boundary
            // with a ConvertDtype instead). Mirrors pinGatherIndexFp32.
            TensorId t = nd.inputs[1];
            for (int hop = 0; t != kNoTensor && !g.isInitializer(t) && g.desc(t).gpuFlat && hop < 64; ++hop)
            {
                if (g.desc(t).storeFp32)
                {
                    break; // already pinned (a shared grid, or a prior hop)
                }
                g.desc(t).storeFp32 = true;
                ++pinned;
                int p = producer[(size_t) t];
                if (p < 0)
                {
                    break; // graph-input boundary: pinned, nothing upstream to follow
                }
                const Node &pn = g.nodes[(size_t) p];
                if (pn.type == OpType::ConvertLayout && pn.inputs.size() == 1)
                {
                    t = pn.inputs[0]; // same values, different layout -- keep pinning toward the source
                } else
                {
                    break; // a real op computes the grid; its (now-pinned) output runs fp32 via nodeFp32
                }
            }
        }
        if (pinned)
        {
            VKNN_INFO << "pinGridSampleGridFp32: pinned " << pinned << " grid tensor(s) to fp32";
        }
    }

} // namespace vknn
