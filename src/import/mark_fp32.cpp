#include "passes_internal.h"

namespace vknn {

    // Selective fp32: mark every activation tensor whose name contains one of the comma-separated
    // substrings (Config::fp32Tensors) so its buffer stays fp32 under fp16 compute, then bridge the
    // fp16/fp32 frontier with ConvertDtype nodes — for each node, any activation input whose storage
    // dtype differs from the node's (its output[0]) gets a convert, exactly mirroring insertLayoutConverts.
    // Initializers are skipped: ops upload them at the node's precision (env.useFp16). Runs at load, after
    // insertLayoutConverts, so it operates on the final flat names.
    void markFp32(Graph &g, const std::string &substrs) {
        if (substrs.empty())
        {
            return;
        }
        // Comma list of substrings; a leading '-' marks an EXCLUDE (a name with an excluded substring is
        // never marked even if it matches an include), so a fragile sub-region can be carved out.
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
        auto matches = [&](const std::string &nm) {
            if (nm.empty())
            {
                return false;
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
        };
        // Only flat tensors are eligible: the flat transformer/geometry kernels all #include precision.glsl
        // so an fp32 SPIR-V variant exists, whereas the NC4HW4 conv family (conv/wino/dwconv/fc/pool) is
        // hand-written fp16-only. Marking an NC4HW4 tensor would request a non-existent fp32 kernel.
        int marked = 0;
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
            return;
        }
        std::map<std::pair<TensorId, bool>, TensorId> cache; // (tensor, wantFp32) -> converted tensor
        std::vector<Node>                             converts;
        int                                           n = 0;
        for (auto &nd: g.nodes)
        {
            if (nd.outputs.empty() || nd.outputs[0] == kNoTensor)
            {
                continue;
            }
            bool nodeFp32 = g.desc(nd.outputs[0]).storeFp32; // the precision this node's kernel runs in
            for (TensorId &in: nd.inputs)
            {
                if (in == kNoTensor || g.isInitializer(in))
                {
                    continue; // initializers upload at the node's precision (env.useFp16)
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
            for (auto &c: converts)
            {
                g.nodes.push_back(std::move(c));
            }
            g.topoSort();
        }
        VKNN_INFO << "markFp32: marked " << marked << " tensor(s) fp32, inserted " << converts.size() << " convert(s)";
    }

} // namespace vknn
