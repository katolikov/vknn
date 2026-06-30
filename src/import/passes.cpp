#include "passes.h"
#include "backend/cpu/cpu_backend.h"
#include "vknn/logging.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>

namespace vknn {

    // Read an int64 list parameter from either a node attribute (older opsets) or an initializer input
    // (opset 10+/13+ moved Slice/Pad/Reduce params to inputs). Returns empty if neither is present.
    std::vector<int64_t> readI64Param(const Graph &g, const Node &nd, const char *attrName, int inputIdx) {
        const auto &av = nd.attr.getints(attrName);
        if (!av.empty())
        {
            return av;
        }
        if (inputIdx >= 0 && inputIdx < (int) nd.inputs.size() && nd.inputs[inputIdx] != kNoTensor)
        {
            auto it = g.initializers.find(nd.inputs[inputIdx]);
            if (it != g.initializers.end())
            {
                const HostBuffer &hb = it->second;
                if (g.tensors[nd.inputs[inputIdx]].dtype == DType::Int64)
                {
                    int64_t n = (int64_t) hb.bytes.size() / 8;
                    return std::vector<int64_t>(hb.i64(), hb.i64() + n);
                }
                int64_t              n = (int64_t) hb.bytes.size() / 4;
                std::vector<int64_t> out;
                const float         *f = hb.f32();
                for (int64_t i = 0; i < n; ++i)
                {
                    out.push_back((int64_t) f[i]);
                }
                return out;
            }
        }
        return {};
    }

    // Redirect every reference to tensor `from` so it points at `to`: node inputs, the fused-residual
    // edge (which is not in the inputs list on every op), and graph outputs. Fusion passes that delete a
    // node and fold its output into a producer must use this; rewiring only node.inputs leaves a stale
    // fusedResidual edge dangling at a dead tensor, which crashes a conv residual read.
    static void rewireTensor(Graph &g, TensorId from, TensorId to) {
        if (from == to || from == kNoTensor)
        {
            return;
        }
        for (auto &nn: g.nodes)
        {
            for (TensorId &in: nn.inputs)
            {
                if (in == from)
                {
                    in = to;
                }
            }
            if (nn.fusedResidual == from)
            {
                nn.fusedResidual = to;
            }
        }
        for (TensorId &go: g.outputs)
        {
            if (go == from)
            {
                go = to;
            }
        }
    }

    void inferShapes(Graph &g, int64_t batch) {
        // Resolve dynamic dims on inputs to `batch`.
        for (TensorId in: g.inputs)
        {
            auto &s = g.desc(in).shape;
            for (auto &d: s)
            {
                if (d < 0)
                {
                    d = batch;
                }
            }
        }
        auto SH = [&](TensorId id) -> Shape & {
            return g.desc(id).shape;
        };
        for (auto &nd: g.nodes)
        {
            if (nd.outputs.empty())
            {
                continue;
            }
            TensorId o = nd.outputs[0];
            switch (nd.type)
            {
                case OpType::Conv: {
                    if (SH(nd.inputs[0]).empty())
                    {
                        break; // input unresolved: leave output empty (don't
                               // fabricate [1,C,1,1] from NCHW::from({}), which would
                               // let constFold freeze a stale Shape() of this output)
                    }
                    NCHW         x = NCHW::from(SH(nd.inputs[0]));
                    const Shape &w = SH(nd.inputs[1]);
                    if (w.size() < 4)
                    {
                        break;
                    }
                    int64_t outC = w[0], kh = w[2], kw = w[3];
                    auto    ints = [&](const char *k, std::vector<int64_t> d) {
                        const auto &v = nd.attr.getints(k);
                        return v.empty() ? d : v;
                    };
                    auto    st  = ints("strides", {1, 1});
                    auto    pad = ints("pads", {0, 0, 0, 0});
                    auto    dil = ints("dilations", {1, 1});
                    int64_t oh  = (x.h + pad[0] + pad[2] - (dil[0] * (kh - 1) + 1)) / st[0] + 1;
                    int64_t ow  = (x.w + pad[1] + pad[3] - (dil[1] * (kw - 1) + 1)) / st[1] + 1;
                    SH(o)       = {x.n, outC, oh, ow};
                    break;
                }
                case OpType::ConvTranspose: {
                    if (SH(nd.inputs[0]).empty())
                    {
                        break;
                    }
                    NCHW         x = NCHW::from(SH(nd.inputs[0]));
                    const Shape &w = SH(nd.inputs[1]); // [Cin, Cout/group, kH, kW]
                    if (w.size() < 4)
                    {
                        break;
                    }
                    int64_t kh = w[2], kw = w[3];
                    auto    ints = [&](const char *k, std::vector<int64_t> d) {
                        const auto &v = nd.attr.getints(k);
                        return v.empty() ? d : v;
                    };
                    int64_t group  = nd.attr.geti("group", 1);
                    int64_t outC   = w[1] * group;
                    auto    st     = ints("strides", {1, 1});
                    auto    pad    = ints("pads", {0, 0, 0, 0});
                    auto    dil    = ints("dilations", {1, 1});
                    auto    outpad = ints("output_padding", {0, 0});
                    // ONNX ConvTranspose output size (output_shape attr, if present, overrides this).
                    const auto &osh = nd.attr.getints("output_shape");
                    int64_t     oh  = (x.h - 1) * st[0] - pad[0] - pad[2] + dil[0] * (kh - 1) + 1 + outpad[0];
                    int64_t     ow  = (x.w - 1) * st[1] - pad[1] - pad[3] + dil[1] * (kw - 1) + 1 + outpad[1];
                    if (osh.size() == 2)
                    {
                        oh = osh[0];
                        ow = osh[1];
                    }
                    SH(o) = {x.n, outC, oh, ow};
                    break;
                }
                case OpType::Clip:
                case OpType::Relu:
                case OpType::BatchNorm:
                case OpType::Identity:
                case OpType::Unary:
                case OpType::Softmax:
                case OpType::LayerNorm:
                case OpType::PRelu:
                case OpType::EyeLike:   // identity-like, same shape as input
                case OpType::ScatterND: // same shape as data (input[0])
                    SH(o) = SH(nd.inputs[0]);
                    break;
                case OpType::Equal:
                case OpType::Greater:
                case OpType::GreaterEqual: {
                    const Shape &a = SH(nd.inputs[0]);
                    const Shape &b = SH(nd.inputs[1]);
                    if (a.empty() && b.empty())
                    {
                        break;
                    }
                    if (a.empty() || b.empty())
                    {
                        SH(o) = a.empty() ? b : a;
                        break;
                    }
                    size_t rank = std::max(a.size(), b.size());
                    Shape  out(rank, 1);
                    auto   dimOf = [&](const Shape &s, size_t i) -> int64_t {
                        size_t off = rank - s.size();
                        return i < off ? 1 : s[i - off];
                    };
                    for (size_t i = 0; i < rank; ++i)
                    {
                        out[i] = std::max(dimOf(a, i), dimOf(b, i));
                    }
                    SH(o) = out;
                    break;
                }
                case OpType::Einsum: {
                    std::string eq;
                    for (char c: nd.attr.gets("equation", ""))
                    {
                        if (c != ' ' && c != '\t')
                        {
                            eq += c;
                        }
                    }
                    const Shape &a = SH(nd.inputs[0]);
                    const Shape &b = SH(nd.inputs[1]);
                    if (eq == "i,j->ij" && !a.empty() && !b.empty())
                    {
                        SH(o) = {a[0], b[0]};
                    } else if (eq == "...ab,...b->...a" && a.size() >= 2 && !b.empty())
                    {
                        // a=[...,A,B], b=[...,B]; out = broadcast(a.batch=a[:-2], b.batch=b[:-1]) + [A]. The
                        // leading `...` dims broadcast between the two operands (per-pixel ray math broadcasts a
                        // [.,3,3] matrix [batch 1,1] against [.,H,W,3] coords), not just a's batch.
                        Shape  ab(a.begin(), a.end() - 2), bb(b.begin(), b.end() - 1);
                        size_t rank = std::max(ab.size(), bb.size());
                        Shape  out(rank, 1);
                        auto   dimOf = [&](const Shape &s, size_t i) -> int64_t {
                            size_t off = rank - s.size();
                            return i < off ? 1 : s[i - off];
                        };
                        for (size_t i = 0; i < rank; ++i)
                        {
                            out[i] = std::max(dimOf(ab, i), dimOf(bb, i));
                        }
                        out.push_back(a[a.size() - 2]); // the free dim A
                        SH(o) = out;
                    } else if (eq == "bij,bnjk->bnik" && a.size() >= 3 && b.size() >= 4)
                    { SH(o) = {a[0], b[1], a[1], b[3]}; }
                    break;
                }
                case OpType::Where: {
                    // Broadcast cond (in[0]), X (in[1]), Y (in[2]); output dims = elementwise-max.
                    size_t rank = 0;
                    for (TensorId t: nd.inputs)
                    {
                        if (t != kNoTensor)
                        {
                            rank = std::max(rank, SH(t).size());
                        }
                    }
                    Shape out(rank, 1);
                    for (TensorId t: nd.inputs)
                    {
                        if (t == kNoTensor)
                        {
                            continue;
                        }
                        const Shape &s   = SH(t);
                        size_t       off = rank - s.size();
                        for (size_t i = 0; i < s.size(); ++i)
                        {
                            out[off + i] = std::max(out[off + i], s[i]);
                        }
                    }
                    SH(o) = out;
                    break;
                }
                case OpType::ConstantOfShape: {
                    // Output shape = the int64 values of the shape initializer input.
                    TensorId sid = nd.inputs[0];
                    if (g.isInitializer(sid))
                    {
                        const HostBuffer &hb = g.initializers[sid];
                        int64_t           r  = numElements(g.desc(sid).shape);
                        if (r <= 0)
                        {
                            r = (int64_t) (hb.bytes.size() / 8);
                        }
                        if (r < 0 || r > 16)
                        {
                            break; // a shape vector has at most a handful of dims
                        }
                        Shape out(r);
                        if (g.tensors[sid].dtype == DType::Int64)
                        {
                            for (int64_t i = 0; i < r; ++i)
                            {
                                out[i] = hb.i64()[i];
                            }
                        } else
                        {
                            for (int64_t i = 0; i < r; ++i)
                            {
                                out[i] = (int64_t) hb.f32()[i];
                            }
                        }
                        SH(o) = out;
                    }
                    break;
                }
                case OpType::GridSample: {
                    const Shape &xs = SH(nd.inputs[0]);
                    const Shape &gs = SH(nd.inputs[1]);
                    if (xs.size() == 4 && gs.size() == 4)
                    {
                        SH(o) = {xs[0], xs[1], gs[1], gs[2]};
                    }
                    break;
                }
                case OpType::Cast:
                case OpType::ConvertLayout:
                case OpType::ConvertDtype: {
                    SH(o) = SH(nd.inputs[0]);
                    break;
                }
                case OpType::FusedSE: {
                    if (SH(nd.inputs[0]).empty())
                    {
                        break;
                    }
                    NCHW x = NCHW::from(SH(nd.inputs[0]));
                    SH(o)  = {x.n, x.c, 1, 1}; // channel scale
                    break;
                }
                case OpType::FusedDwPw: {
                    if (SH(nd.inputs[0]).empty())
                    {
                        break;
                    }
                    NCHW         x  = NCHW::from(SH(nd.inputs[0])); // expanded input [N,E,H,W]
                    const Shape &pw = SH(nd.inputs[3]);             // project weights [Cout,E,1,1]
                    auto         a  = [&](const char *k, std::vector<int64_t> d) {
                        const auto &v = nd.attr.getints(k);
                        return v.empty() ? d : v;
                    };
                    auto    k = a("kernel_shape", {3, 3}), st = a("strides", {1, 1});
                    auto    pad = a("pads", {0, 0, 0, 0}), dil = a("dilations", {1, 1});
                    int64_t oh = (x.h + pad[0] + pad[2] - (dil[0] * (k[0] - 1) + 1)) / st[0] + 1;
                    int64_t ow = (x.w + pad[1] + pad[3] - (dil[1] * (k[1] - 1) + 1)) / st[1] + 1;
                    SH(o)      = {x.n, pw.empty() ? x.c : pw[0], oh, ow};
                    break;
                }
                case OpType::Split: {
                    const Shape &a = SH(nd.inputs[0]);
                    if (a.empty())
                    {
                        break;
                    }
                    int64_t rank = (int64_t) a.size();
                    int64_t axis = nd.attr.geti("axis", 0);
                    if (axis < 0)
                    {
                        axis += rank;
                    }
                    std::vector<int64_t> sp   = readI64Param(g, nd, "split", 1);
                    int64_t              nout = (int64_t) nd.outputs.size();
                    if (sp.empty() && nout > 0)
                    {
                        int64_t each = a[axis] / nout;
                        for (int64_t k = 0; k < nout; ++k)
                        {
                            sp.push_back(each);
                        }
                    }
                    for (int64_t k = 0; k < nout && k < (int64_t) sp.size(); ++k)
                    {
                        if (nd.outputs[k] == kNoTensor)
                        {
                            continue;
                        }
                        Shape os          = a;
                        os[axis]          = sp[k];
                        SH(nd.outputs[k]) = os;
                    }
                    break;
                }
                case OpType::Transpose: {
                    const Shape &a    = SH(nd.inputs[0]);
                    const auto  &perm = nd.attr.getints("perm");
                    Shape        out(a.size());
                    for (size_t i = 0; i < a.size(); ++i)
                    {
                        out[i] = perm.empty() ? a[a.size() - 1 - i] : a[perm[i]];
                    }
                    SH(o) = out;
                    break;
                }
                case OpType::Reduce: {
                    const Shape &a = SH(nd.inputs[0]);
                    if (a.empty())
                    {
                        break;
                    }
                    int64_t              rank = (int64_t) a.size();
                    std::vector<int64_t> axes = readI64Param(g, nd, "axes", 1);
                    if (axes.empty())
                    {
                        for (int64_t i = 0; i < rank; ++i)
                        {
                            axes.push_back(i); // reduce all
                        }
                    }
                    std::set<int64_t> ax;
                    for (int64_t v: axes)
                    {
                        ax.insert(v < 0 ? v + rank : v);
                    }
                    bool  keep = nd.attr.geti("keepdims", 1) != 0;
                    Shape out;
                    for (int64_t i = 0; i < rank; ++i)
                    {
                        if (ax.count(i))
                        {
                            if (keep)
                            {
                                out.push_back(1);
                            }
                        } else
                        {
                            out.push_back(a[i]);
                        }
                    }
                    if (out.empty())
                    {
                        out.push_back(1);
                    }
                    SH(o) = out;
                    break;
                }
                case OpType::DepthToSpace: {
                    // [N,C,H,W] -> [N, C/(b*b), H*b, W*b]
                    if (SH(nd.inputs[0]).empty())
                    {
                        break;
                    }
                    NCHW    x = NCHW::from(SH(nd.inputs[0]));
                    int64_t b = nd.attr.geti("blocksize", 1);
                    if (b < 1)
                    {
                        b = 1;
                    }
                    SH(o) = {x.n, x.c / (b * b), x.h * b, x.w * b};
                    break;
                }
                case OpType::Pad: {
                    const Shape         &a    = SH(nd.inputs[0]);
                    int64_t              rank = (int64_t) a.size();
                    std::vector<int64_t> pads = readI64Param(g, nd, "pads", 1);
                    Shape                out  = a;
                    if ((int64_t) pads.size() >= 2 * rank)
                    {
                        for (int64_t i = 0; i < rank; ++i)
                        {
                            out[i] = a[i] + pads[i] + pads[i + rank];
                        }
                    }
                    SH(o) = out;
                    break;
                }
                case OpType::Slice: {
                    const Shape &a = SH(nd.inputs[0]);
                    if (a.empty())
                    {
                        break;
                    }
                    int64_t              rank   = (int64_t) a.size();
                    std::vector<int64_t> starts = readI64Param(g, nd, "starts", 1);
                    std::vector<int64_t> ends   = readI64Param(g, nd, "ends", 2);
                    std::vector<int64_t> axes   = readI64Param(g, nd, "axes", 3);
                    std::vector<int64_t> steps  = readI64Param(g, nd, "steps", 4);
                    // The number of axes this Slice bounds equals the length of the starts/ends params (known
                    // from the param tensor's shape even while its values are still runtime). When a bound
                    // cannot be read as a constant yet (e.g. a head_dim/2 derived from Shape() arithmetic that
                    // const-folds only on a later pass), leave the output unresolved rather than fabricating
                    // the sliced axis at its full input size: a wrong dim here can be frozen by a downstream
                    // Shape() fold. inferShapes re-runs and resolves it once the bounds const-fold.
                    auto declLen = [&](int idx) -> int64_t {
                        if (idx >= (int) nd.inputs.size() || nd.inputs[idx] == kNoTensor)
                        {
                            return 0;
                        }
                        const Shape &ps = g.desc(nd.inputs[idx]).shape;
                        return ps.empty() ? -1 : numElements(ps);
                    };
                    int64_t want = std::max(declLen(1), declLen(2)); // intended number of (start,end) pairs
                    if (want > 0 && ((int64_t) starts.size() < want || (int64_t) ends.size() < want))
                    {
                        break; // a slice bound is still runtime -> defer rather than fabricate a wrong dim
                    }
                    Shape out = a;
                    // Bound a dim only when BOTH its start and end are known; never index a param past its
                    // length.
                    for (size_t k = 0; k < starts.size() && k < ends.size(); ++k)
                    {
                        int64_t ax = axes.empty() ? (int64_t) k : (k < axes.size() ? axes[k] : (int64_t) k);
                        if (ax < 0)
                        {
                            ax += rank;
                        }
                        if (ax < 0 || ax >= rank)
                        {
                            continue;
                        }
                        int64_t step = (k < steps.size()) ? steps[k] : 1;
                        int64_t dim  = a[ax];
                        int64_t st   = starts[k] < 0 ? starts[k] + dim : starts[k];
                        int64_t en   = ends[k] < 0 ? ends[k] + dim : ends[k];
                        st           = std::max<int64_t>(0, std::min(st, dim));
                        en           = std::max<int64_t>(0, std::min(en, dim));
                        int64_t n    = step > 0 ? (en - st + step - 1) / step : 0;
                        out[ax]      = std::max<int64_t>(0, n);
                    }
                    SH(o) = out;
                    break;
                }
                case OpType::Resize: {
                    // output = round(input * scales) or explicit sizes; scales/sizes are initializer inputs.
                    Shape s = SH(nd.inputs[0]);
                    if (s.size() == 4)
                    {
                        // ONNX Resize inputs: X, roi, scales, sizes (some optional/empty). Prefer sizes if given.
                        auto getInit = [&](int idx, std::vector<float> &f, std::vector<int64_t> &i64) {
                            if (idx >= (int) nd.inputs.size() || nd.inputs[idx] == kNoTensor)
                            {
                                return false;
                            }
                            auto it = g.initializers.find(nd.inputs[idx]);
                            if (it == g.initializers.end())
                            {
                                return false;
                            }
                            int64_t n = (int64_t) it->second.bytes.size() / (g.tensors[nd.inputs[idx]].dtype == DType::Int64 ? 8 : 4);
                            if (g.tensors[nd.inputs[idx]].dtype == DType::Int64)
                            {
                                const int64_t *p = it->second.i64();
                                for (int64_t k = 0; k < n; ++k)
                                {
                                    i64.push_back(p[k]);
                                }
                            } else
                            {
                                const float *p = it->second.f32();
                                for (int64_t k = 0; k < n; ++k)
                                {
                                    f.push_back(p[k]);
                                }
                            }
                            return true;
                        };
                        std::vector<float>   sizesF, scalesF;
                        std::vector<int64_t> sizesI, scalesI;
                        if (nd.inputs.size() >= 4 && getInit(3, sizesF, sizesI) && (sizesI.size() == 4 || sizesF.size() == 4))
                        {
                            for (int k = 0; k < 4; ++k)
                            {
                                s[k] = sizesI.size() == 4 ? sizesI[k] : (int64_t) sizesF[k];
                            }
                        } else if (getInit(2, scalesF, scalesI) && scalesF.size() == 4)
                        {
                            for (int k = 0; k < 4; ++k)
                            {
                                s[k] = (int64_t) (SH(nd.inputs[0])[k] * scalesF[k]);
                            }
                        }
                    }
                    SH(o) = s;
                    break;
                }
                case OpType::Binary:
                case OpType::Add: {
                    // NumPy broadcasting: per-dim max over right-aligned shapes. Required for outer-product
                    // ops ([..,3,1]*[..,1,3]->[..,3,3] in the per-pixel ray math) and trailing broadcasts
                    // ([2,224,224,1]*[3]->[2,224,224,3]).
                    const Shape &a = SH(nd.inputs[0]);
                    const Shape &b = SH(nd.inputs[1]);
                    if (a.empty() && b.empty())
                    {
                        break;
                    }
                    if (a.empty() || b.empty())
                    { // one is a rank-0 scalar (or unresolved): use the other
                        SH(o) = a.empty() ? b : a;
                        break;
                    }
                    size_t rank = std::max(a.size(), b.size());
                    Shape  out(rank, 1);
                    auto   dimOf = [&](const Shape &s, size_t i) -> int64_t {
                        size_t off = rank - s.size();
                        return i < off ? 1 : s[i - off];
                    };
                    for (size_t i = 0; i < rank; ++i)
                    {
                        out[i] = std::max(dimOf(a, i), dimOf(b, i));
                    }
                    SH(o) = out;
                    break;
                }
                case OpType::GlobalAvgPool: {
                    if (SH(nd.inputs[0]).empty())
                    {
                        break;
                    }
                    NCHW x = NCHW::from(SH(nd.inputs[0]));
                    SH(o)  = {x.n, x.c, 1, 1};
                    break;
                }
                case OpType::MaxPool:
                case OpType::AvgPool: {
                    if (SH(nd.inputs[0]).empty())
                    {
                        break;
                    }
                    NCHW x    = NCHW::from(SH(nd.inputs[0]));
                    auto ints = [&](const char *k, std::vector<int64_t> d) {
                        const auto &v = nd.attr.getints(k);
                        return v.empty() ? d : v;
                    };
                    auto    ks  = ints("kernel_shape", {1, 1});
                    auto    st  = ints("strides", {1, 1});
                    auto    pad = ints("pads", {0, 0, 0, 0});
                    int64_t oh  = (x.h + pad[0] + pad[2] - ks[0]) / st[0] + 1;
                    int64_t ow  = (x.w + pad[1] + pad[3] - ks[1]) / st[1] + 1;
                    SH(o)       = {x.n, x.c, oh, ow};
                    break;
                }
                case OpType::Gemm: {
                    const Shape &a = SH(nd.inputs[0]);
                    if (a.empty())
                    {
                        break;
                    }
                    const Shape &w      = SH(nd.inputs[1]);
                    int64_t      transB = nd.attr.geti("transB", 0);
                    int64_t      M      = a.empty() ? 1 : a[0];
                    int64_t      N      = w.size() < 2 ? 0 : (transB ? w[0] : w[1]);
                    SH(o)               = {M, N};
                    break;
                }
                case OpType::MatMul: {
                    Shape a = SH(nd.inputs[0]);
                    Shape b = SH(nd.inputs[1]);
                    if (a.empty() || b.empty())
                    {
                        break;
                    }
                    // 1-D promotion: A[K]->[1,K], B[K]->[K,1]; the prepended/appended dim is stripped from out.
                    bool aWas1D = a.size() == 1, bWas1D = b.size() == 1;
                    if (aWas1D)
                    {
                        a = {1, a[0]};
                    }
                    if (bWas1D)
                    {
                        b = {b[0], 1};
                    }
                    int64_t M = a[a.size() - 2], N = b[b.size() - 1];
                    int64_t aBatch = (int64_t) a.size() - 2, bBatch = (int64_t) b.size() - 2;
                    int64_t batchRank = std::max(aBatch, bBatch);
                    auto    dimOf     = [&](const Shape &s, int64_t batch, int64_t i) -> int64_t {
                        int64_t off = batchRank - batch;
                        return i < off ? 1 : s[i - off];
                    };
                    Shape out;
                    for (int64_t i = 0; i < batchRank; ++i)
                    {
                        out.push_back(std::max(dimOf(a, aBatch, i), dimOf(b, bBatch, i)));
                    }
                    if (!aWas1D)
                    {
                        out.push_back(M);
                    }
                    out.push_back(N);
                    if (bWas1D)
                    {
                        out.pop_back();
                    }
                    if (out.empty())
                    {
                        out.push_back(1); // scalar dot product -> [1]
                    }
                    SH(o) = out;
                    break;
                }
                case OpType::Flatten: {
                    const Shape &a = SH(nd.inputs[0]);
                    if (a.empty())
                    {
                        break;
                    }
                    int64_t axis = nd.attr.geti("axis", 1), d0 = 1, d1 = 1;
                    for (int64_t i = 0; i < (int64_t) a.size(); ++i)
                    {
                        (i < axis ? d0 : d1) *= a[i];
                    }
                    SH(o) = {d0, d1};
                    break;
                }
                case OpType::Concat: {
                    Shape s = SH(nd.inputs[0]);
                    if (!s.empty())
                    {
                        int64_t axis = nd.attr.geti("axis", 1);
                        if (axis < 0)
                        {
                            axis += (int64_t) s.size();
                        }
                        int64_t sum = 0;
                        for (TensorId in: nd.inputs)
                        {
                            const Shape &si = SH(in);
                            if (si.empty())
                            {
                                sum = -1;
                                break;
                            }
                            sum += si[axis];
                        }
                        if (sum >= 0)
                        {
                            s[axis] = sum;
                            SH(o)   = s;
                        }
                    }
                    break;
                }
                case OpType::Reshape: {
                    TensorId sid = nd.inputs[1];
                    if (!g.isInitializer(sid))
                    {
                        break; // shape becomes const after constFold; 2nd pass fills it
                    }
                    const HostBuffer &hb   = g.initializers[sid];
                    const Shape      &in   = SH(nd.inputs[0]);
                    int64_t           rank = numElements(g.desc(sid).shape);
                    if (rank <= 0)
                    {
                        rank = (int64_t) (hb.bytes.size() / 8);
                    }
                    Shape   out(rank);
                    int64_t known = 1, infer = -1;
                    for (int64_t i = 0; i < rank; ++i)
                    {
                        int64_t d = hb.i64()[i];
                        if (d == 0)
                        {
                            d = (i < (int64_t) in.size()) ? in[i] : 1;
                        }
                        out[i] = d;
                        if (d == -1)
                        {
                            infer = i;
                        } else
                        {
                            known *= d;
                        }
                    }
                    if (infer >= 0)
                    {
                        out[infer] = numElements(in) / std::max<int64_t>(known, 1);
                    }
                    SH(o) = out;
                    break;
                }
                case OpType::Expand: {
                    // out = numpy-broadcast(in.shape, target). target is the int64 input[1].
                    const Shape         &in  = SH(nd.inputs[0]);
                    std::vector<int64_t> tgt = readI64Param(g, nd, "shape", 1);
                    if (tgt.empty())
                    {
                        break; // target const after constFold; resolved on a later pass
                    }
                    int   rank = (int) std::max(in.size(), tgt.size());
                    Shape out(rank, 1);
                    for (int k = 0; k < rank; ++k)
                    {
                        int64_t a = (k >= rank - (int) in.size()) ? in[k - (rank - (int) in.size())] : 1;
                        int64_t b = (k >= rank - (int) tgt.size()) ? tgt[k - (rank - (int) tgt.size())] : 1;
                        if (b < 0)
                        {
                            b = 1;
                        }
                        out[k] = std::max<int64_t>(a, b);
                    }
                    SH(o) = out;
                    break;
                }
                case OpType::Tile: {
                    // out.shape[k] = in.shape[k] * repeats[k]. repeats is the int64 input[1].
                    const Shape         &in   = SH(nd.inputs[0]);
                    std::vector<int64_t> reps = readI64Param(g, nd, "repeats", 1);
                    if (reps.empty())
                    {
                        break; // repeats const after constFold; resolved on a later pass
                    }
                    Shape out = in;
                    for (int k = 0; k < (int) in.size(); ++k)
                    {
                        out[k] = in[k] * std::max<int64_t>((k < (int) reps.size()) ? reps[k] : 1, 1);
                    }
                    SH(o) = out;
                    break;
                }
                case OpType::Squeeze: {
                    // remove the listed size-1 axes, or every size-1 dim when axes is absent.
                    const Shape &in = SH(nd.inputs[0]);
                    if (in.empty())
                    {
                        break;
                    }
                    int                  rank = (int) in.size();
                    std::vector<int64_t> axes = readI64Param(g, nd, "axes", 1);
                    std::vector<bool>    drop(rank, false);
                    if (axes.empty())
                    {
                        for (int k = 0; k < rank; ++k)
                        {
                            drop[k] = (in[k] == 1);
                        }
                    } else
                    {
                        for (int64_t ax: axes)
                        {
                            if (ax < 0)
                            {
                                ax += rank;
                            }
                            if (ax >= 0 && ax < rank)
                            {
                                drop[ax] = true;
                            }
                        }
                    }
                    Shape out;
                    for (int k = 0; k < rank; ++k)
                    {
                        if (!drop[k])
                        {
                            out.push_back(in[k]);
                        }
                    }
                    if (out.empty())
                    {
                        out.push_back(1);
                    }
                    SH(o) = out;
                    break;
                }
                case OpType::Gather: {
                    // ONNX Gather: out = data.shape[:axis] + indices.shape + data.shape[axis+1:]. A scalar
                    // index (rank-0, stored here as [1] with one element) removes the axis. Mirrors GatherCpu
                    // exactly so the inferred plan-time shape matches the runtime result (axis-aware).
                    const Shape &d = SH(nd.inputs[0]);
                    if (d.empty())
                    {
                        break;
                    }
                    int64_t rank = (int64_t) d.size();
                    int64_t axis = nd.attr.geti("axis", 0);
                    if (axis < 0)
                    {
                        axis += rank;
                    }
                    if (axis < 0 || axis >= rank)
                    {
                        break;
                    }
                    TensorId     iid = nd.inputs[1];
                    const Shape &is  = SH(iid);
                    if (is.empty() && !g.isInitializer(iid))
                    {
                        break; // indices not resolved yet
                    }
                    int64_t nidx        = is.empty() ? 1 : numElements(is);
                    bool    scalarIndex = is.empty() || (is.size() == 1 && is[0] == 1 && nidx == 1);
                    Shape   out;
                    for (int64_t i = 0; i < axis; ++i)
                    {
                        out.push_back(d[i]);
                    }
                    if (!scalarIndex)
                    {
                        for (int64_t v: is)
                        {
                            out.push_back(v);
                        }
                    }
                    for (int64_t i = axis + 1; i < rank; ++i)
                    {
                        out.push_back(d[i]);
                    }
                    if (out.empty())
                    {
                        out.push_back(1);
                    }
                    SH(o) = out;
                    break;
                }
                case OpType::Unsqueeze: {
                    // Insert size-1 dims at `axes` (attr for opset<13, input[1] for opset>=13). Mirrors
                    // UnsqueezeCpu: sort axes, normalize negatives against the growing rank, insert.
                    TensorId     xid = nd.inputs[0];
                    const Shape &in  = SH(xid);
                    if (in.empty() && !g.isInitializer(xid))
                    {
                        break; // input not resolved (allow rank-0 init)
                    }
                    std::vector<int64_t> axes = readI64Param(g, nd, "axes", 1);
                    if (axes.empty())
                    {
                        break; // a real Unsqueeze always has axes; wait for the param to const-fold
                    }
                    Shape out = in;
                    std::sort(axes.begin(), axes.end());
                    for (int64_t ax: axes)
                    {
                        if (ax < 0)
                        {
                            ax += (int64_t) out.size() + 1;
                        }
                        ax = std::max<int64_t>(0, std::min<int64_t>(ax, (int64_t) out.size()));
                        out.insert(out.begin() + ax, 1);
                    }
                    SH(o) = out;
                    break;
                }
                default:
                    break; // shape-path ops resolved by constFold
            }
        }
    }

    // Lower the two batched-matmul Einsum equations to Unsqueeze + MatMul (+ Squeeze) so they run on
    // the validated flat MatMul GPU kernel instead of the CPU einsum. The remaining "i,j->ij" outer
    // product keeps its own GPU kernel. Run after shapes are resolved (the einsum operand shapes must
    // be known).
    void lowerEinsum(Graph &g) {
        auto axesAttr = [](std::vector<int64_t> ax) {
            Attr a;
            a.kind = Attr::Ints;
            a.ints = std::move(ax);
            return a;
        };
        std::vector<Node> added;
        std::vector<int>  remove;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &n = g.nodes[i];
            if (n.type != OpType::Einsum || n.inputs.size() < 2)
            {
                continue;
            }
            std::string eq;
            for (char c: n.attr.gets("equation", ""))
            {
                if (c != ' ' && c != '\t')
                {
                    eq += c;
                }
            }
            TensorId A = n.inputs[0], B = n.inputs[1], out = n.outputs[0];
            // Copy shapes/descs/names by value up front: g.addTensor() below reallocates g.tensors, which
            // would dangle any reference held into it.
            if (eq == "...ab,...b->...a")
            {
                // y[...,a] = sum_b A[...,a,b]*x[...,b]  ==  Squeeze(MatMul(A, Unsqueeze(x,-1)), -1)
                Shape xs = g.desc(B).shape;
                if (xs.empty())
                {
                    continue;
                }
                int64_t    xrank    = (int64_t) xs.size();
                Shape      outShape = g.desc(out).shape;
                int64_t    outRank  = (int64_t) outShape.size();
                TensorDesc dxp      = g.desc(B);
                dxp.name            = dxp.name + "#e_unsq";
                dxp.isInitializer = dxp.isInput = dxp.isOutput = false;
                dxp.shape                                      = xs;
                dxp.shape.push_back(1);
                TensorId   xp     = g.addTensor(dxp);
                TensorDesc dmm    = g.desc(out);
                dmm.name          = dmm.name + "#e_mm";
                dmm.isInitializer = dmm.isInput = dmm.isOutput = false;
                dmm.shape                                      = outShape;
                dmm.shape.push_back(1);
                TensorId mm = g.addTensor(dmm);
                Node     un;
                un.type             = OpType::Unsqueeze;
                un.name             = n.name + "#unsq";
                un.inputs           = {B};
                un.outputs          = {xp};
                un.attr.map["axes"] = axesAttr({xrank});
                Node mul;
                mul.type    = OpType::MatMul;
                mul.name    = n.name + "#mm";
                mul.inputs  = {A, xp};
                mul.outputs = {mm};
                Node sq;
                sq.type             = OpType::Squeeze;
                sq.name             = n.name + "#sq";
                sq.inputs           = {mm};
                sq.outputs          = {out};
                sq.attr.map["axes"] = axesAttr({outRank});
                added.push_back(un);
                added.push_back(mul);
                added.push_back(sq);
                remove.push_back((int) i);
            } else if (eq == "bij,bnjk->bnik")
            {
                // C[b,n,i,k] = sum_j A[b,i,j]*B[b,n,j,k]  ==  MatMul(Unsqueeze(A,1), B)  (A broadcasts over
                // n)
                Shape as = g.desc(A).shape;
                if (as.empty())
                {
                    continue;
                }
                TensorDesc dap    = g.desc(A);
                dap.name          = dap.name + "#e_unsq";
                dap.isInitializer = dap.isInput = dap.isOutput = false;
                dap.shape                                      = as;
                dap.shape.insert(dap.shape.begin() + 1, 1);
                TensorId ap = g.addTensor(dap);
                Node     un;
                un.type             = OpType::Unsqueeze;
                un.name             = n.name + "#unsq";
                un.inputs           = {A};
                un.outputs          = {ap};
                un.attr.map["axes"] = axesAttr({1});
                Node mul;
                mul.type    = OpType::MatMul;
                mul.name    = n.name + "#mm";
                mul.inputs  = {ap, B};
                mul.outputs = {out};
                added.push_back(un);
                added.push_back(mul);
                remove.push_back((int) i);
            }
        }
        if (!added.empty())
        {
            std::set<int>     rm(remove.begin(), remove.end());
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!rm.count((int) i))
                {
                    kept.push_back(std::move(g.nodes[i]));
                }
            }
            for (auto &a: added)
            {
                kept.push_back(std::move(a));
            }
            g.nodes = std::move(kept);
            g.topoSort();
            VKNN_INFO << "lowerEinsum: lowered " << remove.size() << " batched einsum(s) to MatMul";
        }
    }

    int constFold(Graph &g) {
        std::set<TensorId> known;
        for (auto &kv: g.initializers)
        {
            known.insert(kv.first);
        }
        std::vector<RtTensor> pool(g.tensors.size());
        for (size_t i = 0; i < pool.size(); ++i)
        {
            pool[i].id    = (TensorId) i;
            pool[i].shape = g.tensors[i].shape;
            pool[i].dtype = g.tensors[i].dtype;
            if (g.isInitializer((TensorId) i))
            {
                pool[i].host      = g.initializers[i];
                pool[i].hostValid = true;
            }
        }
        ExecContext ctx;
        ctx.pool  = &pool;
        ctx.graph = &g;
        Config cfg;
        ctx.config = &cfg;

        std::set<int> removeNodes;
        auto          foldable = [&](const Node &nd) {
            switch (nd.type)
            {
                case OpType::Constant:
                    return true;
                case OpType::Shape:
                    return !g.desc(nd.inputs[0]).shape.empty(); // shape known
                // Any op whose every input is a known constant can be evaluated now. This collapses the
                // shape-arithmetic that detection heads (YOLO) build at runtime — Shape/Gather feeding scalar
                // Binary/Add to derive per-level strides — into plain constants, so those ops never need a
                // backend at all (neither CPU nor GPU).
                case OpType::Gather:
                case OpType::Unsqueeze:
                case OpType::Squeeze:
                case OpType::Concat:
                case OpType::Binary:
                case OpType::Add:
                case OpType::Reshape:
                case OpType::Slice:
                case OpType::Transpose:
                case OpType::Cast:
                case OpType::Reduce: {
                    if (nd.inputs.empty())
                    {
                        return false;
                    }
                    for (TensorId in: nd.inputs)
                    {
                        if (in != kNoTensor && !known.count(in))
                        {
                            return false;
                        }
                    }
                    return true;
                }
                // The transformer's dynamic-shape subgraph computes Reshape/Expand/Tile target vectors with
                // float arithmetic that passes through Where/Equal (e.g. Where(Equal(dim,-1), computed, dim))
                // and ConstantOfShape. Fold these when all inputs are constant so the targets become readable
                // initializers — but bound the output size so a large all-const fill/select isn't baked in
                // (its shape is still inferred by inferShapes, and it runs at runtime).
                case OpType::Where:
                case OpType::Equal:
                case OpType::EyeLike: // identity matrix is constant once the (now-known) shape is fixed
                case OpType::ConstantOfShape: {
                    if (nd.inputs.empty())
                    {
                        return false;
                    }
                    int64_t maxElems = 1;
                    for (TensorId in: nd.inputs)
                    {
                        if (in == kNoTensor)
                        {
                            continue;
                        }
                        if (!known.count(in))
                        {
                            return false;
                        }
                        maxElems = std::max(maxElems, numElements(g.desc(in).shape));
                    }
                    if (nd.type == OpType::ConstantOfShape)
                    {
                        // output size = product of the const shape-vector's values
                        auto it = g.initializers.find(nd.inputs[0]);
                        if (it == g.initializers.end())
                        {
                            return false;
                        }
                        const HostBuffer &hb  = it->second;
                        bool              i64 = g.desc(nd.inputs[0]).dtype == DType::Int64;
                        int64_t           r   = numElements(g.desc(nd.inputs[0]).shape);
                        if (r <= 0)
                        {
                            r = (int64_t) (hb.bytes.size() / (i64 ? 8 : 4));
                        }
                        int64_t prod = 1;
                        for (int64_t i = 0; i < r; ++i)
                        {
                            int64_t dv = i64 ? hb.i64()[i] : (int64_t) hb.f32()[i];
                            prod *= (dv > 0 ? dv : 1);
                        }
                        maxElems = prod;
                    }
                    return maxElems <= (1 << 16);
                }
                // Expand/Tile of all-constant operands fold too (bounded). Required for integer index/shape
                // tensors such as the RoPE position arange (int64, built via Expand->Add->Reshape->Gather):
                // on the GPU float path the int64 positions corrupt to zeros and the rotary embedding loses
                // all position information, so folding computes the small constant index on the CPU exactly.
                case OpType::Expand:
                case OpType::Tile: {
                    if (nd.inputs.size() < 2)
                    {
                        return false;
                    }
                    for (TensorId in: nd.inputs)
                    {
                        if (in != kNoTensor && !known.count(in))
                        {
                            return false;
                        }
                    }
                    const Shape         &in  = g.desc(nd.inputs[0]).shape;
                    std::vector<int64_t> p   = readI64Param(g, nd, nd.type == OpType::Tile ? "repeats" : "shape", 1);
                    int64_t              out = 1;
                    if (nd.type == OpType::Tile)
                    {
                        for (size_t k = 0; k < in.size(); ++k)
                        {
                            out *= in[k] * std::max<int64_t>(k < p.size() ? p[k] : 1, 1);
                        }
                    } else
                    { // Expand: numpy broadcast of in.shape against the target
                        int rank = (int) std::max(in.size(), p.size());
                        for (int k = 0; k < rank; ++k)
                        {
                            int64_t a = (k >= rank - (int) in.size()) ? in[k - (rank - (int) in.size())] : 1;
                            int64_t b = (k >= rank - (int) p.size()) ? p[k - (rank - (int) p.size())] : 1;
                            out *= std::max<int64_t>(std::max(a, b), 1);
                        }
                    }
                    // Integer (index / shape) tensors must fold even when large: they cannot run on the GPU's
                    // float buffers, where an int64 value reinterpreted as float corrupts to ~0 (e.g. a large
                    // ScatterND upsample meshgrid index would come out all-zeros and scatter everything to
                    // token 0). Float tensors keep the small bound so a large all-const broadcast is not baked
                    // into the model.
                    DType idt   = g.desc(nd.inputs[0]).dtype;
                    bool  isInt = idt == DType::Int64 || idt == DType::Int32;
                    return out > 0 && out <= (isInt ? (int64_t(1) << 26) : (int64_t(1) << 18));
                }
                default:
                    return false;
            }
        };

        for (size_t ni = 0; ni < g.nodes.size(); ++ni)
        {
            Node &nd = g.nodes[ni];
            if (!foldable(nd))
            {
                continue;
            }
            // ensure Shape's input has a shape-only RtTensor
            if (nd.type == OpType::Shape)
            {
                pool[nd.inputs[0]].shape = g.desc(nd.inputs[0]).shape;
            }
            auto op = CpuOpRegistry::instance().create(nd.type);
            if (!op)
            {
                continue;
            }
            try
            { op->run(nd, ctx); } catch (...)
            { continue; }
            for (TensorId o: nd.outputs)
            {
                if (o == kNoTensor)
                {
                    continue;
                }
                g.initializers[o]       = pool[o].host;
                g.desc(o).isInitializer = true;
                g.desc(o).shape         = pool[o].shape;
                g.desc(o).dtype         = pool[o].dtype;
                known.insert(o);
            }
            removeNodes.insert((int) ni);
        }
        if (!removeNodes.empty())
        {
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!removeNodes.count((int) i))
                {
                    kept.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(kept);
            VKNN_INFO << "constFold: folded " << removeNodes.size() << " shape-path node(s)";
        }
        return (int) removeNodes.size();
    }

    void foldBatchNorm(Graph &g) {
        // Fold BN(Conv(x)) -> Conv with adjusted weights/bias. MobileNetV2 ships BN pre-folded,
        // so this is typically a no-op here, but implemented for correctness on other models.
        int           folded = 0;
        std::set<int> remove;
        // map tensor -> producing node index
        std::vector<int> producer(g.tensors.size(), -1);
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId o: g.nodes[i].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[o] = (int) i;
                }
            }
        }

        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &bn = g.nodes[i];
            if (bn.type != OpType::BatchNorm)
            {
                continue;
            }
            int pi = producer[bn.inputs[0]];
            if (pi < 0 || g.nodes[pi].type != OpType::Conv)
            {
                continue;
            }
            Node &conv = g.nodes[pi];
            // only fold if conv output feeds only this BN
            int consumers = 0;
            for (auto &nn: g.nodes)
            {
                for (TensorId in: nn.inputs)
                {
                    if (in == conv.outputs[0])
                    {
                        consumers++;
                    }
                }
            }
            if (consumers != 1)
            {
                continue;
            }

            const auto &scale = g.initializers[bn.inputs[1]].f32();
            const auto &bias  = g.initializers[bn.inputs[2]].f32();
            const auto &mean  = g.initializers[bn.inputs[3]].f32();
            const auto &var   = g.initializers[bn.inputs[4]].f32();
            float       eps   = bn.attr.getf("epsilon", 1e-5f);
            int64_t     outC  = g.desc(conv.inputs[1]).shape[0];
            int64_t     perOC = numElements(g.desc(conv.inputs[1]).shape) / outC;
            HostBuffer &W     = g.initializers[conv.inputs[1]];
            // ensure bias exists
            TensorId   biasId = (conv.inputs.size() > 2 && conv.inputs[2] != kNoTensor) ? conv.inputs[2] : kNoTensor;
            HostBuffer biasBuf;
            if (biasId == kNoTensor)
            {
                biasBuf.resizeElems(outC, DType::Float32);
            }
            HostBuffer &Bb = (biasId == kNoTensor) ? biasBuf : g.initializers[biasId];
            for (int64_t oc = 0; oc < outC; ++oc)
            {
                float  a = scale[oc] / std::sqrt(var[oc] + eps);
                float *w = W.f32() + oc * perOC;
                for (int64_t k = 0; k < perOC; ++k)
                {
                    w[k] *= a;
                }
                Bb.f32()[oc] = (Bb.f32()[oc] - mean[oc]) * a + bias[oc];
            }
            if (biasId == kNoTensor)
            {
                TensorDesc d;
                d.name             = conv.name + "_bias";
                d.shape            = {outC};
                d.isInitializer    = true;
                TensorId nb        = g.addTensor(d);
                g.initializers[nb] = std::move(biasBuf);
                if (conv.inputs.size() < 3)
                {
                    conv.inputs.resize(3, kNoTensor);
                }
                conv.inputs[2] = nb;
            }
            // rewire: BN consumers now read conv output
            TensorId bnOut = bn.outputs[0], convOut = conv.outputs[0];
            for (auto &nn: g.nodes)
            {
                for (TensorId &in: nn.inputs)
                {
                    if (in == bnOut)
                    {
                        in = convOut;
                    }
                }
            }
            for (TensorId &go: g.outputs)
            {
                if (go == bnOut)
                {
                    go = convOut;
                }
            }
            remove.insert((int) i);
            folded++;
        }
        if (folded)
        {
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!remove.count((int) i))
                {
                    kept.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(kept);
            VKNN_INFO << "foldBatchNorm: folded " << folded << " BN node(s) into Conv";
        }
    }

    void fuseActivations(Graph &g) {
        std::vector<int> producer(g.tensors.size(), -1);
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId o: g.nodes[i].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[o] = (int) i;
                }
            }
        }
        std::set<int> remove;
        int           fused = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &act = g.nodes[i];
            if (act.type != OpType::Clip && act.type != OpType::Relu)
            {
                continue;
            }
            int pi = producer[act.inputs[0]];
            if (pi < 0)
            {
                continue;
            }
            Node &prod = g.nodes[pi];
            if (prod.type != OpType::Conv && prod.type != OpType::Gemm && prod.type != OpType::Add)
            {
                continue;
            }
            if (prod.fusedAct != ActType::None)
            {
                continue;
            }
            // producer output must feed only this activation
            int consumers = 0;
            for (auto &nn: g.nodes)
            {
                for (TensorId in: nn.inputs)
                {
                    if (in == prod.outputs[0])
                    {
                        consumers++;
                    }
                }
            }
            for (TensorId go: g.outputs)
            {
                if (go == prod.outputs[0])
                {
                    consumers++;
                }
            }
            if (consumers != 1)
            {
                continue;
            }

            if (act.type == OpType::Relu)
            {
                prod.fusedAct = ActType::Relu;
            } else
            {
                float lo = 0, hi = 6; // default relu6
                if (act.inputs.size() > 1 && act.inputs[1] != kNoTensor && g.isInitializer(act.inputs[1]))
                {
                    lo = g.initializers[act.inputs[1]].f32()[0];
                }
                if (act.inputs.size() > 2 && act.inputs[2] != kNoTensor && g.isInitializer(act.inputs[2]))
                {
                    hi = g.initializers[act.inputs[2]].f32()[0];
                }
                if (act.attr.has("min"))
                {
                    lo = act.attr.getf("min", lo);
                }
                if (act.attr.has("max"))
                {
                    hi = act.attr.getf("max", hi);
                }
                prod.fusedAct = (lo == 0.f && hi == 6.f) ? ActType::Relu6 : ActType::Clip;
                prod.actLo    = lo;
                prod.actHi    = hi;
            }
            // rewire consumers of act output to producer output
            TensorId actOut = act.outputs[0], prodOut = prod.outputs[0];
            for (auto &nn: g.nodes)
            {
                for (TensorId &in: nn.inputs)
                {
                    if (in == actOut)
                    {
                        in = prodOut;
                    }
                }
            }
            for (TensorId &go: g.outputs)
            {
                if (go == actOut)
                {
                    go = prodOut;
                }
            }
            remove.insert((int) i);
            fused++;
        }
        if (fused)
        {
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!remove.count((int) i))
                {
                    kept.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(kept);
            VKNN_INFO << "fuseActivations: fused " << fused << " activation(s) into Conv/Gemm";
        }
    }

    void eliminateIdentity(Graph &g) {
        // Drop Identity nodes by pointing their consumers (and any graph output) straight at the input.
        std::set<int> remove;
        int           n = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &id = g.nodes[i];
            if (id.type != OpType::Identity)
            {
                continue;
            }
            if (id.inputs.empty() || id.outputs.empty())
            {
                continue;
            }
            TensorId in = id.inputs[0], out = id.outputs[0];
            for (auto &nn: g.nodes)
            {
                for (TensorId &x: nn.inputs)
                {
                    if (x == out)
                    {
                        x = in;
                    }
                }
            }
            for (TensorId &go: g.outputs)
            {
                if (go == out)
                {
                    go = in;
                }
            }
            remove.insert((int) i);
            ++n;
        }
        if (n)
        {
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!remove.count((int) i))
                {
                    kept.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(kept);
            VKNN_INFO << "eliminateIdentity: removed " << n << " Identity node(s)";
        }
    }

    // Drop Cast nodes that convert float -> float. Storage precision is uniform across a segment, so a
    // float->float cast is a same-size buffer copy (CastOp is a vkCmdCopyBuffer) — a wasted dispatch,
    // barrier, and full intermediate round-trip. Transformer graphs emit hundreds (RoPE/attention/
    // layernorm chains). A forward dtype propagation, seeded from initializers and the float graph
    // inputs, gates the removal strictly to a float input and a float ONNX target so genuine
    // int<->float casts (shape / index paths) are left intact.
    void eliminateFloatCast(Graph &g) {
        auto onnxToIsFloat = [](int64_t to) { return to == 1 || to == 10 || to == 11; }; // FLOAT/FLOAT16/DOUBLE
        auto isFloat       = [](DType d) { return d == DType::Float32 || d == DType::Float16; };
        // float-result math ops: their output is float regardless of an int-typed input.
        auto floatResult = [](OpType t) {
            switch (t)
            {
                case OpType::Conv:
                case OpType::Gemm:
                case OpType::MatMul:
                case OpType::Einsum:
                case OpType::Softmax:
                case OpType::LayerNorm:
                case OpType::BatchNorm:
                case OpType::Reduce:
                case OpType::GlobalAvgPool:
                case OpType::AvgPool:
                case OpType::Resize:
                case OpType::GridSample:
                case OpType::FusedSE:
                case OpType::FusedDwPw:
                    return true;
                default:
                    return false;
            }
        };
        std::vector<DType> dt(g.tensors.size(), DType::Float32);
        std::vector<char>  known(g.tensors.size(), 0);
        auto               setk = [&](TensorId id, DType d) {
            if (id >= 0 && id < (TensorId) dt.size())
            {
                dt[id]    = d;
                known[id] = 1;
            }
        };
        for (TensorId id = 0; id < (TensorId) g.tensors.size(); ++id)
        {
            if (g.tensors[id].isInitializer)
            {
                setk(id, g.tensors[id].dtype);
            }
        }
        for (TensorId id: g.inputs)
        {
            setk(id, DType::Float32); // model inputs are float (image / intrinsics / ...)
        }
        // Forward pass (nodes are topo-ordered after import): assign each output a dtype.
        for (const Node &nd: g.nodes)
        {
            if (nd.outputs.empty())
            {
                continue;
            }
            DType out;
            if (nd.type == OpType::Shape)
            {
                out = DType::Int64;
            } else if (nd.type == OpType::Cast)
            {
                out = onnxToIsFloat(nd.attr.geti("to", 1)) ? DType::Float32 : DType::Int64;
            } else if (nd.type == OpType::Equal)
            {
                out = DType::Int32; // boolean result, not float
            } else if (floatResult(nd.type))
            {
                out = DType::Float32;
            } else
            {
                // Elementwise / shape-movement ops carry their primary data input's dtype (Where: the X
                // branch, input[1]). Unknown input -> leave the output unknown (conservative).
                int      pidx = nd.type == OpType::Where ? 1 : 0;
                TensorId pin  = (int) nd.inputs.size() > pidx ? nd.inputs[pidx] : kNoTensor;
                if (pin == kNoTensor || pin >= (TensorId) known.size() || !known[pin])
                {
                    continue;
                }
                out = dt[pin];
            }
            for (TensorId o: nd.outputs)
            {
                setk(o, out);
            }
        }
        // Remove float->float casts, redirecting consumers to the cast's input (graph outputs untouched
        // so a named output is never renamed).
        std::set<int> remove;
        int           n = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &c = g.nodes[i];
            if (c.type != OpType::Cast || c.inputs.size() != 1 || c.outputs.size() != 1)
            {
                continue;
            }
            TensorId in = c.inputs[0], out = c.outputs[0];
            if (in == kNoTensor || !onnxToIsFloat(c.attr.geti("to", 1)))
            {
                continue;
            }
            if (in >= (TensorId) known.size() || !known[in] || !isFloat(dt[in]))
            {
                continue;
            }
            bool isGraphOut = false;
            for (TensorId go: g.outputs)
            {
                if (go == out)
                {
                    isGraphOut = true;
                    break;
                }
            }
            if (isGraphOut)
            {
                continue;
            }
            for (auto &nn: g.nodes)
            {
                for (TensorId &x: nn.inputs)
                {
                    if (x == out)
                    {
                        x = in;
                    }
                }
            }
            remove.insert((int) i);
            ++n;
        }
        if (n)
        {
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!remove.count((int) i))
                {
                    kept.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(kept);
            VKNN_INFO << "eliminateFloatCast: removed " << n << " float->float Cast node(s)";
        }
    }

    void fuseResidualAdd(Graph &g) {
        // Fuse  out = Add(pointwise_conv(x), residual)  into the conv's epilogue, so the conv writes
        // conv+residual directly (saves the Add's full read+write). Only for 1x1 stride-1 pad-0 group-1
        // convs (the ones our conv1x1/split-K kernels run and support a residual on).
        std::vector<int> producer(g.tensors.size(), -1);
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId o: g.nodes[i].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[o] = (int) i;
                }
            }
        }
        auto convEligible = [&](const Node &c) {
            if (c.type != OpType::Conv || c.fusedResidual != kNoTensor)
            {
                return false;
            }
            auto ints = [&](const char *k, std::vector<int64_t> d) {
                const auto &v = c.attr.getints(k);
                return v.empty() ? d : v;
            };
            auto k = ints("kernel_shape", {1, 1}), s = ints("strides", {1, 1});
            auto p = ints("pads", {0, 0, 0, 0});
            return c.attr.geti("group", 1) == 1 && k[0] == 1 && k[1] == 1 && s[0] == 1 && s[1] == 1 && p[0] == 0 && p[1] == 0 && p[2] == 0 && p[3] == 0;
        };
        std::set<int> remove;
        int           fused = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &add = g.nodes[i];
            if (add.type != OpType::Add || add.inputs.size() != 2)
            {
                continue;
            }
            int      p0 = producer[add.inputs[0]], p1 = producer[add.inputs[1]];
            int      ci       = -1;
            TensorId residual = kNoTensor;
            if (p0 >= 0 && convEligible(g.nodes[p0]))
            {
                ci       = p0;
                residual = add.inputs[1];
            } else if (p1 >= 0 && convEligible(g.nodes[p1]))
            {
                ci       = p1;
                residual = add.inputs[0];
            }
            if (ci < 0)
            {
                continue;
            }
            // the conv output must feed only this Add
            int consumers = 0;
            for (auto &nn: g.nodes)
            {
                for (TensorId in: nn.inputs)
                {
                    if (in == g.nodes[ci].outputs[0])
                    {
                        consumers++;
                    }
                }
            }
            for (TensorId go: g.outputs)
            {
                if (go == g.nodes[ci].outputs[0])
                {
                    consumers++;
                }
            }
            if (consumers != 1)
            {
                continue;
            }
            Node &conv         = g.nodes[ci];
            conv.fusedResidual = residual;
            conv.inputs.push_back(residual); // keep it live for DCE / buffer allocation / scheduling
            if (conv.fusedAct == ActType::None)
            { // carry any activation that was folded into the Add
                conv.fusedAct = add.fusedAct;
                conv.actLo    = add.actLo;
                conv.actHi    = add.actHi;
            }
            TensorId addOut = add.outputs[0], convOut = conv.outputs[0];
            for (auto &nn: g.nodes)
            {
                for (TensorId &in: nn.inputs)
                {
                    if (in == addOut)
                    {
                        in = convOut;
                    }
                }
            }
            for (TensorId &go: g.outputs)
            {
                if (go == addOut)
                {
                    go = convOut;
                }
            }
            remove.insert((int) i);
            fused++;
        }
        if (fused)
        {
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!remove.count((int) i))
                {
                    kept.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(kept);
            VKNN_INFO << "fuseResidualAdd: fused " << fused << " residual Add(s) into Conv";
        }
    }

    // Fuse  out = Add(MatMul(A, W), bias)  into the MatMul epilogue (out[...,n] = matmul + bias[n]), so
    // the Linear's bias is added in the fp32 accumulator and the result is stored once — removing the
    // Add's whole-tensor read + write and one fp16 requantization per Linear. The transformer's qkv /
    // proj / fc1 / fc2 layers are all this shape. Eligible only when: the MatMul output feeds only this
    // Add, the other Add operand is a rank-1 [N] (or [1,..,1,N]) initializer matching the output's last
    // dim, and the MatMul is the standard 2-D-operand case (the kernel indexes bias by output column).
    void fuseMatMulBias(Graph &g) {
        std::vector<int> producer(g.tensors.size(), -1);
        std::vector<int> consumers(g.tensors.size(), 0);
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId o: g.nodes[i].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[o] = (int) i;
                }
            }
            for (TensorId in: g.nodes[i].inputs)
            {
                if (in != kNoTensor)
                {
                    consumers[in]++;
                }
            }
        }
        for (TensorId go: g.outputs)
        {
            if (go != kNoTensor)
            {
                consumers[go]++;
            }
        }
        std::set<int> remove;
        int           fused = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &add = g.nodes[i];
            if (add.type != OpType::Add || add.inputs.size() != 2 || add.fusedAct != ActType::None)
            {
                continue;
            }
            // One operand is a MatMul output (single-consumer), the other a rank-1 [N] bias initializer.
            int mi = -1, mmIdx = -1;
            for (int e = 0; e < 2; ++e)
            {
                int p = producer[add.inputs[e]];
                if (p >= 0 && g.nodes[p].type == OpType::MatMul && g.nodes[p].fusedBias == kNoTensor && consumers[add.inputs[e]] == 1)
                {
                    mmIdx = p;
                    mi    = e;
                    break;
                }
            }
            if (mmIdx < 0)
            {
                continue;
            }
            TensorId biasId = add.inputs[1 - mi];
            if (!g.isInitializer(biasId))
            {
                continue;
            }
            Node        &mm  = g.nodes[mmIdx];
            const Shape &os  = g.desc(mm.outputs[0]).shape;
            const Shape &bs  = g.desc(biasId).shape;
            const Shape &as0 = g.desc(mm.inputs[0]).shape;
            const Shape &as1 = g.desc(mm.inputs[1]).shape;
            // Standard 2-D-operand MatMul (no 1-D promotion); bias is a per-output-column vector [.,N].
            if (os.empty() || as0.size() < 2 || as1.size() < 2)
            {
                continue;
            }
            int64_t N = os.back();
            if (bs.empty() || bs.back() != N || numElements(bs) != N)
            {
                continue;
            }
            mm.fusedBias   = biasId;
            mm.inputs.push_back(biasId); // keep the bias live for DCE / buffer allocation / scheduling
            mm.outputs[0]  = add.outputs[0]; // MatMul now produces the (biased) Add output, name intact
            remove.insert((int) i);
            ++fused;
        }
        if (fused)
        {
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!remove.count((int) i))
                {
                    kept.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(kept);
            VKNN_INFO << "fuseMatMulBias: fused " << fused << " bias Add(s) into MatMul";
        }
    }

    void fuseSqueezeExcite(Graph &g) {
        // Collapse the Squeeze-Excite scale chain GlobalAvgPool -> Conv1x1(+relu) -> Conv1x1 ->
        // HardSigmoid into ONE kFusedSE node that emits the channel scale. The following
        // channel-broadcast Mul (scale * feature) is left intact. MobileNetV3 has ~11 of these tiny
        // multi-dispatch chains.
        std::vector<int> producer(g.tensors.size(), -1);
        std::vector<int> consumers(g.tensors.size(), 0);
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId o: g.nodes[i].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[o] = (int) i;
                }
            }
            for (TensorId in: g.nodes[i].inputs)
            {
                if (in != kNoTensor)
                {
                    consumers[in]++;
                }
            }
        }
        auto single = [&](TensorId t) {
            return t != kNoTensor && consumers[t] == 1;
        };
        std::set<int> remove;
        int           fused = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &hs = g.nodes[i];
            if (hs.type != OpType::Unary || (UnaryType) hs.subOp != UnaryType::HardSigmoid)
            {
                continue;
            }
            int p2 = producer[hs.inputs[0]];
            if (p2 < 0 || g.nodes[p2].type != OpType::Conv || !single(g.nodes[p2].outputs[0]))
            {
                continue;
            }
            Node &conv2 = g.nodes[p2];
            int   p1    = producer[conv2.inputs[0]];
            if (p1 < 0 || g.nodes[p1].type != OpType::Conv || !single(g.nodes[p1].outputs[0]))
            {
                continue;
            }
            Node &conv1 = g.nodes[p1];
            if (conv1.fusedAct != ActType::Relu)
            {
                continue;
            }
            // Require a GlobalAvgPool feeding conv1, but KEEP it (its reduction is parallel). We fuse only
            // the tiny [N,C,1,1] middle: conv1(relu)->conv2->hardsigmoid -> one kernel that reads the
            // pooled avg directly. Fusing the pool into one workgroup regressed; this keeps GAP + Mul
            // parallel.
            int pg = producer[conv1.inputs[0]];
            if (pg < 0 || g.nodes[pg].type != OpType::GlobalAvgPool)
            {
                continue;
            }
            TensorId avg  = conv1.inputs[0]; // the pooled [N,C,1,1] tensor
            auto     bias = [](const Node &c) {
                return c.inputs.size() > 2 ? c.inputs[2] : kNoTensor;
            };
            Node se;
            se.type    = OpType::FusedSE;
            se.name    = conv1.name + "#se";
            se.inputs  = {avg, conv1.inputs[1], bias(conv1), conv2.inputs[1], bias(conv2)};
            se.outputs = {hs.outputs[0]}; // the scale tensor (Mul still consumes it)
            se.actLo   = hs.actLo;        // hardsigmoid alpha/beta
            se.actHi   = hs.actHi;
            g.nodes[i] = se;   // replace hardsigmoid node with the fused node
            remove.insert(p1); // remove conv1 + conv2 (gap stays)
            remove.insert(p2);
            fused++;
        }
        if (fused)
        {
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!remove.count((int) i))
                {
                    kept.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(kept);
            VKNN_INFO << "fuseSqueezeExcite: fused " << fused << " SE chain(s)";
        }
    }

    void fuseSwish(Graph &g) {
        // Fuse the elementwise self-gating activation Mul(x, HardSigmoid(x)) = HardSwish and
        // Mul(x, Sigmoid(x)) = SiLU/Swish. If x is produced by a Conv/Gemm (consumed only by the gate +
        // the Mul), fold it into that op's epilogue activation (removes 2 dispatches + the intermediate);
        // otherwise collapse the [gate, Mul] pair into a single unary op. MobileNetV3 has ~21 of these.
        std::vector<int> producer(g.tensors.size(), -1);
        std::vector<int> consumers(g.tensors.size(), 0);
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId o: g.nodes[i].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[o] = (int) i;
                }
            }
            for (TensorId in: g.nodes[i].inputs)
            {
                if (in != kNoTensor)
                {
                    consumers[in]++;
                }
            }
        }
        std::set<int> remove;
        int           fusedC = 0, fusedU = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &M = g.nodes[i];
            if (M.type != OpType::Binary || (BinaryType) M.subOp != BinaryType::Mul || M.inputs.size() != 2)
            {
                continue;
            }
            int      sigIdx = -1;
            TensorId x      = kNoTensor;
            for (int e = 0; e < 2; ++e)
            {
                int pc = producer[M.inputs[e]];
                if (pc < 0 || g.nodes[pc].type != OpType::Unary)
                {
                    continue;
                }
                UnaryType su = (UnaryType) g.nodes[pc].subOp;
                if ((su == UnaryType::HardSigmoid || su == UnaryType::Sigmoid) && g.nodes[pc].inputs[0] == M.inputs[1 - e])
                {
                    sigIdx = pc;
                    x      = M.inputs[1 - e];
                    break;
                }
            }
            if (sigIdx < 0)
            {
                continue;
            }
            if (consumers[g.nodes[sigIdx].outputs[0]] != 1)
            {
                continue; // gate feeds only this Mul
            }
            ActType act = (UnaryType) g.nodes[sigIdx].subOp == UnaryType::HardSigmoid ? ActType::HardSwish : ActType::SiLU;
            int     px  = producer[x];
            bool fuseConv = px >= 0 && (g.nodes[px].type == OpType::Conv || g.nodes[px].type == OpType::Gemm) && g.nodes[px].fusedAct == ActType::None && consumers[x] == 2;
            if (fuseConv)
            {
                g.nodes[px].fusedAct = act;
                // Fold the Mul output onto the conv output. Must also patch any fusedResidual edge that
                // pointed at the Mul output (a residual already folded into a later conv by fuseResidualAdd),
                // else that conv reads a dead tensor -> crash.
                rewireTensor(g, M.outputs[0], x);
                remove.insert((int) i);
                remove.insert(sigIdx);
                fusedC++;
            } else
            {
                // collapse [gate, Mul] -> one unary
                M.type   = OpType::Unary;
                M.subOp  = (int) (act == ActType::HardSwish ? UnaryType::HardSwish : UnaryType::SiLU);
                M.inputs = {x};
                remove.insert(sigIdx);
                fusedU++;
            }
        }
        if (!remove.empty())
        {
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!remove.count((int) i))
                {
                    kept.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(kept);
            VKNN_INFO << "fuseSwish: fused " << fusedC << " into conv, collapsed " << fusedU << " to unary";
        }
    }

    void fuseDwPw(Graph &g) {
        // Fuse depthwise-3x3 conv (D) -> 1x1 project conv (P) into one kFusedDwPw node, so the expanded
        // intermediate (D's output, the block's largest activation) never hits global memory and a
        // dispatch+barrier is removed. Only when D's output feeds ONLY P, D's activation is a plain
        // ActType (Relu/Relu6/Clip/None), and P is a stride-1 pad-0 group-1 pointwise conv.
        std::vector<int> producer(g.tensors.size(), -1);
        std::vector<int> consumers(g.tensors.size(), 0);
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId o: g.nodes[i].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[o] = (int) i;
                }
            }
            for (TensorId in: g.nodes[i].inputs)
            {
                if (in != kNoTensor)
                {
                    consumers[in]++;
                }
            }
        }
        auto ints = [](const Node &n, const char *k, std::vector<int64_t> d) {
            const auto &v = n.attr.getints(k);
            return v.empty() ? d : v;
        };
        std::set<int> remove;
        int           fused = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &P = g.nodes[i];
            if (P.type != OpType::Conv)
            {
                continue;
            }
            const Shape &pw = g.desc(P.inputs[1]).shape; // [Cout, Cin, 1, 1]
            if (pw.size() != 4 || pw[2] != 1 || pw[3] != 1)
            {
                continue;
            }
            if (P.attr.geti("group", 1) != 1)
            {
                continue;
            }
            auto poolStrides = ints(P, "strides", {1, 1}), poolPads = ints(P, "pads", {0, 0, 0, 0});
            if (poolStrides[0] != 1 || poolStrides[1] != 1 || poolPads[0] || poolPads[1] || poolPads[2] || poolPads[3])
            {
                continue;
            }
            int prodIdx = producer[P.inputs[0]];
            if (prodIdx < 0 || g.nodes[prodIdx].type != OpType::Conv)
            {
                continue;
            }
            Node &D = g.nodes[prodIdx];
            if (consumers[D.outputs[0]] != 1)
            {
                continue; // D feeds only P
            }
            const Shape &dw        = g.desc(D.inputs[1]).shape; // [C,1,KH,KW]
            NCHW         dx        = NCHW::from(g.desc(D.inputs[0]).shape);
            bool         depthwise = (D.attr.geti("group", 1) == dx.c && dw.size() == 4 && dw[1] == 1);
            if (!depthwise)
            {
                continue;
            }
            // D's activation must be parameterless (None/Relu/Relu6); skip custom Clip and hardswish-dw.
            if (D.fusedAct != ActType::None && D.fusedAct != ActType::Relu && D.fusedAct != ActType::Relu6)
            {
                continue;
            }
            auto bias = [](const Node &c) {
                return c.inputs.size() > 2 ? c.inputs[2] : kNoTensor;
            };
            Node f;
            f.type   = OpType::FusedDwPw;
            f.name   = D.name + "#dwpw";
            f.inputs = {D.inputs[0], D.inputs[1], bias(D), P.inputs[1], bias(P)};
            if (P.fusedResidual != kNoTensor)
            {
                f.inputs.push_back(P.fusedResidual);
                f.fusedResidual = P.fusedResidual;
            }
            f.outputs  = {P.outputs[0]};
            f.attr     = D.attr;               // carry dw strides/pads/kernel for shape inference + kernel
            f.subOp    = (int32_t) D.fusedAct; // dw activation
            f.fusedAct = P.fusedAct;           // project activation
            f.actLo    = P.actLo;
            f.actHi    = P.actHi;
            g.nodes[i] = f;         // replace P with the fused node
            remove.insert(prodIdx); // remove D
            fused++;
        }
        if (fused)
        {
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!remove.count((int) i))
                {
                    kept.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(kept);
            VKNN_INFO << "fuseDwPw: fused " << fused << " depthwise+project pair(s)";
        }
    }

    void eliminateDeadNodes(Graph &g) {
        std::set<TensorId> live(g.outputs.begin(), g.outputs.end());
        bool               changed = true;
        std::vector<int>   producer(g.tensors.size(), -1);
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId o: g.nodes[i].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[o] = (int) i;
                }
            }
        }
        // propagate liveness backward
        while (changed)
        {
            changed = false;
            for (auto &nd: g.nodes)
            {
                bool nodeLive = false;
                for (TensorId o: nd.outputs)
                {
                    if (o != kNoTensor && live.count(o))
                    {
                        nodeLive = true;
                    }
                }
                if (!nodeLive)
                {
                    continue;
                }
                for (TensorId in: nd.inputs)
                {
                    if (in != kNoTensor && !live.count(in))
                    {
                        live.insert(in);
                        changed = true;
                    }
                }
            }
        }
        std::vector<Node> kept;
        int               removed = 0;
        for (auto &nd: g.nodes)
        {
            bool nodeLive = false;
            for (TensorId o: nd.outputs)
            {
                if (o != kNoTensor && live.count(o))
                {
                    nodeLive = true;
                }
            }
            if (nodeLive)
            {
                kept.push_back(nd);
            } else
            {
                removed++;
            }
        }
        if (removed)
        {
            g.nodes = std::move(kept);
            VKNN_INFO << "eliminateDeadNodes: removed " << removed << " node(s)";
        }
    }

    // Does this op run as a FLAT (row-major) GPU op rather than the NC4HW4 path? Mirrors the cases the
    // Vulkan supportsNode() can't do in NC4HW4: Transpose/Slice always; Softmax on a non-channel axis;
    // Concat that isn't 4D channel-axis 4-aligned; Binary/Add with a constant operand or a broadcast/
    // rank that the packed kernel doesn't handle.
    static bool gpuFlatNode(const Graph &g, const Node &n) {
        auto sh = [&](TensorId t) -> const Shape & {
            return g.desc(t).shape;
        };
        switch (n.type)
        {
            case OpType::Transpose:
            case OpType::Slice:
            case OpType::Expand:       // numpy broadcast to a target shape, flat row-major gather
            case OpType::Tile:         // repeat each dim, flat row-major gather
            case OpType::LayerNorm:    // always flat: reduce over the trailing axes, row-major
            case OpType::DepthToSpace: // spatial<->channel index remap, flat row-major gather
            case OpType::Reduce:       // generic N-D reduce (incl. ReduceL2) runs flat
            case OpType::MatMul:       // batched N-D matmul runs on the flat row-major path
            case OpType::Where:        // cond?X:Y, broadcasting flat select
            case OpType::Equal:        // A==B, broadcasting flat compare
            case OpType::Greater:      // A>B,  broadcasting flat compare
            case OpType::GreaterEqual: // A>=B, broadcasting flat compare
                return true;
            case OpType::ConvTranspose: {
                // Flat row-major transposed conv (one thread per output element, gather form). Needs a
                // 4D input and constant weight (uploaded flat); anything else falls back to the CPU op.
                if (sh(n.inputs[0]).size() != 4)
                {
                    return false;
                }
                if (n.inputs.size() < 2 || !g.isInitializer(n.inputs[1]))
                {
                    return false;
                }
                return !(n.inputs.size() > 2 && n.inputs[2] != kNoTensor && !g.isInitializer(n.inputs[2]));
            }
            case OpType::Pad: {
                // Flat row-major pad (constant/edge/reflect). Needs static pads (attr or a constant
                // input[1]) and rank within the flat limit; a runtime pad value falls back to CPU.
                if (sh(n.outputs[0]).size() > 8)
                {
                    return false;
                }
                std::string mode = n.attr.gets("mode", "constant");
                if (mode != "constant" && mode != "edge" && mode != "reflect")
                {
                    return false;
                }
                bool padsKnown = !n.attr.getints("pads").empty() || (n.inputs.size() > 1 && n.inputs[1] != kNoTensor && g.isInitializer(n.inputs[1]));
                if (!padsKnown)
                {
                    return false;
                }
                return !(n.inputs.size() > 2 && n.inputs[2] != kNoTensor && !g.isInitializer(n.inputs[2]));
            }
            case OpType::Gather:
                // Flat row-major gather along an axis; index may be constant or a runtime float activation
                // (RoPE).
                return n.inputs.size() >= 2;
            case OpType::Clip:
                return true; // flat elementwise clamp (geometry-tail Clip is rank-6, can't be NC4HW4)
            case OpType::Split: {
                // Channel-axis 4-aligned split stays NC4HW4 (contiguous block copy); any other split is flat.
                const Shape &in   = sh(n.inputs[0]);
                int          rank = (int) in.size();
                int64_t      axis = n.attr.geti("axis", 0);
                if (axis < 0)
                {
                    axis += rank;
                }
                if (rank == 4 && axis == 1)
                {
                    for (TensorId o: n.outputs)
                    {
                        if (o != kNoTensor && (sh(o).size() != 4 || sh(o)[1] % 4 != 0))
                        {
                            return true;
                        }
                    }
                    return false;
                }
                return true;
            }
            case OpType::ScatterND:
                // GPU flat scatter; index may be a constant or a runtime float activation.
                return n.inputs.size() >= 3;
            case OpType::Einsum: {
                // Only the "i,j->ij" outer product runs on the GPU; other equations use the CPU op.
                std::string eq;
                for (char c: n.attr.gets("equation", ""))
                {
                    if (c != ' ' && c != '\t')
                    {
                        eq += c;
                    }
                }
                return eq == "i,j->ij";
            }
            case OpType::Softmax: {
                if (n.inputs.empty())
                {
                    return false;
                }
                const Shape &s = sh(n.inputs[0]);
                if (s.size() < 2)
                {
                    return true;
                }
                int     rank = (int) s.size();
                int64_t axis = n.attr.geti("axis", -1);
                if (axis < 0)
                {
                    axis += rank;
                }
                NCHW    x     = NCHW::from(s);
                int64_t inner = 1;
                for (int k = (int) axis; k < rank; ++k)
                {
                    inner *= s[k];
                }
                return !(x.h * x.w == 1 && inner == x.c); // channel softmax stays NC4HW4
            }
            case OpType::Concat: {
                const Shape &o    = sh(n.outputs[0]);
                int          rank = (int) o.size();
                int64_t      axis = n.attr.geti("axis", 1);
                if (axis < 0)
                {
                    axis += rank;
                }
                if (rank != 4 || axis != 1)
                {
                    return true;
                }
                for (TensorId in: n.inputs)
                {
                    if (sh(in).size() != 4 || sh(in)[1] % 4 != 0)
                    {
                        return true;
                    }
                }
                return false;
            }
            case OpType::Binary:
            case OpType::Add: {
                if (n.inputs.size() != 2)
                {
                    return false;
                }
                if (g.isInitializer(n.inputs[0]) || g.isInitializer(n.inputs[1]))
                {
                    return true;
                }
                const Shape &a = sh(n.inputs[0]);
                const Shape &b = sh(n.inputs[1]);
                if (a.size() == 4 && b.size() == 4 && a == b)
                {
                    return false; // NC4HW4 same-shape
                }
                if (n.type == OpType::Binary)
                {
                    auto bc = [](const Shape &s, const Shape &f) {
                        return s.size() == 4 && f.size() == 4 && s[0] == f[0] && s[1] == f[1] && s[2] == 1 && s[3] == 1 && (f[2] > 1 || f[3] > 1);
                    };
                    if (bc(a, b) || bc(b, a))
                    {
                        return false; // NC4HW4 channel-broadcast
                    }
                }
                return true;
            }
            default:
                return false;
        }
    }

    void insertLayoutConverts(Graph &g) {
        // Mark each tensor's GPU format from its producer (flat if produced by a flat op), then for every
        // node input whose format differs from what the consumer needs, splice in a ConvertLayout node.
        auto agnostic = [](const Node &n) {
            // metadata reshape / no-op copy: keeps the producer's layout (input and output bytes
            // identical).
            return n.type == OpType::Reshape || n.type == OpType::Flatten || n.type == OpType::Squeeze || n.type == OpType::Unsqueeze || n.type == OpType::Cast;
        };
        // Mark each tensor's GPU format in topo order. Reshape/Flatten are layout-agnostic: a flat
        // reshape is a plain row-major copy (valid for ANY shape), while the NC4HW4 byte-copy is only
        // valid when the channel count is unchanged (else the vec4 interleave shifts). So a reshape is
        // flat if its input is flat OR it changes the channel count; otherwise it stays NC4HW4.
        for (auto &nd: g.nodes)
        {
            bool f;
            if (agnostic(nd) && !nd.inputs.empty() && nd.inputs[0] != kNoTensor)
            {
                bool    inFlat = g.desc(nd.inputs[0]).gpuFlat;
                int64_t cin    = NCHW::from(g.desc(nd.inputs[0]).shape).c;
                int64_t cout   = NCHW::from(g.desc(nd.outputs[0]).shape).c;
                f              = inFlat || cin != cout;
            } else
            {
                f = gpuFlatNode(g, nd);
            }
            for (TensorId o: nd.outputs)
            {
                if (o != kNoTensor)
                {
                    g.desc(o).gpuFlat = f;
                }
            }
        }
        // NC4HW4 can only represent rank <= 4 (NCHW::from collapses rank>4 to (1,1,1,1)). Any tensor with
        // rank > 4 MUST be a flat row-major buffer — including graph inputs with no producer (the
        // YoNoSplat image input is [1,2,3,224,224]; left NC4HW4 it would be mis-packed and corrupt the
        // whole encoder).
        for (auto &t: g.tensors)
        {
            if (t.shape.size() > 4)
            {
                t.gpuFlat = true;
            }
        }
        std::map<std::pair<TensorId, bool>, TensorId> cache; // (tensor, needFlat) -> converted tensor
        std::vector<Node>                             converts;
        int                                           n = 0;
        for (auto &nd: g.nodes)
        {
            if (nd.outputs.empty() || nd.outputs[0] == kNoTensor)
            {
                continue;
            }
            bool needFlat = g.desc(nd.outputs[0]).gpuFlat; // the format this node operates in
            for (TensorId &in: nd.inputs)
            {
                if (in == kNoTensor || g.isInitializer(in))
                {
                    continue; // constants handled inside flat ops
                }
                if (g.desc(in).gpuFlat == needFlat)
                {
                    continue;
                }
                auto key = std::make_pair(in, needFlat);
                auto it  = cache.find(key);
                if (it == cache.end())
                {
                    TensorDesc d    = g.desc(in);
                    d.name          = g.desc(in).name + (needFlat ? "#flat" : "#nc4") + std::to_string(n);
                    d.isInitializer = d.isInput = d.isOutput = false;
                    d.gpuFlat                                = needFlat;
                    TensorId t2                              = g.addTensor(d);
                    Node     cv;
                    cv.type    = OpType::ConvertLayout;
                    cv.name    = "convert" + std::to_string(n++);
                    cv.subOp   = needFlat ? 0 : 1; // 0: NC4HW4->flat, 1: flat->NC4HW4
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
            VKNN_INFO << "insertLayoutConverts: inserted " << converts.size() << " layout convert(s)";
        }
    }

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
            c = substrs.find(',', p);
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

    void runStandardPasses(Graph &g, const PassOptions &opt) {
        int64_t batch = opt.batch;
        inferShapes(g, batch);
        eliminateIdentity(g);
        foldBatchNorm(g);
        fuseActivations(g);
        fuseResidualAdd(g);
        if (opt.fuseSwish)
        {
            fuseSwish(g); // HardSwish/SiLU into conv epilogue (default on)
        }
        if (opt.fuseSqueezeExcite)
        {
            fuseSqueezeExcite(g);
        }
        if (opt.fuseDwPw)
        {
            fuseDwPw(g);
        }
        // Iterate fold+infer: folding a Shape/Gather/Concat chain turns a dynamic Reshape's shape input
        // into a constant, which lets the next inferShapes resolve that Reshape statically, which in turn
        // exposes more foldable shape ops downstream (YOLO's DFL/box-decode head). Converges in a couple
        // rounds; the loop runs until constFold stops removing nodes.
        for (int iter = 0; iter < 8; ++iter)
        {
            if (constFold(g) == 0)
            {
                break;
            }
            inferShapes(g, batch);
        }
        eliminateFloatCast(g); // drop float->float casts left by transformer import (post-fold)
        fuseMatMulBias(g);     // fold Linear bias-Adds into the MatMul epilogue (Casts now gone)
        eliminateDeadNodes(g);
        inferShapes(g, batch); // refresh shapes after fusion/folding
        lowerEinsum(g);        // batched einsums -> MatMul (needs the operand shapes resolved above)
        inferShapes(g, batch); // resolve the inserted Unsqueeze/MatMul/Squeeze
        if (opt.dumpBig)
        {
            for (const Node &n: g.nodes)
            {
                for (TensorId o: n.outputs)
                {
                    if (o == kNoTensor)
                    {
                        continue;
                    }
                    int64_t ne = numElements(g.desc(o).shape);
                    if (ne > 50000000)
                    {
                        std::string sh;
                        for (int64_t d: g.desc(o).shape)
                        {
                            sh += std::to_string(d) + ",";
                        }
                        VKNN_WARN << "BIG tensor " << ne << " elems from " << opTypeName(n.type) << " " << n.name << " shape=[" << sh << "]";
                    }
                }
            }
        }
    }

} // namespace vknn
