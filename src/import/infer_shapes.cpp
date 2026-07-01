#include "passes_internal.h"

namespace vknn {

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
                    int64_t           outC = w[1] * nd.attr.geti("group", 1);
                    ConvTransposeGeom geom = convTransposeGeom(x.h, x.w, w[2], w[3], nd.attr);
                    SH(o)                  = {x.n, outC, geom.outH, geom.outW};
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
                case OpType::EyeLike:        // identity-like, same shape as input
                case OpType::ScatterND:      // same shape as data (input[0])
                case OpType::FusedPointwise: // per-element chain: same shape/dtype as the primary input
                    SH(o)           = SH(nd.inputs[0]);
                    g.desc(o).dtype = g.desc(nd.inputs[0]).dtype;
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

} // namespace vknn
