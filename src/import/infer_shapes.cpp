#include "core/fused_attention.h"
#include "core/matmul_view.h"
#include "core/slice_bounds.h"
#include "dim_expr.h"
#include "passes_internal.h"

namespace vknn {

    /// Forward shape (and, where it differs, dtype) inference over the whole graph.
    ///
    /// First resolves every dynamic (negative) input dim, then walks the nodes in graph (topological
    /// producer-before-consumer) order and fills each output's `Shape` from its resolved inputs using
    /// the ONNX per-op rules. Node visitation order is load-bearing: a producer's shape must land before
    /// a consumer reads it.
    ///
    /// Input dim resolution, per still-dynamic axis (precedence high to low): a `declared` per-tensor
    /// shape (--shape) overrides everything; else the axis's recorded symbolic dim_param
    /// (TensorDesc::dimParams) is evaluated against `bindings` (--dim NAME=VALUE), which resolves a bare
    /// symbol, an integer literal, and a compound expression like "past_sequence_length + sequence_length"
    /// alike; else the leading axis falls back to `batch` — but ONLY when it carries no symbol or a
    /// batch-named one ("N", "B", or any name containing "batch", case-insensitive). A leading symbol
    /// with any other name (num_frames, views, num_cameras...) is a real extent, not a batch: freezing
    /// it to 1 compiles a 1-frame plan that silently truncates the caller's data (a frame-interp model
    /// fed [num_frames,C,H,W] returns near-zero-cosine output with no diagnostic), so it hard-errors
    /// like a non-leading axis. Any axis that stays unresolved is a hard error rather than a silent
    /// substitution to `batch`: freezing a real spatial/feature axis to 1 compiles the model to a 1x1
    /// plan whose output is quietly wrong (the vit_b16_q8 0.32-cosine bug). The error is aggregated
    /// across every input and lists the unbound symbol names to bind, so a many-input decoder reports
    /// one actionable message instead of failing per tensor.
    ///
    /// Central invariant — an EMPTY shape means "not resolved yet", never a rank-0 scalar, EXCEPT on an
    /// initializer (a constant genuinely may be rank-0). Every case therefore refuses to compute from an
    /// empty non-initializer operand and leaves the output empty rather than fabricating a dim. This
    /// matters because a fabricated (or stale) shape can be frozen by a downstream Shape()/const-fold and
    /// then poison every rank below it. The pass is idempotent and designed to be re-run: bounds that are
    /// still runtime on this pass (e.g. a Reshape/Slice/Expand target that const-folds only later) leave
    /// their output unresolved and resolve on a subsequent call once the operand lands.
    ///
    /// Ops not listed (the `default` arm) are shape-path ops whose outputs are produced by constFold.
    void inferShapes(Graph &g, int64_t batch, const std::map<std::string, Shape> *declared, const std::map<std::string, int64_t> *bindings) {
        static const std::map<std::string, int64_t> kNoBindings;
        const std::map<std::string, int64_t>       &binds = bindings ? *bindings : kNoBindings;

        // Symbols that no binding resolved, and dynamic axes with no recorded symbol at all: accumulated
        // across every input so one aggregated error lists them together (instead of failing on the first
        // tensor with a per-tensor message the caller must then rediscover once per input).
        std::vector<std::string> freeSymbols; // distinct, first-seen order
        std::vector<std::string> freeAxes;    // "name[axis]" for an axis carrying no dim_param symbol
        auto                     addFreeSym = [&](const std::string &sym) {
            for (const auto &f: freeSymbols)
            {
                if (f == sym)
                {
                    return;
                }
            }
            freeSymbols.push_back(sym);
        };

        // Resolve each input's dynamic (negative) dims. A dim resolved on an earlier call is already
        // positive and is skipped, so this is idempotent across the pipeline's repeated inferShapes runs.
        for (TensorId in: g.inputs)
        {
            const TensorDesc &d0   = g.desc(in);
            auto             &s    = g.desc(in).shape;
            const Shape      *decl = nullptr;
            if (declared)
            {
                auto it = declared->find(d0.name);
                if (it != declared->end())
                {
                    decl = &it->second;
                    if (decl->size() != s.size())
                    {
                        throw Error(Status::InvalidArgument, "declared shape for input '" + d0.name + "' has rank " + std::to_string(decl->size()) + " but the model declares rank " + std::to_string(s.size()));
                    }
                }
            }
            const std::vector<std::string> &params = d0.dimParams;
            for (size_t ax = 0; ax < s.size(); ++ax)
            {
                if (s[ax] >= 0)
                {
                    continue; // already resolved (or statically known)
                }
                // 1. Per-tensor declared shape (--shape) overrides everything for this input.
                if (decl)
                {
                    if ((*decl)[ax] < 0)
                    {
                        throw Error(Status::InvalidArgument, "declared shape for input '" + d0.name + "' leaves axis " + std::to_string(ax) + " dynamic (" + std::to_string((*decl)[ax]) + "); declare a concrete extent");
                    }
                    s[ax] = (*decl)[ax];
                    continue;
                }
                // 2. A recorded dim_param symbol/expression resolved from `bindings` (the compound
                //    "past_sequence_length + sequence_length" evaluates from its two bound symbols).
                const std::string sym = ax < params.size() ? params[ax] : std::string();
                if (!sym.empty())
                {
                    DimEval e = evalDimExpr(sym, binds);
                    if (e.ok)
                    {
                        if (e.value < 0)
                        {
                            throw Error(Status::InvalidArgument, "input '" + d0.name + "' axis " + std::to_string(ax) + " (" + sym + ") resolves to a negative extent " + std::to_string(e.value));
                        }
                        s[ax] = e.value;
                        continue;
                    }
                    // Only a batch-NAMED leading symbol takes the batch fallback; any other leading
                    // symbol (num_frames, views...) is a real extent and hard-errors below when unbound.
                    if (ax == 0 && batchLikeDimSymbol(sym))
                    {
                        s[ax] = batch;
                        continue;
                    }
                    for (const std::string &fs: e.freeSymbols)
                    {
                        addFreeSym(fs);
                    }
                    continue; // stays dynamic; the aggregated error below aborts the compile
                }
                // 3. No recorded symbol: only the leading (batch) axis falls back; any other is an error.
                if (ax == 0)
                {
                    s[ax] = batch; // leading axis = batch, the documented dynamic-batch fallback
                    continue;
                }
                freeAxes.push_back(d0.name + "[" + std::to_string(ax) + "]");
            }
        }

        if (!freeSymbols.empty() || !freeAxes.empty())
        {
            std::string msg = "cannot resolve dynamic input shapes: ";
            if (!freeSymbols.empty())
            {
                msg += "unbound symbolic dim(s) {";
                for (size_t i = 0; i < freeSymbols.size(); ++i)
                {
                    if (i)
                    {
                        msg += ", ";
                    }
                    msg += freeSymbols[i];
                }
                msg += "} -- bind each with --dim NAME=VALUE (or Config::dimBindings)";
            }
            if (!freeAxes.empty())
            {
                if (!freeSymbols.empty())
                {
                    msg += "; ";
                }
                msg += "undeclared dynamic axis/axes {";
                for (size_t i = 0; i < freeAxes.size(); ++i)
                {
                    if (i)
                    {
                        msg += ", ";
                    }
                    msg += freeAxes[i];
                }
                msg += "} -- declare with --shape NAME=D0xD1x... (or Config::inputShapes)";
            }
            throw Error(Status::InvalidArgument, msg);
        }
        for (auto &nd: g.nodes)
        {
            inferNodeShape(g, nd);
        }
    }

    /// The per-node forward shape rule of inferShapes, callable on its own so constFold can interleave
    /// it with value folding inside ONE program-order walk: folding a Shape/Gather/Concat chain makes a
    /// dynamic Reshape's target constant, this rule then resolves the Reshape (and the activations
    /// behind it) immediately, which unblocks folding the NEXT block's Shape() later in the same walk.
    /// Without the interleave each fold/infer alternation advanced exactly one such block per round (a
    /// deep encoder took dozens of full-graph rounds to converge, re-warning on every still-unresolved
    /// tensor each round). Idempotent; only ever fills shapes that are derivable from resolved inputs.
    namespace {
        /// Report a node whose rule resolved an output axis to zero while every data operand was
        /// non-empty. Such a tensor allocates nothing, so every kernel downstream of it runs over zero
        /// elements and the model produces a silently empty (not wrong-valued) result — with no CPU
        /// fallback to announce it, because the node itself is perfectly supported. The message names
        /// the node, its type, and the operand it collapsed, which is the whole diagnosis for a model
        /// whose bytes are not available to inspect.
        void warnCollapsedAxis(const Graph &g, const Node &nd) {
            auto hasZero = [](const Shape &s) {
                return std::find(s.begin(), s.end(), 0) != s.end();
            };
            for (TensorId o: nd.outputs)
            {
                if (o == kNoTensor)
                {
                    continue;
                }
                const Shape &out = g.desc(o).shape;
                if (out.empty() || !hasZero(out))
                {
                    continue; // empty = "not resolved yet"; a zero-free shape is fine
                }
                std::string operands;
                bool        fromInput = false;
                for (TensorId in: nd.inputs)
                {
                    if (in == kNoTensor)
                    {
                        continue;
                    }
                    const Shape &s = g.desc(in).shape;
                    fromInput      = fromInput || hasZero(s);
                    operands += (operands.empty() ? "" : ", ") + g.desc(in).name + shapeStr(s);
                }
                if (fromInput)
                {
                    continue; // inherited, not introduced here: the producer already reported it
                }
                VKNN_WARN << "node '" << nd.name << "' (" << opTypeName(nd.type) << ") resolved output '" << g.desc(o).name << "' to " << shapeStr(out)
                          << " -- a zero-length axis, so this tensor and everything downstream of it computes nothing. Operands: " << operands;
            }
        }
    } // namespace

    void inferNodeShape(Graph &g, Node &nd) {
        auto SH = [&](TensorId id) -> Shape & {
            return g.desc(id).shape;
        };
        {
            if (nd.outputs.empty())
            {
                return;
            }
            // A view-folded MatMul (core/matmul_view.h) reads its operands through attr strides;
            // its operand shapes are the chain SOURCES, so the forward matmul rule would derive a
            // wrong output shape. Every shape on such a node is already final. The same holds for
            // a FusedAttention node (core/fused_attention.h): the load-time fusion pass takes over
            // the decomposed chain's final output tensor, whose shape kFaOut also records.
            if (nd.attr.has(kMmView) || nd.attr.has(kFa))
            {
                return;
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
                    int64_t     outC = w[0], kh = w[2], kw = w[3];
                    const auto &st  = nd.attr.getints("strides");
                    const auto &pad = nd.attr.getints("pads");
                    const auto &dil = nd.attr.getints("dilations");
                    if ((!st.empty() && st.size() < 2) || (!pad.empty() && pad.size() < 4) || (!dil.empty() && dil.size() < 2))
                    {
                        break; // 1-spatial-dim attributes: normalizeConv1d has not run (runtime weight)
                    }
                    ConvGeom cg = convGeom(x.h, x.w, kh, kw, nd.attr); // resolves auto_pad
                    // A rank-3 input is a normalized 1-D conv (NCHW::from maps [N,C,L] to h=L, w=1,
                    // and the weight's kW extent is 1, so outW == 1); the output keeps rank 3.
                    if (SH(nd.inputs[0]).size() == 3 && cg.outW == 1)
                    {
                        SH(o) = {x.n, outC, cg.outH};
                    } else
                    {
                        SH(o) = {x.n, outC, cg.outH, cg.outW};
                    }
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
                case OpType::InstanceNorm: // normalize over the spatial dims: same shape as input
                case OpType::Identity:
                case OpType::Unary:
                case OpType::IsNaN: // elementwise NaN test: bool output, same shape as input
                case OpType::Softmax:
                case OpType::LayerNorm:
                case OpType::RMSNorm: // y = x*rsqrt(mean(x^2)+eps)*gamma over the last axis: same shape as input
                case OpType::Rope:    // rotate-half rotary embedding over the last axis: same shape as input
                case OpType::PRelu:
                case OpType::EyeLike:        // identity-like, same shape as input
                case OpType::ScatterND:      // same shape as data (input[0])
                case OpType::ChannelShuffle: // channel permutation: same shape as input
                case OpType::FusedPointwise: // per-element chain: same shape/dtype as the primary input
                    SH(o)           = SH(nd.inputs[0]);
                    g.desc(o).dtype = g.desc(nd.inputs[0]).dtype;
                    break;
                case OpType::Det: { // [..., n, n] -> [...]; rank-2 -> {1} (no rank-0 activations)
                    const Shape &in = g.desc(nd.inputs[0]).shape;
                    if (in.size() >= 2)
                    {
                        Shape out(in.begin(), in.end() - 2);
                        if (out.empty())
                        {
                            out.push_back(1);
                        }
                        SH(o) = out;
                    }
                    break;
                }
                case OpType::Equal:
                case OpType::Greater:
                case OpType::GreaterEqual:
                case OpType::Less:
                case OpType::LessEqual:
                case OpType::And: {
                    // Same empty-shape discrimination as Binary/Add: scalar only if initializer.
                    const Shape &a = SH(nd.inputs[0]);
                    const Shape &b = SH(nd.inputs[1]);
                    if ((a.empty() && !g.isInitializer(nd.inputs[0])) || (b.empty() && !g.isInitializer(nd.inputs[1])))
                    {
                        break; // unresolved operand
                    }
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
                        int64_t da = dimOf(a, i), db = dimOf(b, i);
                        out[i] = (da == 0 || db == 0) ? 0 : std::max(da, db); // a 0 dim broadcasts to 0 (NumPy), never to 1
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
                    // An empty-shaped non-initializer input is unresolved, not a scalar: refuse.
                    bool unresolved = false;
                    for (TensorId t: nd.inputs)
                    {
                        if (t != kNoTensor && SH(t).empty() && !g.isInitializer(t))
                        {
                            unresolved = true;
                            break;
                        }
                    }
                    if (unresolved)
                    {
                        break;
                    }
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
                            // a 0 dim broadcasts to 0 (NumPy), never to 1
                            out[off + i] = (s[i] == 0 || out[off + i] == 0) ? 0 : std::max(out[off + i], s[i]);
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
                        // never read past the initializer's real payload, whatever the desc claims
                        r = std::min<int64_t>(r, (int64_t) (hb.bytes.size() / (g.tensors[sid].dtype == DType::Int64 ? 8 : 4)));
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
                case OpType::Range: {
                    // Output is 1-D [ceil((limit-start)/delta)] once all three scalars are constant;
                    // a runtime start/limit/delta stays unresolved (the CPU op sizes it at run time).
                    if (nd.inputs.size() < 3)
                    {
                        break;
                    }
                    double vals[3];
                    bool   ok = true;
                    for (int i = 0; i < 3; ++i)
                    {
                        TensorId t = nd.inputs[i];
                        if (t == kNoTensor || !g.isInitializer(t))
                        {
                            ok = false;
                            break;
                        }
                        const HostBuffer &hb  = g.initializers[t];
                        bool              i64 = g.tensors[t].dtype == DType::Int64;
                        if (hb.bytes.size() < (i64 ? 8u : 4u))
                        {
                            ok = false; // never read past the initializer's real payload
                            break;
                        }
                        vals[i] = i64 ? (double) hb.i64()[0] : (double) hb.f32()[0];
                    }
                    if (ok && vals[2] != 0.0)
                    {
                        SH(o) = {std::max<int64_t>((int64_t) std::ceil((vals[1] - vals[0]) / vals[2]), 0)};
                    }
                    break;
                }
                case OpType::GridSample: {
                    const Shape &xs = SH(nd.inputs[0]);
                    const Shape &gs = SH(nd.inputs[1]);
                    // Warp mode (fuseGridSampleWarp): input 1 is the NCHW flow [N,2,Hout,Wout], so the
                    // output spatial dims come from the flow's spatial axes, not a channels-last grid.
                    if (nd.attr.has("warp"))
                    {
                        if (xs.size() == 4 && gs.size() == 4)
                        {
                            SH(o) = {xs[0], xs[1], gs[2], gs[3]};
                        }
                    } else if (xs.size() == 4 && gs.size() == 4)
                    { SH(o) = {xs[0], xs[1], gs[1], gs[2]}; }
                    break;
                }
                case OpType::Cast:
                case OpType::ConvertLayout:
                case OpType::ConvertDtype: {
                    SH(o) = SH(nd.inputs[0]);
                    break;
                }
                case OpType::DequantizeLinear:
                    // y = (x - zp) * scale: same shape as x, real-valued fp32 output.
                    SH(o)           = SH(nd.inputs[0]);
                    g.desc(o).dtype = DType::Float32;
                    break;
                case OpType::QuantizeLinear: {
                    // Same shape as x; the output is the integer quant type, read from the zero_point
                    // input's dtype (ONNX defaults an absent zero_point to uint8).
                    SH(o)     = SH(nd.inputs[0]);
                    DType out = DType::UInt8;
                    if (nd.inputs.size() > 2 && nd.inputs[2] != kNoTensor)
                    {
                        out = g.desc(nd.inputs[2]).dtype;
                    }
                    g.desc(o).dtype = out;
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
                    const Shape &dw = SH(nd.inputs[1]);             // depthwise weights [E,1,KH,KW]
                    auto         a  = [&](const char *k, std::vector<int64_t> d) {
                        const auto &v = nd.attr.getints(k);
                        return v.empty() ? d : v;
                    };
                    // Kernel size from the depthwise weight shape: the kernel_shape attr is
                    // optional in ONNX and the runtime kernel reads the weight dims too — a {3,3}
                    // fallback mis-sizes every 5x5 pair whose conv omitted the attr.
                    auto     k  = dw.size() == 4 ? std::vector<int64_t> {dw[2], dw[3]} : a("kernel_shape", {3, 3});
                    ConvGeom cg = convGeom(x.h, x.w, k[0], k[1], nd.attr); // attrs come from the depthwise Conv, auto_pad included
                    SH(o)       = {x.n, pw.empty() ? x.c : pw[0], cg.outH, cg.outW};
                    break;
                }
                case OpType::ConvGemm: {
                    if (SH(nd.inputs[0]).empty())
                    {
                        break;
                    }
                    NCHW         x  = NCHW::from(SH(nd.inputs[0]));
                    const Shape &wt = SH(nd.inputs[1]); // repacked [K, Cout]
                    auto         a  = [&](const char *k, std::vector<int64_t> d) {
                        const auto &v = nd.attr.getints(k);
                        return v.empty() ? d : v;
                    };
                    auto     k  = a("kernel_shape", {1, 1});
                    ConvGeom cg = convGeom(x.h, x.w, k[0], k[1], nd.attr);
                    SH(o)       = {x.n, wt.size() == 2 ? wt[1] : x.c, cg.outH, cg.outW};
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
                    if (axis < 0 || axis >= rank)
                    {
                        break; // out-of-range axis (untrusted graph): a[axis]/os[axis] would be OOB
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
                case OpType::TopK: {
                    // Both outputs (values, indices) share the input shape with the axis dim replaced
                    // by k. k is the `k` attribute (opset < 10) or the const int64 input[1] (opset 10+);
                    // a runtime k leaves the outputs unresolved. Indices are int64 regardless of the
                    // data dtype — the session readback keys off this for a graph-output index tensor.
                    const Shape &a = SH(nd.inputs[0]);
                    if (a.empty())
                    {
                        break;
                    }
                    int64_t rank = (int64_t) a.size();
                    int64_t axis = nd.attr.geti("axis", -1);
                    if (axis < 0)
                    {
                        axis += rank;
                    }
                    if (axis < 0 || axis >= rank)
                    {
                        break;
                    }
                    int64_t k = -1;
                    if (nd.attr.has("k"))
                    {
                        k = nd.attr.geti("k", -1);
                    } else
                    {
                        std::vector<int64_t> kv = readI64Param(g, nd, "k", 1);
                        if (!kv.empty())
                        {
                            k = kv[0];
                        }
                    }
                    if (k < 0)
                    {
                        break; // k const-folds later; resolved on a subsequent pass
                    }
                    Shape out       = a;
                    out[axis]       = std::min(k, a[axis]); // an oversized k saturates at the axis length
                    SH(o)           = out;
                    g.desc(o).dtype = g.desc(nd.inputs[0]).dtype;
                    if (nd.outputs.size() > 1 && nd.outputs[1] != kNoTensor)
                    {
                        SH(nd.outputs[1])           = out;
                        g.desc(nd.outputs[1]).dtype = DType::Int64;
                    }
                    break;
                }
                case OpType::Transpose: {
                    const Shape &a = SH(nd.inputs[0]);
                    if (a.empty())
                    {
                        break; // input unresolved: defer silently, a later fold/infer round fills it
                    }
                    const auto &perm = nd.attr.getints("perm");
                    if (!perm.empty() && perm.size() != a.size())
                    {
                        // A RESOLVED input whose rank disagrees with perm is a real model/import
                        // defect (an unresolved input is the silent deferral above, never this).
                        VKNN_WARN << "Transpose '" << nd.name << "': perm rank " << perm.size() << " != input rank " << a.size() << " (shape " << shapeStr(a) << "); leaving unresolved";
                        break; // indexing a mismatched perm would read past the shape vector
                    }
                    Shape out(a.size());
                    for (size_t i = 0; i < a.size(); ++i)
                    {
                        int64_t src = perm.empty() ? (int64_t) (a.size() - 1 - i) : perm[i];
                        if (src < 0 || src >= (int64_t) a.size())
                        {
                            VKNN_WARN << "Transpose '" << nd.name << "': perm[" << i << "]=" << src << " out of range for rank " << a.size();
                            out.clear();
                            break;
                        }
                        out[i] = a[src];
                    }
                    if (!out.empty())
                    {
                        SH(o) = out;
                    }
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
                    if (want < 0 || declLen(3) < 0 || declLen(4) < 0)
                    {
                        break; // a param tensor's own shape is unresolved -> defer (out=input would
                               // fabricate an unsliced dim that a downstream Shape() fold freezes)
                    }
                    if (want > 0 && ((int64_t) starts.size() < want || (int64_t) ends.size() < want))
                    {
                        break; // a slice bound is still runtime -> defer rather than fabricate a wrong dim
                    }
                    if ((declLen(3) > 0 && (int64_t) axes.size() < declLen(3)) || (declLen(4) > 0 && (int64_t) steps.size() < declLen(4)))
                    {
                        break; // axes/steps exist as tensors but are still runtime -> defer
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
                        const int64_t step = (k < steps.size()) ? steps[k] : 1;
                        out[ax]            = resolveSliceAxis(a[ax], starts[k], ends[k], step).count;
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
                            if (idx >= (int) pwCoreInputs(nd) || nd.inputs[idx] == kNoTensor)
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
                    // ([2,224,224,1]*[3]->[2,224,224,3]). An empty shape is a rank-0 scalar only on an
                    // initializer; on a produced tensor it means "not resolved yet" and the output must
                    // stay unresolved (adopting the constant operand's shape here poisons every rank
                    // downstream of a transformer's MatMul-bias Adds).
                    const Shape &a = SH(nd.inputs[0]);
                    const Shape &b = SH(nd.inputs[1]);
                    if ((a.empty() && !g.isInitializer(nd.inputs[0])) || (b.empty() && !g.isInitializer(nd.inputs[1])))
                    {
                        break; // unresolved operand
                    }
                    if (a.empty() && b.empty())
                    {
                        break;
                    }
                    if (a.empty() || b.empty())
                    { // one is a rank-0 scalar constant: use the other
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
                        int64_t da = dimOf(a, i), db = dimOf(b, i);
                        out[i] = (da == 0 || db == 0) ? 0 : std::max(da, db); // a 0 dim broadcasts to 0 (NumPy), never to 1
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
                    NCHW     x  = NCHW::from(SH(nd.inputs[0]));
                    ConvGeom cg = poolGeom(x.h, x.w, nd.attr); // resolves auto_pad
                    SH(o)       = {x.n, x.c, cg.outH, cg.outW};
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
                        int64_t da = dimOf(a, aBatch, i), db = dimOf(b, bBatch, i);
                        out.push_back((da == 0 || db == 0) ? 0 : std::max(da, db)); // a 0 dim broadcasts to 0 (NumPy), never to 1
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
                    if (axis < 0)
                    {
                        axis += (int64_t) a.size(); // ONNX allows a negative Flatten axis; match cpu/ops/flatten.cpp
                    }
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
                        // Sum the axis over the concatenated parts only: inputs from pwCoreInputs
                        // on are fused-epilogue operands, and this case re-runs after fusion.
                        int64_t sum   = 0;
                        int64_t parts = pwCoreInputs(nd);
                        for (int64_t e = 0; e < parts && e < (int64_t) nd.inputs.size(); ++e)
                        {
                            const Shape &si = SH(nd.inputs[e]);
                            if (si.empty() || axis < 0 || axis >= (int64_t) si.size())
                            {
                                sum = -1; // out-of-range axis (untrusted graph): si[axis]/s[axis] would be OOB
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
                    int64_t avail = (int64_t) (hb.bytes.size() / 8);
                    if (rank > avail)
                    {
                        VKNN_WARN << "Reshape target '" << g.tensors[sid].name << "' declares " << rank << " dims but holds " << avail << " values; clamping (desc/payload mismatch upstream)";
                        rank = avail;
                    }
                    // A -1 (deduce) or 0 (copy) target dim needs the data input's shape. While that
                    // input is unresolved, numElements({}) = 0 would fabricate a zero dim here — and a
                    // downstream Shape-of-this then const-folds the lie into the shape arithmetic.
                    // Leave unresolved; a later pass resolves it once the input shape lands.
                    if (in.empty())
                    {
                        bool needsIn = false;
                        for (int64_t i = 0; i < rank; ++i)
                        {
                            if (hb.i64()[i] <= 0)
                            {
                                needsIn = true;
                                break;
                            }
                        }
                        if (needsIn)
                        {
                            break;
                        }
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
                    // ONNX Gather: out = data.shape[:axis] + indices.shape + data.shape[axis+1:]. Only a
                    // true rank-0 scalar index (empty shape — the importer preserves 0-D dims) removes the
                    // axis; a rank-1 [1] index KEEPS it as size 1 (treating [1] as scalar collapsed the rank
                    // and mis-broadcast everything downstream). Mirrors GatherCpu exactly.
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
                    bool    scalarIndex = is.empty() || nd.attr.geti("idx_scalar", 0) != 0;
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
            warnCollapsedAxis(g, nd);
        }
    }

} // namespace vknn
