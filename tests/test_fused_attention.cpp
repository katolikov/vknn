// fuseDecodeAttention (src/import/fuse_decode_attention.cpp + core/fused_attention.h): the M == 1
// decode-attention chain — view MatMul (+ scale/mask, as a fused epilogue or standalone nodes) ->
// Softmax -> view MatMul -> Transpose -> Reshape — collapses into one FusedAttention node whose
// strides address q/k/v/mask exactly; the CPU op matches the decomposed chain to fp32 rounding; a
// chain carrying foreign fused work, a non-M=1 (prefill) chain, an fp32-pinned interior tensor,
// or a pattern-free graph is refused; Hint::FusedAttention Off keeps the decomposed plan.
#include "core/fused_attention.h"
#include "core/matmul_view.h"
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/io_link.h"
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    constexpr int64_t kB = 1, kKvHeads = 2, kGroup = 3, kHeads = kKvHeads * kGroup, kTokens = 7, kHd = 4;
    constexpr float   kScaleValue = 0.25f;

    TensorId addInput(Graph &g, const std::string &name, Shape shape) {
        TensorDesc td;
        td.name    = name;
        td.shape   = std::move(shape);
        td.isInput = true;
        TensorId t = g.addTensor(td);
        g.inputs.push_back(t);
        return t;
    }

    TensorId addTemp(Graph &g, const std::string &name) {
        TensorDesc td;
        td.name = name;
        return g.addTensor(td);
    }

    TensorId addI64Init(Graph &g, const std::string &name, const std::vector<int64_t> &vals) {
        TensorDesc td;
        td.name          = name;
        td.shape         = {(int64_t) vals.size()};
        td.dtype         = DType::Int64;
        td.isInitializer = true;
        TensorId   t     = g.addTensor(td);
        HostBuffer hb;
        hb.resizeElems((int64_t) vals.size(), DType::Int64);
        std::memcpy(hb.i64(), vals.data(), vals.size() * sizeof(int64_t));
        g.initializers[t] = hb;
        return t;
    }

    TensorId addF32Init(Graph &g, const std::string &name, Shape shape, const std::vector<float> &vals) {
        TensorDesc td;
        td.name          = name;
        td.shape         = std::move(shape);
        td.dtype         = DType::Float32;
        td.isInitializer = true;
        TensorId   t     = g.addTensor(td);
        HostBuffer hb;
        hb.resizeElems((int64_t) vals.size(), DType::Float32);
        std::memcpy(hb.f32(), vals.data(), vals.size() * sizeof(float));
        g.initializers[t] = hb;
        return t;
    }

    Node *addNode(Graph &g, OpType type, const std::string &name, std::vector<TensorId> ins, TensorId out) {
        Node n;
        n.type    = type;
        n.name    = name;
        n.inputs  = std::move(ins);
        n.outputs = {out};
        g.nodes.push_back(n);
        return &g.nodes.back();
    }

    // A repeat_kv chain: cache [B,kv,S,hd] -> Reshape [B,kv,1,S,hd] -> Expand [B,kv,g,S,hd] ->
    // Reshape [B,H,S,hd] (-> Transpose to [B,H,hd,S] for the QK orientation).
    TensorId repeatKv(Graph &g, TensorId cache, const std::string &tag, bool transposeToQk) {
        TensorId r1 = addTemp(g, tag + "_r1");
        TensorId ex = addTemp(g, tag + "_ex");
        TensorId r2 = addTemp(g, tag + "_rep");
        addNode(g, OpType::Reshape, tag + "_reshape1", {cache, addI64Init(g, tag + "_s1", {kB, kKvHeads, 1, kTokens, kHd})}, r1);
        addNode(g, OpType::Expand, tag + "_expand", {r1, addI64Init(g, tag + "_s2", {kB, kKvHeads, kGroup, kTokens, kHd})}, ex);
        addNode(g, OpType::Reshape, tag + "_reshape2", {ex, addI64Init(g, tag + "_s3", {kB, kHeads, kTokens, kHd})}, r2);
        if (!transposeToQk)
        {
            return r2;
        }
        TensorId tr = addTemp(g, tag + "_kT");
        Node    *tn = addNode(g, OpType::Transpose, tag + "_transpose", {r2}, tr);
        Attr     perm;
        perm.kind            = Attr::Ints;
        perm.ints            = {0, 1, 3, 2};
        tn->attr.map["perm"] = perm;
        return tr;
    }

    // The decode-attention chain at toy scale, mirroring the production decode bucket:
    //   q [1,H,1,hd]; k,v caches [1,kv,S,hd] through repeat_kv; scores * scale + mask;
    //   Softmax(-1); probs @ v; Transpose(0,2,1,3); Reshape [1,1,H*hd].
    // withScaleMask=false emits the bare MatMul->Softmax->MatMul core. m>1 builds the prefill
    // (non-decode) form with that query length.
    Graph attnGraph(bool withScaleMask, int64_t m = 1) {
        Graph    g;
        TensorId q  = addInput(g, "q", {kB, kHeads, m, kHd});
        TensorId kc = addInput(g, "kcache", {kB, kKvHeads, kTokens, kHd});
        TensorId vc = addInput(g, "vcache", {kB, kKvHeads, kTokens, kHd});
        TensorId kT = repeatKv(g, kc, "k", true);
        TensorId vR = repeatKv(g, vc, "v", false);

        TensorId scores = addTemp(g, "scores");
        addNode(g, OpType::MatMul, "qk", {q, kT}, scores);

        TensorId smIn = scores;
        if (withScaleMask)
        {
            TensorId mask   = addInput(g, "mask", {1, 1, 1, kTokens});
            TensorId scale  = addF32Init(g, "scalec", {1}, {kScaleValue});
            TensorId scaled = addTemp(g, "scaled");
            Node    *mn     = addNode(g, OpType::Binary, "scalemul", {scores, scale}, scaled);
            mn->subOp       = (int32_t) BinaryType::Mul;
            TensorId masked = addTemp(g, "masked");
            addNode(g, OpType::Add, "maskadd", {scaled, mask}, masked);
            smIn = masked;
        }

        TensorId probs = addTemp(g, "probs");
        Node    *sn    = addNode(g, OpType::Softmax, "softmax", {smIn}, probs);
        Attr     ax;
        ax.kind              = Attr::Int;
        ax.i                 = -1;
        sn->attr.map["axis"] = ax;

        TensorId ctx = addTemp(g, "ctx");
        addNode(g, OpType::MatMul, "pv", {probs, vR}, ctx);

        TensorId ctxT = addTemp(g, "ctxT");
        Node    *tn   = addNode(g, OpType::Transpose, "ctx_transpose", {ctx}, ctxT);
        Attr     perm;
        perm.kind            = Attr::Ints;
        perm.ints            = {0, 2, 1, 3};
        tn->attr.map["perm"] = perm;

        TensorDesc yo;
        yo.name     = "out";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        addNode(g, OpType::Reshape, "ctx_reshape", {ctxT, addI64Init(g, "so", {kB, m, kHeads * kHd})}, y);
        g.outputs = {y};
        return g;
    }

    const Node *findNode(const Graph &g, OpType t) {
        for (const Node &n: g.nodes)
        {
            if (n.type == t)
            {
                return &n;
            }
        }
        return nullptr;
    }

    int countNodes(const Graph &g, OpType t) {
        int c = 0;
        for (const Node &n: g.nodes)
        {
            c += n.type == t ? 1 : 0;
        }
        return c;
    }

    std::vector<float> ramp(int64_t n, float scale, float phase = 0.f) {
        std::vector<float> v((size_t) n);
        for (int64_t i = 0; i < n; ++i)
        {
            v[(size_t) i] = scale * std::sin(0.37f * (float) i + phase) + 0.05f * (float) (i % 11);
        }
        return v;
    }

    // The decode mask idiom: additive 0 / very-negative columns.
    std::vector<float> maskValues() {
        std::vector<float> m((size_t) kTokens, 0.f);
        m[1] = -65504.f;
        m[5] = -65504.f;
        return m;
    }

    IOTensor ioTensor(const std::string &name, Shape shape, const std::vector<float> &data) {
        IOTensor io;
        io.name  = name;
        io.shape = std::move(shape);
        io.dtype = DType::Float32;
        io.data.resize(data.size() * 4);
        std::memcpy(io.data.data(), data.data(), data.size() * 4);
        return io;
    }

    std::vector<IOTensor> decodeInputs(bool withScaleMask) {
        std::vector<IOTensor> ins = {
            ioTensor("q", {kB, kHeads, 1, kHd}, ramp(kHeads * kHd, 1.f)),
            ioTensor("kcache", {kB, kKvHeads, kTokens, kHd}, ramp(kKvHeads * kTokens * kHd, 0.7f, 0.9f)),
            ioTensor("vcache", {kB, kKvHeads, kTokens, kHd}, ramp(kKvHeads * kTokens * kHd, 0.5f, 2.1f)),
        };
        if (withScaleMask)
        {
            ins.push_back(ioTensor("mask", {1, 1, 1, kTokens}, maskValues()));
        }
        return ins;
    }

    std::vector<float> runGraph(Graph g, bool fuse, const std::vector<IOTensor> &ins) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        if (!fuse)
        {
            cfg.setHint(Hint::FusedAttention, (int) Mode::Off);
        }
        auto sess = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run(ins, outs), Status::Ok);
        for (const IOTensor &o: outs)
        {
            if (o.name == "out")
            {
                return std::vector<float>(o.f32(), o.f32() + numElements(o.shape));
            }
        }
        ADD_FAILURE() << "no output tensor";
        return {};
    }

    // Ground-truth decode attention straight off the raw inputs, in double.
    std::vector<float> attnReference(const std::vector<float> &q, const std::vector<float> &kc, const std::vector<float> &vc, const std::vector<float> *mask, float scale) {
        std::vector<float> out((size_t) (kHeads * kHd));
        for (int64_t h = 0; h < kHeads; ++h)
        {
            const int64_t       kv = h / kGroup;
            std::vector<double> sc((size_t) kTokens);
            double              mx = -1e300;
            for (int64_t s = 0; s < kTokens; ++s)
            {
                double dot = 0;
                for (int64_t d = 0; d < kHd; ++d)
                {
                    dot += (double) q[(size_t) (h * kHd + d)] * (double) kc[(size_t) ((kv * kTokens + s) * kHd + d)];
                }
                sc[(size_t) s] = dot * scale + (mask ? (double) (*mask)[(size_t) s] : 0.0);
                mx             = std::max(mx, sc[(size_t) s]);
            }
            double sum = 0;
            for (int64_t s = 0; s < kTokens; ++s)
            {
                sc[(size_t) s] = std::exp(sc[(size_t) s] - mx);
                sum += sc[(size_t) s];
            }
            for (int64_t n = 0; n < kHd; ++n)
            {
                double acc = 0;
                for (int64_t s = 0; s < kTokens; ++s)
                {
                    acc += sc[(size_t) s] / sum * (double) vc[(size_t) ((kv * kTokens + s) * kHd + n)];
                }
                out[(size_t) (h * kHd + n)] = (float) acc;
            }
        }
        return out;
    }

    float maxRelErr(const std::vector<float> &a, const std::vector<float> &b) {
        EXPECT_EQ(a.size(), b.size());
        float worst = 0.f;
        for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        {
            float denom = std::max(1e-3f, std::fabs(b[i]));
            worst       = std::max(worst, std::fabs(a[i] - b[i]) / denom);
        }
        return worst;
    }

    // The compiled decode graph with the load-time folds applied, as session load produces it.
    Graph compiledAndFolded(bool withScaleMask, int64_t m = 1, const std::string &fp32Pins = "", bool pointwiseFusion = true) {
        Graph       g = attnGraph(withScaleMask, m);
        PassOptions opt;
        opt.fusePointwiseChains = pointwiseFusion;
        runStandardPasses(g, opt);
        foldMatMulViews(g);
        fuseDecodeAttention(g, fp32Pins);
        return g;
    }

} // namespace

// The production form — scale and mask fused as a pointwise epilogue on the QK MatMul — collapses
// to one FusedAttention node carrying the composed view strides, the scalar scale, and the mask
// operand; the whole decomposed chain (both MatMuls, the Softmax, the output Transpose/Reshape)
// leaves the graph.
TEST(FusedAttention, FusesEpilogueForm) {
    Graph g = compiledAndFolded(true);

    const Node *fa = findNode(g, OpType::FusedAttention);
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(countNodes(g, OpType::MatMul), 0);
    EXPECT_EQ(countNodes(g, OpType::Softmax), 0);
    EXPECT_EQ(countNodes(g, OpType::Transpose), 0);
    EXPECT_EQ(countNodes(g, OpType::Binary), 0);
    EXPECT_EQ(countNodes(g, OpType::Add), 0);

    ASSERT_EQ(fa->inputs.size(), 4u); // q, k cache, v cache, mask
    EXPECT_EQ(g.desc(fa->inputs[0]).name, "q");
    EXPECT_EQ(g.desc(fa->inputs[1]).name, "kcache");
    EXPECT_EQ(g.desc(fa->inputs[2]).name, "vcache");
    EXPECT_EQ(g.desc(fa->inputs[3]).name, "mask");
    EXPECT_EQ(g.desc(fa->outputs[0]).name, "out");

    EXPECT_EQ(fa->attr.geti(kFa), 1);
    EXPECT_EQ(fa->attr.getints(kFaDims), (std::vector<int64_t> {1, kKvHeads, kGroup}));
    EXPECT_EQ(fa->attr.getints(kFaQStride), (std::vector<int64_t> {0, kGroup * kHd, kHd}));
    EXPECT_EQ(fa->attr.getints(kFaKStride), (std::vector<int64_t> {0, kTokens * kHd, 0}));
    EXPECT_EQ(fa->attr.getints(kFaVStride), (std::vector<int64_t> {0, kTokens * kHd, 0}));
    EXPECT_EQ(fa->attr.getints(kFaMStride), (std::vector<int64_t> {0, 0, 0}));
    EXPECT_EQ(fa->attr.geti(kFaQK), 1);
    EXPECT_EQ(fa->attr.geti(kFaKN), kHd);
    EXPECT_EQ(fa->attr.geti(kFaKK), 1);
    EXPECT_EQ(fa->attr.geti(kFaVN), 1);
    EXPECT_EQ(fa->attr.geti(kFaVK), kHd);
    EXPECT_EQ(fa->attr.geti(kFaMN), 1);
    EXPECT_EQ(fa->attr.geti(kFaC), kTokens);
    EXPECT_EQ(fa->attr.geti(kFaHd), kHd);
    EXPECT_FLOAT_EQ(fa->attr.getf(kFaScale, 1.f), kScaleValue);
    EXPECT_FLOAT_EQ(fa->attr.getf(kFaMaskScale, 1.f), 1.f);
}

// The standalone form — scale/mask left as their own Binary/Add nodes (a no-pointwise-fusion
// compile) — fuses through the same node-walk path with the same composed result.
TEST(FusedAttention, FusesStandaloneScaleMaskForm) {
    Graph g = compiledAndFolded(true, 1, "", /*pointwiseFusion=*/false);

    const Node *fa = findNode(g, OpType::FusedAttention);
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(countNodes(g, OpType::MatMul), 0);
    EXPECT_EQ(countNodes(g, OpType::Softmax), 0);
    EXPECT_EQ(countNodes(g, OpType::Binary), 0);
    EXPECT_EQ(countNodes(g, OpType::Add), 0);
    ASSERT_EQ(fa->inputs.size(), 4u);
    EXPECT_FLOAT_EQ(fa->attr.getf(kFaScale, 1.f), kScaleValue);
    EXPECT_EQ(fa->attr.geti(kFaC), kTokens);
}

// The bare core (no scale, no mask) fuses with the neutral scale and three inputs.
TEST(FusedAttention, FusesMasklessCore) {
    Graph       g  = compiledAndFolded(false);
    const Node *fa = findNode(g, OpType::FusedAttention);
    ASSERT_NE(fa, nullptr);
    ASSERT_EQ(fa->inputs.size(), 3u);
    EXPECT_FLOAT_EQ(fa->attr.getf(kFaScale, 1.f), 1.f);
    EXPECT_FALSE(fa->attr.has(kFaMStride));
}

// The fused CPU op matches the decomposed CPU chain (fp32 vs fp32-with-fp16-free storage: same
// math, reassociated softmax/PV order only) and the double-precision ground truth.
TEST(FusedAttention, CpuMatchesDecomposedChain) {
    for (bool withScaleMask: {true, false})
    {
        std::vector<IOTensor> ins   = decodeInputs(withScaleMask);
        std::vector<float>    fused = runGraph(attnGraph(withScaleMask), true, ins);
        std::vector<float>    plain = runGraph(attnGraph(withScaleMask), false, ins);
        std::vector<float>    mask  = maskValues();
        std::vector<float> ref = attnReference(ramp(kHeads * kHd, 1.f), ramp(kKvHeads * kTokens * kHd, 0.7f, 0.9f), ramp(kKvHeads * kTokens * kHd, 0.5f, 2.1f), withScaleMask ? &mask : nullptr, withScaleMask ? kScaleValue : 1.f);
        ASSERT_EQ(fused.size(), (size_t) (kHeads * kHd));
        EXPECT_LT(maxRelErr(fused, plain), 1e-3f) << (withScaleMask ? "scale+mask" : "bare");
        EXPECT_LT(maxRelErr(fused, ref), 1e-3f) << (withScaleMask ? "scale+mask" : "bare") << " vs reference";
    }
}

// Hint::FusedAttention Off keeps the decomposed plan: no FusedAttention node in the session graph
// and identical-to-tolerance outputs.
TEST(FusedAttention, HintOffKeepsDecomposedPlan) {
    Config cfg;
    cfg.backend = BackendKind::Cpu;
    cfg.setHint(Hint::FusedAttention, (int) Mode::Off);
    auto off = Session::create(attnGraph(true), cfg);
    ASSERT_TRUE(off);
    EXPECT_EQ(countNodes(off->graph(), OpType::FusedAttention), 0);
    EXPECT_NE(countNodes(off->graph(), OpType::Softmax), 0);

    Config cfgOn;
    cfgOn.backend = BackendKind::Cpu;
    auto on       = Session::create(attnGraph(true), cfgOn);
    ASSERT_TRUE(on);
    EXPECT_EQ(countNodes(on->graph(), OpType::FusedAttention), 1);
}

// A chain node carrying fused work it does not understand is refused: foreign pw_steps on the PV
// MatMul (the bareNode guard), and an unsupported binary code in the standalone scale node.
TEST(FusedAttention, ForeignFusedWorkRefused) {
    {
        Graph g = attnGraph(true);
        runStandardPasses(g, PassOptions {});
        foldMatMulViews(g);
        for (Node &n: g.nodes)
        {
            if (n.name == "pv")
            {
                Attr marker;
                marker.kind            = Attr::Int;
                marker.i               = 1;
                n.attr.map["pw_steps"] = marker;
            }
        }
        fuseDecodeAttention(g);
        EXPECT_EQ(findNode(g, OpType::FusedAttention), nullptr);
        EXPECT_NE(findNode(g, OpType::Softmax), nullptr);
    }
    {
        Graph       g = attnGraph(true);
        PassOptions opt;
        opt.fusePointwiseChains = false;
        runStandardPasses(g, opt);
        foldMatMulViews(g);
        for (Node &n: g.nodes)
        {
            if (n.name == "scalemul")
            {
                n.subOp = (int32_t) BinaryType::Div; // not a scale multiply anymore
            }
        }
        fuseDecodeAttention(g);
        EXPECT_EQ(findNode(g, OpType::FusedAttention), nullptr);
        EXPECT_NE(findNode(g, OpType::Softmax), nullptr);
    }
}

// The prefill form (query length > 1) keeps the primitive path.
TEST(FusedAttention, PrefillRefused) {
    Graph g = compiledAndFolded(true, /*m=*/2);
    EXPECT_EQ(findNode(g, OpType::FusedAttention), nullptr);
    EXPECT_NE(findNode(g, OpType::Softmax), nullptr);
}

// An interior tensor matching the markFp32 pin set keeps the decomposed form (erasing it would
// erase the fp32 store the pin exists for).
TEST(FusedAttention, Fp32PinnedInteriorRefused) {
    Graph g = compiledAndFolded(true, 1, "probs");
    EXPECT_EQ(findNode(g, OpType::FusedAttention), nullptr);
    EXPECT_NE(findNode(g, OpType::Softmax), nullptr);
}

// A graph without the decode-attention pattern is untouched: a MatMul -> Softmax -> MatMul chain
// whose operands are plain (no folded view attrs) stays as-is, node for node.
TEST(FusedAttention, NoViewPatternUntouched) {
    Graph    g;
    TensorId q = addInput(g, "q", {1, kHeads, 1, kHd});
    TensorId k = addInput(g, "k", {1, kHeads, kHd, kTokens});
    TensorId v = addInput(g, "v", {1, kHeads, kTokens, kHd});
    TensorId s = addTemp(g, "scores");
    TensorId p = addTemp(g, "probs");
    TensorId c = addTemp(g, "ctx");
    addNode(g, OpType::MatMul, "qk", {q, k}, s);
    Node *sn = addNode(g, OpType::Softmax, "softmax", {s}, p);
    Attr  ax;
    ax.kind              = Attr::Int;
    ax.i                 = -1;
    sn->attr.map["axis"] = ax;
    addNode(g, OpType::MatMul, "pv", {p, v}, c);
    g.desc(c).isOutput = true;
    g.outputs          = {c};
    runStandardPasses(g, PassOptions {});
    foldMatMulViews(g);
    const size_t before = g.nodes.size();
    fuseDecodeAttention(g);
    EXPECT_EQ(g.nodes.size(), before);
    EXPECT_EQ(findNode(g, OpType::FusedAttention), nullptr);
}

// A CNN-shaped graph (no attention pattern at all) passes through the fusion untouched.
TEST(FusedAttention, CnnGraphUntouched) {
    Graph    g;
    TensorId x = addInput(g, "x", {1, 8, 4, 4});
    TensorId r = addTemp(g, "relu_out");
    TensorId p = addTemp(g, "probs");
    addNode(g, OpType::Relu, "relu", {x}, r);
    Node *sn = addNode(g, OpType::Softmax, "softmax", {r}, p);
    Attr  ax;
    ax.kind              = Attr::Int;
    ax.i                 = 1;
    sn->attr.map["axis"] = ax;
    g.desc(p).isOutput   = true;
    g.outputs            = {p};
    runStandardPasses(g, PassOptions {});
    foldMatMulViews(g);
    const size_t before = g.nodes.size();
    fuseDecodeAttention(g);
    EXPECT_EQ(g.nodes.size(), before);
    EXPECT_EQ(findNode(g, OpType::FusedAttention), nullptr);
}

// foldFusedAttentionKvConcat: with the caches produced by past‖new Concats (the with-past decoder
// step), the fold rewires the fused node onto the two sources, rewrites the present outputs to the
// new-rows tensors under the present names, and the attention output stays BYTE-identical — the
// fold moves the copy, never the math.
TEST(FusedAttention, KvConcatFoldsToSplitSources) {
    auto buildConcatGraph = [&]() {
        Graph    g;
        TensorId q     = addInput(g, "q", {kB, kHeads, 1, kHd});
        TensorId kPast = addInput(g, "kpast", {kB, kKvHeads, kTokens - 1, kHd});
        TensorId kNew  = addInput(g, "knew", {kB, kKvHeads, 1, kHd});
        TensorId vPast = addInput(g, "vpast", {kB, kKvHeads, kTokens - 1, kHd});
        TensorId vNew  = addInput(g, "vnew", {kB, kKvHeads, 1, kHd});
        TensorDesc kco;
        kco.name     = "present_key";
        kco.isOutput = true;
        TensorId kc  = g.addTensor(kco);
        TensorDesc vco;
        vco.name     = "present_value";
        vco.isOutput = true;
        TensorId vc  = g.addTensor(vco);
        Attr axis;
        axis.kind = Attr::Int;
        axis.i    = 2;
        addNode(g, OpType::Concat, "kcat", {kPast, kNew}, kc)->attr.map["axis"] = axis;
        addNode(g, OpType::Concat, "vcat", {vPast, vNew}, vc)->attr.map["axis"] = axis;

        TensorId kT = repeatKv(g, kc, "k", true);
        TensorId vR = repeatKv(g, vc, "v", false);
        TensorId scores = addTemp(g, "scores");
        addNode(g, OpType::MatMul, "qk", {q, kT}, scores);
        TensorId probs = addTemp(g, "probs");
        Node    *sn    = addNode(g, OpType::Softmax, "softmax", {scores}, probs);
        Attr     ax;
        ax.kind              = Attr::Int;
        ax.i                 = -1;
        sn->attr.map["axis"] = ax;
        TensorId ctx = addTemp(g, "ctx");
        addNode(g, OpType::MatMul, "pv", {probs, vR}, ctx);
        TensorId ctxT = addTemp(g, "ctxT");
        Node    *tn   = addNode(g, OpType::Transpose, "ctx_transpose", {ctx}, ctxT);
        Attr     perm;
        perm.kind            = Attr::Ints;
        perm.ints            = {0, 2, 1, 3};
        tn->attr.map["perm"] = perm;
        TensorDesc yo;
        yo.name     = "out";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        addNode(g, OpType::Reshape, "ctx_reshape", {ctxT, addI64Init(g, "so", {kB, 1, kHeads * kHd})}, y);
        g.outputs = {y, kc, vc};
        inferShapes(g);
        foldMatMulViews(g);
        fuseDecodeAttention(g);
        return g;
    };

    auto randTensor = [](const std::string &name, const Shape &shape, unsigned seed) {
        IOTensor io;
        io.name  = name;
        io.shape = shape;
        io.dtype = DType::Float32;
        int64_t elems = numElements(shape);
        io.data.resize((size_t) elems * 4);
        float *f = reinterpret_cast<float *>(io.data.data());
        unsigned s = seed;
        for (int64_t i = 0; i < elems; ++i)
        {
            s    = s * 1664525u + 1013904223u;
            f[i] = ((float) (s >> 8) / (float) (1 << 24)) - 0.5f;
        }
        return io;
    };
    const std::vector<IOTensor> inputs = {
        randTensor("q", {kB, kHeads, 1, kHd}, 1),
        randTensor("kpast", {kB, kKvHeads, kTokens - 1, kHd}, 2),
        randTensor("knew", {kB, kKvHeads, 1, kHd}, 3),
        randTensor("vpast", {kB, kKvHeads, kTokens - 1, kHd}, 4),
        randTensor("vnew", {kB, kKvHeads, 1, kHd}, 5),
    };
    auto outBytes = [](const std::vector<IOTensor> &outs, const std::string &name) {
        for (const IOTensor &o: outs)
        {
            if (o.name == name)
            {
                return o.data;
            }
        }
        ADD_FAILURE() << "missing output " << name;
        return std::vector<uint8_t> {};
    };

    // Unsplit reference: the fused node reads the concatenated caches.
    Graph gRef = buildConcatGraph();
    // Folded: the fused node reads the two sources; the Concats die.
    Graph gFold = buildConcatGraph();
    EXPECT_EQ(foldFusedAttentionKvConcat(gFold), 1);
    const Node *fa = findNode(gFold, OpType::FusedAttention);
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(fa->attr.geti(kFaSplit, 0), 1);
    EXPECT_EQ(fa->attr.geti(kFaPastLen), kTokens - 1);
    ASSERT_EQ(fa->inputs.size(), 6u);
    EXPECT_EQ(findNode(gFold, OpType::Concat), nullptr); // dead concats removed
    // The present outputs became the new-rows tensors under the present names.
    bool foundKey = false, foundValue = false;
    for (TensorId t: gFold.outputs)
    {
        foundKey   = foundKey || (gFold.desc(t).name == "present_key" && gFold.desc(t).shape == Shape {kB, kKvHeads, 1, kHd});
        foundValue = foundValue || (gFold.desc(t).name == "present_value" && gFold.desc(t).shape == Shape {kB, kKvHeads, 1, kHd});
    }
    EXPECT_TRUE(foundKey);
    EXPECT_TRUE(foundValue);

    // Session::create re-runs the load passes, so the reference session must pin the fold OFF
    // (the folded session leaves the default ON — either way the graph is already folded above).
    Config cfgRef;
    cfgRef.backend = BackendKind::Cpu;
    cfgRef.setHint(Hint::KvConcatFold, (int) Mode::Off);
    Config cfgFold;
    cfgFold.backend = BackendKind::Cpu;
    auto sRef  = Session::create(std::move(gRef), cfgRef);
    auto sFold = Session::create(std::move(gFold), cfgFold);
    ASSERT_TRUE(sRef);
    ASSERT_TRUE(sFold);
    std::vector<IOTensor> oRef, oFold;
    ASSERT_EQ(sRef->run(inputs, oRef), Status::Ok);
    ASSERT_EQ(sFold->run(inputs, oFold), Status::Ok);
    // The attention output is byte-identical; the folded present is the last (new) row of the
    // reference present.
    EXPECT_EQ(outBytes(oRef, "out"), outBytes(oFold, "out"));
    const std::vector<uint8_t> refKey  = outBytes(oRef, "present_key");
    const std::vector<uint8_t> foldKey = outBytes(oFold, "present_key");
    ASSERT_EQ(refKey.size(), (size_t) kB * kKvHeads * kTokens * kHd * 4);
    ASSERT_EQ(foldKey.size(), (size_t) kB * kKvHeads * kHd * 4);
    for (int64_t h = 0; h < kKvHeads; ++h)
    {
        const uint8_t *refRow  = refKey.data() + ((h * kTokens + (kTokens - 1)) * kHd) * 4;
        const uint8_t *foldRow = foldKey.data() + (h * kHd) * 4;
        EXPECT_EQ(std::memcmp(refRow, foldRow, (size_t) kHd * 4), 0) << "present head " << h;
    }
}

namespace {

    // A with-past decode step whose past K/V are the cache INPUTS (the linked-decoder shape class):
    // present.{key,value} = past ‖ new (Concat outputs under the present names), attention reads the
    // concats through repeat_kv views, and an additive mask marks the valid slots. The new rows are
    // INTERNAL tensors (an identity-projection MatMul stands in for the production k/v projections),
    // so the folded present output is a mid-graph tensor exactly like a real decoder's. The load
    // passes (foldMatMulViews -> fuseDecodeAttention -> foldFusedAttentionKvConcat when enabled)
    // turn this into the split-source FusedAttention with a rows-only present.
    Graph withPastDecodeGraph() {
        constexpr int64_t kSlots = kTokens - 1;
        Graph             g;
        TensorId          q      = addInput(g, "q", {kB, kHeads, 1, kHd});
        TensorId          pastK  = addInput(g, "past_key", {kB, kKvHeads, kSlots, kHd});
        TensorId          pastV  = addInput(g, "past_value", {kB, kKvHeads, kSlots, kHd});
        TensorId          kNewIn = addInput(g, "knew", {kB, kKvHeads, 1, kHd});
        TensorId          vNewIn = addInput(g, "vnew", {kB, kKvHeads, 1, kHd});
        std::vector<float> eye((size_t) (kHd * kHd), 0.f);
        for (int64_t d = 0; d < kHd; ++d)
        {
            eye[(size_t) (d * kHd + d)] = 1.f;
        }
        TensorId kNew = addTemp(g, "knew_rows");
        TensorId vNew = addTemp(g, "vnew_rows");
        addNode(g, OpType::MatMul, "k_proj", {kNewIn, addF32Init(g, "k_eye", {kHd, kHd}, eye)}, kNew);
        addNode(g, OpType::MatMul, "v_proj", {vNewIn, addF32Init(g, "v_eye", {kHd, kHd}, eye)}, vNew);
        TensorDesc kco;
        kco.name     = "present_key";
        kco.isOutput = true;
        TensorId kc  = g.addTensor(kco);
        TensorDesc vco;
        vco.name     = "present_value";
        vco.isOutput = true;
        TensorId vc  = g.addTensor(vco);
        Attr     axis;
        axis.kind                                                                = Attr::Int;
        axis.i                                                                   = 2;
        addNode(g, OpType::Concat, "kcat", {pastK, kNew}, kc)->attr.map["axis"]  = axis;
        addNode(g, OpType::Concat, "vcat", {pastV, vNew}, vc)->attr.map["axis"]  = axis;

        TensorId kT     = repeatKv(g, kc, "k", true);
        TensorId vR     = repeatKv(g, vc, "v", false);
        TensorId scores = addTemp(g, "scores");
        addNode(g, OpType::MatMul, "qk", {q, kT}, scores);
        TensorId mask   = addInput(g, "mask", {1, 1, 1, kTokens});
        TensorId scale  = addF32Init(g, "scalec", {1}, {kScaleValue});
        TensorId scaled = addTemp(g, "scaled");
        addNode(g, OpType::Binary, "scalemul", {scores, scale}, scaled)->subOp = (int32_t) BinaryType::Mul;
        TensorId masked                                                        = addTemp(g, "masked");
        addNode(g, OpType::Add, "maskadd", {scaled, mask}, masked);
        TensorId probs       = addTemp(g, "probs");
        Node    *sn          = addNode(g, OpType::Softmax, "softmax", {masked}, probs);
        Attr     ax;
        ax.kind              = Attr::Int;
        ax.i                 = -1;
        sn->attr.map["axis"] = ax;
        TensorId ctx         = addTemp(g, "ctx");
        addNode(g, OpType::MatMul, "pv", {probs, vR}, ctx);
        TensorId ctxT        = addTemp(g, "ctxT");
        Node    *tn          = addNode(g, OpType::Transpose, "ctx_transpose", {ctx}, ctxT);
        Attr     perm;
        perm.kind            = Attr::Ints;
        perm.ints            = {0, 2, 1, 3};
        tn->attr.map["perm"] = perm;
        TensorDesc yo;
        yo.name     = "out";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        addNode(g, OpType::Reshape, "ctx_reshape", {ctxT, addI64Init(g, "so", {kB, 1, kHeads * kHd})}, y);
        g.outputs = {y, kc, vc};
        return g;
    }

    IOTensor pseudoTensor(const std::string &name, const Shape &shape, unsigned seed) {
        IOTensor io;
        io.name       = name;
        io.shape      = shape;
        io.dtype      = DType::Float32;
        int64_t elems = numElements(shape);
        io.data.resize((size_t) elems * 4);
        float   *f = reinterpret_cast<float *>(io.data.data());
        unsigned s = seed;
        for (int64_t i = 0; i < elems; ++i)
        {
            s    = s * 1664525u + 1013904223u;
            f[i] = ((float) (s >> 8) / (float) (1 << 24)) - 0.5f;
        }
        return io;
    }

} // namespace

// The folded rows-only present drives the engine-resident KV link: a fold-On session linking
// present -> past (ranges from kvFoldRanges over the ACTUAL present shape) streams byte-identical
// attention outputs to the fold-Off host-cache loop across a multi-step decode, and cache-concat
// source offsets applied to the rows-only present stay rejected.
TEST(FusedAttention, KvConcatFoldedPresentDrivesResidentLink) {
    constexpr int64_t kSlots = kTokens - 1;
    Config            cfgRef;
    cfgRef.backend = BackendKind::Cpu;
    cfgRef.setHint(Hint::KvConcatFold, (int) Mode::Off);
    Config cfgFold;
    cfgFold.backend = BackendKind::Cpu;
    cfgFold.setHint(Hint::KvConcatFold, (int) Mode::On);
    auto sRef  = Session::create(withPastDecodeGraph(), cfgRef);
    auto sFold = Session::create(withPastDecodeGraph(), cfgFold);
    ASSERT_TRUE(sRef);
    ASSERT_TRUE(sFold);

    // The fold rewrote the linked session's present outputs to rows-only; the reference keeps the
    // cache-concat convention. Both row counts come from the plans' own output shapes, exactly like
    // the decode drivers.
    int64_t refRows = 0, foldRows = 0;
    for (const IOInfo &out: sRef->outputInfo(0))
    {
        refRows = out.name == "present_key" ? out.shape[2] : refRows;
    }
    for (const IOInfo &out: sFold->outputInfo(0))
    {
        foldRows = out.name == "present_key" ? out.shape[2] : foldRows;
    }
    ASSERT_EQ(refRows, kTokens);
    ASSERT_EQ(foldRows, 1);

    // The rows-only present links; cache-concat source offsets against it stay rejected.
    ASSERT_EQ(sFold->linkOutputToInput("present_key", "past_key", {}), Status::Ok);
    ASSERT_EQ(sFold->linkOutputToInput("present_value", "past_value", {}), Status::Ok);
    EXPECT_EQ(sFold->linkOutputToInput("present_key", "past_key", kvFoldRanges(kKvHeads, refRows, kSlots, kHd, 0)), Status::InvalidArgument);

    std::vector<float> pastKeyRef((size_t) (kKvHeads * kSlots * kHd), 0.f);
    std::vector<float> pastValueRef((size_t) (kKvHeads * kSlots * kHd), 0.f);
    auto               outBytes = [](const std::vector<IOTensor> &outs, const std::string &name) {
        for (const IOTensor &o: outs)
        {
            if (o.name == name)
            {
                return o.data;
            }
        }
        ADD_FAILURE() << "missing output " << name;
        return std::vector<uint8_t> {};
    };
    for (int64_t p = 0; p < kSlots; ++p)
    {
        // Additive decode mask: slots < p valid, empty slots masked, the current token (the last
        // concat column) valid.
        std::vector<float> mask((size_t) kTokens, -65504.f);
        for (int64_t j = 0; j < p; ++j)
        {
            mask[(size_t) j] = 0.f;
        }
        mask[(size_t) kTokens - 1]  = 0.f;
        const IOTensor     qT       = pseudoTensor("q", {kB, kHeads, 1, kHd}, (unsigned) (7 * p + 1));
        const IOTensor     kNewT    = pseudoTensor("knew", {kB, kKvHeads, 1, kHd}, (unsigned) (7 * p + 2));
        const IOTensor     vNewT    = pseudoTensor("vnew", {kB, kKvHeads, 1, kHd}, (unsigned) (7 * p + 3));
        const IOTensor     maskT    = ioTensor("mask", {1, 1, 1, kTokens}, mask);

        // Linked: fold the PREVIOUS step's present row into slot p-1 at the head of this run.
        ASSERT_EQ(sFold->linkOutputToInput("present_key", "past_key", kvFoldRanges(kKvHeads, foldRows, kSlots, kHd, kvFoldSlot(p, kSlots))), Status::Ok);
        ASSERT_EQ(sFold->linkOutputToInput("present_value", "past_value", kvFoldRanges(kKvHeads, foldRows, kSlots, kHd, kvFoldSlot(p, kSlots))), Status::Ok);
        std::vector<IOTensor> foldOuts;
        ASSERT_EQ(sFold->run({qT, kNewT, vNewT, maskT}, foldOuts), Status::Ok);

        // Reference: bind the host cache, then fold the newest present row into slot p by hand.
        std::vector<IOTensor> refOuts;
        ASSERT_EQ(sRef->run({qT, ioTensor("past_key", {kB, kKvHeads, kSlots, kHd}, pastKeyRef), ioTensor("past_value", {kB, kKvHeads, kSlots, kHd}, pastValueRef), kNewT, vNewT, maskT}, refOuts),
                  Status::Ok);
        for (int part = 0; part < 2; ++part)
        {
            const std::vector<uint8_t> present = outBytes(refOuts, part ? "present_value" : "present_key");
            std::vector<float>        &cache   = part ? pastValueRef : pastKeyRef;
            const float               *rows    = reinterpret_cast<const float *>(present.data());
            for (int64_t h = 0; h < kKvHeads; ++h)
            {
                std::memcpy(cache.data() + (size_t) ((h * kSlots + p) * kHd), rows + (h * refRows + (refRows - 1)) * kHd, (size_t) kHd * 4);
            }
        }

        EXPECT_EQ(outBytes(foldOuts, "out"), outBytes(refOuts, "out")) << "step " << p;
        // The linked present is engine-resident: metadata only, no payload.
        EXPECT_TRUE(outBytes(foldOuts, "present_key").empty());
    }
}
