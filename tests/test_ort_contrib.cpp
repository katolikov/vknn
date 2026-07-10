// ORT contrib-operator lowering (lowerOrtContribOps) and the fold-path scalar-rank restoration it
// depends on. The expansions are pure graph rewrites onto existing primitives, so each test builds
// the contrib node directly, runs the standard passes, and checks both the structure and the CPU
// value against an in-test reference.
#include "core/quant_int4.h"
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    TensorId addInput(Graph &g, const char *name, Shape s) {
        TensorDesc d;
        d.name    = name;
        d.shape   = std::move(s);
        d.isInput = true;
        TensorId id = g.addTensor(d);
        g.inputs.push_back(id);
        return id;
    }
    TensorId addOutput(Graph &g, const char *name) {
        TensorDesc d;
        d.name     = name;
        d.isOutput = true;
        TensorId id = g.addTensor(d);
        g.outputs.push_back(id);
        return id;
    }
    TensorId addInitF32(Graph &g, const char *name, Shape s, const std::vector<float> &v) {
        TensorDesc d;
        d.name          = name;
        d.shape         = std::move(s);
        d.isInitializer = true;
        TensorId id     = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) v.size(), DType::Float32);
        std::memcpy(hb.f32(), v.data(), v.size() * 4);
        g.initializers[id] = hb;
        return id;
    }
    TensorId addInitI64(Graph &g, const char *name, Shape s, const std::vector<int64_t> &v) {
        TensorDesc d;
        d.name          = name;
        d.shape         = std::move(s);
        d.dtype         = DType::Int64;
        d.isInitializer = true;
        TensorId id     = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) v.size(), DType::Int64);
        std::memcpy(hb.i64(), v.data(), v.size() * 8);
        g.initializers[id] = hb;
        return id;
    }
    void setI(Node &n, const char *k, int64_t v) {
        Attr a;
        a.kind        = Attr::Int;
        a.i           = v;
        n.attr.map[k] = a;
    }
    void setF(Node &n, const char *k, float v) {
        Attr a;
        a.kind        = Attr::Float;
        a.f           = v;
        n.attr.map[k] = a;
    }
    void setInts(Node &n, const char *k, std::vector<int64_t> v) {
        Attr a;
        a.kind        = Attr::Ints;
        a.ints        = std::move(v);
        n.attr.map[k] = a;
    }

    std::vector<float> runCpu(Graph g, const std::map<std::string, std::pair<Shape, std::vector<float>>> &feeds,
                              const std::string &outName) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        std::vector<IOTensor> ins;
        for (const auto &kv: feeds)
        {
            IOTensor t;
            t.name  = kv.first;
            t.shape = kv.second.first;
            t.dtype = DType::Float32;
            t.data.resize(kv.second.second.size() * 4);
            std::memcpy(t.data.data(), kv.second.second.data(), t.data.size());
            ins.push_back(std::move(t));
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run(ins, outs), Status::Ok);
        for (const IOTensor &o: outs)
        {
            if (o.name == outName)
            {
                return std::vector<float>(o.f32(), o.f32() + std::max<int64_t>(numElements(o.shape), 1));
            }
        }
        ADD_FAILURE() << "no output " << outName;
        return {};
    }

} // namespace

// A folded Gather-of-Shape with a rank-0 index stays conceptually scalar through Unsqueeze/Concat:
// the ORT transformer mask subgraph builds Expand/Reshape targets this way, and a [1]-normalized
// scalar used to inflate the chain by one rank per step (a [4,1] target broadcast against a genuine
// 1-D vector then explodes into a matrix-shaped "shape").
TEST(OrtContrib, ScalarGatherFoldKeepsOnnxRank) {
    Graph    g;
    TensorId x   = addInput(g, "x", {1, 257});
    TensorId idx = addInitI64(g, "i0", {}, {1}); // rank-0 index
    TensorId one = addInitI64(g, "one", {1}, {1});
    TensorId shp = g.addTensor([] { TensorDesc d; d.name = "shp"; d.dtype = DType::Int64; return d; }());
    TensorId gth = g.addTensor([] { TensorDesc d; d.name = "gth"; d.dtype = DType::Int64; return d; }());
    TensorId u1  = g.addTensor([] { TensorDesc d; d.name = "u1"; d.dtype = DType::Int64; return d; }());
    TensorId cat = g.addTensor([] { TensorDesc d; d.name = "cat"; d.dtype = DType::Int64; return d; }());
    Node sh;
    sh.type    = OpType::Shape;
    sh.name    = "shape";
    sh.inputs  = {x};
    sh.outputs = {shp};
    g.nodes.push_back(sh);
    Node gn;
    gn.type    = OpType::Gather;
    gn.name    = "gather";
    gn.inputs  = {shp, idx};
    gn.outputs = {gth};
    setI(gn, "axis", 0);
    g.nodes.push_back(gn);
    Node uq;
    uq.type    = OpType::Unsqueeze;
    uq.name    = "unsq";
    uq.inputs  = {gth};
    uq.outputs = {u1};
    setInts(uq, "axes", {0});
    g.nodes.push_back(uq);
    Node cc;
    cc.type    = OpType::Concat;
    cc.name    = "concat";
    cc.inputs  = {u1, one, u1};
    cc.outputs = {cat};
    setI(cc, "axis", 0);
    g.nodes.push_back(cc);
    // Keep the chain alive: route it through a Reshape target so DCE cannot drop it.
    TensorId y  = addOutput(g, "y");
    Node     rs;
    rs.type    = OpType::Reshape;
    rs.name    = "reshape";
    rs.inputs  = {x, cat};
    rs.outputs = {y};
    g.nodes.push_back(rs);

    runStandardPasses(g);
    ASSERT_NE(g.find("gth"), kNoTensor);
    EXPECT_TRUE(g.desc(g.find("gth")).shape.empty()) << "scalar gather result must stay rank-0";
    EXPECT_EQ(g.desc(g.find("u1")).shape, (Shape {1}));
    EXPECT_EQ(g.desc(g.find("cat")).shape, (Shape {3}));
    const int64_t *v = g.initializers.at(g.find("cat")).i64();
    EXPECT_EQ(v[0], 257);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 257);
    EXPECT_EQ(g.desc(y).shape, (Shape {257, 1, 257}));
}

// SkipSimplifiedLayerNormalization expands to Add + RMSNorm, with output 3 (the residual sum)
// produced by the Add, and the values match the in-test reference.
TEST(OrtContrib, SkipSimplifiedLayerNormExpands) {
    const int64_t      E = 8;
    std::vector<float> gamma(E);
    for (int64_t i = 0; i < E; ++i)
    {
        gamma[(size_t) i] = 0.5f + 0.1f * (float) i;
    }
    Graph    g;
    TensorId x  = addInput(g, "x", {1, 2, E});
    TensorId sk = addInput(g, "skip", {1, 2, E});
    TensorId gm = addInitF32(g, "gamma", {E}, gamma);
    TensorId y  = addOutput(g, "y");
    TensorId sm = addOutput(g, "sum");
    Node     nd;
    nd.type    = OpType::SkipSimplifiedLayerNorm;
    nd.name    = "sln";
    nd.inputs  = {x, sk, gm};
    nd.outputs = {y, kNoTensor, kNoTensor, sm};
    setF(nd, "epsilon", 1e-6f);
    g.nodes.push_back(nd);

    Graph structural = g;
    runStandardPasses(structural);
    bool sawRms = false, sawContrib = false;
    for (const Node &n: structural.nodes)
    {
        sawRms     = sawRms || n.type == OpType::RMSNorm;
        sawContrib = sawContrib || n.type == OpType::SkipSimplifiedLayerNorm;
    }
    EXPECT_TRUE(sawRms);
    EXPECT_FALSE(sawContrib);

    std::vector<float> xv(16), sv(16);
    for (size_t i = 0; i < 16; ++i)
    {
        xv[i] = std::sin(0.3f * (float) i);
        sv[i] = std::cos(0.7f * (float) i);
    }
    auto got = runCpu(std::move(g), {{"x", {{1, 2, E}, xv}}, {"skip", {{1, 2, E}, sv}}}, "y");
    ASSERT_EQ((int64_t) got.size(), 16);
    for (int64_t r = 0; r < 2; ++r)
    {
        double ss = 0;
        for (int64_t c = 0; c < E; ++c)
        {
            const double s = xv[(size_t) (r * E + c)] + sv[(size_t) (r * E + c)];
            ss += s * s;
        }
        const double inv = 1.0 / std::sqrt(ss / E + 1e-6);
        for (int64_t c = 0; c < E; ++c)
        {
            const double want = (xv[(size_t) (r * E + c)] + sv[(size_t) (r * E + c)]) * inv * gamma[(size_t) c];
            EXPECT_NEAR(got[(size_t) (r * E + c)], want, 1e-4) << "r=" << r << " c=" << c;
        }
    }
}

// MatMulNBits repacks the ORT blockwise-uint4 weight into the int4 wq MatMul form, and the CPU run
// (through the load-time materialization) matches the dequantized reference exactly at fp16.
TEST(OrtContrib, MatMulNBitsRepacksToWqMatMul) {
    const int64_t        K = 64, N = 16, BS = 32, kb = K / BS;
    std::vector<uint8_t> q((size_t) (N * K));
    std::vector<float>   scales((size_t) (N * kb));
    for (size_t i = 0; i < q.size(); ++i)
    {
        q[i] = (uint8_t) ((i * 7) % 16);
    }
    for (size_t i = 0; i < scales.size(); ++i)
    {
        scales[i] = 0.125f * (float) (1 + i % 3); // exact in fp16
    }
    // ORT packing: B[N][kb][BS/2], low nibble first.
    std::vector<float> packedAsF32((size_t) (N * kb * BS / 2), 0.0f); // the importer's widened form
    for (int64_t n = 0; n < N; ++n)
    {
        for (int64_t k = 0; k < K; ++k)
        {
            size_t  at   = (size_t) (n * kb * (BS / 2) + (k / BS) * (BS / 2) + (k % BS) / 2);
            uint8_t byte = (uint8_t) packedAsF32[at];
            byte |= (k & 1) ? (uint8_t) (q[(size_t) (n * K + k)] << 4) : q[(size_t) (n * K + k)];
            packedAsF32[at] = (float) byte;
        }
    }
    Graph    g;
    TensorId a  = addInput(g, "a", {2, K});
    TensorId bq = addInitF32(g, "bq", {N, kb, BS / 2}, packedAsF32);
    g.desc(bq).dtype = DType::UInt8; // as the importer stamps a widened uint8 initializer
    TensorId sc = addInitF32(g, "sc", {N * kb}, scales);
    TensorId y  = addOutput(g, "y");
    Node     nd;
    nd.type    = OpType::MatMulNBits;
    nd.name    = "nbits";
    nd.inputs  = {a, bq, sc};
    nd.outputs = {y};
    setI(nd, "K", K);
    setI(nd, "N", N);
    setI(nd, "bits", 4);
    setI(nd, "block_size", BS);
    g.nodes.push_back(nd);

    Graph structural = g;
    runStandardPasses(structural);
    const Node *mm = nullptr;
    for (const Node &n: structural.nodes)
    {
        if (n.type == OpType::MatMul)
        {
            mm = &n;
        }
        EXPECT_NE(n.type, OpType::MatMulNBits);
    }
    ASSERT_TRUE(mm);
    EXPECT_TRUE(mm->attr.has(kWq));
    EXPECT_EQ(mm->attr.geti(kWqGroup, 0), BS);
    EXPECT_EQ(mm->attr.geti(kWqNOut, -1), 0);

    std::vector<float> av((size_t) (2 * K));
    for (size_t i = 0; i < av.size(); ++i)
    {
        av[i] = 0.25f * (float) ((int) (i % 9) - 4); // exact fp16 values
    }
    auto got = runCpu(std::move(g), {{"a", {{2, K}, av}}}, "y");
    ASSERT_EQ((int64_t) got.size(), 2 * N);
    for (int64_t r = 0; r < 2; ++r)
    {
        for (int64_t n = 0; n < N; ++n)
        {
            double acc = 0;
            for (int64_t k = 0; k < K; ++k)
            {
                const double w = (double) ((int) q[(size_t) (n * K + k)] - 8) * scales[(size_t) (n * kb + k / BS)];
                acc += (double) av[(size_t) (r * K + k)] * w;
            }
            EXPECT_NEAR(got[(size_t) (r * N + n)], acc, 1e-3 * std::max(1.0, std::fabs(acc))) << "r=" << r << " n=" << n;
        }
    }
}

// RotaryEmbedding (interleaved=0) expands to the rotate-half subgraph over the cos/sin caches and
// matches the reference rotation.
TEST(OrtContrib, RotaryEmbeddingExpands) {
    const int64_t      B = 1, S = 2, H = 2, hd = 4, half = hd / 2, E = H * hd, maxPos = 4;
    std::vector<float> cosC((size_t) (maxPos * half)), sinC((size_t) (maxPos * half));
    for (size_t i = 0; i < cosC.size(); ++i)
    {
        cosC[i] = std::cos(0.1f * (float) i);
        sinC[i] = std::sin(0.1f * (float) i);
    }
    Graph    g;
    TensorId x = addInput(g, "x", {B, S, E});
    // position_ids as a constant input (int64 [1,S]): folds, and the expansion's Gather reads it.
    TensorId pos = addInitI64(g, "pos", {1, S}, {2, 3});
    TensorId cc  = addInitF32(g, "cos_cache", {maxPos, half}, cosC);
    TensorId sc  = addInitF32(g, "sin_cache", {maxPos, half}, sinC);
    TensorId y   = addOutput(g, "y");
    Node     nd;
    nd.type    = OpType::RotaryEmbedding;
    nd.name    = "rope";
    nd.inputs  = {x, pos, cc, sc};
    nd.outputs = {y};
    setI(nd, "interleaved", 0);
    g.nodes.push_back(nd);

    std::vector<float> xv((size_t) (B * S * E));
    for (size_t i = 0; i < xv.size(); ++i)
    {
        xv[i] = std::sin(0.37f * (float) i) + 0.2f;
    }
    auto got = runCpu(std::move(g), {{"x", {{B, S, E}, xv}}}, "y");
    ASSERT_EQ((int64_t) got.size(), B * S * E);
    const int64_t posv[2] = {2, 3};
    for (int64_t s = 0; s < S; ++s)
    {
        for (int64_t h = 0; h < H; ++h)
        {
            for (int64_t c = 0; c < half; ++c)
            {
                const double x1 = xv[(size_t) (s * E + h * hd + c)];
                const double x2 = xv[(size_t) (s * E + h * hd + half + c)];
                const double cv = cosC[(size_t) (posv[s] * half + c)];
                const double sv = sinC[(size_t) (posv[s] * half + c)];
                EXPECT_NEAR(got[(size_t) (s * E + h * hd + c)], x1 * cv - x2 * sv, 1e-5);
                EXPECT_NEAR(got[(size_t) (s * E + h * hd + half + c)], x1 * sv + x2 * cv, 1e-5);
            }
        }
    }
}

// MultiHeadAttention in the pure q/k/v(+additive bias) form expands to the primitive attention
// subgraph and matches the reference softmax(QK^T*scale + bias)V.
TEST(OrtContrib, MultiHeadAttentionExpands) {
    const int64_t B = 1, S = 2, T = 3, H = 2, hd = 2, E = H * hd;
    Graph         g;
    TensorId      q = addInput(g, "q", {B, S, E});
    TensorId      k = addInput(g, "k", {B, T, E});
    TensorId      v = addInput(g, "v", {B, T, E});
    TensorId      m = addInput(g, "m", {B, 1, S, T});
    TensorId      y = addOutput(g, "y");
    Node          nd;
    nd.type    = OpType::MultiHeadAttention;
    nd.name    = "mha";
    nd.inputs  = {q, k, v, kNoTensor, kNoTensor, m};
    nd.outputs = {y};
    setI(nd, "num_heads", H);
    setF(nd, "scale", 0.5f);
    g.nodes.push_back(nd);

    auto fill = [&](int64_t n, float f0, float f1) {
        std::vector<float> r((size_t) n);
        for (int64_t i = 0; i < n; ++i)
        {
            r[(size_t) i] = std::sin(f0 * (float) i) * f1;
        }
        return r;
    };
    auto qv = fill(B * S * E, 0.31f, 0.9f), kv = fill(B * T * E, 0.57f, 0.8f), vv = fill(B * T * E, 0.83f, 1.1f);
    std::vector<float> mv((size_t) (S * T));
    for (size_t i = 0; i < mv.size(); ++i)
    {
        mv[i] = i % 3 == 2 ? -10.0f : 0.0f;
    }
    auto got = runCpu(std::move(g),
                      {{"q", {{B, S, E}, qv}}, {"k", {{B, T, E}, kv}}, {"v", {{B, T, E}, vv}}, {"m", {{B, 1, S, T}, mv}}},
                      "y");
    ASSERT_EQ((int64_t) got.size(), B * S * E);
    for (int64_t h = 0; h < H; ++h)
    {
        for (int64_t s = 0; s < S; ++s)
        {
            double logits[3], mx = -1e30;
            for (int64_t t = 0; t < T; ++t)
            {
                double d = 0;
                for (int64_t c = 0; c < hd; ++c)
                {
                    d += (double) qv[(size_t) (s * E + h * hd + c)] * kv[(size_t) (t * E + h * hd + c)];
                }
                logits[t] = d * 0.5 + mv[(size_t) (s * T + t)];
                mx        = std::max(mx, logits[t]);
            }
            double Z = 0;
            for (int64_t t = 0; t < T; ++t)
            {
                Z += std::exp(logits[t] - mx);
            }
            for (int64_t c = 0; c < hd; ++c)
            {
                double acc = 0;
                for (int64_t t = 0; t < T; ++t)
                {
                    acc += std::exp(logits[t] - mx) / Z * vv[(size_t) (t * E + h * hd + c)];
                }
                EXPECT_NEAR(got[(size_t) (s * E + h * hd + c)], acc, 1e-4) << "h=" << h << " s=" << s << " c=" << c;
            }
        }
    }
}
