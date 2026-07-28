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
        d.name      = name;
        d.shape     = std::move(s);
        d.isInput   = true;
        TensorId id = g.addTensor(d);
        g.inputs.push_back(id);
        return id;
    }
    TensorId addOutput(Graph &g, const char *name) {
        TensorDesc d;
        d.name      = name;
        d.isOutput  = true;
        TensorId id = g.addTensor(d);
        g.outputs.push_back(id);
        return id;
    }
    TensorId addInitF32(Graph &g, const char *name, Shape s, const std::vector<float> &v) {
        TensorDesc d;
        d.name          = name;
        d.shape         = std::move(s);
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) v.size(), DType::Float32);
        std::memcpy(hb.f32(), v.data(), v.size() * 4);
        g.initializers[id] = hb;
        return id;
    }
    // A native uint8 initializer (one byte per element), exactly as the importer stores a MatMulNBits
    // packed int4 payload (materializeInitializers -> fillHostBytes). `bytes` carries the byte values as
    // integer-valued floats; initFloats decodes the lanes back to fp32 for the repack.
    TensorId addInitU8(Graph &g, const char *name, Shape s, const std::vector<float> &bytes) {
        TensorDesc d;
        d.name           = name;
        d.shape          = std::move(s);
        d.isInitializer  = true;
        TensorId id      = g.addTensor(d);
        g.desc(id).dtype = DType::UInt8;
        HostBuffer hb;
        hb.resizeElems((int64_t) bytes.size(), DType::UInt8);
        for (size_t i = 0; i < bytes.size(); ++i)
        {
            hb.bytes.data()[i] = (uint8_t) bytes[i];
        }
        g.initializers[id] = hb;
        return id;
    }
    TensorId addInitI64(Graph &g, const char *name, Shape s, const std::vector<int64_t> &v) {
        TensorDesc d;
        d.name          = name;
        d.shape         = std::move(s);
        d.dtype         = DType::Int64;
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
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

    TensorId addInputI64(Graph &g, const char *name, Shape s) {
        TensorDesc d;
        d.name      = name;
        d.shape     = std::move(s);
        d.dtype     = DType::Int64;
        d.isInput   = true;
        TensorId id = g.addTensor(d);
        g.inputs.push_back(id);
        return id;
    }
    TensorId addInitI32(Graph &g, const char *name, Shape s, const std::vector<int32_t> &v) {
        TensorDesc d;
        d.name          = name;
        d.shape         = std::move(s);
        d.dtype         = DType::Int32;
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) v.size(), DType::Int32);
        std::memcpy(hb.bytes.data(), v.data(), v.size() * 4);
        g.initializers[id] = hb;
        return id;
    }
    IOTensor mkFeedF32(const char *name, Shape s, const std::vector<float> &v) {
        IOTensor t;
        t.name  = name;
        t.shape = std::move(s);
        t.dtype = DType::Float32;
        t.data.resize(v.size() * 4);
        std::memcpy(t.data.data(), v.data(), t.data.size());
        return t;
    }
    IOTensor mkFeedI64(const char *name, Shape s, const std::vector<int64_t> &v) {
        IOTensor t;
        t.name  = name;
        t.shape = std::move(s);
        t.dtype = DType::Int64;
        t.data.resize(v.size() * 8);
        std::memcpy(t.data.data(), v.data(), t.data.size());
        return t;
    }
    // Run on the CPU backend with caller-built typed feeds; returns every output.
    std::vector<IOTensor> runCpuAll(Graph g, const std::vector<IOTensor> &ins, bool kvConcatFold = true) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        // KvConcatFold rewrites a decode attention's present output to the rows-only convention; a
        // test asserting the canonical Concat(past, new) present pins it off to read the unfolded form.
        if (!kvConcatFold)
        {
            cfg.setHint(Hint::KvConcatFold, (int) Mode::Off);
        }
        auto sess = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        std::vector<IOTensor> outs;
        if (sess)
        {
            EXPECT_EQ(sess->run(ins, outs), Status::Ok);
        }
        return outs;
    }
    const IOTensor *findOutput(const std::vector<IOTensor> &outs, const std::string &name) {
        for (const IOTensor &o: outs)
        {
            if (o.name == name)
            {
                return &o;
            }
        }
        ADD_FAILURE() << "no output " << name;
        return nullptr;
    }

    std::vector<float> runCpu(Graph g, const std::map<std::string, std::pair<Shape, std::vector<float>>> &feeds, const std::string &outName) {
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
    TensorId shp = g.addTensor([] {
        TensorDesc d;
        d.name  = "shp";
        d.dtype = DType::Int64;
        return d;
    }());
    TensorId gth = g.addTensor([] {
        TensorDesc d;
        d.name  = "gth";
        d.dtype = DType::Int64;
        return d;
    }());
    TensorId u1  = g.addTensor([] {
        TensorDesc d;
        d.name  = "u1";
        d.dtype = DType::Int64;
        return d;
    }());
    TensorId cat = g.addTensor([] {
        TensorDesc d;
        d.name  = "cat";
        d.dtype = DType::Int64;
        return d;
    }());
    Node     sh;
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
    TensorId y = addOutput(g, "y");
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
    TensorId bq = addInitU8(g, "bq", {N, kb, BS / 2}, packedAsF32); // native uint8, as the importer stores it
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

namespace {

    // In-test GroupQueryAttention reference (the ORT contract, probe-validated: past_len =
    // seqlens_k + 1 - S valid past rows, new token i rotates at position past_len + i and attends
    // past rows [0, past_len) plus new tokens [0, i]) under the engine's Concat(past, new) present
    // convention (new rows AFTER the full P-row past block).
    struct GqaRef {
        int64_t            B = 1, S, P, Hq, Hkv, hd;
        float              scale;
        std::vector<float> q, k, v, pastK, pastV; // q [S,Hq*hd]; k/v [S,Hkv*hd]; past [Hkv,P,hd]
        std::vector<float> cosC, sinC;            // [maxPos, hd/2]
        int64_t            seqlens;

        std::vector<float> rot(const float *x, int64_t pos) const {
            const int64_t      half = hd / 2;
            std::vector<float> y((size_t) hd);
            for (int64_t c = 0; c < half; ++c)
            {
                const float cv         = cosC[(size_t) (pos * half + c)];
                const float sv         = sinC[(size_t) (pos * half + c)];
                y[(size_t) c]          = x[c] * cv - x[half + c] * sv;
                y[(size_t) (half + c)] = x[c] * sv + x[half + c] * cv;
            }
            return y;
        }
        // Key/value row j (concat order: past rows then new rows) for kv head kh; new keys rotated.
        std::vector<float> keyRow(int64_t j, int64_t kh) const {
            if (j < P)
            {
                return std::vector<float>(&pastK[(size_t) ((kh * P + j) * hd)], &pastK[(size_t) ((kh * P + j) * hd)] + hd);
            }
            const int64_t i = j - P;
            return rot(&k[(size_t) (i * Hkv * hd + kh * hd)], seqlens + 1 - S + i);
        }
        std::vector<float> valueRow(int64_t j, int64_t kh) const {
            if (j < P)
            {
                return std::vector<float>(&pastV[(size_t) ((kh * P + j) * hd)], &pastV[(size_t) ((kh * P + j) * hd)] + hd);
            }
            const int64_t i = j - P;
            return std::vector<float>(&v[(size_t) (i * Hkv * hd + kh * hd)], &v[(size_t) (i * Hkv * hd + kh * hd)] + hd);
        }
        // Attention output [S, Hq*hd].
        std::vector<float> attention() const {
            const int64_t      T = P + S, grp = Hq / Hkv, pastLen = seqlens + 1 - S;
            std::vector<float> out((size_t) (S * Hq * hd), 0.0f);
            for (int64_t i = 0; i < S; ++i)
            {
                for (int64_t h = 0; h < Hq; ++h)
                {
                    const int64_t            kh = h / grp;
                    const std::vector<float> qr = rot(&q[(size_t) (i * Hq * hd + h * hd)], pastLen + i);
                    std::vector<double>      logits((size_t) T, -1e30);
                    double                   mx = -1e30;
                    for (int64_t j = 0; j < T; ++j)
                    {
                        const bool valid = j < P ? (j < pastLen) : (j - P <= i);
                        if (!valid)
                        {
                            continue;
                        }
                        const std::vector<float> kr = keyRow(j, kh);
                        double                   d  = 0;
                        for (int64_t c = 0; c < hd; ++c)
                        {
                            d += (double) qr[(size_t) c] * kr[(size_t) c];
                        }
                        logits[(size_t) j] = d * scale;
                        mx                 = std::max(mx, logits[(size_t) j]);
                    }
                    double Z = 0;
                    for (int64_t j = 0; j < T; ++j)
                    {
                        if (logits[(size_t) j] > -1e29)
                        {
                            Z += std::exp(logits[(size_t) j] - mx);
                        }
                    }
                    for (int64_t j = 0; j < T; ++j)
                    {
                        if (logits[(size_t) j] <= -1e29)
                        {
                            continue;
                        }
                        const double             p  = std::exp(logits[(size_t) j] - mx) / Z;
                        const std::vector<float> vr = valueRow(j, kh);
                        for (int64_t c = 0; c < hd; ++c)
                        {
                            out[(size_t) (i * Hq * hd + h * hd + c)] += (float) (p * vr[(size_t) c]);
                        }
                    }
                }
            }
            return out;
        }
    };

    // Build the export-shaped GQA graph: attention_mask int64 input feeding the seqlens_k
    // reformat chain (ReduceSum - 1 -> int32), a constant total_sequence_length, fp32 q/k/v and
    // past inputs, cos/sin cache initializers, and the contrib node with do_rotary=1.
    Graph buildGqaGraph(const GqaRef &r, int64_t maxPos) {
        const int64_t T = r.P + r.S;
        Graph         g;
        TensorId      q   = addInput(g, "q", {r.B, r.S, r.Hq * r.hd});
        TensorId      k   = addInput(g, "k", {r.B, r.S, r.Hkv * r.hd});
        TensorId      v   = addInput(g, "v", {r.B, r.S, r.Hkv * r.hd});
        TensorId      pk  = addInput(g, "past_key", {r.B, r.Hkv, r.P, r.hd});
        TensorId      pv  = addInput(g, "past_value", {r.B, r.Hkv, r.P, r.hd});
        TensorId      am  = addInputI64(g, "attention_mask", {r.B, T});
        TensorId      one = addInitI64(g, "one", {1}, {1});
        TensorId      rs  = g.addTensor([] {
            TensorDesc d;
            d.name  = "mask_sum";
            d.dtype = DType::Int64;
            return d;
        }());
        Node          red;
        red.type    = OpType::Reduce;
        red.subOp   = (int32_t) ReduceType::Sum;
        red.name    = "mask_reduce";
        red.inputs  = {am, one};
        red.outputs = {rs};
        setI(red, "keepdims", 1);
        g.nodes.push_back(red);
        TensorId sub = g.addTensor([] {
            TensorDesc d;
            d.name  = "mask_sub";
            d.dtype = DType::Int64;
            return d;
        }());
        Node     sb;
        sb.type    = OpType::Binary;
        sb.subOp   = (int32_t) BinaryType::Sub;
        sb.name    = "mask_sub_node";
        sb.inputs  = {rs, one};
        sb.outputs = {sub};
        g.nodes.push_back(sb);
        TensorId sl = g.addTensor([] {
            TensorDesc d;
            d.name  = "seqlens_k";
            d.dtype = DType::Int32;
            return d;
        }());
        Node     cs;
        cs.type    = OpType::Cast;
        cs.name    = "mask_cast";
        cs.inputs  = {sub};
        cs.outputs = {sl};
        setI(cs, "to", 6);
        g.nodes.push_back(cs);
        TensorId total = addInitI32(g, "total_seq_len", {}, {(int32_t) T});
        TensorId cosT  = addInitF32(g, "cos_cache", {maxPos, r.hd / 2}, r.cosC);
        TensorId sinT  = addInitF32(g, "sin_cache", {maxPos, r.hd / 2}, r.sinC);
        TensorId y     = addOutput(g, "y");
        TensorId prk   = addOutput(g, "present_key");
        TensorId prv   = addOutput(g, "present_value");
        Node     nd;
        nd.type    = OpType::GroupQueryAttention;
        nd.name    = "gqa";
        nd.inputs  = {q, k, v, pk, pv, sl, total, cosT, sinT};
        nd.outputs = {y, prk, prv};
        setI(nd, "num_heads", r.Hq);
        setI(nd, "kv_num_heads", r.Hkv);
        setI(nd, "do_rotary", 1);
        setI(nd, "rotary_interleaved", 0);
        setI(nd, "local_window_size", -1);
        setF(nd, "softcap", 0.0f);
        setF(nd, "scale", r.scale);
        g.nodes.push_back(nd);
        return g;
    }

    std::vector<float> sinFill(int64_t n, float f, float a) {
        std::vector<float> r((size_t) n);
        for (int64_t i = 0; i < n; ++i)
        {
            r[(size_t) i] = std::sin(f * (float) i) * a;
        }
        return r;
    }

} // namespace

// GroupQueryAttention (decode: S=1 over a partially valid past) expands to the primitive
// rope/concat/attention subgraph: the seqlens_k-derived rotary position and past-row mask match
// the ORT contract, and the present outputs carry the Concat(past, new) convention.
TEST(OrtContrib, GroupQueryAttentionDecodeExpands) {
    GqaRef r;
    r.S = 1, r.P = 4, r.Hq = 4, r.Hkv = 2, r.hd = 4;
    r.scale              = 0.5f;
    const int64_t maxPos = 8, T = r.P + r.S, half = r.hd / 2;
    r.q     = sinFill(r.S * r.Hq * r.hd, 0.31f, 0.9f);
    r.k     = sinFill(r.S * r.Hkv * r.hd, 0.57f, 0.8f);
    r.v     = sinFill(r.S * r.Hkv * r.hd, 0.83f, 1.1f);
    r.pastK = sinFill(r.Hkv * r.P * r.hd, 0.41f, 0.7f);
    r.pastV = sinFill(r.Hkv * r.P * r.hd, 0.67f, 1.2f);
    r.cosC.resize((size_t) (maxPos * half));
    r.sinC.resize((size_t) (maxPos * half));
    for (size_t i = 0; i < r.cosC.size(); ++i)
    {
        r.cosC[i] = std::cos(0.13f * (float) i);
        r.sinC[i] = std::sin(0.13f * (float) i);
    }
    // 2 valid past rows + the new token: seqlens_k = 2, so the token rotates at position 2 and
    // past rows 2..3 are masked out.
    r.seqlens = 2;
    std::vector<int64_t> mask((size_t) T, 0);
    mask[0] = mask[1]      = 1;
    mask[(size_t) (T - 1)] = 1;

    Graph g = buildGqaGraph(r, maxPos);

    Graph structural = g;
    runStandardPasses(structural);
    for (const Node &n: structural.nodes)
    {
        EXPECT_NE(n.type, OpType::GroupQueryAttention);
    }

    auto outs = runCpuAll(std::move(g), {mkFeedF32("q", {r.B, r.S, r.Hq * r.hd}, r.q), mkFeedF32("k", {r.B, r.S, r.Hkv * r.hd}, r.k), mkFeedF32("v", {r.B, r.S, r.Hkv * r.hd}, r.v), mkFeedF32("past_key", {r.B, r.Hkv, r.P, r.hd}, r.pastK), mkFeedF32("past_value", {r.B, r.Hkv, r.P, r.hd}, r.pastV), mkFeedI64("attention_mask", {r.B, T}, mask)},
                          /*kvConcatFold=*/false);
    const IOTensor *y = findOutput(outs, "y");
    ASSERT_TRUE(y);
    const std::vector<float> want = r.attention();
    ASSERT_EQ(numElements(y->shape), (int64_t) want.size());
    for (size_t i = 0; i < want.size(); ++i)
    {
        EXPECT_NEAR(y->f32()[i], want[i], 1e-4) << "i=" << i;
    }
    // present = Concat(past, new): the past block passes through untouched, the last row is the
    // position-2 rotated key (value row unrotated).
    const IOTensor *prk = findOutput(outs, "present_key");
    const IOTensor *prv = findOutput(outs, "present_value");
    ASSERT_TRUE(prk && prv);
    ASSERT_EQ(prk->shape, (Shape {r.B, r.Hkv, T, r.hd}));
    for (int64_t kh = 0; kh < r.Hkv; ++kh)
    {
        for (int64_t j = 0; j < T; ++j)
        {
            const std::vector<float> wk = r.keyRow(j, kh), wv = r.valueRow(j, kh);
            for (int64_t c = 0; c < r.hd; ++c)
            {
                EXPECT_NEAR(prk->f32()[(kh * T + j) * r.hd + c], wk[(size_t) c], 1e-5) << "key kh=" << kh << " j=" << j;
                EXPECT_NEAR(prv->f32()[(kh * T + j) * r.hd + c], wv[(size_t) c], 1e-5) << "value kh=" << kh << " j=" << j;
            }
        }
    }
}

// GroupQueryAttention (prefill: S=3 with one valid past row) applies the causal triangle to the
// new-token block and the seqlens_k mask to the past block, with per-token rotary positions
// past_len + i.
TEST(OrtContrib, GroupQueryAttentionPrefillExpands) {
    GqaRef r;
    r.S = 3, r.P = 2, r.Hq = 2, r.Hkv = 1, r.hd = 4;
    r.scale              = 0.5f;
    const int64_t maxPos = 8, T = r.P + r.S, half = r.hd / 2;
    r.q     = sinFill(r.S * r.Hq * r.hd, 0.29f, 1.0f);
    r.k     = sinFill(r.S * r.Hkv * r.hd, 0.53f, 0.9f);
    r.v     = sinFill(r.S * r.Hkv * r.hd, 0.79f, 1.1f);
    r.pastK = sinFill(r.Hkv * r.P * r.hd, 0.37f, 0.8f);
    r.pastV = sinFill(r.Hkv * r.P * r.hd, 0.61f, 1.3f);
    r.cosC.resize((size_t) (maxPos * half));
    r.sinC.resize((size_t) (maxPos * half));
    for (size_t i = 0; i < r.cosC.size(); ++i)
    {
        r.cosC[i] = std::cos(0.11f * (float) i);
        r.sinC[i] = std::sin(0.11f * (float) i);
    }
    // 1 valid past row + 3 new tokens: seqlens_k = 3, past_len = 1, positions 1..3; past row 1 is
    // masked out for every query.
    r.seqlens = 3;
    std::vector<int64_t> mask((size_t) T, 1);
    mask[1] = 0;

    Graph g = buildGqaGraph(r, maxPos);
    auto outs = runCpuAll(std::move(g), {mkFeedF32("q", {r.B, r.S, r.Hq * r.hd}, r.q), mkFeedF32("k", {r.B, r.S, r.Hkv * r.hd}, r.k), mkFeedF32("v", {r.B, r.S, r.Hkv * r.hd}, r.v), mkFeedF32("past_key", {r.B, r.Hkv, r.P, r.hd}, r.pastK), mkFeedF32("past_value", {r.B, r.Hkv, r.P, r.hd}, r.pastV), mkFeedI64("attention_mask", {r.B, T}, mask)});
    const IOTensor *y = findOutput(outs, "y");
    ASSERT_TRUE(y);
    const std::vector<float> want = r.attention();
    ASSERT_EQ(numElements(y->shape), (int64_t) want.size());
    for (size_t i = 0; i < want.size(); ++i)
    {
        EXPECT_NEAR(y->f32()[i], want[i], 1e-4) << "i=" << i;
    }
}

// A GroupQueryAttention variant outside the expansion (softcap) keeps its node — loud at plan,
// never silently miscomputed.
TEST(OrtContrib, GroupQueryAttentionUnsupportedVariantKept) {
    GqaRef r;
    r.S = 1, r.P = 2, r.Hq = 2, r.Hkv = 1, r.hd = 4;
    r.scale              = 0.5f;
    r.seqlens            = 2;
    const int64_t maxPos = 4, half = r.hd / 2;
    r.q     = sinFill(r.S * r.Hq * r.hd, 0.3f, 1.0f);
    r.k     = sinFill(r.S * r.Hkv * r.hd, 0.5f, 1.0f);
    r.v     = sinFill(r.S * r.Hkv * r.hd, 0.7f, 1.0f);
    r.pastK = sinFill(r.Hkv * r.P * r.hd, 0.4f, 1.0f);
    r.pastV = sinFill(r.Hkv * r.P * r.hd, 0.6f, 1.0f);
    r.cosC.assign((size_t) (maxPos * half), 1.0f);
    r.sinC.assign((size_t) (maxPos * half), 0.0f);
    Graph g = buildGqaGraph(r, maxPos);
    for (Node &n: g.nodes)
    {
        if (n.type == OpType::GroupQueryAttention)
        {
            setF(n, "softcap", 50.0f);
        }
    }
    runStandardPasses(g);
    bool kept = false;
    for (const Node &n: g.nodes)
    {
        kept = kept || n.type == OpType::GroupQueryAttention;
    }
    EXPECT_TRUE(kept);
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
    auto               qv = fill(B * S * E, 0.31f, 0.9f), kv = fill(B * T * E, 0.57f, 0.8f), vv = fill(B * T * E, 0.83f, 1.1f);
    std::vector<float> mv((size_t) (S * T));
    for (size_t i = 0; i < mv.size(); ++i)
    {
        mv[i] = i % 3 == 2 ? -10.0f : 0.0f;
    }
    auto got = runCpu(std::move(g), {{"q", {{B, S, E}, qv}}, {"k", {{B, T, E}, kv}}, {"v", {{B, T, E}, vv}}, {"m", {{B, 1, S, T}, mv}}}, "y");
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
