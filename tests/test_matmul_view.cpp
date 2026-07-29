// foldMatMulViews (src/import/fold_matmul_views.cpp + core/matmul_view.h): a GQA-style
// Reshape/Expand/Reshape/Transpose chain feeding a non-tiled MatMul folds into operand-view attrs
// with the chain source rewired in, the composed strides address the source exactly (bit-identical
// outputs vs the materialized chain, same ascending-k order), tiled-class and inexpressible chains
// stay materialized, and the fold is idempotent. GPU parity is verified on device against this
// same CPU path as the oracle.
#include "core/matmul_view.h"
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

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

    void addNode(Graph &g, OpType type, const std::string &name, std::vector<TensorId> ins, TensorId out) {
        Node n;
        n.type    = type;
        n.name    = name;
        n.inputs  = std::move(ins);
        n.outputs = {out};
        g.nodes.push_back(n);
    }

    // The decode QK idiom at toy size: kv=2 heads repeat to H=6 over S=5 tokens of hd=4, then the
    // K operand transposes into [B,H,hd,S] so the matmul is plain row-major.
    //   q [1,6,1,4], k [1,2,5,4]
    //   k -> Reshape [1,2,1,5,4] -> Expand [1,2,3,5,4] -> Reshape [1,6,5,4] -> Transpose{0,1,3,2}
    //     -> [1,6,4,5];  MatMul(q, .) -> [1,6,1,5]
    // `finalPerm` empty keeps the [1,6,5,4] orientation instead (the PV idiom, with q as [1,6,1,5]).
    Graph gqaGraph(bool qkOrientation) {
        Graph    g;
        TensorId q  = addInput(g, "q", qkOrientation ? Shape {1, 6, 1, 4} : Shape {1, 6, 1, 5});
        TensorId k  = addInput(g, "k", {1, 2, 5, 4});
        TensorId r1 = addTemp(g, "r1");
        TensorId ex = addTemp(g, "ex");
        TensorId r2 = addTemp(g, "r2");
        addNode(g, OpType::Reshape, "reshape1", {k, addI64Init(g, "s1", {1, 2, 1, 5, 4})}, r1);
        addNode(g, OpType::Expand, "expand", {r1, addI64Init(g, "s2", {1, 2, 3, 5, 4})}, ex);
        addNode(g, OpType::Reshape, "reshape2", {ex, addI64Init(g, "s3", {1, 6, 5, 4})}, r2);
        TensorId b = r2;
        if (qkOrientation)
        {
            b = addTemp(g, "kt");
            Node tr;
            tr.type    = OpType::Transpose;
            tr.name    = "kT";
            tr.inputs  = {r2};
            tr.outputs = {b};
            Attr perm;
            perm.kind           = Attr::Ints;
            perm.ints           = {0, 1, 3, 2};
            tr.attr.map["perm"] = perm;
            g.nodes.push_back(tr);
        }
        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        addNode(g, OpType::MatMul, "mm", {q, b}, y);
        g.outputs = {y};
        return g;
    }

    std::vector<float> ramp(int64_t n, float scale) {
        std::vector<float> v((size_t) n);
        for (int64_t i = 0; i < n; ++i)
        {
            v[(size_t) i] = scale * std::sin(0.31f * (float) i) + 0.01f * (float) (i % 17);
        }
        return v;
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

    std::vector<float> runGraph(Graph g, bool viewFold, const std::vector<IOTensor> &ins) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        if (!viewFold)
        {
            cfg.setHint(Hint::MatMulViewFold, (int) Mode::Off);
        }
        auto sess = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run(ins, outs), Status::Ok);
        for (const IOTensor &o: outs)
        {
            if (o.name == "y")
            {
                return std::vector<float>(o.f32(), o.f32() + numElements(o.shape));
            }
        }
        ADD_FAILURE() << "no output y";
        return {};
    }

    const Node *findNode(const Graph &g, const std::string &name) {
        for (const Node &n: g.nodes)
        {
            if (n.name == name)
            {
                return &n;
            }
        }
        return nullptr;
    }

    // Reference dot with the CPU oracle's exact FP semantics: strict IEEE mul+add over ascending k
    // with contraction pinned off (backend/cpu/ops/matmul.cpp's matmulRow pins the same), so
    // equality against the engine is exact, not approximate.
    VKNN_NOINLINE float refDot(const float *a, int64_t aStep, const float *b, int64_t bStep, int64_t K) {
#pragma clang fp contract(off)
        float acc = 0.f;
        for (int64_t k = 0; k < K; ++k)
        {
            acc += a[k * aStep] * b[k * bStep];
        }
        return acc;
    }

    // Ground-truth GQA attention output for gqaGraph(qk) straight off the raw inputs:
    // qk: y[h,n] = sum_k q[h,k] * ksrc[h/3, n, k];  pv: y[h,n] = sum_s p[h,s] * ksrc[h/3, s, n].
    std::vector<float> gqaReference(bool qk, const std::vector<float> &q, const std::vector<float> &ks) {
        const int64_t      N = qk ? 5 : 4, K = qk ? 4 : 5;
        std::vector<float> y((size_t) (6 * N));
        for (int64_t h = 0; h < 6; ++h)
        {
            for (int64_t n = 0; n < N; ++n)
            {
                const float *src        = ks.data() + (h / 3) * 20;
                y[(size_t) (h * N + n)] = qk ? refDot(q.data() + h * 4, 1, src + n * 4, 1, K) : refDot(q.data() + h * 5, 1, src + n, 4, K);
            }
        }
        return y;
    }

} // namespace

// The QK-orientation chain folds: the MatMul's B rewires to the chain source, the head axis splits
// into (kv, group) with a zero group stride, and the whole chain leaves the graph.
TEST(MatMulView, FoldsGqaQkChain) {
    Graph g = gqaGraph(true);
    runStandardPasses(g, PassOptions {});
    foldMatMulViews(g);

    const Node *mm = findNode(g, "mm");
    ASSERT_NE(mm, nullptr);
    ASSERT_TRUE(mm->attr.has(kMmView));
    // B addresses the k source [1,2,5,4] (dense strides kv=20, s=4, hd=1) as [1,(2,3),4,5]:
    // group stride 0, k walk contiguous over hd, n over tokens.
    EXPECT_EQ(mm->attr.getints(kMmViewDims), (std::vector<int64_t> {1, 2, 3, 1, 5}));
    EXPECT_EQ(mm->attr.getints(kMmViewAStride), (std::vector<int64_t> {0, 12, 4, 0, 0}));
    EXPECT_EQ(mm->attr.getints(kMmViewBStride), (std::vector<int64_t> {0, 20, 0, 0, 4}));
    EXPECT_EQ(mm->attr.geti(kMmViewAK), 1);
    EXPECT_EQ(mm->attr.geti(kMmViewBK), 1);
    EXPECT_EQ(mm->attr.geti(kMmViewM), 1);
    EXPECT_EQ(mm->attr.geti(kMmViewN), 5);
    EXPECT_EQ(mm->attr.geti(kMmViewK), 4);
    EXPECT_EQ(g.desc(mm->inputs[1]).name, "k");
    EXPECT_EQ(findNode(g, "expand"), nullptr);
    EXPECT_EQ(findNode(g, "kT"), nullptr);

    // Idempotent: a second run leaves the folded node alone.
    Graph copy = g;
    foldMatMulViews(g);
    EXPECT_EQ(g.nodes.size(), copy.nodes.size());
}

// Folded and materialized runs produce byte-identical fp32 outputs in both attention orientations
// (same values, same ascending-k accumulation).
TEST(MatMulView, FoldedMatchesMaterializedBitExact) {
    for (bool qk: {true, false})
    {
        std::vector<IOTensor> ins = {
            ioTensor("q", qk ? Shape {1, 6, 1, 4} : Shape {1, 6, 1, 5}, ramp(qk ? 24 : 30, 1.f)),
            ioTensor("k", {1, 2, 5, 4}, ramp(40, 0.7f)),
        };
        std::vector<float> folded = runGraph(gqaGraph(qk), true, ins);
        std::vector<float> mat    = runGraph(gqaGraph(qk), false, ins);
        std::vector<float> ref    = gqaReference(qk, ramp(qk ? 24 : 30, 1.f), ramp(40, 0.7f));
        ASSERT_EQ(folded.size(), mat.size());
        ASSERT_EQ(folded.size(), ref.size());
        EXPECT_EQ(std::memcmp(folded.data(), mat.data(), folded.size() * 4), 0) << (qk ? "qk" : "pv");
        EXPECT_EQ(std::memcmp(folded.data(), ref.data(), folded.size() * 4), 0) << (qk ? "qk" : "pv") << " vs reference";
    }
}

// A tiled-class shape keeps its materialized chain (the tiled kernels need dense panels), and a
// reshape whose boundary splits a source block at a non-divisible point is not view-expressible.
TEST(MatMulView, TiledClassAndInexpressibleChainsKeepMaterialized) {
    // Tiled class: [1,48,48] x (Transpose-fed [1,48,48]) with M,N,K all >= 32.
    {
        Graph    g;
        TensorId a  = addInput(g, "q", {1, 48, 48});
        TensorId b0 = addInput(g, "k", {1, 48, 48});
        TensorId bt = addTemp(g, "kt");
        Node     tr;
        tr.type    = OpType::Transpose;
        tr.name    = "kT";
        tr.inputs  = {b0};
        tr.outputs = {bt};
        Attr perm;
        perm.kind           = Attr::Ints;
        perm.ints           = {0, 2, 1};
        tr.attr.map["perm"] = perm;
        g.nodes.push_back(tr);
        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        addNode(g, OpType::MatMul, "mm", {a, bt}, y);
        g.outputs = {y};
        runStandardPasses(g, PassOptions {});
        foldMatMulViews(g);
        const Node *mm = findNode(g, "mm");
        ASSERT_NE(mm, nullptr);
        EXPECT_FALSE(mm->attr.has(kMmView));
        EXPECT_NE(findNode(g, "kT"), nullptr);
    }
    // Inexpressible: [1,6,10] -> Reshape [1,4,15] crosses the (6,10) block boundary at a
    // non-divisible point; the chain stays and the run still computes through it.
    {
        Graph    g;
        TensorId a  = addInput(g, "q", {1, 1, 15});
        TensorId b0 = addInput(g, "k", {1, 6, 10});
        TensorId r  = addTemp(g, "r");
        TensorId bt = addTemp(g, "kt");
        addNode(g, OpType::Reshape, "reshape", {b0, addI64Init(g, "s", {1, 4, 15})}, r);
        Node tr;
        tr.type    = OpType::Transpose;
        tr.name    = "kT";
        tr.inputs  = {r};
        tr.outputs = {bt};
        Attr perm;
        perm.kind           = Attr::Ints;
        perm.ints           = {0, 2, 1};
        tr.attr.map["perm"] = perm;
        g.nodes.push_back(tr);
        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        addNode(g, OpType::MatMul, "mm", {a, bt}, y);
        g.outputs = {y};
        runStandardPasses(g, PassOptions {});
        Graph unfolded = g;
        foldMatMulViews(g);
        const Node *mm = findNode(g, "mm");
        ASSERT_NE(mm, nullptr);
        EXPECT_FALSE(mm->attr.has(kMmView));
        EXPECT_EQ(g.nodes.size(), unfolded.nodes.size());
    }
}

// A chain node carrying fused work (a pointwise epilogue or activation) is not pure data movement
// — folding it away would drop that computation. The walk stops there and the MatMul keeps its
// materialized operand.
TEST(MatMulView, ChainNodeWithFusedWorkStopsTheFold) {
    Graph g = gqaGraph(true);
    runStandardPasses(g, PassOptions {});
    for (Node &n: g.nodes)
    {
        if (n.name == "kT")
        {
            Attr marker;
            marker.kind            = Attr::Int;
            marker.i               = 1;
            n.attr.map["pw_steps"] = marker;
        }
    }
    foldMatMulViews(g);
    const Node *mm = findNode(g, "mm");
    ASSERT_NE(mm, nullptr);
    EXPECT_FALSE(mm->attr.has(kMmView));
    EXPECT_NE(findNode(g, "kT"), nullptr);
    EXPECT_NE(findNode(g, "expand"), nullptr);
}

// A rank-2 transpose-fed B broadcasts over every batch axis and still folds bit-exactly.
TEST(MatMulView, Rank2BroadcastBOperandFolds) {
    auto build = [] {
        Graph    g;
        TensorId a  = addInput(g, "q", {1, 6, 1, 4});
        TensorId b0 = addInput(g, "k", {5, 4});
        TensorId bt = addTemp(g, "kt");
        Node     tr;
        tr.type    = OpType::Transpose;
        tr.name    = "kT";
        tr.inputs  = {b0};
        tr.outputs = {bt};
        Attr perm;
        perm.kind           = Attr::Ints;
        perm.ints           = {1, 0};
        tr.attr.map["perm"] = perm;
        g.nodes.push_back(tr);
        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        addNode(g, OpType::MatMul, "mm", {a, bt}, y);
        g.outputs = {y};
        return g;
    };
    {
        Graph g = build();
        runStandardPasses(g, PassOptions {});
        foldMatMulViews(g);
        const Node *mm = findNode(g, "mm");
        ASSERT_NE(mm, nullptr);
        ASSERT_TRUE(mm->attr.has(kMmView));
        EXPECT_EQ(mm->attr.getints(kMmViewBStride), (std::vector<int64_t> {0, 0, 0, 4}));
        EXPECT_EQ(mm->attr.geti(kMmViewBK), 1);
    }
    std::vector<IOTensor> ins = {
        ioTensor("q", {1, 6, 1, 4}, ramp(24, 1.f)),
        ioTensor("k", {5, 4}, ramp(20, 0.9f)),
    };
    std::vector<float> folded = runGraph(build(), true, ins);
    std::vector<float> mat    = runGraph(build(), false, ins);
    ASSERT_EQ(folded.size(), mat.size());
    EXPECT_EQ(std::memcmp(folded.data(), mat.data(), folded.size() * 4), 0);
}

// A graph folded ahead of session load (the structural test pins these exact attrs) executes
// bit-identically to the ground-truth reference even with the load-time fold disabled — the attrs
// alone carry the whole addressing, and a later inferShapes leaves the folded node's final shapes
// untouched.
TEST(MatMulView, PreFoldedGraphMatchesReference) {
    Graph g = gqaGraph(true);
    runStandardPasses(g, PassOptions {});
    foldMatMulViews(g);
    std::vector<IOTensor> ins = {
        ioTensor("q", {1, 6, 1, 4}, ramp(24, 1.f)),
        ioTensor("k", {1, 2, 5, 4}, ramp(40, 0.7f)),
    };
    std::vector<float> got = runGraph(std::move(g), false, ins);
    std::vector<float> ref = gqaReference(true, ramp(24, 1.f), ramp(40, 0.7f));
    ASSERT_EQ(got.size(), ref.size());
    EXPECT_EQ(std::memcmp(got.data(), ref.data(), got.size() * 4), 0);
}
