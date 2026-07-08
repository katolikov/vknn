// Config::cpuThreads is an effort knob, never a numeric one: the CPU backend only partitions loops
// whose iterations write disjoint outputs and carry no cross-iteration accumulation, so every output
// byte must be identical for every thread count. These tests run each threaded op at 1 thread and at
// several thread counts and memcmp the raw output buffers. Shapes are chosen so the partitioned
// extent (output planes, matrix rows, elements) does NOT divide evenly by the thread count, which is
// where an off-by-one chunk boundary or a shared accumulator would show up.
//
// A drift here breaks the verification ladder: the CPU backend is the byte oracle the Vulkan kernels
// are diffed against on-device.
#include "backend/cpu/parallel.h"
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

using namespace vknn;

namespace {

    // Thread counts probed against the 1-thread reference. Deliberately includes counts larger than
    // the partitioned extent of some shapes (parallelFor must then run fewer chunks, or inline).
    const std::vector<int> kThreadCounts {2, 3, 5, 8};

    Attr ints(std::vector<int64_t> v) {
        Attr a;
        a.kind = Attr::Ints;
        a.ints = std::move(v);
        return a;
    }
    Attr integer(int64_t v) {
        Attr a;
        a.kind = Attr::Int;
        a.i    = v;
        return a;
    }

    // Reproducible non-trivial values: a plain LCG, so a chunk boundary lands on differing data
    // rather than on a repeating pattern that would hide a mis-partition.
    std::vector<float> lcg(size_t n, uint32_t seed) {
        std::vector<float> v(n);
        uint32_t           s = seed;
        for (size_t i = 0; i < n; ++i)
        {
            s    = s * 1664525u + 1013904223u;
            v[i] = (float) ((int32_t) (s >> 8) % 2001 - 1000) * 0.001f;
        }
        return v;
    }

    TensorId addInit(Graph &g, const std::string &name, const Shape &shape, const std::vector<float> &data) {
        TensorDesc d;
        d.name          = name;
        d.shape         = shape;
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems(data.size(), DType::Float32);
        std::memcpy(hb.f32(), data.data(), data.size() * sizeof(float));
        g.initializers[id] = hb;
        return id;
    }

    // Run `g` on the CPU backend with `threads` workers and return the output tensor's raw bytes.
    std::vector<uint8_t> runBytes(Graph g, const Shape &xshape, const std::vector<float> &xdata, int threads) {
        Config cfg;
        cfg.backend    = BackendKind::Cpu;
        cfg.cpuThreads = threads;
        auto sess      = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = "x";
        in.shape = xshape;
        in.dtype = DType::Float32;
        in.data.resize(xdata.size() * sizeof(float));
        std::memcpy(in.data.data(), xdata.data(), in.data.size());
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        EXPECT_FALSE(outs.empty());
        return outs.empty() ? std::vector<uint8_t> {} : outs[0].data;
    }

    // The whole point of the suite: the byte image of the output is invariant under cpuThreads.
    void expectBytesInvariant(const std::function<Graph()> &build, const Shape &xshape, const std::vector<float> &xdata) {
        std::vector<uint8_t> ref = runBytes(build(), xshape, xdata, 1);
        ASSERT_FALSE(ref.empty());
        for (int t: kThreadCounts)
        {
            std::vector<uint8_t> got = runBytes(build(), xshape, xdata, t);
            ASSERT_EQ(got.size(), ref.size()) << "threads=" << t;
            EXPECT_EQ(0, std::memcmp(got.data(), ref.data(), ref.size())) << "threads=" << t;
        }
    }

    // Byte-identical output across thread counts, plus proof that the partition actually engaged:
    // parallelFor runs a loop inline when a chunk would not be worth kMinChunkOps, so a test on
    // too-small shapes would pass vacuously. Every shape below is sized past that threshold.
    void expectByteIdenticalAcrossThreads(const std::function<Graph()> &build, const Shape &xshape, const std::vector<float> &xdata) {
        int64_t dispatchesBefore = cpu::detail::poolDispatches();
        expectBytesInvariant(build, xshape, xdata);
        EXPECT_GT(cpu::detail::poolDispatches(), dispatchesBefore) << "shape too small to partition: the byte comparison is vacuous";
    }

    TensorId addInput(Graph &g, const Shape &shape) {
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = shape;
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs.push_back(x);
        return x;
    }
    TensorId addOutput(Graph &g) {
        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        g.outputs   = {y};
        return y;
    }

} // namespace

// Conv partitions x.n*outC output planes: 1*7 planes divides evenly by none of 2/3/5/8.
TEST(CpuThreading, ConvBitExact) {
    Shape              xs {1, 5, 40, 40};
    std::vector<float> xd    = lcg(1 * 5 * 40 * 40, 1);
    auto               build = [&] {
        Graph    g;
        TensorId x = addInput(g, xs);
        TensorId w = addInit(g, "w", {7, 5, 3, 3}, lcg(7 * 5 * 3 * 3, 2));
        TensorId b = addInit(g, "b", {7}, lcg(7, 3));
        TensorId y = addOutput(g);
        Node     n;
        n.type                  = OpType::Conv;
        n.name                  = "conv";
        n.inputs                = {x, w, b};
        n.outputs               = {y};
        n.attr.map["strides"]   = ints({1, 1});
        n.attr.map["pads"]      = ints({1, 1, 1, 1});
        n.attr.map["dilations"] = ints({1, 1});
        g.nodes.push_back(n);
        return g;
    };
    expectByteIdenticalAcrossThreads(build, xs, xd);
}

// Grouped conv exercises the same plane partition with a per-group input-channel window.
TEST(CpuThreading, GroupedConvBitExact) {
    Shape              xs {1, 6, 48, 48};
    std::vector<float> xd    = lcg(1 * 6 * 48 * 48, 4);
    auto               build = [&] {
        Graph    g;
        TensorId x = addInput(g, xs);
        TensorId w = addInit(g, "w", {9, 2, 3, 3}, lcg(9 * 2 * 3 * 3, 5));
        TensorId y = addOutput(g);
        Node     n;
        n.type                = OpType::Conv;
        n.name                = "conv";
        n.inputs              = {x, w};
        n.outputs             = {y};
        n.attr.map["group"]   = integer(3);
        n.attr.map["strides"] = ints({1, 1});
        n.attr.map["pads"]    = ints({1, 1, 1, 1});
        g.nodes.push_back(n);
        return g;
    };
    expectByteIdenticalAcrossThreads(build, xs, xd);
}

// Gemm partitions its M output rows: 11 rows, none of 2/3/5/8 divides it.
TEST(CpuThreading, GemmBitExact) {
    Shape              xs {11, 512};
    std::vector<float> xd    = lcg(11 * 512, 6);
    auto               build = [&] {
        Graph    g;
        TensorId x = addInput(g, xs);
        TensorId w = addInit(g, "w", {64, 512}, lcg(64 * 512, 7)); // transB=1: [N,K]
        TensorId c = addInit(g, "c", {64}, lcg(64, 8));
        TensorId y = addOutput(g);
        Node     n;
        n.type               = OpType::Gemm;
        n.name               = "gemm";
        n.inputs             = {x, w, c};
        n.outputs            = {y};
        n.attr.map["transB"] = integer(1);
        g.nodes.push_back(n);
        return g;
    };
    expectByteIdenticalAcrossThreads(build, xs, xd);
}

// MatMul partitions batchElems*M rows: 3*5 = 15 rows, plus a broadcast batch dim on B.
TEST(CpuThreading, MatMulBatchedBitExact) {
    Shape              xs {3, 5, 128};
    std::vector<float> xd    = lcg(3 * 5 * 128, 9);
    auto               build = [&] {
        Graph    g;
        TensorId x = addInput(g, xs);
        TensorId w = addInit(g, "w", {1, 128, 128}, lcg(128 * 128, 10)); // batch dim 1 broadcasts over 3
        TensorId y = addOutput(g);
        Node     n;
        n.type    = OpType::MatMul;
        n.name    = "matmul";
        n.inputs  = {x, w};
        n.outputs = {y};
        g.nodes.push_back(n);
        return g;
    };
    expectByteIdenticalAcrossThreads(build, xs, xd);
}

// Equal-shape Add takes the NEON block plus a scalar tail; the element count is prime-ish and well
// above the pointwise grain so several chunks actually run, none of them 4-element aligned.
TEST(CpuThreading, AddEqualShapeBitExact) {
    const int64_t      n = 131111;
    Shape              xs {n};
    std::vector<float> xd    = lcg((size_t) n, 11);
    auto               build = [&] {
        Graph    g;
        TensorId x = addInput(g, xs);
        TensorId b = addInit(g, "b", {n}, lcg((size_t) n, 12));
        TensorId y = addOutput(g);
        Node     nd;
        nd.type    = OpType::Add;
        nd.name    = "add";
        nd.inputs  = {x, b};
        nd.outputs = {y};
        g.nodes.push_back(nd);
        return g;
    };
    expectByteIdenticalAcrossThreads(build, xs, xd);
}

// Broadcast Mul: every chunk seeks its own BroadcastWalk start, so a seek/advance mismatch on a
// non-aligned boundary would corrupt exactly the chunk heads.
TEST(CpuThreading, BroadcastBinaryBitExact) {
    Shape              xs {7, 13, 1447};
    std::vector<float> xd    = lcg(7 * 13 * 1447, 13);
    auto               build = [&] {
        Graph    g;
        TensorId x = addInput(g, xs);
        TensorId b = addInit(g, "b", {13, 1}, lcg(13, 14));
        TensorId y = addOutput(g);
        Node     nd;
        nd.type    = OpType::Binary;
        nd.name    = "mul";
        nd.subOp   = (int) BinaryType::Mul;
        nd.inputs  = {x, b};
        nd.outputs = {y};
        g.nodes.push_back(nd);
        return g;
    };
    expectByteIdenticalAcrossThreads(build, xs, xd);
}

// A depthwise conv feeding a 1x1 projection fuses to FusedDwPw, whose two stages partition over the
// E depthwise channels and the Cout projected channels respectively.
TEST(CpuThreading, FusedDwPwBitExact) {
    Shape              xs {1, 13, 45, 45};
    std::vector<float> xd    = lcg(1 * 13 * 45 * 45, 15);
    auto               build = [&] {
        Graph      g;
        TensorId   x  = addInput(g, xs);
        TensorId   dw = addInit(g, "dw", {13, 1, 3, 3}, lcg(13 * 9, 16));
        TensorId   pw = addInit(g, "pw", {11, 13, 1, 1}, lcg(11 * 13, 17));
        TensorDesc t0;
        t0.name    = "t";
        TensorId t = g.addTensor(t0);
        TensorId y = addOutput(g);
        Node     d;
        d.type              = OpType::Conv;
        d.name              = "dconv";
        d.inputs            = {x, dw};
        d.outputs           = {t};
        d.attr.map["group"] = integer(13);
        d.attr.map["pads"]  = ints({1, 1, 1, 1});
        Node p;
        p.type    = OpType::Conv;
        p.name    = "pconv";
        p.inputs  = {t, pw};
        p.outputs = {y};
        g.nodes.push_back(d);
        g.nodes.push_back(p);
        PassOptions opt;
        opt.fuseDwPw = true;
        runStandardPasses(g, opt);
        bool fused = false;
        for (const Node &n: g.nodes)
        {
            fused = fused || n.type == OpType::FusedDwPw;
        }
        EXPECT_TRUE(fused) << "the depthwise + 1x1 pair must fuse for this test to cover FusedDwPw";
        return g;
    };
    expectByteIdenticalAcrossThreads(build, xs, xd);
}
