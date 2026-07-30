// Multi-bucket VXM container round-trips. A compiled model carries one full pass+plan graph per
// declared input-shape set (a "bucket"); every bucket shares ONE content-deduped initializer pool
// because weights are shape-independent. These tests pin the container contract:
//   - a single-bucket save is byte-identical to the legacy single-graph .vxm (the fixed-shape path
//     must not change on disk),
//   - an old single-bucket file still loads through the bucket reader (as one bucket),
//   - a two-bucket container reloads both graphs with their own shapes/nodes,
//   - the shared initializer pool is stored once, not once per bucket (asserted on file size).
#include "vknn/graph.h"
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <vector>

using namespace vknn;

namespace {

    // Read a whole file into a byte vector (empty on failure).
    std::vector<uint8_t> readFile(const std::string &path) {
        FILE *f = fopen(path.c_str(), "rb");
        if (!f)
        {
            return {};
        }
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<uint8_t> b((size_t) (n > 0 ? n : 0));
        if (!b.empty())
        {
            size_t got = fread(b.data(), 1, b.size(), f);
            b.resize(got);
        }
        fclose(f);
        return b;
    }

    // A tiny graph: one input x[dims], one weight w (constWeight bytes), one Relu-ish node producing
    // y[dims]. `dims` differs per bucket so the two buckets carry distinct shapes; `w` is identical
    // across buckets so the initializer pool can dedupe it.
    Graph makeGraph(const Shape &dims, const std::vector<float> &constWeight) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = dims;
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs.push_back(x);

        TensorDesc wi;
        wi.name          = "w";
        wi.shape         = {(int64_t) constWeight.size()};
        wi.isInitializer = true;
        TensorId   w     = g.addTensor(wi);
        HostBuffer hb;
        hb.resizeElems((int64_t) constWeight.size(), DType::Float32);
        for (size_t i = 0; i < constWeight.size(); ++i)
        {
            hb.f32()[i] = constWeight[i];
        }
        g.initializers[w] = hb;

        TensorDesc yo;
        yo.name     = "y";
        yo.shape    = dims;
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        g.outputs.push_back(y);

        Node n;
        n.type    = OpType::Relu;
        n.name    = "relu";
        n.inputs  = {x};
        n.outputs = {y};
        g.nodes.push_back(n);
        return g;
    }

    std::string tmp(const char *base) {
        return testing::TempDir() + base;
    }

} // namespace

// A single-bucket save through the bucket API is byte-identical to the legacy single-graph save.
// The fixed-shape (one-bucket) path must not change the on-disk container at all.
TEST(VxmBuckets, SingleBucketByteIdenticalToLegacy) {
    Graph g = makeGraph({1, 3, 4, 4}, {1.f, 2.f, 3.f});

    std::string legacy = tmp("vxm_legacy.vxm");
    ASSERT_TRUE(saveGraphBin(g, legacy));

    std::vector<Graph> one;
    one.push_back(g);
    std::string bucketed = tmp("vxm_one_bucket.vxm");
    ASSERT_TRUE(saveGraphBinBuckets(one, {"b0"}, bucketed));

    std::vector<uint8_t> a = readFile(legacy);
    std::vector<uint8_t> b = readFile(bucketed);
    std::remove(legacy.c_str());
    std::remove(bucketed.c_str());
    EXPECT_FALSE(a.empty());
    EXPECT_EQ(a, b);
}

// A legacy single-bucket .vxm loads through the bucket reader as exactly one bucket.
TEST(VxmBuckets, LegacyFileLoadsAsOneBucket) {
    Graph       g    = makeGraph({1, 3, 8, 8}, {5.f, 6.f});
    std::string path = tmp("vxm_legacy_load.vxm");
    ASSERT_TRUE(saveGraphBin(g, path));

    std::vector<Graph>       buckets;
    std::vector<std::string> names;
    ASSERT_TRUE(loadGraphBinBuckets(buckets, names, path));
    std::remove(path.c_str());
    ASSERT_EQ(buckets.size(), 1u);
    ASSERT_EQ(names.size(), 1u);
    EXPECT_EQ(buckets[0].tensors.size(), g.tensors.size());
    EXPECT_EQ(buckets[0].nodes.size(), g.nodes.size());
    EXPECT_EQ(buckets[0].desc(buckets[0].inputs[0]).shape, (Shape {1, 3, 8, 8}));
    EXPECT_EQ(buckets[0].initializers.size(), 1u);
}

// Two buckets round-trip: distinct shapes/graphs, both reload correctly, sharing one weight pool.
TEST(VxmBuckets, TwoBucketRoundTrip) {
    Graph g224 = makeGraph({1, 3, 224, 224}, {1.f, 2.f, 3.f, 4.f});
    Graph g320 = makeGraph({1, 3, 320, 320}, {1.f, 2.f, 3.f, 4.f}); // identical weight bytes

    std::vector<Graph>       save   = {g224, g320};
    std::vector<std::string> snames = {"s224", "s320"};
    std::string              path   = tmp("vxm_two_bucket.vxm");
    ASSERT_TRUE(saveGraphBinBuckets(save, snames, path));

    std::vector<Graph>       buckets;
    std::vector<std::string> names;
    ASSERT_TRUE(loadGraphBinBuckets(buckets, names, path));
    std::remove(path.c_str());

    ASSERT_EQ(buckets.size(), 2u);
    ASSERT_EQ(names.size(), 2u);
    EXPECT_EQ(names[0], "s224");
    EXPECT_EQ(names[1], "s320");
    EXPECT_EQ(buckets[0].desc(buckets[0].inputs[0]).shape, (Shape {1, 3, 224, 224}));
    EXPECT_EQ(buckets[1].desc(buckets[1].inputs[0]).shape, (Shape {1, 3, 320, 320}));
    // Each bucket recovers its own weight bytes from the shared pool.
    ASSERT_EQ(buckets[0].initializers.size(), 1u);
    ASSERT_EQ(buckets[1].initializers.size(), 1u);
    TensorId w0 = buckets[0].inputs[0]; // not the weight; look it up by name instead
    (void) w0;
    TensorId wid0 = buckets[0].find("w");
    TensorId wid1 = buckets[1].find("w");
    ASSERT_NE(wid0, kNoTensor);
    ASSERT_NE(wid1, kNoTensor);
    std::vector<float> f0 = initFloats(buckets[0], wid0);
    std::vector<float> f1 = initFloats(buckets[1], wid1);
    EXPECT_EQ(f0, (std::vector<float> {1.f, 2.f, 3.f, 4.f}));
    EXPECT_EQ(f1, (std::vector<float> {1.f, 2.f, 3.f, 4.f}));
}

// The shared initializer pool is stored ONCE for identical weights, not once per bucket. Compare a
// 2-bucket container that shares a big weight against the size of a single-bucket file: the delta is
// the second bucket's graph structure only, far below one extra copy of the weight payload.
TEST(VxmBuckets, SharedPoolNotDuplicated) {
    // A weight large enough that duplicating it would dominate the file size.
    std::vector<float> bigW(4096, 3.14f);
    Graph              gA = makeGraph({1, 3, 16, 16}, bigW);
    Graph              gB = makeGraph({1, 3, 32, 32}, bigW); // same weight bytes, different shape

    std::string oneP = tmp("vxm_pool_one.vxm");
    std::string twoP = tmp("vxm_pool_two.vxm");
    ASSERT_TRUE(saveGraphBinBuckets({gA}, {"a"}, oneP));
    ASSERT_TRUE(saveGraphBinBuckets({gA, gB}, {"a", "b"}, twoP));

    long oneSz = (long) readFile(oneP).size();
    long twoSz = (long) readFile(twoP).size();
    std::remove(oneP.c_str());
    std::remove(twoP.c_str());

    long weightBytes = (long) (bigW.size() * sizeof(float)); // 16384
    // The second bucket adds its graph structure (tensor/node tables, a pool-index reference) but NOT
    // a second copy of the weight payload. The growth must be well under one weight copy.
    EXPECT_LT(twoSz - oneSz, weightBytes / 2);
}

// Buckets from DIFFERENT source graphs (disjoint input/weight names) round-trip in one container:
// each bucket keeps its own input name, shape, and weight bytes. This is the multi-graph .vxm a
// vision tower + decoder ship as.
TEST(VxmBuckets, MultiGraphBucketsRoundTrip) {
    Graph gx = makeGraph({1, 3, 4, 4}, {1.f, 2.f});
    Graph gp = makeGraph({1, 3, 384, 384}, {9.f, 8.f, 7.f});
    // Rename gp's IO/weight so the two buckets are genuinely different graphs.
    gp.tensors[gp.inputs[0]].name  = "pix";
    gp.tensors[gp.outputs[0]].name = "emb";
    gp.tensors[gp.find("w")].name  = "wp";

    std::string path = tmp("vxm_multi_graph.vxm");
    ASSERT_TRUE(saveGraphBinBuckets({gx, gp}, {"gx", "gp"}, path));

    std::vector<Graph>       buckets;
    std::vector<std::string> names;
    ASSERT_TRUE(loadGraphBinBuckets(buckets, names, path));
    std::remove(path.c_str());

    ASSERT_EQ(buckets.size(), 2u);
    EXPECT_EQ(buckets[0].desc(buckets[0].inputs[0]).name, "x");
    EXPECT_EQ(buckets[1].desc(buckets[1].inputs[0]).name, "pix");
    EXPECT_EQ(buckets[1].desc(buckets[1].inputs[0]).shape, (Shape {1, 3, 384, 384}));
    TensorId wp = buckets[1].find("wp");
    ASSERT_NE(wp, kNoTensor);
    EXPECT_EQ(initFloats(buckets[1], wp), (std::vector<float> {9.f, 8.f, 7.f}));
    TensorId wx = buckets[0].find("w");
    ASSERT_NE(wx, kNoTensor);
    EXPECT_EQ(initFloats(buckets[0], wx), (std::vector<float> {1.f, 2.f}));
}

// Identical weight bytes dedupe in the pool even when they belong to DIFFERENT graphs: the pool is
// content-keyed, so a weight shared by two unrelated buckets is stored once.
TEST(VxmBuckets, CrossGraphContentDedup) {
    std::vector<float> bigW(4096, 2.71f);
    Graph              gA          = makeGraph({1, 3, 16, 16}, bigW);
    Graph              gB          = makeGraph({1, 3, 32, 32}, bigW);
    gB.tensors[gB.inputs[0]].name  = "in2";
    gB.tensors[gB.outputs[0]].name = "out2";
    gB.tensors[gB.find("w")].name  = "w2"; // different name, same bytes

    std::string oneP = tmp("vxm_xg_one.vxm");
    std::string twoP = tmp("vxm_xg_two.vxm");
    ASSERT_TRUE(saveGraphBinBuckets({gA}, {"a"}, oneP));
    ASSERT_TRUE(saveGraphBinBuckets({gA, gB}, {"a", "b"}, twoP));
    long oneSz = (long) readFile(oneP).size();
    long twoSz = (long) readFile(twoP).size();
    std::remove(oneP.c_str());
    std::remove(twoP.c_str());
    EXPECT_LT(twoSz - oneSz, (long) (bigW.size() * sizeof(float)) / 2);
}

// The streamed reader delivers the same buckets the bulk reader does (same names, structure, weight
// bytes), one at a time and in order; returning false from the consumer stops the stream cleanly
// after the current bucket with a true (non-error) result.
TEST(VxmBuckets, StreamedLoadMatchesBulkLoad) {
    Graph g1 = makeGraph({1, 3, 8, 8}, {1.f, 2.f, 3.f});
    Graph g2 = makeGraph({1, 3, 16, 16}, {1.f, 2.f, 3.f});
    Graph g3 = makeGraph({1, 3, 24, 24}, {4.f});

    std::string path = tmp("vxm_streamed.vxm");
    ASSERT_TRUE(saveGraphBinBuckets({g1, g2, g3}, {"b1", "b2", "b3"}, path));

    std::vector<Graph>       bulk;
    std::vector<std::string> bulkNames;
    ASSERT_TRUE(loadGraphBinBuckets(bulk, bulkNames, path));

    std::vector<Graph>       streamed;
    std::vector<std::string> streamedNames;
    std::vector<size_t>      counts;
    ASSERT_TRUE(loadGraphBinBucketsStreamed(path, [&](Graph &&g, const std::string &n, size_t idx, size_t count) {
        EXPECT_EQ(idx, streamed.size());
        counts.push_back(count);
        streamed.push_back(std::move(g));
        streamedNames.push_back(n);
        return true;
    }));
    ASSERT_EQ(streamed.size(), bulk.size());
    EXPECT_EQ(streamedNames, bulkNames);
    for (size_t b = 0; b < bulk.size(); ++b)
    {
        EXPECT_EQ(counts[b], 3u);
        EXPECT_EQ(streamed[b].nodes.size(), bulk[b].nodes.size());
        EXPECT_EQ(streamed[b].tensors.size(), bulk[b].tensors.size());
        TensorId ws = streamed[b].find("w");
        TensorId wb = bulk[b].find("w");
        ASSERT_NE(ws, kNoTensor);
        EXPECT_EQ(initFloats(streamed[b], ws), initFloats(bulk[b], wb));
    }

    // Early stop: the consumer takes bucket 0 only; the stream ends true with one bucket delivered.
    size_t seen = 0;
    ASSERT_TRUE(loadGraphBinBucketsStreamed(path, [&](Graph &&, const std::string &, size_t, size_t) {
        ++seen;
        return false;
    }));
    EXPECT_EQ(seen, 1u);

    // loadGraphBin takes the first bucket of a multi-bucket file.
    Graph first;
    ASSERT_TRUE(loadGraphBin(first, path));
    EXPECT_EQ(first.desc(first.inputs[0]).shape, (Shape {1, 3, 8, 8}));
    std::remove(path.c_str());
}

// A non-VXM3/VXM4 file loads to a clean failure (returns false, no crash). The loader distinguishes an
// incompatible-version VXM container (a "VXM<n>" magic) from a file that is not a .vxm at all so the
// log names the fix; both paths must simply refuse the file.
TEST(VxmBuckets, IncompatibleAndForeignFilesRejected) {
    auto write = [](const std::string &path, const std::vector<uint8_t> &bytes) {
        FILE *f = fopen(path.c_str(), "wb");
        if (f)
        {
            if (!bytes.empty())
            {
                fwrite(bytes.data(), 1, bytes.size(), f);
            }
            fclose(f);
        }
    };
    std::vector<Graph>       buckets;
    std::vector<std::string> names;

    // A "VXM2" container (0x324d5856) -- a recognizable but incompatible engine version.
    std::string v2 = tmp("vxm_v2.vxm");
    write(v2, {0x56, 0x58, 0x4d, 0x32, 0, 0, 0, 0});
    EXPECT_FALSE(loadGraphBinBuckets(buckets, names, v2));

    // A non-VXM magic -- not a .vxm at all (wrong file / corrupt).
    std::string junk = tmp("vxm_junk.vxm");
    write(junk, {0xde, 0xad, 0xbe, 0xef, 1, 2, 3, 4});
    EXPECT_FALSE(loadGraphBinBuckets(buckets, names, junk));

    // An empty file -- the first-word read cannot complete.
    std::string empty = tmp("vxm_empty.vxm");
    write(empty, {});
    EXPECT_FALSE(loadGraphBinBuckets(buckets, names, empty));

    std::remove(v2.c_str());
    std::remove(junk.c_str());
    std::remove(empty.c_str());
}
