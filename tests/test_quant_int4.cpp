// -Os INT4 weight quantization (quantizeWeights + core/quant_int4.h): the packed-payload
// layout round-trips, the pass quantizes exactly the eligible weights with calibrated group scales
// and outlier columns, the mixed-precision guard keeps hostile layers fp16, the quantized graph
// stamps VXM5 and round-trips through a .vxm, and the materialized CPU run tracks the fp32
// reference within the int4 error envelope. GPU-kernel parity is verified on device (the host
// suite has no Vulkan), against this same CPU path as the oracle.
#include "core/quant_int4.h"
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    // a[M,K] -> Relu -> t -> MatMul(t, W[K,N]) -> y. The Relu keeps the captured calibration
    // activation an intermediate (the common shape), not the graph input itself. `M` differentiates
    // shape buckets of one weight set (M = 1 is the plain single-graph fixture).
    Graph matmulGraph(int64_t K, int64_t N, float wScale = 1.0f, int64_t M = 1) {
        Graph      g;
        TensorDesc ai;
        ai.name    = "a";
        ai.shape   = {M, K};
        ai.isInput = true;
        TensorId a = g.addTensor(ai);
        g.inputs.push_back(a);
        TensorDesc td;
        td.name      = "t";
        TensorId   t = g.addTensor(td);
        TensorDesc wd;
        wd.name          = "w";
        wd.shape         = {K, N};
        wd.isInitializer = true;
        TensorId   w     = g.addTensor(wd);
        HostBuffer hb;
        hb.resizeElems(K * N, DType::Float32);
        for (int64_t k = 0; k < K; ++k)
        {
            for (int64_t n = 0; n < N; ++n)
            {
                hb.f32()[k * N + n] = wScale * std::sin(0.37f * (float) k + 1.13f * (float) n);
            }
        }
        g.initializers[w] = hb;
        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     relu;
        relu.type    = OpType::Relu;
        relu.name    = "relu";
        relu.inputs  = {a};
        relu.outputs = {t};
        g.nodes.push_back(relu);
        Node mm;
        mm.type    = OpType::MatMul;
        mm.name    = "mm";
        mm.inputs  = {t, w};
        mm.outputs = {y};
        g.nodes.push_back(mm);
        g.outputs = {y};
        return g;
    }

    std::vector<float> runCpu(Graph g, const std::vector<float> &adata, int64_t K, int64_t M = 1) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        IOTensor in;
        in.name  = "a";
        in.shape = {M, K};
        in.dtype = DType::Float32;
        in.data.resize(adata.size() * 4);
        std::memcpy(in.data.data(), adata.data(), adata.size() * 4);
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
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

    double relL2(const std::vector<float> &got, const std::vector<float> &want) {
        EXPECT_EQ(got.size(), want.size());
        double num = 0, den = 0;
        for (size_t i = 0; i < got.size() && i < want.size(); ++i)
        {
            num += ((double) got[i] - want[i]) * ((double) got[i] - want[i]);
            den += (double) want[i] * want[i];
        }
        return den > 0 ? std::sqrt(num / den) : 0.0;
    }

    std::vector<float> testInput(int64_t K, int64_t M = 1) {
        std::vector<float> a((size_t) (M * K));
        for (int64_t i = 0; i < M * K; ++i)
        {
            a[(size_t) i] = std::cos(0.11f * (float) i) + 0.3f * std::sin(0.05f * (float) i * (float) i);
        }
        return a;
    }

} // namespace

// The packed nibble layout round-trips every int4 value through int4Pack/int4At, including the odd-N
// word-padding tail (padding nibbles read 0).
TEST(QuantInt4, PackUnpackRoundTrip) {
    const int64_t       K = 3, N = 11; // ragged: rowBytes pads 11 nibbles to 8 bytes (2 words)
    std::vector<int8_t> q((size_t) (K * N));
    for (int64_t i = 0; i < K * N; ++i)
    {
        q[(size_t) i] = (int8_t) ((i % 15) - 7); // the full [-7,7] alphabet
    }
    std::vector<uint8_t> packed = int4Pack(q, K, N);
    EXPECT_EQ((int64_t) packed.size(), K * int4RowBytes(N));
    for (int64_t k = 0; k < K; ++k)
    {
        for (int64_t n = 0; n < N; ++n)
        {
            EXPECT_EQ(int4At(packed.data(), int4RowBytes(N), k, n), (int) q[(size_t) (k * N + n)]) << "k=" << k << " n=" << n;
        }
        for (int64_t n = N; n < int4RowBytes(N) * 2; ++n)
        {
            EXPECT_EQ(int4At(packed.data(), int4RowBytes(N), k, n), 0) << "padding nibble k=" << k << " n=" << n;
        }
    }
}

// int4Dequant applies the per-(group, column) fp16 scale to each nibble and overwrites outlier rows
// with their kept fp16 values.
TEST(QuantInt4, DequantAppliesScalesAndOutliers) {
    const int64_t       K = 4, N = 8, group = 2;
    std::vector<int8_t> q((size_t) (K * N));
    for (int64_t i = 0; i < K * N; ++i)
    {
        q[(size_t) i] = (int8_t) ((i % 5) - 2);
    }
    std::vector<uint8_t>  packed = int4Pack(q, K, N);
    std::vector<uint16_t> scales((size_t) (int4GroupCount(K, group) * N));
    for (size_t i = 0; i < scales.size(); ++i)
    {
        scales[i] = floatToHalf(0.5f + 0.25f * (float) (i % 3));
    }
    const int32_t         oidx[1] = {2};
    std::vector<uint16_t> oval((size_t) N);
    for (int64_t n = 0; n < N; ++n)
    {
        oval[(size_t) n] = floatToHalf(10.0f + (float) n);
    }
    std::vector<float> w = int4Dequant(packed.data(), scales.data(), oidx, oval.data(), K, N, group, 1);
    for (int64_t k = 0; k < K; ++k)
    {
        for (int64_t n = 0; n < N; ++n)
        {
            float want = k == 2 ? 10.0f + (float) n : (float) q[(size_t) (k * N + n)] * halfToFloat(scales[(size_t) ((k / group) * N + n)]);
            EXPECT_FLOAT_EQ(w[(size_t) (k * N + n)], want) << "k=" << k << " n=" << n;
        }
    }
}

// The pass quantizes an eligible MatMul weight: the payload packs to K*rowBytes, the side
// initializers appear with the declared extents, the node carries the full attribute set, and the
// desc keeps its logical shape with an fp16 dtype stamp.
TEST(QuantInt4, PassQuantizesEligibleMatMul) {
    const int64_t K = 256, N = 64;
    Graph         g = matmulGraph(K, N);
    runStandardPasses(g);
    QuantOptions opt;
    QuantStats   st = quantizeWeights(g, opt);
    EXPECT_EQ(st.sites, 1);
    EXPECT_EQ(st.quantized, 1);
    EXPECT_TRUE(st.calibrated);
    const Node *mm = nullptr;
    for (const Node &nd: g.nodes)
    {
        if (nd.type == OpType::MatMul)
        {
            mm = &nd;
        }
    }
    ASSERT_TRUE(mm);
    ASSERT_TRUE(mm->attr.has(kWq));
    EXPECT_EQ(mm->attr.geti(kWqK, 0), K);
    EXPECT_EQ(mm->attr.geti(kWqN, 0), N);
    EXPECT_EQ(mm->attr.geti(kWqLayout, -1), 0);
    const int64_t nOut = mm->attr.geti(kWqNOut, -1);
    EXPECT_EQ(nOut, (int64_t) ((double) K * opt.outlierFrac));
    const TensorId w = mm->inputs[1];
    EXPECT_EQ((int64_t) g.initializers.at(w).bytes.size(), K * int4RowBytes(N));
    EXPECT_EQ(g.desc(w).dtype, DType::Float16);
    EXPECT_EQ(g.desc(w).shape, (Shape {K, N})); // logical shape survives for shape-reading gates
    const TensorId scaleId = (TensorId) mm->attr.geti(kWqScales, kNoTensor);
    ASSERT_NE(scaleId, kNoTensor);
    EXPECT_EQ((int64_t) g.initializers.at(scaleId).bytes.size(), int4GroupCount(K, opt.group) * N * 2);
    if (nOut > 0)
    {
        const TensorId oidxId = (TensorId) mm->attr.geti(kWqOidx, kNoTensor);
        const TensorId ovalId = (TensorId) mm->attr.geti(kWqOval, kNoTensor);
        EXPECT_EQ((int64_t) g.initializers.at(oidxId).bytes.size(), nOut * 4);
        EXPECT_EQ((int64_t) g.initializers.at(ovalId).bytes.size(), nOut * N * 2);
    }
}

// The mixed-precision guard: with an impossible error bar every layer stays fp16 and the graph is
// structurally untouched (no attrs, no side tensors, original payload).
TEST(QuantInt4, GuardKeepsHostileLayerFp16) {
    const int64_t K = 256, N = 64;
    Graph         g = matmulGraph(K, N);
    runStandardPasses(g);
    const size_t tensorsBefore = g.tensors.size();
    QuantOptions opt;
    opt.maxLayerRelErr = 1e-6; // int4 cannot meet this: the guard must keep the layer fp16
    QuantStats st      = quantizeWeights(g, opt);
    EXPECT_EQ(st.quantized, 0);
    EXPECT_EQ(st.guardKept, 1);
    EXPECT_EQ(g.tensors.size(), tensorsBefore);
    for (const Node &nd: g.nodes)
    {
        EXPECT_FALSE(nd.attr.has(kWq));
    }
}

// Ineligible shapes stay fp16 by construction: a shallow reduction (K < minK) and a shared weight
// are both skipped.
TEST(QuantInt4, IneligibleWeightsSkipped) {
    {
        Graph g = matmulGraph(128, 256); // K=128 < minK
        runStandardPasses(g);
        QuantStats st = quantizeWeights(g, QuantOptions {});
        EXPECT_EQ(st.sites, 0);
    }
    {
        Graph g = matmulGraph(256, 64);
        // A second consumer of the weight makes it shared: attrs live on one node but the payload
        // is one tensor, so shared weights are ineligible.
        TensorId   w = g.find("w");
        TensorDesc y2d;
        y2d.name     = "y2";
        y2d.isOutput = true;
        TensorId y2  = g.addTensor(y2d);
        Node     mm2;
        mm2.type    = OpType::MatMul;
        mm2.name    = "mm2";
        mm2.inputs  = {g.find("t"), w};
        mm2.outputs = {y2};
        g.nodes.push_back(mm2);
        g.outputs.push_back(y2);
        runStandardPasses(g);
        QuantStats st = quantizeWeights(g, QuantOptions {});
        EXPECT_EQ(st.sites, 0);
    }
}

// End-to-end through the exact compile pipeline (-Os order: passes, quantize, fp16 sweep, save):
// the .vxm stamps VXM5, loads back, and the materialized CPU run tracks the fp32 reference inside
// the int4 error envelope while a plain fp16 compile of the same graph stays VXM3.
TEST(QuantInt4, QuantizedVxmRoundTripsAndTracksFp32) {
    const int64_t K = 512, N = 96;
    const auto    a   = testInput(K);
    const auto    ref = runCpu(matmulGraph(K, N), a, K);
    Graph         g   = matmulGraph(K, N);
    runStandardPasses(g);
    QuantStats st = quantizeWeights(g, QuantOptions {});
    EXPECT_EQ(st.quantized, 1);
    convertInitializersFp16(g);
    const std::string path = testing::TempDir() + "vknn_quant_int4_e2e.vxm";
    ASSERT_TRUE(saveGraphBin(g, path));
    {
        FILE *f = fopen(path.c_str(), "rb");
        ASSERT_TRUE(f);
        char magic[5] = {};
        ASSERT_EQ(fread(magic, 1, 4, f), 4u);
        fclose(f);
        EXPECT_EQ(std::string(magic, 4), "VXM5");
    }
    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::createFromVxm(path, cfg);
    std::remove(path.c_str());
    ASSERT_TRUE(sess);
    IOTensor in;
    in.name  = "a";
    in.shape = {1, K};
    in.dtype = DType::Float32;
    in.data.resize(a.size() * 4);
    std::memcpy(in.data.data(), a.data(), a.size() * 4);
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_FALSE(outs.empty());
    std::vector<float> got(outs[0].f32(), outs[0].f32() + numElements(outs[0].shape));
    for (float v: got)
    {
        EXPECT_FALSE(std::isnan(v));
    }
    // Grouped int4 sits near 10% weight-space error, and this synthetic weight (a low-rank sine
    // pattern) amplifies it: the structured signal partially cancels under the test input while the
    // unstructured quantization noise does not. The envelope is a garbage detector — a packing,
    // scale-indexing, or outlier bug produces relL2 near 1, not a tenth — and the sharp value-level
    // check is ExactlyRepresentableWeightsRoundTripExactly below.
    EXPECT_LT(relL2(got, ref), 0.25) << "quantized CPU run drifted from the fp32 reference";

    Graph plain = matmulGraph(K, N);
    runStandardPasses(plain);
    convertInitializersFp16(plain);
    const std::string plainPath = testing::TempDir() + "vknn_quant_int4_plain.vxm";
    ASSERT_TRUE(saveGraphBin(plain, plainPath));
    FILE *f = fopen(plainPath.c_str(), "rb");
    ASSERT_TRUE(f);
    char magic[5] = {};
    ASSERT_EQ(fread(magic, 1, 4, f), 4u);
    fclose(f);
    std::remove(plainPath.c_str());
    EXPECT_EQ(std::string(magic, 4), "VXM3") << "non-quantized output must keep the legacy container";
}

// A weight whose every value is an int4 multiple of a power-of-two step must survive quantization
// EXACTLY: the min-MSE search finds the zero-cost step (maxAbs/7 = the true step, exact in fp16),
// every value requantizes to itself, and the materialized fp16 payload equals the fp16 sweep of the
// original — so the quantized CPU run is bit-identical to the plain fp16 compile.
TEST(QuantInt4, ExactlyRepresentableWeightsRoundTripExactly) {
    const int64_t K = 256, N = 64;
    const float   step       = 0.125f; // 2^-3: exact in fp16, and q*step is exact for q in [-7,7]
    auto          buildExact = [&] {
        Graph    g = matmulGraph(K, N);
        TensorId w = g.find("w");
        float   *v = g.initializers.at(w).f32();
        for (int64_t i = 0; i < K * N; ++i)
        {
            v[i] = (float) ((i * 7919) % 15 - 7) * step; // the full [-7,7] alphabet times the step
        }
        return g;
    };
    const auto a     = testInput(K);
    Graph      plain = buildExact();
    runStandardPasses(plain);
    convertInitializersFp16(plain);
    const auto ref = runCpu(std::move(plain), a, K);
    Graph      g   = buildExact();
    runStandardPasses(g);
    ASSERT_EQ(quantizeWeights(g, QuantOptions {}).quantized, 1);
    convertInitializersFp16(g);
    const auto got = runCpu(std::move(g), a, K);
    ASSERT_EQ(got.size(), ref.size());
    for (size_t i = 0; i < got.size(); ++i)
    {
        EXPECT_EQ(got[i], ref[i]) << "output " << i << " not bit-identical on exactly representable weights";
    }
}

// materializeInt4Weights with no keep predicate reconstructs a plain fp16 weight: attrs stripped,
// side tensors dropped, payload sized to the logical desc — indistinguishable from an fp16 compile
// to any consumer.
TEST(QuantInt4, MaterializeReconstructsPlainFp16Weight) {
    const int64_t K = 256, N = 64;
    Graph         g = matmulGraph(K, N);
    runStandardPasses(g);
    ASSERT_EQ(quantizeWeights(g, QuantOptions {}).quantized, 1);
    const int64_t materialized = materializeInt4Weights(g, nullptr);
    EXPECT_EQ(materialized, 1);
    TensorId w = g.find("w");
    ASSERT_NE(w, kNoTensor);
    EXPECT_EQ(g.desc(w).dtype, DType::Float16);
    EXPECT_EQ((int64_t) g.initializers.at(w).bytes.size(), K * N * 2);
    for (const Node &nd: g.nodes)
    {
        EXPECT_FALSE(nd.attr.has(kWq));
    }
    for (const auto &kv: g.initializers)
    {
        // The scale/outlier side tensors must be gone; a bias-correction bias (#i4b) is a real
        // operand and legitimately survives.
        for (const char *suffix: {"#i4s", "#i4oi", "#i4ov"})
        {
            EXPECT_EQ(g.desc(kv.first).name.find(suffix), std::string::npos) << "side tensor payload survived materialization";
        }
    }
}

// The .vxm-input re-quantization pipeline (vknn_compile <in.vxm> <out.vxm> -Os --quant-samples 0,
// minus the flag plumbing): EVERY bucket of a multi-bucket input is loaded, quantized, swept to
// fp16, and saved with bucket order and names preserved over one content-deduped initializer pool.
// Weight-only quantization is deterministic per payload, so the buckets' identical weights quantize
// to identical packed bytes and the two-bucket output grows by graph structure only — never by a
// second copy of the payloads. Each bucket of the output then loads and runs on the CPU at its own
// shape.
TEST(QuantInt4, VxmInputRequantPreservesAllBuckets) {
    const int64_t K = 512, N = 96;
    const int64_t seqs[2] = {2, 5};

    // The multi-bucket INPUT .vxm: one passes-applied graph per shape bucket, identical weights.
    std::vector<Graph>       inBuckets;
    std::vector<std::string> inNames = {"dim:seq=2", "dim:seq=5"};
    for (int64_t m: seqs)
    {
        Graph g = matmulGraph(K, N, 1.0f, m);
        runStandardPasses(g);
        inBuckets.push_back(std::move(g));
    }
    const std::string inPath = testing::TempDir() + "vknn_requant_in.vxm";
    ASSERT_TRUE(saveGraphBinBuckets(inBuckets, inNames, inPath));

    // The re-quantization pipeline over the .vxm input: load ALL buckets, quantize and fp16-sweep
    // each, save them back under their stored names.
    QuantOptions opt;
    opt.calibSamples = 0; // weight-only (--quant-samples 0)
    std::vector<Graph>       buckets;
    std::vector<std::string> names;
    ASSERT_TRUE(loadGraphBinBuckets(buckets, names, inPath));
    std::remove(inPath.c_str());
    ASSERT_EQ(buckets.size(), 2u);
    EXPECT_EQ(names, inNames);
    for (Graph &gb: buckets)
    {
        QuantStats st = quantizeWeights(gb, opt);
        EXPECT_EQ(st.quantized, 1);
        EXPECT_FALSE(st.calibrated);
        convertInitializersFp16(gb);
    }
    const std::string outPath = testing::TempDir() + "vknn_requant_out.vxm";
    ASSERT_TRUE(saveGraphBinBuckets(buckets, names, outPath));

    // Quantized multi-bucket containers stamp VXM5.
    {
        FILE *f = fopen(outPath.c_str(), "rb");
        ASSERT_TRUE(f);
        char magic[5] = {};
        ASSERT_EQ(fread(magic, 1, 4, f), 4u);
        fclose(f);
        EXPECT_EQ(std::string(magic, 4), "VXM5");
    }

    // Bucket count, order, names, and per-bucket I/O names+shapes survive; every bucket carries its
    // quantization.
    std::vector<Graph>       reloaded;
    std::vector<std::string> reloadedNames;
    ASSERT_TRUE(loadGraphBinBuckets(reloaded, reloadedNames, outPath));
    ASSERT_EQ(reloaded.size(), 2u);
    EXPECT_EQ(reloadedNames, inNames);
    for (size_t b = 0; b < reloaded.size(); ++b)
    {
        ASSERT_EQ(reloaded[b].inputs.size(), 1u);
        EXPECT_EQ(reloaded[b].desc(reloaded[b].inputs[0]).name, "a");
        EXPECT_EQ(reloaded[b].desc(reloaded[b].inputs[0]).shape, (Shape {seqs[b], K}));
        ASSERT_EQ(reloaded[b].outputs.size(), 1u);
        EXPECT_EQ(reloaded[b].desc(reloaded[b].outputs[0]).name, "y");
        EXPECT_EQ(reloaded[b].desc(reloaded[b].outputs[0]).shape, (Shape {seqs[b], N}));
        bool hasQuantizedNode = false;
        for (const Node &nd: reloaded[b].nodes)
        {
            hasQuantizedNode = hasQuantizedNode || nd.attr.has(kWq);
        }
        EXPECT_TRUE(hasQuantizedNode) << "bucket " << b << " lost its quantization";
    }

    // Content dedup: a single-bucket product of the same pipeline is the yardstick — the second
    // bucket adds its graph structure but never a second copy of the packed weight payload.
    const std::string onePath = testing::TempDir() + "vknn_requant_one.vxm";
    {
        std::vector<Graph> one;
        one.push_back(std::move(reloaded[0]));
        ASSERT_TRUE(saveGraphBinBuckets(one, {reloadedNames[0]}, onePath));
    }
    auto fileSize = [](const std::string &p) {
        FILE *f = fopen(p.c_str(), "rb");
        EXPECT_TRUE(f);
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fclose(f);
        return n;
    };
    const long twoSz = fileSize(outPath);
    const long oneSz = fileSize(onePath);
    std::remove(onePath.c_str());
    EXPECT_LT(twoSz - oneSz, (long) (K * int4RowBytes(N)) / 2) << "quantized payloads did not dedup across buckets";

    // Both buckets run on the CPU from the saved file, dispatched by input shape, inside the int4
    // error envelope of their fp32 references.
    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::createFromVxm(outPath, cfg);
    std::remove(outPath.c_str());
    ASSERT_TRUE(sess);
    for (int64_t m: seqs)
    {
        const auto a   = testInput(K, m);
        const auto ref = runCpu(matmulGraph(K, N, 1.0f, m), a, K, m);
        IOTensor   in;
        in.name  = "a";
        in.shape = {m, K};
        in.dtype = DType::Float32;
        in.data.resize(a.size() * 4);
        std::memcpy(in.data.data(), a.data(), a.size() * 4);
        std::vector<IOTensor> outs;
        ASSERT_EQ(sess->run({in}, outs), Status::Ok) << "bucket m=" << m;
        ASSERT_FALSE(outs.empty());
        EXPECT_EQ(outs[0].shape, (Shape {m, N}));
        std::vector<float> got(outs[0].f32(), outs[0].f32() + numElements(outs[0].shape));
        EXPECT_LT(relL2(got, ref), 0.25) << "bucket m=" << m << " drifted from its fp32 reference";
    }
}

// A weight holding a subnormal-magnitude group (every candidate quantization step underflows to
// fp16 zero) keeps the guard metric finite and meaningful: the flushed group contributes its own
// energy — never the 1e300 search sentinel — so the healthy groups decide and the layer quantizes,
// encoding the degenerate group as zeros at scale 1.
TEST(QuantInt4, SubnormalGroupKeepsGuardMetricFinite) {
    const int64_t K = 256, N = 64; // default group 128: rows [0,128) form group 0, [128,256) group 1
    Graph         g = matmulGraph(K, N);
    {
        TensorId w = g.find("w");
        float   *v = g.initializers.at(w).f32();
        for (int64_t k = 0; k < 128; ++k)
        {
            for (int64_t n = 0; n < N; ++n)
            {
                // Below every candidate step's fp16 threshold: maxAbs / 7 rounds to fp16 zero.
                v[k * N + n] = 1e-7f * (float) ((k + n) % 3 - 1);
            }
        }
    }
    runStandardPasses(g);
    QuantStats st = quantizeWeights(g, QuantOptions {});
    EXPECT_EQ(st.sites, 1);
    EXPECT_EQ(st.quantized, 1) << "the flushed group must not veto the layer through the error metric";
    EXPECT_EQ(st.guardKept, 0);

    const Node *mm = nullptr;
    for (const Node &nd: g.nodes)
    {
        if (nd.type == OpType::MatMul)
        {
            mm = &nd;
        }
    }
    ASSERT_TRUE(mm);
    ASSERT_TRUE(mm->attr.has(kWq));
    // The subnormal group encodes as zero nibbles at scale 1 (outlier columns rank by magnitude and
    // land in the healthy group).
    const TensorId w      = mm->inputs[1];
    const uint8_t *packed = g.initializers.at(w).bytes.data();
    for (int64_t k = 0; k < 128; ++k)
    {
        for (int64_t n = 0; n < N; ++n)
        {
            ASSERT_EQ(int4At(packed, int4RowBytes(N), k, n), 0) << "k=" << k << " n=" << n;
        }
    }
    const TensorId scaleId = (TensorId) mm->attr.geti(kWqScales, kNoTensor);
    ASSERT_NE(scaleId, kNoTensor);
    const uint16_t *scales = reinterpret_cast<const uint16_t *>(g.initializers.at(scaleId).bytes.data());
    for (int64_t n = 0; n < N; ++n)
    {
        EXPECT_EQ(halfToFloat(scales[n]), 1.0f) << "group 0 scale n=" << n;
    }
}

// Fully degenerate weights (all zeros, and ~1e-30 — below any representable fp16 step) make a
// defined decision with a finite metric: the layer quantizes, dequantizes to zeros, and runs.
TEST(QuantInt4, DegenerateWeightsQuantizeToZeros) {
    const int64_t K = 256, N = 64;
    for (float scale: {0.0f, 1e-30f})
    {
        Graph g = matmulGraph(K, N);
        {
            TensorId w = g.find("w");
            float   *v = g.initializers.at(w).f32();
            for (int64_t i = 0; i < K * N; ++i)
            {
                v[i] = scale * (float) ((i % 5) - 2);
            }
        }
        runStandardPasses(g);
        QuantStats st = quantizeWeights(g, QuantOptions {});
        EXPECT_EQ(st.sites, 1) << "scale " << scale;
        EXPECT_EQ(st.quantized, 1) << "scale " << scale;
        EXPECT_EQ(st.guardKept, 0) << "scale " << scale;
        convertInitializersFp16(g);
        const auto a   = testInput(K);
        const auto got = runCpu(std::move(g), a, K);
        for (float y: got)
        {
            ASSERT_TRUE(std::isfinite(y));
            // The bias correction may carry the (sub-1e-28) systematic shift of flushing the tiny
            // weights; the output stays zero at any observable magnitude.
            EXPECT_LT(std::fabs(y), 1e-20f) << "scale " << scale;
        }
    }
}
