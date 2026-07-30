// Bit-width-agnostic weight quantization core (core/quant_weights.h): the int8 payload layout
// round-trips, the LUT4 decode applies its codebook, the format-dispatching materialization
// reconstructs fp16 weights and rejects unknown format ids, and the container versioning holds —
// int4-only content stays VXM5, any extended format stamps VXM6, and unknown VXM5/VXM6 subtags or
// foreign VXM versions are rejected at load instead of being misread.
#include "core/quant_weights.h"
#include "import/passes.h"
#include "vknn/error.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    // a[M,K] -> Relu -> t -> MatMul(t, W[K,N]) -> y, the test_quant_int4.cpp fixture shape. `M` is
    // the activation row count one calibration sample yields — the sequence length a bucket compiles
    // at, which is what makes calibration statistics shape-dependent.
    Graph matmulGraph(int64_t K, int64_t N, int64_t M = 1) {
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
                hb.f32()[k * N + n] = std::sin(0.37f * (float) k + 1.13f * (float) n);
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

    // Replace the graph's [K, N] fp32 MatMul weight with a hand-packed int8 payload (per-group
    // symmetric scales, no outliers) and stamp the kWq attribute set with format kWqFormatInt8 —
    // the serialized product the int8 quantization pass emits, built directly so the container and
    // materialization contracts are testable without the pass.
    void packWeightInt8ByHand(Graph &g, int64_t K, int64_t N, int64_t group) {
        TensorId              w       = g.find("w");
        std::vector<float>    vals    = initFloats(g, w);
        const int64_t         nGroups = int4GroupCount(K, group);
        std::vector<uint16_t> scales((size_t) (nGroups * N));
        std::vector<int8_t>   q((size_t) (K * N), 0);
        for (int64_t gp = 0; gp < nGroups; ++gp)
        {
            const int64_t k0 = gp * group, k1 = std::min(K, k0 + group);
            for (int64_t n = 0; n < N; ++n)
            {
                double maxAbs = 0;
                for (int64_t k = k0; k < k1; ++k)
                {
                    maxAbs = std::max(maxAbs, (double) std::fabs(vals[(size_t) (k * N + n)]));
                }
                const float s                 = halfToFloat(floatToHalfSat((float) (maxAbs / 127.0)));
                scales[(size_t) (gp * N + n)] = floatToHalfSat(s <= 0 ? 1.0f : s);
                if (s <= 0)
                {
                    continue;
                }
                for (int64_t k = k0; k < k1; ++k)
                {
                    double qq               = std::nearbyint((double) vals[(size_t) (k * N + n)] / (double) s);
                    qq                      = std::min(127.0, std::max(-127.0, qq));
                    q[(size_t) (k * N + n)] = (int8_t) qq;
                }
            }
        }
        {
            HostBuffer hb;
            hb.bytes          = int8Pack(q, K, N);
            g.initializers[w] = std::move(hb);
        }
        g.desc(w).dtype = DType::Float16;
        TensorDesc sd;
        sd.name                      = "w#i8s";
        sd.shape                     = {nGroups * N};
        sd.dtype                     = DType::Float16;
        sd.isInitializer             = true;
        TensorId             scaleId = g.addTensor(sd);
        HostBuffer           shb;
        std::vector<uint8_t> scaleBytes((const uint8_t *) scales.data(), (const uint8_t *) scales.data() + scales.size() * 2);
        shb.bytes               = std::move(scaleBytes);
        g.initializers[scaleId] = std::move(shb);
        for (Node &nd: g.nodes)
        {
            if (nd.type != OpType::MatMul)
            {
                continue;
            }
            auto seti = [&](const char *key, int64_t v) {
                Attr attr;
                attr.kind        = Attr::Int;
                attr.i           = v;
                nd.attr.map[key] = attr;
            };
            seti(kWq, kWqFormatInt8);
            seti(kWqK, K);
            seti(kWqN, N);
            seti(kWqGroup, group);
            seti(kWqNOut, 0);
            seti(kWqLayout, 0);
            seti(kWqScales, scaleId);
        }
    }

    std::vector<float> runCpu(Graph g, const std::vector<float> &adata, int64_t K) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        IOTensor in;
        in.name  = "a";
        in.shape = {1, K};
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

    std::vector<float> testInput(int64_t K) {
        std::vector<float> a((size_t) K);
        for (int64_t i = 0; i < K; ++i)
        {
            a[(size_t) i] = std::cos(0.11f * (float) i) + 0.3f * std::sin(0.05f * (float) i * (float) i);
        }
        return a;
    }

    std::string readMagic(const std::string &path) {
        FILE *f = fopen(path.c_str(), "rb");
        EXPECT_TRUE(f);
        char magic[5] = {};
        EXPECT_EQ(fread(magic, 1, 4, f), 4u);
        fclose(f);
        return std::string(magic, 4);
    }

} // namespace

// The packed int8 byte layout round-trips every value through int8Pack/int8At, including the
// word-padding tail of a ragged N (padding bytes read 0).
TEST(QuantWeights, Int8PackUnpackRoundTrip) {
    const int64_t       K = 3, N = 11; // ragged: rowBytes pads 11 bytes to 12 (3 words)
    std::vector<int8_t> q((size_t) (K * N));
    for (int64_t i = 0; i < K * N; ++i)
    {
        q[(size_t) i] = (int8_t) ((i * 37) % 255 - 127); // spans the full [-127, 127] alphabet
    }
    std::vector<uint8_t> packed = int8Pack(q, K, N);
    EXPECT_EQ((int64_t) packed.size(), K * int8RowBytes(N));
    for (int64_t k = 0; k < K; ++k)
    {
        for (int64_t n = 0; n < N; ++n)
        {
            EXPECT_EQ(int8At(packed.data(), int8RowBytes(N), k, n), (int) q[(size_t) (k * N + n)]) << "k=" << k << " n=" << n;
        }
        for (int64_t n = N; n < int8RowBytes(N); ++n)
        {
            EXPECT_EQ(int8At(packed.data(), int8RowBytes(N), k, n), 0) << "padding byte k=" << k << " n=" << n;
        }
    }
}

// int8Dequant applies the per-(group, column) fp16 scale to each byte and overwrites outlier rows
// with their kept fp16 values — the int4Dequant contract at 8 bits.
TEST(QuantWeights, Int8DequantAppliesScalesAndOutliers) {
    const int64_t       K = 4, N = 8, group = 2;
    std::vector<int8_t> q((size_t) (K * N));
    for (int64_t i = 0; i < K * N; ++i)
    {
        q[(size_t) i] = (int8_t) ((i * 11) % 200 - 100);
    }
    std::vector<uint8_t>  packed = int8Pack(q, K, N);
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
    std::vector<float> w = int8Dequant(packed.data(), scales.data(), oidx, oval.data(), K, N, group, 1);
    for (int64_t k = 0; k < K; ++k)
    {
        for (int64_t n = 0; n < N; ++n)
        {
            float want = k == 2 ? 10.0f + (float) n : (float) q[(size_t) (k * N + n)] * halfToFloat(scales[(size_t) ((k / group) * N + n)]);
            EXPECT_FLOAT_EQ(w[(size_t) (k * N + n)], want) << "k=" << k << " n=" << n;
        }
    }
}

// lut4Dequant reads the payload as UNSIGNED 4-bit codebook indices and applies
// codebook[index] * scale, with outlier rows overwritten like every format.
TEST(QuantWeights, Lut4DequantAppliesCodebookAndScales) {
    const int64_t       K = 2, N = 16, group = 2; // one group; every index value appears once per row
    std::vector<int8_t> idx((size_t) (K * N));
    for (int64_t k = 0; k < K; ++k)
    {
        for (int64_t n = 0; n < N; ++n)
        {
            idx[(size_t) (k * N + n)] = (int8_t) ((n + k) % 16); // 0..15, int4Pack keeps the low nibble
        }
    }
    std::vector<uint8_t>  packed = int4Pack(idx, K, N);
    std::vector<uint16_t> codebook(16);
    for (int i = 0; i < 16; ++i)
    {
        codebook[(size_t) i] = floatToHalf(-1.0f + 0.125f * (float) i); // exact fp16 entries
    }
    std::vector<uint16_t> scales((size_t) (int4GroupCount(K, group) * N));
    for (size_t i = 0; i < scales.size(); ++i)
    {
        scales[i] = floatToHalf(2.0f);
    }
    std::vector<float> w = lut4Dequant(packed.data(), codebook.data(), scales.data(), nullptr, nullptr, K, N, group, 0);
    for (int64_t k = 0; k < K; ++k)
    {
        for (int64_t n = 0; n < N; ++n)
        {
            const int   ci   = (int) ((n + k) % 16);
            const float want = halfToFloat(codebook[(size_t) ci]) * 2.0f;
            EXPECT_FLOAT_EQ(w[(size_t) (k * N + n)], want) << "k=" << k << " n=" << n;
        }
    }
}

// A graph carrying an int8-packed weight stamps VXM6, reloads, materializes to a plain fp16 weight
// through the format dispatch, and the CPU run tracks the fp32 reference inside the (tight) int8
// error envelope. Int4-only content keeps VXM5 — VXM6 exists exactly so pre-VXM6 engines reject
// extended-format containers instead of dequantizing them as int4.
TEST(QuantWeights, Int8ContentStampsVxm6AndRoundTrips) {
    const int64_t K = 512, N = 96;
    const auto    a   = testInput(K);
    const auto    ref = runCpu(matmulGraph(K, N), a, K);

    Graph g = matmulGraph(K, N);
    runStandardPasses(g);
    packWeightInt8ByHand(g, K, N, 128);
    convertInitializersFp16(g);
    const std::string path = testing::TempDir() + "vknn_quant_weights_i8.vxm";
    ASSERT_TRUE(saveGraphBin(g, path));
    EXPECT_EQ(readMagic(path), "VXM6");

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
    // Hand-packed round-to-nearest int8 at group 128 sits near 0.3% weight-space error; 0.05 is the
    // garbage detector (a byte-layout or dispatch bug lands near 1).
    EXPECT_LT(relL2(got, ref), 0.05) << "int8 CPU run drifted from the fp32 reference";
}

// materializeQuantWeights on an int8 graph reconstructs a plain fp16 weight: attrs stripped, side
// tensors dropped, payload sized to the logical desc.
TEST(QuantWeights, MaterializeReconstructsInt8Weight) {
    const int64_t K = 256, N = 64;
    Graph         g = matmulGraph(K, N);
    runStandardPasses(g);
    packWeightInt8ByHand(g, K, N, 128);
    const int64_t materialized = materializeQuantWeights(g, nullptr);
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
        EXPECT_EQ(g.desc(kv.first).name.find("#i8s"), std::string::npos) << "side tensor payload survived materialization";
    }
}

// An unimplemented kWq format id fails the load loudly (Error(Unsupported)) — a reader must never
// fall back to another format's decode.
TEST(QuantWeights, UnknownFormatIdFailsLoudly) {
    const int64_t K = 256, N = 64;
    Graph         g = matmulGraph(K, N);
    runStandardPasses(g);
    packWeightInt8ByHand(g, K, N, 128);
    for (Node &nd: g.nodes)
    {
        if (nd.attr.has(kWq))
        {
            nd.attr.map[kWq].i = 9; // a format id this build does not implement
        }
    }
    EXPECT_THROW(materializeQuantWeights(g, nullptr), Error);
}

// The -Os pass at --quant-bits 8 quantizes an eligible MatMul weight to format kWqFormatInt8: the
// payload packs to K*int8RowBytes, the side initializers appear with the declared extents, and the
// desc keeps its logical shape with an fp16 dtype stamp — the int4 pass contract at 8 bits.
TEST(QuantWeights, PassQuantizesEligibleMatMulInt8) {
    const int64_t K = 256, N = 64;
    Graph         g = matmulGraph(K, N);
    runStandardPasses(g);
    QuantOptions opt;
    opt.bits      = 8;
    QuantStats st = quantizeWeights(g, opt);
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
    EXPECT_EQ(mm->attr.geti(kWq, 0), kWqFormatInt8);
    EXPECT_EQ(mm->attr.geti(kWqK, 0), K);
    EXPECT_EQ(mm->attr.geti(kWqN, 0), N);
    EXPECT_EQ(mm->attr.geti(kWqLayout, -1), 0);
    const int64_t nOut = mm->attr.geti(kWqNOut, -1);
    EXPECT_EQ(nOut, (int64_t) ((double) K * opt.outlierFrac));
    const TensorId w = mm->inputs[1];
    EXPECT_EQ((int64_t) g.initializers.at(w).bytes.size(), K * int8RowBytes(N));
    EXPECT_EQ(g.desc(w).dtype, DType::Float16);
    EXPECT_EQ(g.desc(w).shape, (Shape {K, N}));
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

// End-to-end through the exact compile pipeline at 8 bits (-Os --quant-bits 8 order: passes,
// quantize, fp16 sweep, save): the .vxm stamps VXM6, loads back, and the materialized CPU run
// tracks the fp32 reference inside the int8 error envelope — an order of magnitude tighter than
// int4's 0.25 (the same low-rank sine fixture measures ~4e-3 at 8 bits vs ~1e-1 at 4; the 0.02
// bound keeps 5x headroom while any packing/scale-indexing bug still lands near 1).
TEST(QuantWeights, Int8PassVxmRoundTripsAndTracksFp32) {
    const int64_t K = 512, N = 96;
    const auto    a   = testInput(K);
    const auto    ref = runCpu(matmulGraph(K, N), a, K);
    Graph         g   = matmulGraph(K, N);
    runStandardPasses(g);
    QuantOptions opt;
    opt.bits      = 8;
    QuantStats st = quantizeWeights(g, opt);
    EXPECT_EQ(st.quantized, 1);
    convertInitializersFp16(g);
    const std::string path = testing::TempDir() + "vknn_quant_weights_i8_e2e.vxm";
    ASSERT_TRUE(saveGraphBin(g, path));
    EXPECT_EQ(readMagic(path), "VXM6");
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
    const double rel = relL2(got, ref);
    printf("[QuantWeights] int8 e2e relL2 = %.6f\n", rel);
    EXPECT_LT(rel, 0.02) << "int8 quantized CPU run drifted from the fp32 reference";
}

// A weight whose every value is an int8 multiple of a power-of-two step must survive 8-bit
// quantization EXACTLY, mirroring the int4 bit gate: the min-MSE search finds the zero-cost step
// (maxAbs/127 = the true step, exact in fp16), every value requantizes to itself, and the
// materialized fp16 payload equals the fp16 sweep of the original — so the quantized CPU run is
// bit-identical to the plain fp16 compile. Outliers are off: an outlier row is kept fp16 verbatim
// (exact by construction), so the gate pins the quantized grid itself — and each group's anchor
// row (k % group == 0, all +127*step) then pins maxAbs to exactly 127 steps in every group-column
// (a 255-letter alphabet cannot cover a 128-row group the way int4's 15-letter one does).
TEST(QuantWeights, Int8ExactlyRepresentableWeightsRoundTripExactly) {
    const int64_t K = 256, N = 64;
    const float   step = 0.125f; // 2^-3: exact in fp16, and q*step is exact for q in [-127,127]
    QuantOptions  opt;
    opt.bits        = 8;
    opt.outlierFrac = 0.0;
    auto buildExact = [&] {
        Graph    g = matmulGraph(K, N);
        TensorId w = g.find("w");
        float   *v = g.initializers.at(w).f32();
        for (int64_t k = 0; k < K; ++k)
        {
            for (int64_t n = 0; n < N; ++n)
            {
                const int64_t i = k * N + n;
                v[i]            = k % opt.group == 0 ? 127.0f * step // the group's maxAbs anchor
                                                       :
                                                       (float) ((i * 7919) % 255 - 127) * step;
            }
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
    ASSERT_EQ(quantizeWeights(g, opt).quantized, 1);
    convertInitializersFp16(g);
    const auto got = runCpu(std::move(g), a, K);
    ASSERT_EQ(got.size(), ref.size());
    for (size_t i = 0; i < got.size(); ++i)
    {
        EXPECT_EQ(got[i], ref[i]) << "output " << i << " not bit-identical on exactly representable weights";
    }
}

// The mixed-precision guard holds at 8 bits: with an impossible error bar every layer stays fp16
// and the graph is structurally untouched.
TEST(QuantWeights, Int8GuardKeepsHostileLayerFp16) {
    const int64_t K = 256, N = 64;
    Graph         g = matmulGraph(K, N);
    runStandardPasses(g);
    const size_t tensorsBefore = g.tensors.size();
    QuantOptions opt;
    opt.bits           = 8;
    opt.maxLayerRelErr = 1e-9; // below int8's floor: the guard must keep the layer fp16
    QuantStats st      = quantizeWeights(g, opt);
    EXPECT_EQ(st.quantized, 0);
    EXPECT_EQ(st.guardKept, 1);
    EXPECT_EQ(g.tensors.size(), tensorsBefore);
    for (const Node &nd: g.nodes)
    {
        EXPECT_FALSE(nd.attr.has(kWq));
    }
}

// The -Os pass at --quant-bits lut4 quantizes to format kWqFormatLut4: nibble indices into a
// fitted 16-entry fp16 codebook whose entry 0 is EXACT zero (the padding/outlier contract the GPU
// kernels rely on), outlier rows packing index 0, and the e2e .vxm stamping VXM6 with a CPU run
// inside the int4-class envelope.
TEST(QuantWeights, Lut4PassQuantizesAndRoundTrips) {
    const int64_t K = 512, N = 96;
    const auto    a   = testInput(K);
    const auto    ref = runCpu(matmulGraph(K, N), a, K);
    Graph         g   = matmulGraph(K, N);
    runStandardPasses(g);
    QuantOptions opt;
    opt.lut4      = true;
    QuantStats st = quantizeWeights(g, opt);
    EXPECT_EQ(st.quantized, 1);

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
    EXPECT_EQ(mm->attr.geti(kWq, 0), kWqFormatLut4);
    const TensorId w = mm->inputs[1];
    EXPECT_EQ((int64_t) g.initializers.at(w).bytes.size(), K * int4RowBytes(N));
    const TensorId lutId = (TensorId) mm->attr.geti(kWqLut, kNoTensor);
    ASSERT_NE(lutId, kNoTensor);
    ASSERT_EQ(g.initializers.at(lutId).bytes.size(), 32u);
    const uint16_t *codebook = reinterpret_cast<const uint16_t *>(g.initializers.at(lutId).bytes.data());
    EXPECT_EQ(codebook[0], floatToHalf(0.0f)) << "codebook[0] must be the exact zero entry";

    // Outlier rows pack index 0 (the zero entry): the GPU kernels never skip outlier k rows in the
    // packed loop — they rely on the grid contribution being zero there.
    const int64_t nOut = mm->attr.geti(kWqNOut, 0);
    ASSERT_GT(nOut, 0);
    const TensorId oidxId = (TensorId) mm->attr.geti(kWqOidx, kNoTensor);
    const int32_t *oidx   = reinterpret_cast<const int32_t *>(g.initializers.at(oidxId).bytes.data());
    const uint8_t *packed = g.initializers.at(w).bytes.data();
    for (int64_t j = 0; j < nOut; ++j)
    {
        for (int64_t n = 0; n < N; ++n)
        {
            ASSERT_EQ(int4At(packed, int4RowBytes(N), oidx[j], n) & 0xF, 0) << "outlier row " << oidx[j] << " n=" << n;
        }
    }

    convertInitializersFp16(g);
    const std::string path = testing::TempDir() + "vknn_quant_weights_lut4_e2e.vxm";
    ASSERT_TRUE(saveGraphBin(g, path));
    EXPECT_EQ(readMagic(path), "VXM6");
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
    const double rel = relL2(got, ref);
    printf("[QuantWeights] lut4 e2e relL2 = %.6f\n", rel);
    EXPECT_LT(rel, 0.25) << "lut4 quantized CPU run drifted from the fp32 reference";
}

// The mixed-precision guard holds for LUT4: with an impossible error bar every layer stays fp16.
TEST(QuantWeights, Lut4GuardKeepsHostileLayerFp16) {
    const int64_t K = 256, N = 64;
    Graph         g = matmulGraph(K, N);
    runStandardPasses(g);
    QuantOptions opt;
    opt.lut4           = true;
    opt.maxLayerRelErr = 1e-9;
    QuantStats st      = quantizeWeights(g, opt);
    EXPECT_EQ(st.quantized, 0);
    EXPECT_EQ(st.guardKept, 1);
    for (const Node &nd: g.nodes)
    {
        EXPECT_FALSE(nd.attr.has(kWq));
    }
}

// materializeQuantWeights on a LUT4 graph reconstructs a plain fp16 weight, dropping the codebook
// side tensor with the rest.
TEST(QuantWeights, MaterializeReconstructsLut4Weight) {
    const int64_t K = 256, N = 64;
    Graph         g = matmulGraph(K, N);
    runStandardPasses(g);
    QuantOptions opt;
    opt.lut4 = true;
    ASSERT_EQ(quantizeWeights(g, opt).quantized, 1);
    EXPECT_EQ(materializeQuantWeights(g, nullptr), 1);
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
        for (const char *suffix: {"#l4s", "#l4oi", "#l4ov", "#l4cb"})
        {
            EXPECT_EQ(g.desc(kv.first).name.find(suffix), std::string::npos) << "side tensor payload survived materialization";
        }
    }
}

// quantizeWeights rejects a width it does not implement instead of guessing.
TEST(QuantWeights, UnsupportedBitsRejected) {
    Graph g = matmulGraph(256, 64);
    runStandardPasses(g);
    QuantOptions opt;
    opt.bits = 5;
    EXPECT_THROW(quantizeWeights(g, opt), Error);
}

// --- Multi-bucket weight identity ----------------------------------------------------------------
//
// The buckets of one compile are the SAME model at different input shapes, so a weight they share
// has to be one weight. Calibration statistics are shape-dependent (the synthetic samples are sized
// by the graph's input shapes), so quantizing each bucket on its own gives that weight a different
// packed payload per bucket: the decode and prefill buckets stop being the same model, and the
// .vxm's content-deduped initializer pool has to store both copies.

namespace {

    // A whole file as bytes (empty on failure), for the container-size comparison below.
    std::vector<uint8_t> readWholeFile(const std::string &path) {
        FILE *f = fopen(path.c_str(), "rb");
        if (!f)
        {
            return {};
        }
        fseek(f, 0, SEEK_END);
        const long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<uint8_t> bytes((size_t) (n > 0 ? n : 0));
        if (!bytes.empty())
        {
            bytes.resize(fread(bytes.data(), 1, bytes.size(), f));
        }
        fclose(f);
        return bytes;
    }

    // The tensor named `name` holds byte-identical payloads in both buckets — the exact condition under
    // which the .vxm's pool stores one blob instead of one per bucket.
    void expectSamePayload(const Graph &a, const Graph &b, const std::string &name) {
        const TensorId ia = a.find(name), ib = b.find(name);
        ASSERT_NE(ia, kNoTensor) << name << " is missing from the first bucket";
        ASSERT_NE(ib, kNoTensor) << name << " is missing from the second bucket";
        const ByteStorage &pa = a.initializers.at(ia).bytes;
        const ByteStorage &pb = b.initializers.at(ib).bytes;
        ASSERT_EQ(pa.size(), pb.size()) << name << " differs in payload size across the buckets";
        EXPECT_EQ(0, std::memcmp(pa.data(), pb.data(), pa.size())) << name << " differs in payload bytes across the buckets";
    }

    // The int-valued attribute `key` of the single MatMul, for cross-bucket comparison.
    int64_t matmulAttr(const Graph &g, const char *key) {
        for (const Node &nd: g.nodes)
        {
            if (nd.type == OpType::MatMul)
            {
                return nd.attr.geti(key, -1);
            }
        }
        return -1;
    }

} // namespace

// A weight two buckets of one compile share quantizes ONCE: the decode bucket (one activation row
// per sample) reuses the prefill bucket's payload instead of calibrating its own, and the packed
// weight, the group scales, and the outlier columns come out byte-identical.
TEST(QuantWeights, SharedWeightQuantizesIdenticallyAcrossBuckets) {
    const int64_t K = 512, N = 96;
    Graph         decode  = matmulGraph(K, N, 1);  // input_ids [1, 1]-style: one row per sample
    Graph         prefill = matmulGraph(K, N, 64); // the prefill bucket: 64 rows per sample
    runStandardPasses(decode);
    runStandardPasses(prefill);
    std::vector<Graph *>          buckets {&decode, &prefill};
    const std::vector<QuantStats> stats = quantizeWeightsShared(buckets, QuantOptions {});
    ASSERT_EQ(stats.size(), buckets.size());
    EXPECT_EQ(stats[0].quantized, 1);
    EXPECT_EQ(stats[1].quantized, 1);
    // The longest-activation bucket calibrates and the other one reuses its result, whatever order
    // the buckets are declared in: the richest statistics are the ones the shared weight carries.
    EXPECT_TRUE(stats[1].calibrated);
    EXPECT_EQ(stats[1].shared, 0);
    EXPECT_EQ(stats[0].shared, 1);
    EXPECT_FALSE(stats[0].calibrated) << "the decode bucket recalibrated a weight it should have reused";
    // The payload and every side tensor: identical bytes, so the pool holds one copy.
    expectSamePayload(decode, prefill, "w");
    expectSamePayload(decode, prefill, "w#i4s");
    ASSERT_GT(matmulAttr(prefill, kWqNOut), 0) << "the fixture must select outlier columns for the outlier check to bite";
    expectSamePayload(decode, prefill, "w#i4oi");
    expectSamePayload(decode, prefill, "w#i4ov");
    // ... and the attribute set the runtime dequantizes with agrees too.
    for (const char *key: {kWq, kWqK, kWqN, kWqGroup, kWqNOut, kWqLayout})
    {
        EXPECT_EQ(matmulAttr(decode, key), matmulAttr(prefill, key)) << key << " differs across the buckets";
    }
    // The installed quantization is a working one, not just matching bytes: materializing the shared
    // bucket reconstructs the weight inside the int4 envelope, which a payload wired to the wrong
    // group size or scale tensor could not.
    Graph reconstructed = decode;
    EXPECT_EQ(materializeQuantWeights(reconstructed, nullptr), 1);
    const std::vector<float> got = initFloats(reconstructed, reconstructed.find("w"));
    std::vector<float>       want((size_t) (K * N));
    for (int64_t k = 0; k < K; ++k)
    {
        for (int64_t n = 0; n < N; ++n)
        {
            want[(size_t) (k * N + n)] = std::sin(0.37f * (float) k + 1.13f * (float) n);
        }
    }
    EXPECT_LT(relL2(got, want), 0.25);
}

// The size consequence of that identity: a second bucket costs the .vxm its graph metadata, not a
// second copy of the weights. Quantized per bucket the payloads differ, the content-deduped pool
// cannot collapse them, and the two-bucket file carries nearly two full weight sets.
TEST(QuantWeights, SecondBucketAddsGraphMetadataNotWeights) {
    const int64_t K = 512, N = 96;
    Graph         decode  = matmulGraph(K, N, 1);
    Graph         prefill = matmulGraph(K, N, 64);
    runStandardPasses(decode);
    runStandardPasses(prefill);
    std::vector<Graph *> refs {&decode, &prefill};
    quantizeWeightsShared(refs, QuantOptions {});
    convertInitializersFp16(decode);
    convertInitializersFp16(prefill);
    const std::string onePath = testing::TempDir() + "vknn_quant_shared_one.vxm";
    const std::string twoPath = testing::TempDir() + "vknn_quant_shared_two.vxm";
    ASSERT_TRUE(saveGraphBin(decode, onePath));
    ASSERT_TRUE(saveGraphBinBuckets({decode, prefill}, {"decode", "prefill"}, twoPath));
    const std::vector<uint8_t> one = readWholeFile(onePath);
    const std::vector<uint8_t> two = readWholeFile(twoPath);
    std::remove(onePath.c_str());
    std::remove(twoPath.c_str());
    ASSERT_FALSE(one.empty());
    ASSERT_FALSE(two.empty());
    // Weights are the bulk of this fixture, so a second stored copy would push the ratio toward 2.
    EXPECT_LT((double) two.size(), 1.25 * (double) one.size()) << "two-bucket .vxm is " << two.size() << " bytes against a single bucket's " << one.size() << ": the buckets' weights are not deduping";
}

// One bucket is the single-graph pass exactly: sharing is a CROSS-bucket rule, so a lone graph
// quantizes to the same bytes quantizeWeights produces, with nothing shared.
TEST(QuantWeights, SharedQuantizationOfOneBucketMatchesTheSingleGraphPass) {
    const int64_t K = 512, N = 96;
    Graph         alone = matmulGraph(K, N);
    Graph         solo  = matmulGraph(K, N);
    runStandardPasses(alone);
    runStandardPasses(solo);
    std::vector<Graph *>          buckets {&alone};
    const std::vector<QuantStats> stats = quantizeWeightsShared(buckets, QuantOptions {});
    const QuantStats              ref   = quantizeWeights(solo, QuantOptions {});
    ASSERT_EQ(stats.size(), 1u);
    EXPECT_EQ(stats[0].quantized, ref.quantized);
    EXPECT_EQ(stats[0].shared, 0);
    expectSamePayload(alone, solo, "w");
    expectSamePayload(alone, solo, "w#i4s");
    expectSamePayload(alone, solo, "w#i4oi");
    expectSamePayload(alone, solo, "w#i4ov");
}

// A weight the calibration bucket does not hold (a second model's tower compiled into the same
// .vxm) is not left fp16: the first bucket that does hold it quantizes it on its own statistics.
TEST(QuantWeights, WeightAbsentFromTheCalibrationBucketStillQuantizes) {
    const int64_t K = 512, N = 96;
    Graph         wide   = matmulGraph(K, N, 64); // calibrates first: the most activation rows
    Graph         narrow = matmulGraph(K, N, 1);
    runStandardPasses(wide);
    runStandardPasses(narrow);
    // Rename the second bucket's weight: same payload, different tensor, so it is genuinely absent
    // from the bucket that calibrates.
    const TensorId other = narrow.find("w");
    ASSERT_NE(other, kNoTensor);
    narrow.tensorByName.erase("w");
    narrow.desc(other).name        = "w_other";
    narrow.tensorByName["w_other"] = other;
    std::vector<Graph *>          buckets {&wide, &narrow};
    const std::vector<QuantStats> stats = quantizeWeightsShared(buckets, QuantOptions {});
    EXPECT_EQ(stats[0].quantized, 1);
    EXPECT_EQ(stats[1].quantized, 1);
    EXPECT_EQ(stats[0].shared, 0);
    EXPECT_EQ(stats[1].shared, 0) << "a weight the calibration bucket never held cannot be shared from it";
    EXPECT_TRUE(stats[1].calibrated) << "the bucket holding it must calibrate it itself";
    EXPECT_EQ(matmulAttr(narrow, kWq), kWqFormatInt4);
    EXPECT_NE(narrow.find("w_other#i4s"), kNoTensor);
}

// The VXM5/VXM6 subtag guard: a quantized container whose subcontainer tag is not 3 or 4 is
// rejected at load (the tag remaps to the bad-magic diagnostics), and a foreign future VXM version
// is rejected through the magic-prefix check — never parsed as some other container.
TEST(QuantWeights, UnknownSubtagsAndForeignVersionsRejected) {
    auto writeHeader = [](const std::string &path, uint32_t magic, bool withSubtag, uint32_t subtag) {
        FILE *f = fopen(path.c_str(), "wb");
        ASSERT_TRUE(f);
        fwrite(&magic, 4, 1, f);
        if (withSubtag)
        {
            fwrite(&subtag, 4, 1, f);
        }
        const uint32_t filler[4] = {0, 0, 0, 0}; // body bytes; never reached past the rejection
        fwrite(filler, 4, 4, f);
        fclose(f);
    };
    const std::string path = testing::TempDir() + "vknn_quant_weights_reject.vxm";
    Graph             g;
    writeHeader(path, 0x354d5856u, true, 7); // VXM5, unknown subtag
    EXPECT_FALSE(loadGraphBin(g, path));
    writeHeader(path, 0x364d5856u, true, 7); // VXM6, unknown subtag
    EXPECT_FALSE(loadGraphBin(g, path));
    writeHeader(path, 0x374d5856u, false, 0); // "VXM7": a future container version
    EXPECT_FALSE(loadGraphBin(g, path));
    std::remove(path.c_str());
}
