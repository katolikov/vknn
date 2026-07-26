// Int8 KV-cache quantization (Hint::KvCacheQuant; src/core/kv_quant.h): the host codec —
// per-(token,head) absmax, fp16 scale, roundEven int8 codes — its NaN/zero/denormal edge
// behavior and byte-determinism; the fold-range/scale-buffer indexing rule
// (scaleIndex == destElem / headDim against kvFoldSlot/kvFoldRanges); the shared eligibility
// rule the Vulkan segment and both FusedAttention backends key off; the config/cache-variant
// plumbing; and the CPU-oracle stream (hint On vs Off: near-lossless cosine, engaged path,
// byte-deterministic across runs and sessions). The hint's default (Off) is covered by the entire
// existing suite running with the feature absent.
#include "core/cache_codec.h"
#include "core/kv_quant.h"
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/io_link.h"
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

using namespace vknn;

namespace {

    constexpr int64_t kB = 1, kKvHeads = 2, kGroup = 3, kHeads = kKvHeads * kGroup, kTokens = 7, kHd = 4;
    constexpr int64_t kSlots = kTokens - 1;

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

    // The with-past decode step whose caches are graph inputs feeding past ‖ new Concats — the
    // linked-decoder shape class. The load passes (foldMatMulViews -> fuseDecodeAttention ->
    // foldFusedAttentionKvConcat) turn it into the split-source FusedAttention whose past sources
    // are the cache inputs, i.e. the exact form the int8 scheme targets.
    Graph splitDecodeGraph() {
        Graph      g;
        TensorId   q     = addInput(g, "q", {kB, kHeads, 1, kHd});
        TensorId   kPast = addInput(g, "kpast", {kB, kKvHeads, kSlots, kHd});
        TensorId   kNew  = addInput(g, "knew", {kB, kKvHeads, 1, kHd});
        TensorId   vPast = addInput(g, "vpast", {kB, kKvHeads, kSlots, kHd});
        TensorId   vNew  = addInput(g, "vnew", {kB, kKvHeads, 1, kHd});
        TensorDesc kco;
        kco.name     = "present_key";
        kco.isOutput = true;
        TensorId kc  = g.addTensor(kco);
        TensorDesc vco;
        vco.name     = "present_value";
        vco.isOutput = true;
        TensorId vc  = g.addTensor(vco);
        Attr     axis;
        axis.kind                                                               = Attr::Int;
        axis.i                                                                  = 2;
        addNode(g, OpType::Concat, "kcat", {kPast, kNew}, kc)->attr.map["axis"] = axis;
        addNode(g, OpType::Concat, "vcat", {vPast, vNew}, vc)->attr.map["axis"] = axis;

        TensorId kT     = repeatKv(g, kc, "k", true);
        TensorId vR     = repeatKv(g, vc, "v", false);
        TensorId scores = addTemp(g, "scores");
        addNode(g, OpType::MatMul, "qk", {q, kT}, scores);
        TensorId probs       = addTemp(g, "probs");
        Node    *sn          = addNode(g, OpType::Softmax, "softmax", {scores}, probs);
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

    // splitDecodeGraph after the load passes: the split-source FusedAttention form.
    Graph foldedSplitDecodeGraph() {
        Graph g = splitDecodeGraph();
        inferShapes(g);
        foldMatMulViews(g);
        fuseDecodeAttention(g);
        foldFusedAttentionKvConcat(g);
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

    IOTensor randTensor(const std::string &name, const Shape &shape, unsigned seed) {
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

    std::vector<uint8_t> outBytes(const std::vector<IOTensor> &outs, const std::string &name) {
        for (const IOTensor &o: outs)
        {
            if (o.name == name)
            {
                return o.data;
            }
        }
        ADD_FAILURE() << "missing output " << name;
        return {};
    }

    double cosine(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) {
        EXPECT_EQ(a.size(), b.size());
        const float *fa = reinterpret_cast<const float *>(a.data());
        const float *fb = reinterpret_cast<const float *>(b.data());
        const size_t n  = a.size() / 4;
        double       dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += (double) fa[i] * fb[i];
            na += (double) fa[i] * fa[i];
            nb += (double) fb[i] * fb[i];
        }
        return na > 0 && nb > 0 ? dot / std::sqrt(na * nb) : 0.0;
    }

} // namespace

// Round-trip accuracy: every element lands within the code's half-step of the original (plus the
// fp16 rounding of the scale itself), and the row's largest magnitude uses the full code range.
TEST(KvQuant, CodecRoundTrip) {
    constexpr int64_t kRows = 8;
    std::vector<float> src((size_t) (kRows * kHd));
    unsigned           s = 7;
    for (float &v: src)
    {
        s = s * 1664525u + 1013904223u;
        v = ((float) (s >> 8) / (float) (1 << 24)) * 4.f - 2.f;
    }
    std::vector<int8_t> payload(src.size());
    std::vector<fp16_t> scaleBits((size_t) kRows);
    kvQuantRows(src.data(), kRows, kHd, payload.data(), scaleBits.data());
    std::vector<float> back(src.size());
    kvDequantRows(payload.data(), scaleBits.data(), kRows, kHd, back.data());
    for (int64_t r = 0; r < kRows; ++r)
    {
        const float scale = halfToFloat(scaleBits[(size_t) r]);
        ASSERT_GT(scale, 0.f);
        for (int64_t i = 0; i < kHd; ++i)
        {
            const size_t at = (size_t) (r * kHd + i);
            // Half a code step, plus the fp16 scale rounding acting on up-to-127 steps.
            EXPECT_NEAR(back[at], src[at], 0.6f * scale) << "row " << r << " elem " << i;
        }
    }
}

// Edge cases: all-zero and negative-zero rows encode zero scale + zero codes; a NaN element is
// skipped by the absmax and encodes 0; an infinite element saturates the ceiling and encodes 127;
// a denormal-absmax row never produces NaN. Ties round to even, matching the GPU's roundEven.
TEST(KvQuant, CodecEdgeCases) {
    // Zero and signed-zero rows.
    const float         zeros[kHd] = {0.f, -0.f, 0.f, -0.f};
    std::vector<int8_t> payload(kHd);
    std::vector<fp16_t> scaleBits(1);
    kvQuantRows(zeros, 1, kHd, payload.data(), scaleBits.data());
    EXPECT_EQ(scaleBits[0], (fp16_t) 0);
    for (int8_t code: payload)
    {
        EXPECT_EQ(code, 0);
    }

    // NaN skipped by absmax, encoded as 0; the finite elements stay exact code multiples.
    const float nanRow[kHd] = {std::nanf(""), 127.f, -127.f, 63.5f};
    kvQuantRows(nanRow, 1, kHd, payload.data(), scaleBits.data());
    const float nanScale = halfToFloat(scaleBits[0]);
    EXPECT_FLOAT_EQ(nanScale, halfToFloat(floatToHalf(1.f)));
    EXPECT_EQ(payload[0], 0);
    EXPECT_EQ(payload[1], 127);
    EXPECT_EQ(payload[2], -127);
    // 63.5 / 1.0 ties to even: 64.
    EXPECT_EQ(payload[3], 64);

    // Infinity saturates the fp16 ceiling; its code clamps to 127 and nothing is NaN.
    const float infRow[kHd] = {std::numeric_limits<float>::infinity(), 1.f, -2.f, 0.f};
    kvQuantRows(infRow, 1, kHd, payload.data(), scaleBits.data());
    EXPECT_FLOAT_EQ(halfToFloat(scaleBits[0]), halfToFloat(floatToHalf(kKvQuantAbsMaxCeil / kKvQuantMaxCode)));
    EXPECT_EQ(payload[0], 127);
    std::vector<float> back(kHd);
    kvDequantRows(payload.data(), scaleBits.data(), 1, kHd, back.data());
    for (float v: back)
    {
        EXPECT_TRUE(std::isfinite(v));
    }

    // A denormal absmax underflows the fp16 scale to zero -> zero codes, zero dequant, no NaN.
    const float denormRow[kHd] = {1e-42f, -1e-42f, 5e-43f, 0.f};
    kvQuantRows(denormRow, 1, kHd, payload.data(), scaleBits.data());
    EXPECT_EQ(scaleBits[0], (fp16_t) 0);
    kvDequantRows(payload.data(), scaleBits.data(), 1, kHd, back.data());
    for (float v: back)
    {
        EXPECT_EQ(v, 0.f);
    }

    // Ties-to-even at the half-step: with scale exactly 1.0, 0.5 -> 0, 1.5 -> 2, 2.5 -> 2.
    const float tieRow[kHd] = {127.f, 0.5f, 1.5f, 2.5f};
    kvQuantRows(tieRow, 1, kHd, payload.data(), scaleBits.data());
    EXPECT_FLOAT_EQ(halfToFloat(scaleBits[0]), 1.f);
    EXPECT_EQ(payload[1], 0);
    EXPECT_EQ(payload[2], 2);
    EXPECT_EQ(payload[3], 2);
}

// The codec is a pure function: two runs over the same bytes produce identical payload + scales.
TEST(KvQuant, CodecDeterministic) {
    constexpr int64_t  kRows = 16;
    std::vector<float> src((size_t) (kRows * kHd));
    unsigned           s = 1234;
    for (float &v: src)
    {
        s = s * 1664525u + 1013904223u;
        v = ((float) (s >> 8) / (float) (1 << 24)) - 0.5f;
    }
    std::vector<int8_t> payloadA(src.size()), payloadB(src.size());
    std::vector<fp16_t> scalesA((size_t) kRows), scalesB((size_t) kRows);
    kvQuantRows(src.data(), kRows, kHd, payloadA.data(), scalesA.data());
    kvQuantRows(src.data(), kRows, kHd, payloadB.data(), scalesB.data());
    EXPECT_EQ(std::memcmp(payloadA.data(), payloadB.data(), payloadA.size()), 0);
    EXPECT_EQ(std::memcmp(scalesA.data(), scalesB.data(), scalesA.size() * sizeof(fp16_t)), 0);
}

// The seed path pre-rounds through fp16 storage: on fp16-representable values it equals the plain
// codec (the production hand-off, whose mirror came from an fp16 buffer), and on arbitrary fp32
// values it equals the codec applied to the fp16-rounded values — exactly what a GPU fold of the
// packed rows would see.
TEST(KvQuant, SeedViaFp16MatchesFoldOfStoredRows) {
    constexpr int64_t  kRows = 4;
    std::vector<float> raw((size_t) (kRows * kHd)), stored(raw.size());
    unsigned           s = 99;
    for (size_t i = 0; i < raw.size(); ++i)
    {
        s         = s * 1664525u + 1013904223u;
        raw[i]    = ((float) (s >> 8) / (float) (1 << 24)) * 3.001f - 1.5f; // generally not fp16-exact
        stored[i] = halfToFloat(floatToHalf(raw[i]));
    }
    std::vector<int8_t> seedPayload(raw.size()), foldPayload(raw.size());
    std::vector<fp16_t> seedScales((size_t) kRows), foldScales((size_t) kRows);
    kvQuantRowsFromFp32ViaFp16(raw.data(), kRows, kHd, seedPayload.data(), seedScales.data());
    kvQuantRows(stored.data(), kRows, kHd, foldPayload.data(), foldScales.data());
    EXPECT_EQ(std::memcmp(seedPayload.data(), foldPayload.data(), seedPayload.size()), 0);
    EXPECT_EQ(std::memcmp(seedScales.data(), foldScales.data(), seedScales.size() * sizeof(fp16_t)), 0);
}

// The fold-range/scale-buffer indexing rule: applying kvFoldRanges (the engine's own fold driver)
// with the host codec lands each present row's payload at its (head, slot) row and its scale at
// destElem / headDim — the exact address link_copy_kvq.comp and the attention kernels use.
TEST(KvQuant, FoldRangesScaleIndexing) {
    constexpr int64_t kPresentRows = 1;
    const int64_t     slot         = kvFoldSlot(4, kSlots); // decode position 4 folds into slot 3
    ASSERT_EQ(slot, 3);
    const std::vector<LinkRange> ranges = kvFoldRanges(kKvHeads, kPresentRows, kSlots, kHd, slot);
    ASSERT_EQ(ranges.size(), (size_t) kKvHeads);

    // Present rows (the fold source), one per head.
    std::vector<float> present((size_t) (kKvHeads * kPresentRows * kHd));
    for (size_t i = 0; i < present.size(); ++i)
    {
        present[i] = 0.25f * (float) (i + 1);
    }
    // The int8 cache: payload [kvHeads, slots, hd], scales [kvHeads, slots].
    std::vector<int8_t> payload((size_t) (kKvHeads * kSlots * kHd), 0);
    std::vector<fp16_t> scales((size_t) (kKvHeads * kSlots), 0);
    for (const LinkRange &range: ranges)
    {
        ASSERT_EQ(range.count % kHd, 0);
        ASSERT_EQ(range.destElem % kHd, 0);
        ASSERT_EQ(range.sourceElem % kHd, 0);
        kvQuantRows(present.data() + range.sourceElem, range.count / kHd, kHd, payload.data() + range.destElem, scales.data() + range.destElem / kHd);
    }
    for (int64_t head = 0; head < kKvHeads; ++head)
    {
        const int64_t cacheRow = head * kSlots + slot;
        EXPECT_NE(scales[(size_t) cacheRow], (fp16_t) 0) << "head " << head;
        // Every other row of this head stays untouched (zero scale, zero codes).
        for (int64_t other = 0; other < kSlots; ++other)
        {
            if (other == slot)
            {
                continue;
            }
            EXPECT_EQ(scales[(size_t) (head * kSlots + other)], (fp16_t) 0);
        }
        // The folded row dequantizes back to the present row within the codec's half-step.
        const float scale = halfToFloat(scales[(size_t) cacheRow]);
        for (int64_t i = 0; i < kHd; ++i)
        {
            const float sourceValue = present[(size_t) (head * kPresentRows * kHd + i)];
            const float backValue   = kvQuantDecode(payload[(size_t) (cacheRow * kHd + i)], scale);
            EXPECT_NEAR(backValue, sourceValue, 0.6f * scale);
        }
    }
}

// The shared eligibility rule: the folded split-source decode form qualifies with the hint On,
// and every gate (hint, backend eligibility, flat requirement, split form, row-contiguous token
// strides) refuses independently.
TEST(KvQuant, EligibilityRules) {
    Graph       g  = foldedSplitDecodeGraph();
    const Node *fa = findNode(g, OpType::FusedAttention);
    ASSERT_NE(fa, nullptr);
    ASSERT_EQ(fa->attr.geti(kFaSplit, 0), 1);
    EXPECT_TRUE(kvQuantNodeEligible(g, *fa));

    Config cfgOn;
    cfgOn.setHint(Hint::KvCacheQuant, Mode::On);
    const std::set<TensorId> tensors = kvQuantCacheTensors(g, cfgOn, /*backendEligible=*/true, /*requireFlat=*/false);
    EXPECT_EQ(tensors.size(), 2u);
    EXPECT_TRUE(tensors.count(fa->inputs[1]));
    EXPECT_TRUE(tensors.count(fa->inputs[2]));

    // The unset default (Off), an explicit Off, and an ineligible backend each empty the set. An
    // explicit Auto also refuses here because this synthetic cache is far below
    // kKvQuantAutoMinCacheBytes — the size-driven Auto resolution has its own case below.
    Config cfgDefault;
    EXPECT_TRUE(kvQuantCacheTensors(g, cfgDefault, true, false).empty());
    Config cfgAuto;
    cfgAuto.setHint(Hint::KvCacheQuant, Mode::Auto);
    EXPECT_TRUE(kvQuantCacheTensors(g, cfgAuto, true, false).empty());
    Config cfgOff;
    cfgOff.setHint(Hint::KvCacheQuant, Mode::Off);
    EXPECT_TRUE(kvQuantCacheTensors(g, cfgOff, true, false).empty());
    EXPECT_TRUE(kvQuantCacheTensors(g, cfgOn, /*backendEligible=*/false, false).empty());
    // The Vulkan flat requirement: these test tensors carry no gpuFlat marking.
    EXPECT_TRUE(kvQuantCacheTensors(g, cfgOn, true, /*requireFlat=*/true).empty());

    // The unsplit form (no KV-concat fold) never qualifies.
    Graph gUnsplit = splitDecodeGraph();
    inferShapes(gUnsplit);
    foldMatMulViews(gUnsplit);
    fuseDecodeAttention(gUnsplit);
    const Node *faUnsplit = findNode(gUnsplit, OpType::FusedAttention);
    ASSERT_NE(faUnsplit, nullptr);
    EXPECT_FALSE(kvQuantNodeEligible(gUnsplit, *faUnsplit));
    EXPECT_TRUE(kvQuantCacheTensors(gUnsplit, cfgOn, true, false).empty());

    // A non-row-contiguous token stride (kN != headDim) breaks the scale-index rule and refuses.
    Graph gBadStride = foldedSplitDecodeGraph();
    for (Node &node: gBadStride.nodes)
    {
        if (node.type == OpType::FusedAttention)
        {
            Attr a;
            a.kind               = Attr::Int;
            a.i                  = kHd + 1;
            node.attr.map[kFaKN] = a;
            EXPECT_FALSE(kvQuantNodeEligible(gBadStride, node));
        }
    }
}

// Auto resolves on the segment's eligible cache size: the same graph is refused below the measured
// threshold and accepted at or above it, with no timing input — the decision is a pure function of
// the compiled shapes, so every load of one plan on one device picks the same path. Auto is an
// explicit opt-in (the unset default is Off), so every case here sets the hint.
TEST(KvQuant, AutoResolvesOnCacheSize) {
    Graph       g  = foldedSplitDecodeGraph();
    const Node *fa = findNode(g, OpType::FusedAttention);
    ASSERT_NE(fa, nullptr);
    const int64_t cacheBytes = kvQuantCacheTensorFp16Bytes(g, fa->inputs[1]) + kvQuantCacheTensorFp16Bytes(g, fa->inputs[2]);
    ASSERT_GT(cacheBytes, 0);
    ASSERT_LT(cacheBytes, kKvQuantAutoMinCacheBytes) << "the synthetic decode cache must sit below the Auto threshold";

    Config autoCfg;
    autoCfg.setHint(Hint::KvCacheQuant, Mode::Auto);
    EXPECT_TRUE(kvQuantCacheTensors(g, autoCfg, /*backendEligible=*/true, /*requireFlat=*/false).empty());
    // The unset default refuses the widened cache below too — it is Off, not Auto.
    Config unsetCfg;
    EXPECT_EQ(unsetCfg.kvCacheQuantMode(), (int) Mode::Off);

    // Widen the same cache past the threshold: Auto now engages on exactly the two past tensors.
    Graph wide = g;
    for (TensorId past: {fa->inputs[1], fa->inputs[2]})
    {
        Shape shape = wide.desc(past).shape;
        while (kvQuantCacheTensorFp16Bytes(wide, past) * 2 < kKvQuantAutoMinCacheBytes)
        {
            shape[shape.size() - 2] *= 2; // more cache slots, same row geometry
            wide.desc(past).shape = shape;
        }
    }
    const std::set<TensorId> engaged = kvQuantCacheTensors(wide, autoCfg, true, false);
    EXPECT_EQ(engaged.size(), 2u);
    EXPECT_TRUE(engaged.count(fa->inputs[1]));
    // An explicit Off still refuses the widened cache.
    Config offCfg;
    offCfg.setHint(Hint::KvCacheQuant, Mode::Off);
    EXPECT_TRUE(kvQuantCacheTensors(wide, offCfg, true, false).empty());
}

// Config plumbing and the cache-variant key: the JSON knob parses and round-trips, and a variant
// with a different kvCacheQuant value never matches (so flipping the hint reselects a variant
// instead of serving a stale plan) while the codec round-trips the field.
TEST(KvQuant, ConfigAndCacheVariantKey) {
    // Unset is Off: the scheme is lossy (measured perplexity cost in src/core/kv_quant.h) and is
    // never taken on a caller's behalf, so it takes an explicit "on" or "auto" to engage.
    EXPECT_EQ(Config().kvCacheQuantMode(), (int) Mode::Off);
    EXPECT_EQ(Config::fromJsonString("{}").kvCacheQuantMode(), (int) Mode::Off);
    Config on = Config::fromJsonString("{\"kvCacheQuant\": \"on\"}");
    EXPECT_EQ(on.kvCacheQuantMode(), (int) Mode::On);
    EXPECT_EQ(Config::fromJsonString("{\"kvCacheQuant\": \"off\"}").kvCacheQuantMode(), (int) Mode::Off);
    EXPECT_EQ(Config::fromJsonString("{\"kvCacheQuant\": \"auto\"}").kvCacheQuantMode(), (int) Mode::Auto);
    // toJson -> fromJsonString round-trips the knob, including the default and an explicit Auto.
    EXPECT_EQ(Config::fromJsonString(on.toJson()).kvCacheQuantMode(), (int) Mode::On);
    EXPECT_EQ(Config::fromJsonString(Config().toJson()).kvCacheQuantMode(), (int) Mode::Off);
    Config autoCfg;
    autoCfg.setHint(Hint::KvCacheQuant, Mode::Auto);
    EXPECT_EQ(Config::fromJsonString(autoCfg.toJson()).kvCacheQuantMode(), (int) Mode::Auto);

    // The resolution: On/Off are literal at any size; Auto engages only from the measured
    // cache-size threshold up, so a small cache keeps the fp16 path without any timing input; the
    // unset default keeps it at every size.
    Config offCfg = Config::fromJsonString("{\"kvCacheQuant\": \"off\"}");
    EXPECT_FALSE(kvQuantEnabled(autoCfg, kKvQuantAutoMinCacheBytes - 1));
    EXPECT_TRUE(kvQuantEnabled(autoCfg, kKvQuantAutoMinCacheBytes));
    EXPECT_TRUE(kvQuantEnabled(on, 0));
    EXPECT_FALSE(kvQuantEnabled(offCfg, kKvQuantAutoMinCacheBytes * 4));
    EXPECT_FALSE(kvQuantEnabled(Config(), kKvQuantAutoMinCacheBytes * 4));

    CacheVariant base, quantized;
    quantized.kvCacheQuant = (int) Mode::On;
    EXPECT_TRUE(base.sameKey(CacheVariant {}));
    EXPECT_FALSE(base.sameKey(quantized));

    CacheDoc doc;
    doc.variants.push_back(quantized);
    CacheDoc decoded;
    const std::vector<uint8_t> bytes = cacheEncode(doc);
    ASSERT_TRUE(cacheDecode(bytes.data(), bytes.size(), decoded));
    ASSERT_EQ(decoded.variants.size(), 1u);
    EXPECT_EQ(decoded.variants[0].kvCacheQuant, (int) Mode::On);
    EXPECT_TRUE(decoded.variants[0].sameKey(quantized));
}

// The CPU oracle stream: a decode step with the hint On tracks the fp16-cache stream near-
// losslessly (cosine), visibly engages the quantized path (bytes differ from Off), and is
// byte-deterministic across repeated runs and across sessions — the reference the GPU kvq path
// is validated against on device.
TEST(KvQuant, CpuOracleCosineAndDeterminism) {
    const std::vector<IOTensor> inputs = {
        randTensor("q", {kB, kHeads, 1, kHd}, 11),
        randTensor("kpast", {kB, kKvHeads, kSlots, kHd}, 12),
        randTensor("knew", {kB, kKvHeads, 1, kHd}, 13),
        randTensor("vpast", {kB, kKvHeads, kSlots, kHd}, 14),
        randTensor("vnew", {kB, kKvHeads, 1, kHd}, 15),
    };
    Config cfgOff;
    cfgOff.backend = BackendKind::Cpu;
    Config cfgOn;
    cfgOn.backend = BackendKind::Cpu;
    cfgOn.setHint(Hint::KvCacheQuant, Mode::On);

    auto sOff = Session::create(splitDecodeGraph(), cfgOff);
    auto sOn  = Session::create(splitDecodeGraph(), cfgOn);
    auto sOn2 = Session::create(splitDecodeGraph(), cfgOn);
    ASSERT_TRUE(sOff);
    ASSERT_TRUE(sOn);
    ASSERT_TRUE(sOn2);

    std::vector<IOTensor> outsOff, outsOn, outsOnAgain, outsOn2;
    ASSERT_EQ(sOff->run(inputs, outsOff), Status::Ok);
    ASSERT_EQ(sOn->run(inputs, outsOn), Status::Ok);
    ASSERT_EQ(sOn->run(inputs, outsOnAgain), Status::Ok);
    ASSERT_EQ(sOn2->run(inputs, outsOn2), Status::Ok);

    const std::vector<uint8_t> off = outBytes(outsOff, "out");
    const std::vector<uint8_t> on  = outBytes(outsOn, "out");
    ASSERT_FALSE(off.empty());
    ASSERT_FALSE(on.empty());
    // Near-lossless: 8-bit per-(token,head) symmetric quantization of the past barely moves the
    // context rows.
    EXPECT_GT(cosine(on, off), 0.999);
    // ... but it MUST move them: identical bytes would mean the quantized path never engaged.
    EXPECT_NE(on, off);
    // int8-domain determinism: repeated runs and a fresh session are byte-identical.
    EXPECT_EQ(on, outBytes(outsOnAgain, "out"));
    EXPECT_EQ(on, outBytes(outsOn2, "out"));
}
