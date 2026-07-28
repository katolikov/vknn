// Host tests for the MessagePack model-cache codec (round-trip + rejection of non-MessagePack input).
#include "core/cache_codec.h"
#include <gtest/gtest.h>

using namespace vknn;

static CacheVariant makeVariant(const std::string &prec, bool flat) {
    CacheVariant v;
    v.precision                 = prec;
    v.flatLayout                = flat;
    v.gpuIslandFold             = true;
    v.matmulViewFold            = flat; // exercise a non-default value through the round-trip
    v.fusedAttention            = flat;
    v.fp32Tensors               = "/enc/Foo,/enc/Bar";
    v.winograd                  = 1;
    v.winogradVariant           = 2;
    v.winogradUnit              = 4;
    v.directConv3x3             = 1;
    v.pipeline                  = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02};
    v.weights["conv1.weight"]   = {1.5f, -2.25f, 3.0f, 0.0f};
    v.weights["conv2.bias"]     = {0.125f};
    v.tune["sig-a/64x3x3"]      = 128;
    v.tune["sig-b/1x1"]         = -1;
    v.tuneLevel["sig-a/64x3x3"] = 2; // measured at Heavy
    v.tuneLevel["sig-b/1x1"]    = 1; // measured at Fast
    return v;
}

// A fully-populated document survives encode -> decode byte-for-byte, and findVariant resolves a
// variant by its cache-affecting key alone (precision here) while rejecting an unmatched key.
TEST(CacheCodec, RoundTrip) {
    CacheDoc doc;
    doc.format            = kCacheFormat;
    doc.kernelHash        = "abc123def456";
    doc.vendorId          = 0x1234;
    doc.deviceId          = 0x5678;
    doc.driverVersion     = 0x25022739;
    doc.pipelineCacheUUID = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    doc.model             = "modelhash-deadbeef";
    doc.variants.push_back(makeVariant("low", true));
    doc.variants.push_back(makeVariant("high", false));

    std::vector<uint8_t> bytes = cacheEncode(doc);
    ASSERT_FALSE(bytes.empty());

    CacheDoc got;
    ASSERT_TRUE(cacheDecode(bytes.data(), bytes.size(), got));
    EXPECT_EQ(got.format, doc.format);
    EXPECT_EQ(got.kernelHash, doc.kernelHash);
    EXPECT_EQ(got.vendorId, doc.vendorId);
    EXPECT_EQ(got.deviceId, doc.deviceId);
    EXPECT_EQ(got.driverVersion, doc.driverVersion);
    EXPECT_EQ(got.pipelineCacheUUID, doc.pipelineCacheUUID);
    EXPECT_EQ(got.model, doc.model);
    ASSERT_EQ(got.variants.size(), 2u);

    for (size_t i = 0; i < 2; ++i)
    {
        const CacheVariant &a = doc.variants[i];
        const CacheVariant &b = got.variants[i];
        EXPECT_TRUE(a.sameKey(b));
        EXPECT_EQ(a.pipeline, b.pipeline);
        EXPECT_EQ(a.weights, b.weights); // exact float bytes preserved
        EXPECT_EQ(a.tune, b.tune);
        EXPECT_EQ(a.tuneLevel, b.tuneLevel); // append-only "tunelvl" companion survives the round trip
    }

    // findVariant keys on the cache-affecting config only.
    CacheVariant        key = makeVariant("high", false);
    const CacheVariant *hit = got.findVariant(key);
    ASSERT_NE(hit, nullptr);
    EXPECT_EQ(hit->precision, "high");
    CacheVariant miss = makeVariant("normal", false);
    EXPECT_EQ(got.findVariant(miss), nullptr);
    // Flipping a single pass hint (the decode-attention fusion) is a distinct variant key: a
    // fused-plan cache entry must never serve an unfused config or vice versa.
    CacheVariant faFlip   = makeVariant("high", false);
    faFlip.fusedAttention = !faFlip.fusedAttention;
    EXPECT_EQ(got.findVariant(faFlip), nullptr);
}

// Input that is not a well-formed MessagePack top-level map is rejected: a legacy "VKNNCAC1" container,
// null/empty input, and a truncated map all return false rather than yielding a partial document.
TEST(CacheCodec, RejectsLegacyAndGarbage) {
    CacheDoc got;
    // A legacy "VKNNCAC1..." container is not valid MessagePack top-level map -> rejected.
    const char *legacy = "VKNNCAC1\x10\x00\x00\x00somepipelinebytes";
    EXPECT_FALSE(cacheDecode((const uint8_t *) legacy, 30, got));
    // Empty / truncated input.
    EXPECT_FALSE(cacheDecode(nullptr, 0, got));
    const uint8_t truncated[] = {0x81, 0xA6, 'f', 'o'}; // map(1){ str6 "fo... truncated
    EXPECT_FALSE(cacheDecode(truncated, sizeof(truncated), got));
}

// The load-time graph-rewrite hints are key fields: a variant compiled with a fold/fusion pass off
// must never satisfy the default-on key (and vice versa), and the flag survives a round-trip — the
// stale-cache guard for a hint flip between runs.
TEST(CacheCodec, LoadTimePassHintsDistinguishVariants) {
    CacheVariant off = makeVariant("low", true);
    off.ropeFusion   = false;
    CacheVariant on  = makeVariant("low", true);
    EXPECT_FALSE(on.sameKey(off));

    CacheVariant viewOff   = makeVariant("low", true);
    viewOff.matmulViewFold = false;
    EXPECT_FALSE(on.sameKey(viewOff));

    CacheDoc doc;
    doc.variants.push_back(off);
    std::vector<uint8_t> bytes = cacheEncode(doc);
    CacheDoc             got;
    ASSERT_TRUE(cacheDecode(bytes.data(), bytes.size(), got));
    ASSERT_EQ(got.variants.size(), 1u);
    EXPECT_FALSE(got.variants[0].ropeFusion);
    EXPECT_TRUE(got.variants[0].sameKey(off));
    EXPECT_EQ(got.findVariant(on), nullptr);
}

// A document carrying zero variants round-trips: the variants array decodes empty and the scalar
// guard fields are still preserved.
TEST(CacheCodec, EmptyVariants) {
    CacheDoc doc;
    doc.format                 = kCacheFormat;
    doc.kernelHash             = "k";
    doc.model                  = "m";
    std::vector<uint8_t> bytes = cacheEncode(doc);
    CacheDoc             got;
    ASSERT_TRUE(cacheDecode(bytes.data(), bytes.size(), got));
    EXPECT_EQ(got.variants.size(), 0u);
    EXPECT_EQ(got.kernelHash, "k");
}
