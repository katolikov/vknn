// Host tests for the MessagePack model-cache codec (round-trip + rejection of non-MessagePack input).
#include "core/cache_codec.h"
#include <gtest/gtest.h>

using namespace vknn;

static CacheVariant makeVariant(const std::string &prec, bool flat) {
    CacheVariant v;
    v.precision       = prec;
    v.flatLayout      = flat;
    v.gpuIslandFold   = true;
    v.fp32Tensors     = "/enc/Foo,/enc/Bar";
    v.winograd        = 1;
    v.winogradVariant = 2;
    v.winogradUnit    = 4;
    v.directConv3x3   = 1;
    v.pipeline        = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02};
    v.weights["conv1.weight"] = {1.5f, -2.25f, 3.0f, 0.0f};
    v.weights["conv2.bias"]   = {0.125f};
    v.tune["sig-a/64x3x3"]    = 128;
    v.tune["sig-b/1x1"]       = -1;
    return v;
}

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
    }

    // findVariant keys on the cache-affecting config only.
    CacheVariant key = makeVariant("high", false);
    const CacheVariant *hit = got.findVariant(key);
    ASSERT_NE(hit, nullptr);
    EXPECT_EQ(hit->precision, "high");
    CacheVariant miss = makeVariant("normal", false);
    EXPECT_EQ(got.findVariant(miss), nullptr);
}

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

TEST(CacheCodec, EmptyVariants) {
    CacheDoc doc;
    doc.format     = kCacheFormat;
    doc.kernelHash = "k";
    doc.model      = "m";
    std::vector<uint8_t> bytes = cacheEncode(doc);
    CacheDoc             got;
    ASSERT_TRUE(cacheDecode(bytes.data(), bytes.size(), got));
    EXPECT_EQ(got.variants.size(), 0u);
    EXPECT_EQ(got.kernelHash, "k");
}
