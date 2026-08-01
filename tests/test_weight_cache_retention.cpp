// Host tests for the model-cache retention rules: the prepacked-weight cache's byte accounting and
// release (src/backend/vulkan/vk_weight_cache.h), and the cache-file image fingerprint + variant
// merge the backend saves through (src/backend/vulkan/vk_cache_image.h).
//
// Both are deliberately free of Vulkan handles -- the weight cache maps to/from one CacheVariant and
// the image rules work on raw bytes -- so the accounting the backend's reclaim is measured against is
// exercised here. The Vulkan-side wiring (loadCache keeping only the matched variant, saveCaches
// re-reading the file and releasing after a confirmed write) needs a device gate.
#include "backend/vulkan/vk_cache_image.h"
#include "backend/vulkan/vk_weight_cache.h"
#include "core/cache_codec.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    std::vector<float> blob(size_t elems, float first) {
        std::vector<float> v(elems);
        for (size_t i = 0; i < elems; ++i)
        {
            v[i] = first + (float) i;
        }
        return v;
    }

    // A variant carrying two prepacked weights and one autotune entry, as one session's save would.
    CacheVariant storedVariant(const std::string &precision) {
        CacheVariant v;
        v.precision               = precision;
        v.weights["/conv1#w"]     = blob(64, 1.f);
        v.weights["/conv2#wino"]  = blob(128, 2.f);
        v.tune["/conv1#sig"]      = 7;
        v.tuneLevel["/conv1#sig"] = 1;
        return v;
    }

} // namespace

// A retained prepacked blob is accounted in bytes, and the release drops the accounting to zero
// without disabling the cache: this is the reclaim point the backend reaches once a save has put the
// blobs in the cache file.
TEST(WeightCacheRetention, ReleaseDropsRetainedBytesButKeepsCollecting) {
    WeightCache cache;
    cache.reset(/*enabled=*/true);
    EXPECT_EQ(cache.retainedBytes(), 0u);

    cache.put("/conv1#w", blob(64, 1.f));
    cache.put("/conv2#wino", blob(128, 2.f));
    cache.setTuned("/conv1#sig", 7, 1);
    EXPECT_EQ(cache.retainedCount(), 2u);
    EXPECT_EQ(cache.retainedBytes(), (64u + 128u) * sizeof(float));
    EXPECT_TRUE(cache.dirty());

    // Marking the content saved is bookkeeping only: the blobs are still resident, and stay resident
    // for a save that could not be confirmed. The release is the reclaim point.
    cache.markSaved();
    EXPECT_EQ(cache.retainedBytes(), (64u + 128u) * sizeof(float));

    cache.releaseRetained();

    EXPECT_EQ(cache.retainedBytes(), 0u);
    EXPECT_EQ(cache.retainedCount(), 0u);
    EXPECT_FALSE(cache.dirty());
    // Still enabled, and the autotune table (key-sized) survives: the cache keeps serving and keeps
    // collecting whatever a later bucket produces.
    EXPECT_TRUE(cache.enabled());
    int level = -1;
    EXPECT_EQ(cache.tuned("/conv1#sig", -1, &level), 7);
    EXPECT_EQ(level, 1);

    std::vector<float> got;
    EXPECT_FALSE(cache.get("/conv1#w", got)); // a released blob misses; the op recomputes its prepack

    cache.put("/conv3#w", blob(32, 3.f));
    EXPECT_TRUE(cache.dirty());
    EXPECT_EQ(cache.retainedBytes(), 32u * sizeof(float));
}

// What accumulates after a release is a DELTA, and writeInto emits exactly that delta. Merging it
// into the stored variant reproduces the full table, which is what lets the backend release and still
// write a complete cache file later.
TEST(WeightCacheRetention, DeltaAfterReleaseMergesBackToTheFullTable) {
    WeightCache cache;
    cache.reset(/*enabled=*/true);
    cache.put("/conv1#w", blob(64, 1.f));
    cache.put("/conv2#wino", blob(128, 2.f));
    cache.setTuned("/conv1#sig", 7, 1);

    CacheVariant firstSave;
    cache.writeInto(firstSave);
    ASSERT_EQ(firstSave.weights.size(), 2u);
    cache.markSaved();
    cache.releaseRetained();

    // A bucket added later produces one new prepack and re-tunes one signature.
    cache.put("/conv3#w", blob(32, 3.f));
    cache.setTuned("/conv1#sig", 9, 2);
    CacheVariant delta;
    cache.writeInto(delta);
    EXPECT_EQ(delta.weights.size(), 1u) << "only what was produced since the release is retained";

    mergeSessionArtifacts(firstSave, std::move(delta));

    ASSERT_EQ(firstSave.weights.size(), 3u);
    EXPECT_EQ(firstSave.weights["/conv1#w"], blob(64, 1.f));
    EXPECT_EQ(firstSave.weights["/conv2#wino"], blob(128, 2.f));
    EXPECT_EQ(firstSave.weights["/conv3#w"], blob(32, 3.f));
    EXPECT_EQ(firstSave.tune["/conv1#sig"], 9); // the re-tune wins
    EXPECT_EQ(firstSave.tuneLevel["/conv1#sig"], 2);
}

// loadFrom consumes the variant it is given: a warm start moves the decoded weight map in rather than
// leaving a second copy behind in the document it came from.
TEST(WeightCacheRetention, WarmLoadMovesTheDecodedWeights) {
    CacheVariant stored = storedVariant("low");
    WeightCache  cache;
    cache.loadFrom(std::move(stored));

    EXPECT_EQ(cache.retainedCount(), 2u);
    EXPECT_EQ(cache.retainedBytes(), (64u + 128u) * sizeof(float));
    EXPECT_TRUE(cache.enabled());
    EXPECT_FALSE(cache.dirty()) << "a warm load owes the file nothing";
    EXPECT_TRUE(stored.weights.empty()) << "the decoded map moved rather than doubling";

    std::vector<float> got;
    ASSERT_TRUE(cache.get("/conv1#w", got));
    EXPECT_EQ(got, blob(64, 1.f));
}

// A cache written without a persistent file retains nothing: put() still records the blob for the
// caller that asked for it, but the enabled flag is what the upload path consults before putting.
TEST(WeightCacheRetention, PathlessCacheIsNotEnabled) {
    WeightCache cache;
    cache.reset(/*enabled=*/false);
    EXPECT_FALSE(cache.enabled());
    EXPECT_EQ(cache.retainedBytes(), 0u);
}

// The image fingerprint answers the save path's only question about the previous file: equal bytes
// compare equal, and any single-byte, length or ordering change compares different.
TEST(CacheImage, FingerprintDistinguishesImages) {
    std::vector<uint8_t> image(1024);
    for (size_t i = 0; i < image.size(); ++i)
    {
        image[i] = (uint8_t) ((i * 31 + 7) & 0xFF);
    }
    const CacheImageFingerprint base = fingerprintCacheImage(image.data(), image.size());
    EXPECT_EQ(base.byteCount, image.size());
    EXPECT_EQ(fingerprintCacheImage(image.data(), image.size()), base);

    std::vector<uint8_t> flipped = image;
    flipped[500] ^= 0x01;
    EXPECT_NE(fingerprintCacheImage(flipped.data(), flipped.size()), base);

    std::vector<uint8_t> tailFlipped = image;
    tailFlipped.back() ^= 0x80; // inside the sub-word tail
    EXPECT_NE(fingerprintCacheImage(tailFlipped.data(), tailFlipped.size()), base);

    std::vector<uint8_t> shorter(image.begin(), image.end() - 1);
    EXPECT_NE(fingerprintCacheImage(shorter.data(), shorter.size()), base);

    std::vector<uint8_t> swapped = image;
    std::swap_ranges(swapped.begin(), swapped.begin() + 8, swapped.begin() + 64);
    EXPECT_NE(fingerprintCacheImage(swapped.data(), swapped.size()), base);

    EXPECT_TRUE(CacheImageFingerprint {}.empty());
    EXPECT_FALSE(base.empty());
}

// Encoding the same document twice yields the same fingerprint, so a session with nothing new to add
// recognizes its encode as the image already on disk and leaves the file untouched.
TEST(CacheImage, UnchangedDocumentReEncodesToTheSameFingerprint) {
    CacheDoc doc;
    doc.kernelHash        = "kernelhash";
    doc.vendorId          = 1;
    doc.deviceId          = 2;
    doc.driverVersion     = 3;
    doc.pipelineCacheUUID = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    doc.model             = "modelhash";
    doc.variants.push_back(storedVariant("low"));

    const std::vector<uint8_t> first  = cacheEncode(doc);
    const std::vector<uint8_t> second = cacheEncode(doc);
    EXPECT_EQ(fingerprintCacheImage(first.data(), first.size()), fingerprintCacheImage(second.data(), second.size()));

    doc.variants.front().weights["/conv3#w"] = blob(16, 5.f);
    const std::vector<uint8_t> grown         = cacheEncode(doc);
    EXPECT_NE(fingerprintCacheImage(grown.data(), grown.size()), fingerprintCacheImage(first.data(), first.size()));
}

// Merging this session's variant into a document read back from disk keeps every other
// configuration's variant intact -- the property that lets the backend stop holding them decoded.
TEST(CacheImage, MergePreservesOtherConfigurationsVariants) {
    CacheDoc doc;
    doc.variants.push_back(storedVariant("high")); // another configuration's variant
    doc.variants.push_back(storedVariant("low"));  // this session's

    CacheVariant fresh;
    fresh.precision           = "low";
    fresh.pipeline            = {9, 9, 9};
    fresh.weights["/conv3#w"] = blob(16, 5.f);
    mergeVariantIntoDoc(doc, std::move(fresh));

    ASSERT_EQ(doc.variants.size(), 2u);
    const CacheVariant &other = doc.variants[0];
    EXPECT_EQ(other.precision, "high");
    EXPECT_EQ(other.weights.size(), 2u) << "an untouched configuration keeps its whole weight table";

    const CacheVariant &mine = doc.variants[1];
    EXPECT_EQ(mine.precision, "low");
    EXPECT_EQ(mine.weights.size(), 3u) << "the delta completed what disk held";
    EXPECT_EQ(mine.pipeline, (std::vector<uint8_t> {9, 9, 9})) << "the driver blob is replaced, not merged";
}

// A variant whose key the document does not hold yet is appended, so a configuration built for the
// first time joins the file instead of overwriting another one.
TEST(CacheImage, MergeAppendsAnUnseenVariantKey) {
    CacheDoc doc;
    doc.variants.push_back(storedVariant("high"));

    CacheVariant fresh;
    fresh.precision           = "low";
    fresh.weights["/conv9#w"] = blob(8, 1.f);
    mergeVariantIntoDoc(doc, std::move(fresh));

    ASSERT_EQ(doc.variants.size(), 2u);
    EXPECT_EQ(doc.variants[0].precision, "high");
    EXPECT_EQ(doc.variants[1].precision, "low");
}

// The merge survives a full encode/decode round trip: what a later session reads back off disk is the
// union of what every session before it wrote.
TEST(CacheImage, MergedDocumentSurvivesTheCodecRoundTrip) {
    CacheDoc doc;
    doc.kernelHash = "kernelhash";
    doc.model      = "modelhash";
    doc.variants.push_back(storedVariant("low"));

    CacheVariant fresh;
    fresh.precision           = "low";
    fresh.weights["/conv3#w"] = blob(16, 5.f);
    fresh.tune["/conv3#sig"]  = 4;
    mergeVariantIntoDoc(doc, std::move(fresh));

    const std::vector<uint8_t> image = cacheEncode(doc);
    CacheDoc                   back;
    ASSERT_TRUE(cacheDecode(image.data(), image.size(), back));
    ASSERT_EQ(back.variants.size(), 1u);
    EXPECT_EQ(back.variants[0].weights.size(), 3u);
    EXPECT_EQ(back.variants[0].weights["/conv3#w"], blob(16, 5.f));
    EXPECT_EQ(back.variants[0].tune["/conv3#sig"], 4);
    EXPECT_EQ(back.variants[0].tune["/conv1#sig"], 7);
}
