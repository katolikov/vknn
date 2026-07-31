// What the backend remembers about the model-cache FILE between saves, and how a session's compiled
// artifacts fold into the variant the file already holds.
//
// Free of Vulkan types (it works on CacheDoc/CacheVariant and raw bytes), so both rules are host
// testable and have exactly one definition. The backend keeps a fingerprint of the file image instead
// of the image itself, and rebuilds the document from disk at save time instead of holding every
// decoded variant for the session lifetime.
#pragma once
#include "core/cache_codec.h"
#include <cstdint>
#include <cstring>

namespace vknn {

    /// Identity of a serialized cache-file image: its length plus a 128-bit content digest. Comparing
    /// two fingerprints answers the only question the save path asks of the previous image -- "are the
    /// bytes I am about to write the bytes already on disk?" -- at 24 bytes instead of the whole file.
    struct CacheImageFingerprint {
        size_t   byteCount  = 0;
        uint64_t digestLow  = 0;
        uint64_t digestHigh = 0;

        bool operator==(const CacheImageFingerprint &o) const {
            return byteCount == o.byteCount && digestLow == o.digestLow && digestHigh == o.digestHigh;
        }
        bool operator!=(const CacheImageFingerprint &o) const {
            return !(*this == o);
        }
        /// True before any image has been fingerprinted, i.e. no cache file was read or written yet.
        bool empty() const {
            return byteCount == 0 && digestLow == 0 && digestHigh == 0;
        }
    };

    /// 128-bit content digest of `bytes`: two independent FNV-1a folds over 8-byte words, tail bytes
    /// folded individually. Same construction as the flat-weight pool's content key
    /// (contentDigest, backend/vulkan/ops/vk_op_common.h); it depends on the content and the length
    /// only, so two runs producing equal bytes produce equal fingerprints.
    inline CacheImageFingerprint fingerprintCacheImage(const uint8_t *bytes, size_t byteCount) {
        constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ull;
        constexpr uint64_t kFnvPrime       = 1099511628211ull;
        // Second fold: a different basis, prime and per-word salt, so a pair of words swapping places
        // cannot cancel in both folds at once.
        constexpr uint64_t kSecondBasis  = 0x2b992ddfa23249d6ull;
        constexpr uint64_t kSecondPrime  = 0x100000001b3ull;
        constexpr uint64_t kSecondSalt   = 0x9e3779b97f4a7c15ull;
        constexpr size_t   kBytesPerWord = 8;

        uint64_t     low       = kFnvOffsetBasis;
        uint64_t     high      = kSecondBasis ^ (uint64_t) byteCount;
        const size_t wordCount = byteCount / kBytesPerWord;
        for (size_t i = 0; i < wordCount; ++i)
        {
            // memcpy per word: the buffer carries no alignment guarantee, so a word-at-a-time read
            // of it would be undefined.
            uint64_t word;
            std::memcpy(&word, bytes + i * kBytesPerWord, kBytesPerWord);
            low  = (low ^ word) * kFnvPrime;
            high = (high ^ (word + kSecondSalt)) * kSecondPrime;
        }
        for (size_t i = wordCount * kBytesPerWord; i < byteCount; ++i)
        {
            low  = (low ^ bytes[i]) * kFnvPrime;
            high = (high ^ (bytes[i] + kSecondSalt)) * kSecondPrime;
        }
        CacheImageFingerprint f;
        f.byteCount  = byteCount;
        f.digestLow  = low ^ (uint64_t) byteCount;
        f.digestHigh = high;
        return f;
    }

    /// Fold this session's compiled artifacts into the variant the cache file already holds.
    ///
    /// The prepacked-weight and autotune tables are ADDITIVE: a key names one deterministic blob for a
    /// given model hash and variant key (both guard the whole document), so an entry this session
    /// produced replaces an equal entry and an entry it did not touch survives. That is what lets the
    /// backend release its retained weights after a save and still write a complete file later --
    /// whatever it no longer holds is read back from disk and merged here.
    ///
    /// The pipeline blob is REPLACED: the driver cache was primed from the stored blob at load, so what
    /// it serializes now covers it.
    ///
    /// @param stored The variant read back from the cache file (or a fresh one keyed for this config).
    /// @param fresh  This session's variant. Its tables may hold only the entries produced since the
    ///               last save.
    inline void mergeSessionArtifacts(CacheVariant &stored, CacheVariant &&fresh) {
        stored.pipeline = std::move(fresh.pipeline);
        for (auto &kv: fresh.weights)
        {
            stored.weights[kv.first] = std::move(kv.second);
        }
        for (const auto &kv: fresh.tune)
        {
            stored.tune[kv.first] = kv.second;
        }
        for (const auto &kv: fresh.tuneLevel)
        {
            stored.tuneLevel[kv.first] = kv.second;
        }
    }

    /// Place `fresh` into `doc`, merging into the variant with an equal key when the document already
    /// holds one and appending it otherwise. The document's other variants are untouched, so a cache
    /// file shared by several configurations keeps every one of them.
    inline void mergeVariantIntoDoc(CacheDoc &doc, CacheVariant &&fresh) {
        for (CacheVariant &stored: doc.variants)
        {
            if (stored.sameKey(fresh))
            {
                mergeSessionArtifacts(stored, std::move(fresh));
                return;
            }
        }
        doc.variants.push_back(std::move(fresh));
    }

} // namespace vknn
