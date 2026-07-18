// Typed warm-start model-cache document and its MessagePack (de)serialization.
//
// Backend-neutral: the document holds opaque byte blobs (a serialized VkPipelineCache), prepacked
// weight arrays, and small tables — no Vulkan types — so the codec is host-testable. The Vulkan backend
// fills/consumes the document; validation (kernel hash + device + model) lives at the call site.
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vknn {

    // The current cache format version. Bumped on any structural change so older files fail validation
    // and are transparently recomputed. Version 2 is the first MessagePack format (replaces "VKNNCAC1").
    constexpr uint32_t kCacheFormat = 2;

    // One cache variant: the compiled artifacts for one distinct cache-affecting configuration. Two
    // variants with an equal key() are interchangeable; the backend keeps one variant per config it has
    // built and appends new ones on demand.
    //
    // The codec is name-keyed (each field maps to a MessagePack key equal to its member name), so the
    // member names below are the on-disk contract, not their declaration order: reordering these members
    // is harmless, but renaming one or changing its wire type breaks older files. Adding a key-guard
    // field also requires extending sameKey() so the new dimension actually distinguishes variants.
    struct CacheVariant {
        // The cache-affecting configuration (the variant key).
        std::string precision;             // "low" / "normal" / "high"
        bool        flatLayout     = true; // Config::flatLayout()
        bool        gpuIslandFold  = true; // Config::gpuIslandFold()
        bool        matmulViewFold = true; // Config::matmulViewFold()
        bool        ropeFusion     = true; // Config::ropeFusion()
        bool        fusedAttention = true; // Config::fusedAttention()
        bool        kvConcatFold   = false; // Config::kvConcatFold()
        std::string fp32Tensors;           // Config::fp32Tensors
        int         winograd        = 0;   // Hint::Winograd
        int         winogradVariant = 0;   // Hint::WinogradVariant
        int         winogradUnit    = 0;   // Hint::WinogradUnit
        int         directConv3x3   = 0;   // Hint::DirectConv3x3
        int         splitKConv      = 0;   // Hint::SplitKConv
        int         coopmatGemm     = 0;   // Hint::CoopmatGemm (default equals Auto so an older file keys identically)

        // The compiled artifacts.
        std::vector<uint8_t>                      pipeline; // serialized VkPipelineCache blob
        std::map<std::string, std::vector<float>> weights;  // prepacked / Winograd-transformed weights by name
        std::map<std::string, int32_t>            tune;      // conv autotune table (op signature -> chosen value)
        std::map<std::string, int32_t>            tuneLevel; // append-only "tunelvl": op signature -> Tuning level it was measured at

        bool sameKey(const CacheVariant &o) const {
            return precision == o.precision && flatLayout == o.flatLayout && gpuIslandFold == o.gpuIslandFold && matmulViewFold == o.matmulViewFold && ropeFusion == o.ropeFusion && fusedAttention == o.fusedAttention && kvConcatFold == o.kvConcatFold && fp32Tensors == o.fp32Tensors && winograd == o.winograd && winogradVariant == o.winogradVariant && winogradUnit == o.winogradUnit && directConv3x3 == o.directConv3x3 && splitKConv == o.splitKConv && coopmatGemm == o.coopmatGemm;
        }
    };

    // The whole cache file. The top-level fields (format, kernelHash, device, model) guard every variant;
    // a mismatch of any invalidates the whole file. cacheDecode only reconstructs this layout — it does
    // not compare these guards against the running engine, so the call site checks them and recomputes on
    // any mismatch (see kCacheFormat).
    struct CacheDoc {
        uint32_t                  format = kCacheFormat;
        std::string               kernelHash; // md5 of all embedded SPIR-V (embeddedShadersHash())
        uint32_t                  vendorId      = 0;
        uint32_t                  deviceId      = 0;
        uint32_t                  driverVersion = 0;
        std::vector<uint8_t>      pipelineCacheUUID;
        std::string               model; // hash of the compiled graph
        std::vector<CacheVariant> variants;

        // The variant matching key `k`, or nullptr if none is cached yet.
        const CacheVariant *findVariant(const CacheVariant &k) const {
            for (auto &v: variants)
            {
                if (v.sameKey(k))
                {
                    return &v;
                }
            }
            return nullptr;
        }
    };

    // Serialize a CacheDoc to MessagePack bytes.
    std::vector<uint8_t> cacheEncode(const CacheDoc &doc);
    // Parse MessagePack bytes into `out`. Returns false on malformed / truncated / non-MessagePack input
    // (e.g. a legacy "VKNNCAC1" file), leaving `out` unspecified.
    bool cacheDecode(const uint8_t *data, size_t n, CacheDoc &out);

} // namespace vknn
