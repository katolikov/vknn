// Shared bits for the Vulkan operators: push-constant blocks (kept byte-for-byte in sync
// with the matching shaders/*.comp) and a few small upload/dispatch helpers. Each operator
// lives in its own .cpp next to this header.
#pragma once
#include "backend/vulkan/vk_backend.h"
#include "vknn/dtype.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace vknn {

    // Push-constant layouts, byte-matched to the matching shaders/*.comp.
    struct ConvPC {
        int   N, Cin, H, W, Cout, OH, OW, KH, KW, SH, SW, PT, PL, DH, DW, act;
        float actLo, actHi;
    };
    struct DwPC {
        // pad0 is a reserved slot present in dwconv.comp's push_constant block too: it pads the int run
        // up to a 16-byte multiple so the trailing actLo/actHi floats land at the byte offset the shader
        // reads them from. Keep it in lockstep with the shader.
        int   N, C, H, W, OH, OW, KH, KW, SH, SW, PT, PL, DH, DW, act, pad0;
        float actLo, actHi;
    };
    struct PoolPC {
        int N, C, H, W;
    };
    struct MaxPC {
        int N, C, H, W, OH, OW, KH, KW, SH, SW, PT, PL;
    };
    struct AvgPC {
        int N, C, H, W, OH, OW, KH, KW, SH, SW, PT, PL, countIncludePad;
    };
    struct FcPC {
        // M = batch rows (>1 for the YoNoSplat 2-view camera head; 1 for classifiers).
        // srcStride/dstStride = per-row element stride: padded channels for NC4HW4 (H=W=1), or exact C
        // for a gpuFlat operand.
        int   Cin, Cout, M, srcStride, dstStride, act;
        float actLo, actHi;
    };
    // Split-K 1x1 conv (for deep, small-spatial convs that otherwise have too few threads).
    struct SplitKPC {
        int Cin, Cout, HW, KPARTS, chunk;
    };
    struct ReducePC {
        int   Cout, HW, KPARTS, act;
        float actLo, actHi;
    };
    // Winograd F(2x2,3x3) push constants.
    struct WinoInPC {
        int N, C, H, W, OH, OW, nTH, nTW;
    };
    struct WinoMmPC {
        int Cin, Cout, nT;
    };
    struct WinoOutPC {
        // pad0: same role as in DwPC — a reserved int mirrored in the winograd-output shader that pads the
        // int run to a 16-byte multiple so actLo/actHi sit at the offset the shader expects.
        int   N, Cout, OH, OW, nTH, nTW, act, pad0;
        float actLo, actHi;
    };
    struct WinoFusedPC {
        int   N, Cin, Cout, OH, OW, nTH, nTW, act;
        float actLo, actHi;
    };
    struct WinoGemmPC {
        int Cin, Cout, nT;
    };
    // Implicit-GEMM conv (conv_gemm.comp); shared by the ConvGemm op and Conv's raced gemm path.
    struct ConvGemmPC {
        int   C, H, W, Cout, OH, OW, KH, KW, SH, SW, PT, PL, DH, DW;
        int   act, hasBias;
        float actLo, actHi;
    };
    // Split-K partial pass (conv_gemm_ksplit.comp) + reduce pass (conv_gemm_kreduce.comp).
    struct ConvGemmKsPC {
        int C, H, W, Cout, OH, OW, KH, KW, SH, SW, PT, PL, DH, DW;
        int S, chunkK;
    };
    struct ConvGemmKrPC {
        int   N, Cout, M, S, act, hasBias;
        float actLo, actHi;
    };

    // conv_gemm.comp tile geometry: TN/TK are fixed in the shader; the M tile is specialization
    // constant 0 (16/32/64 — bit-neutral, it only remaps threads to outputs). The heuristic picks a
    // narrow tile for small-M shapes so the M axis still spreads across workgroups; Tuning::Fast/
    // Heavy race the variants per shape instead (see conv_gemm.cpp / conv.cpp).
    constexpr int kConvGemmTileN = 64, kConvGemmTileK = 16;
    inline int convGemmTileM(int64_t M) {
        return M < 24 ? 16 : (M < 48 ? 32 : 64);
    }

    // attribute ints with a fallback
    inline std::vector<int64_t> attrInts(const Node &n, const char *k, std::vector<int64_t> dflt) {
        const auto &v = n.attr.getints(k);
        return v.empty() ? dflt : v;
    }

    // 1D dispatch group counts
    inline uint32_t groups(int64_t total, uint32_t local) {
        return (uint32_t) ((total + local - 1) / local);
    }

    // shader name, with the fp16 suffix when running in half precision
    inline std::string shader(const char *base, bool fp16) {
        return fp16 ? std::string(base) + "_fp16" : std::string(base);
    }

    // Push a float vector into a fresh HOST-VISIBLE device buffer, converting to fp16 when asked.
    // For small operand constants (norm scales, comparison operands, scalar clip bounds); weight
    // payloads go through uploadWeight() below so they stay out of host-mapped memory. The fp16
    // convert saturates to +/-65504 like an imported constant (clampToFp16Range), so the ONNX-load
    // path uploads the same bytes a --fp16 .vxm compile would store.
    inline std::shared_ptr<vk::Buffer> upload(vk::VulkanContext &ctx, const std::vector<float> &data, bool fp16) {
        if (fp16)
        {
            std::vector<uint16_t> h(data.size());
            floatToHalfSatBulk(data.data(), h.data(), (int64_t) data.size());
            auto b = std::make_shared<vk::Buffer>(ctx, std::max<size_t>(h.size(), 4) * 2, vk::MemPref::kAuto);
            b->upload(h.data(), h.size() * 2);
            return b;
        }
        auto b = std::make_shared<vk::Buffer>(ctx, std::max<size_t>(data.size(), 4) * 4, vk::MemPref::kAuto);
        b->upload(data.data(), data.size() * 4);
        return b;
    }

    // upload() variant for weight payloads: the destination is DEVICE-ONLY memory filled through the
    // backend's staging buffer (VulkanBackend::stageWeightToDevice, which carries the rationale), so
    // weights do not count against the driver's per-process host-mappable budget. Same bytes and same
    // sizing floor (>= 4 elements) as upload(); only the transport differs.
    inline std::shared_ptr<vk::Buffer> uploadWeight(VkOpEnv &env, const std::vector<float> &data, bool fp16) {
        if (fp16)
        {
            std::vector<uint16_t> halfWords(data.size());
            floatToHalfSatBulk(data.data(), halfWords.data(), (int64_t) data.size());
            return env.uploadWeightDeviceOnly(halfWords.data(), halfWords.size() * 2, std::max<size_t>(halfWords.size(), 4) * 2);
        }
        return env.uploadWeightDeviceOnly(data.data(), data.size() * 4, std::max<size_t>(data.size(), 4) * 4);
    }

    // Upload a prepacked weight/bias/transformed-weight blob, reusing (1) the device buffer across op
    // instances and plan buckets via the backend weight pool, and (2) the prepacked host blob from the
    // weight cache on warm starts. `compute` runs only when neither is available.
    //
    // The device pool is keyed by (weight-cache key, precision); a hit returns the shared device buffer
    // with no host lookup and no upload — this is what lets N shape buckets share ONE uploaded copy of
    // each weight. A miss falls through to today's path (host-cache consult, prepack, upload), so a
    // single-bucket model uploads exactly the same buffer it did before the pool existed.
    template <typename Fn> inline std::shared_ptr<vk::Buffer> uploadCached(VkOpEnv &env, const std::string &rawKey, Fn compute) {
        std::string key = env.modelTag.empty() ? rawKey : env.modelTag + "/" + rawKey;
        return env.acquireWeight(key, env.useFp16, [&] {
            std::vector<float> v;
            if (!(env.weights && env.weights->get(key, v)))
            {
                v = compute();
                // Only retain the prepacked blob when the cache is persistent (a disk path). Without a path the
                // cache would still balloon RAM with every weight (a 965M model: ~3.85GB of prepacked fp32) for
                // no warm-start benefit — so a pathless run (WeightCache::enabled() false) computes + uploads + frees.
                if (env.weights && env.weights->enabled())
                {
                    env.weights->put(key, v);
                }
            }
            return uploadWeight(env, v, env.useFp16);
        });
    }

    // Initializer payloads decode to fp32 through initFloats (vknn/graph.h) — the shared decode
    // path for host-side payload reads. Ops prepack/transpose weights in fp32, then uploadWeight()
    // re-encodes to fp16 for the GPU (fp16->fp32->fp16 is exact).

    // 128-bit content digest of a byte payload: two independent FNV-1a folds over 8-byte words (tail
    // bytes folded individually). Keys the flat-weight device pool by what the bytes ARE — never by
    // which graph/bucket referenced them — so the digest must depend on content and length only.
    inline void contentDigest(const uint8_t *bytes, size_t byteCount, uint64_t out[2]) {
        // Byte-wise: a mapped payload carries no alignment guarantee, so a word-at-a-time read of it
        // would be undefined. memcpy of each word keeps the digest identical and alignment-safe.
        uint64_t     h1 = 1469598103934665603ull, h2 = 0x2b992ddfa23249d6ull ^ (uint64_t) byteCount;
        const size_t wordCount = byteCount / 8;
        for (size_t i = 0; i < wordCount; ++i)
        {
            uint64_t word;
            std::memcpy(&word, bytes + i * 8, 8);
            h1 = (h1 ^ word) * 1099511628211ull;
            h2 = (h2 ^ (word + 0x9e3779b97f4a7c15ull)) * 0x100000001b3ull;
        }
        for (size_t i = wordCount * 8; i < byteCount; ++i)
        {
            h1 = (h1 ^ bytes[i]) * 1099511628211ull;
            h2 = (h2 ^ (bytes[i] + 0x9e3779b97f4a7c15ull)) * 0x100000001b3ull;
        }
        out[0] = h1 ^ (uint64_t) byteCount;
        out[1] = h2;
    }

    // Upload an initializer uploaded FLAT (no transpose/prepack) to a device buffer with at most one
    // element conversion. When the stored dtype already matches the compute precision the raw bytes are
    // copied straight through, skipping the fp16->fp32->fp16 round-trip that initFloats()+uploadWeight()
    // would otherwise do over every weight (the dominant model-load cost for transformer matmuls).
    // fp16->fp32->fp16 is exact, so the direct copy is bit-identical to the round-trip.
    //
    // The upload is pooled across op instances and plan buckets through the backend weight pool: a
    // flat upload has no shape-dependent layout (every kernel variant reads the same row-major bytes),
    // so buckets referencing the same payload share ONE device copy — for an LLM whose prefill and
    // decode buckets carry the same weights, this halves weight VRAM. The key is a CONTENT digest of
    // the stored payload plus its dtype and the bound element count (the pool splits by precision):
    // content identity is the only key that both survives per-bucket graph divergence (shape-gated
    // fusion renames/removes nodes, so any graph-derived tag differs between an S=128 and an S=1 plan
    // of one model) and refuses to alias same-named tensors whose BYTES were shape-baked differently.
    inline std::shared_ptr<vk::Buffer> uploadInit(VkOpEnv &env, TensorId id, const Shape &shape) {
        const Graph      &g  = *env.graph;
        const HostBuffer &hb = g.initializers.at(id);
        int64_t           n  = numElements(shape);
        if (n <= 0)
        {
            n = (int64_t) (hb.bytes.size() / (g.desc(id).dtype == DType::Float16 ? 2 : 4)); // 0-D scalar
        }
        auto make = [&] {
            if (env.useFp16 && g.desc(id).dtype == DType::Float16 && hb.bytes.size() == (size_t) n * 2)
            {
                return env.uploadWeightDeviceOnly(hb.bytes.data(), (size_t) n * 2, std::max<size_t>((size_t) n, 4) * 2);
            }
            std::vector<float> v = initFloats(g, id);
            v.resize((size_t) std::max<int64_t>(n, 0));
            return uploadWeight(env, v, env.useFp16);
        };
        // A second consumer of this weight resolves through the memo: after the first upload released the
        // host bytes, the content digest below could not be recomputed from them.
        if (env.lookupFlatWeight)
        {
            if (std::shared_ptr<vk::Buffer> memo = env.lookupFlatWeight(id))
            {
                return memo;
            }
        }
        uint64_t d[2];
        contentDigest(hb.bytes.data(), hb.bytes.size(), d);
        char key[64];
        snprintf(key, sizeof key, "flat#%d#%lld#%016llx%016llx", (int) g.desc(id).dtype, (long long) n, (unsigned long long) d[0], (unsigned long long) d[1]);
        std::shared_ptr<vk::Buffer> deviceBuffer = env.acquireWeight(key, env.useFp16, make);
        if (env.rememberFlatWeight)
        {
            env.rememberFlatWeight(id, deviceBuffer);
        }
        // The host bytes are dead once the device copy exists. Releasing here (rather than after the
        // whole segment builds) halves the load-time peak of a multi-GB weight set: the host and device
        // copy of one weight never coexist beyond this call. Small payloads stay resident — a
        // record-time constant operand re-reads them and their total is negligible.
        constexpr size_t kReleaseThresholdBytes = 1u << 20;
        if (env.releaseInitializer && hb.bytes.size() >= kReleaseThresholdBytes)
        {
            env.releaseInitializer(id);
        }
        return deviceBuffer;
    }

    // Upload an initializer's RAW payload bytes to a device-only buffer, unconverted — for payloads
    // whose bytes are not element-typed by the desc (an int4-packed weight, an int32 index table).
    // Pooled by content digest like uploadInit, with `tag` namespacing the key so a raw upload never
    // aliases a flat element-typed upload of identical bytes; the same host-release rule applies.
    inline std::shared_ptr<vk::Buffer> uploadInitRaw(VkOpEnv &env, TensorId id, const char *tag) {
        const HostBuffer &hb = env.graph->initializers.at(id);
        if (env.lookupFlatWeight)
        {
            if (std::shared_ptr<vk::Buffer> memo = env.lookupFlatWeight(id))
            {
                return memo;
            }
        }
        uint64_t d[2];
        contentDigest(hb.bytes.data(), hb.bytes.size(), d);
        char key[64];
        snprintf(key, sizeof key, "%s#%016llx%016llx", tag, (unsigned long long) d[0], (unsigned long long) d[1]);
        std::shared_ptr<vk::Buffer> deviceBuffer = env.acquireWeight(key, env.useFp16, [&] {
            return env.uploadWeightDeviceOnly(hb.bytes.data(), hb.bytes.size(), std::max<size_t>(hb.bytes.size(), 16));
        });
        if (env.rememberFlatWeight)
        {
            env.rememberFlatWeight(id, deviceBuffer);
        }
        constexpr size_t kReleaseThresholdBytes = 1u << 20;
        if (env.releaseInitializer && hb.bytes.size() >= kReleaseThresholdBytes)
        {
            env.releaseInitializer(id);
        }
        return deviceBuffer;
    }

    // Resolve an op's DATA operand to a GPU buffer. An activation has a device buffer (env.devBuf); a
    // constant initializer has none, so upload it flat (decoding fp16) into `hold` on first use. Lets
    // any elementwise/data-movement op accept a constant operand (e.g. the RoPE freq tables computed
    // from constants, or a concatenated constant token) without a null-buffer crash.
    inline vk::Buffer *operandBuf(VkOpEnv &env, TensorId t, std::shared_ptr<vk::Buffer> &hold) {
        const Graph &g = *env.graph;
        if (g.isInitializer(t))
        {
            if (!hold)
            {
                hold = uploadInit(env, t, g.desc(t).shape);
            }
            return hold.get();
        }
        return env.devBuf(t);
    }

} // namespace vknn
