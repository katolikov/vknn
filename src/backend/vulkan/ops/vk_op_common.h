// Shared bits for the Vulkan operators: push-constant blocks (kept byte-for-byte in sync
// with the matching shaders/*.comp) and a few small upload/dispatch helpers. Each operator
// lives in its own .cpp next to this header.
#pragma once
#include "backend/vulkan/vk_backend.h"
#include "vknn/dtype.h"
#include <cstdio>
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

    // Push a float vector into a fresh device buffer, converting to fp16 when asked.
    inline std::shared_ptr<vk::Buffer> upload(vk::VulkanContext &ctx, const std::vector<float> &data, bool fp16) {
        if (fp16)
        {
            std::vector<uint16_t> h(data.size());
            for (size_t i = 0; i < data.size(); ++i)
            {
                h[i] = floatToHalf(data[i]);
            }
            auto b = std::make_shared<vk::Buffer>(ctx, std::max<size_t>(h.size(), 4) * 2, vk::MemPref::kAuto);
            b->upload(h.data(), h.size() * 2);
            return b;
        }
        auto b = std::make_shared<vk::Buffer>(ctx, std::max<size_t>(data.size(), 4) * 4, vk::MemPref::kAuto);
        b->upload(data.data(), data.size() * 4);
        return b;
    }

    // Same as upload(), but reuse the prepacked blob from the weight cache on warm starts.
    // `compute` only runs on a cache miss.
    template <typename Fn> inline std::shared_ptr<vk::Buffer> uploadCached(VkOpEnv &env, const std::string &rawKey, Fn compute) {
        std::string        key = env.modelTag.empty() ? rawKey : env.modelTag + "/" + rawKey;
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
        return upload(*env.ctx, v, env.useFp16);
    }

    // Initializer payloads decode to fp32 through initFloats (vknn/graph.h) — the shared decode
    // path for host-side payload reads. Ops prepack/transpose weights in fp32, then upload()
    // re-encodes to fp16 for the GPU (fp16->fp32->fp16 is exact).

    // Upload an initializer uploaded FLAT (no transpose/prepack) to a device buffer with at most one
    // element conversion. When the stored dtype already matches the compute precision the raw bytes are
    // memcpy'd straight through, skipping the fp16->fp32->fp16 round-trip that initFloats()+upload()
    // would otherwise do over every weight (the dominant model-load cost for transformer matmuls).
    // fp16->fp32->fp16 is exact, so the direct copy is bit-identical to the round-trip.
    inline std::shared_ptr<vk::Buffer> uploadInit(VkOpEnv &env, TensorId id, const Shape &shape) {
        const Graph      &g  = *env.graph;
        const HostBuffer &hb = g.initializers.at(id);
        int64_t           n  = numElements(shape);
        if (n <= 0)
        {
            n = (int64_t) (hb.bytes.size() / (g.desc(id).dtype == DType::Float16 ? 2 : 4)); // 0-D scalar
        }
        if (env.useFp16 && g.desc(id).dtype == DType::Float16 && hb.bytes.size() == (size_t) n * 2)
        {
            auto b = std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>((size_t) n, 4) * 2, vk::MemPref::kAuto);
            b->upload(hb.bytes.data(), (size_t) n * 2);
            return b;
        }
        std::vector<float> v = initFloats(g, id);
        v.resize((size_t) std::max<int64_t>(n, 0));
        return upload(*env.ctx, v, env.useFp16);
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
