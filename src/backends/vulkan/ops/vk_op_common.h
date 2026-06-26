// Shared bits for the Vulkan operators: push-constant blocks (kept byte-for-byte in sync
// with the matching shaders/*.comp) and a few small upload/dispatch helpers. Each operator
// lives in its own .cpp next to this header.
#pragma once
#include <cstdio>
#include <string>
#include <vector>

#include "backends/vulkan/vk_backend.h"
#include "vx/dtype.h"

namespace vx {

// Push-constant layouts. If you touch one of these, change the matching shader too.
struct ConvPC {
  int N, Cin, H, W, Cout, OH, OW, KH, KW, SH, SW, PT, PL, DH, DW, act;
  float actLo, actHi;
};
struct DwPC {
  int N, C, H, W, OH, OW, KH, KW, SH, SW, PT, PL, DH, DW, act, pad0;
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
  int Cin, Cout, act;
  float actLo, actHi;
};
// Split-K 1x1 conv (for deep, small-spatial convs that otherwise have too few threads).
struct SplitKPC {
  int Cin, Cout, HW, KPARTS, chunk;
};
struct ReducePC {
  int Cout, HW, KPARTS, act;
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
  int N, Cout, OH, OW, nTH, nTW, act, pad0;
  float actLo, actHi;
};
struct WinoFusedPC {
  int N, Cin, Cout, OH, OW, nTH, nTW, act;
  float actLo, actHi;
};

// attribute ints with a fallback
inline std::vector<int64_t> attrInts(const Node& n, const char* k, std::vector<int64_t> dflt) {
  const auto& v = n.attr.getints(k);
  return v.empty() ? dflt : v;
}

// 1D dispatch group counts
inline uint32_t groups(int64_t total, uint32_t local) {
  return (uint32_t)((total + local - 1) / local);
}

// shader name, with the fp16 suffix when running in half precision
inline std::string shader(const char* base, bool fp16) {
  return fp16 ? std::string(base) + "_fp16" : std::string(base);
}

// Push a float vector into a fresh device buffer, converting to fp16 when asked.
inline std::shared_ptr<vk::Buffer> upload(vk::VulkanContext& ctx, const std::vector<float>& data,
                                          bool fp16) {
  if (fp16) {
    std::vector<uint16_t> h(data.size());
    for (size_t i = 0; i < data.size(); ++i) h[i] = floatToHalf(data[i]);
    auto b =
        std::make_shared<vk::Buffer>(ctx, std::max<size_t>(h.size(), 4) * 2, vk::MemPref::kAuto);
    b->upload(h.data(), h.size() * 2);
    return b;
  }
  auto b =
      std::make_shared<vk::Buffer>(ctx, std::max<size_t>(data.size(), 4) * 4, vk::MemPref::kAuto);
  b->upload(data.data(), data.size() * 4);
  return b;
}

// Same as upload(), but reuse the prepacked blob from the weight cache on warm starts.
// `compute` only runs on a cache miss.
template <typename Fn>
inline std::shared_ptr<vk::Buffer> uploadCached(VkOpEnv& env, const std::string& rawKey, Fn compute) {
  std::string key = env.modelTag.empty() ? rawKey : env.modelTag + "/" + rawKey;
  std::vector<float> v;
  if (!(env.weights && env.weights->get(key, v))) {
    v = compute();
    // Only retain the prepacked blob when the cache is persistent (a disk path). Without a path the
    // cache would still balloon RAM with every weight (a 965M model: ~3.85GB of prepacked fp32) for
    // no warm-start benefit — so a one-shot run (cacheWeights=false) computes + uploads + frees.
    if (env.weights && env.weights->enabled()) env.weights->put(key, v);
  }
  return upload(*env.ctx, v, env.useFp16);
}

// Read an initializer as fp32, decoding from fp16 if it was stored fp16 (an fp16 .vxm from
// vx_convert). Ops prepack/transpose weights in fp32, then upload() re-encodes to fp16 for the GPU
// (fp16->fp32->fp16 is exact). Use this instead of `g.initializers.at(id).f32()` so a model loaded
// from an fp16 .vxm works unchanged; for an fp32 source it's a plain copy.
inline std::vector<float> initFloats(const Graph& g, TensorId id) {
  const HostBuffer& hb = g.initializers.at(id);
  int64_t n = numElements(g.desc(id).shape);
  std::vector<float> out((size_t)std::max<int64_t>(n, 0));
  if (g.desc(id).dtype == DType::kFloat16) {
    const fp16_t* h = reinterpret_cast<const fp16_t*>(hb.bytes.data());
    for (int64_t i = 0; i < n; ++i) out[i] = halfToFloat(h[i]);
  } else {
    const float* f = hb.f32();
    for (int64_t i = 0; i < n; ++i) out[i] = f[i];
  }
  return out;
}

// Resolve an op's DATA operand to a GPU buffer. An activation has a device buffer (env.devBuf); a
// constant initializer has none, so upload it flat (decoding fp16) into `hold` on first use. Lets any
// elementwise/data-movement op accept a constant operand (e.g. the RoPE freq tables computed from
// constants, or a concatenated constant token) without a null-buffer crash.
inline vk::Buffer* operandBuf(VkOpEnv& env, TensorId t, std::shared_ptr<vk::Buffer>& hold) {
  const Graph& g = *env.graph;
  if (g.isInitializer(t)) {
    if (!hold) {
      std::vector<float> v = initFloats(g, t);
      v.resize(numElements(g.desc(t).shape));
      hold = upload(*env.ctx, v, env.useFp16);
    }
    return hold.get();
  }
  return env.devBuf(t);
}

}  // namespace vx
