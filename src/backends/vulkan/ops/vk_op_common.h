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
struct FcPC {
  int Cin, Cout, act;
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
inline std::shared_ptr<vk::Buffer> uploadCached(VkOpEnv& env, const std::string& key, Fn compute) {
  std::vector<float> v;
  if (!(env.weights && env.weights->get(key, v))) {
    v = compute();
    if (env.weights) env.weights->put(key, v);
  }
  return upload(*env.ctx, v, env.useFp16);
}

}  // namespace vx
