// Vulkan backend: device tensors (NC4HW4), op registry, pre-recorded segments.
#pragma once
#include <functional>
#include <map>
#include <memory>
#include <vector>
#include "vx/backend.h"
#include "vk_buffer.h"
#include "vk_command.h"
#include "vk_context.h"
#include "vk_pipeline.h"

namespace vx {

/// Opaque (to core) device storage = a Vulkan buffer holding an NC4HW4 tensor.
struct DeviceStorage {
  std::shared_ptr<vk::Buffer> buffer;
};

/// Disk cache of prepacked weights (keyed by op+role+shape) and autotuned workgroup sizes.
/// Skips the host repacking work on warm session creation. Format is a simple length-prefixed
/// blob written/read atomically.
class WeightCache {
 public:
  void load(const std::string& path);
  void save() const;
  bool enabled() const { return !path_.empty(); }
  bool get(const std::string& key, std::vector<float>& out) const;
  void put(const std::string& key, const std::vector<float>& data);
  // autotune table: op-signature -> chosen local_size_x
  int tuned(const std::string& sig, int dflt) const;
  void setTuned(const std::string& sig, int val);

 private:
  std::string path_;
  std::map<std::string, std::vector<float>> weights_;
  std::map<std::string, int> tune_;
  mutable bool dirty_ = false;
};

class VulkanBackend;

/// Environment passed to Vulkan operators during prepare/record.
struct VkOpEnv {
  VulkanBackend* backend = nullptr;
  vk::VulkanContext* ctx = nullptr;
  vk::PipelineCache* cache = nullptr;
  const Graph* graph = nullptr;
  const Config* config = nullptr;
  std::function<vk::Buffer*(TensorId)> devBuf;  // activation buffer for a tensor id
  bool useFp16 = false;
  WeightCache* weights = nullptr;       // prepacked-weight + tuning cache (may be null)
  vk::CommandRunner* runner = nullptr;  // for on-device autotuning benchmarks
  TuningLevel tuning = TuningLevel::kFast;
};

/// One operator on the Vulkan backend. Adding an op: subclass + VX_REGISTER_VK_OP.
class VulkanOp {
 public:
  virtual ~VulkanOp() = default;
  /// Create pipeline(s), prepack + upload weights, allocate op-private buffers.
  virtual void prepare(const Node& node, VkOpEnv& env) = 0;
  /// Record dispatch(es) into the command buffer.
  virtual void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) = 0;
};

using VkOpFactory = std::function<std::unique_ptr<VulkanOp>()>;

class VkOpRegistry {
 public:
  static VkOpRegistry& instance();
  void reg(OpType t, VkOpFactory f) { factories_[t] = std::move(f); }
  bool has(OpType t) const { return factories_.count(t) > 0; }
  std::unique_ptr<VulkanOp> create(OpType t) const {
    auto it = factories_.find(t);
    return it == factories_.end() ? nullptr : it->second();
  }

 private:
  std::map<OpType, VkOpFactory> factories_;
};

struct VkOpRegistrar {
  VkOpRegistrar(OpType t, VkOpFactory f) { VkOpRegistry::instance().reg(t, std::move(f)); }
};
#define VX_REGISTER_VK_OP(OPTYPE, CLASS)           \
  static ::vx::VkOpRegistrar _vx_vkop_reg_##CLASS( \
      OPTYPE, []() { return std::unique_ptr<::vx::VulkanOp>(new CLASS()); })

// NC4HW4 element count for a logical NCHW shape.
inline int64_t packedElems(const Shape& shape) {
  NCHW x = NCHW::from(shape);
  return x.n * cBlocks(x.c) * 4 * x.h * x.w;
}

}  // namespace vx
