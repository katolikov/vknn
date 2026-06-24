// vxrt — command pool, one-shot + recordable command buffers, barriers, timestamps.
#pragma once
#include "vk_context.h"
#include <functional>
#include <vector>

namespace vx {
namespace vk {

/// Inserts a barrier so a following compute dispatch sees the previous dispatch's writes.
inline void computeBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

class CommandRunner {
 public:
  explicit CommandRunner(VulkanContext& ctx);
  ~CommandRunner();

  /// Record `fn` into a transient primary command buffer, submit, and wait.
  void oneShot(const std::function<void(VkCommandBuffer)>& fn);

  /// Allocate a reusable primary command buffer (for pre-recorded static graphs).
  VkCommandBuffer allocate();
  void begin(VkCommandBuffer cmd);
  void end(VkCommandBuffer cmd);
  /// Submit a pre-recorded buffer and wait on a fence. Returns wall time in ms.
  double submitAndWait(VkCommandBuffer cmd);

  VkCommandPool pool() const { return pool_; }

 private:
  VulkanContext& ctx_;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
};

}  // namespace vk
}  // namespace vx
