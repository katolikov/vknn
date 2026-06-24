#include "vk_command.h"
#include <chrono>

namespace vx {
namespace vk {

CommandRunner::CommandRunner(VulkanContext& ctx) : ctx_(ctx) {
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = ctx_.computeQueueFamily();
  VK_CHECK(vkCreateCommandPool(ctx_.device(), &pci, nullptr, &pool_));
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VK_CHECK(vkCreateFence(ctx_.device(), &fci, nullptr, &fence_));
}

CommandRunner::~CommandRunner() {
  if (fence_) vkDestroyFence(ctx_.device(), fence_, nullptr);
  if (pool_) vkDestroyCommandPool(ctx_.device(), pool_, nullptr);
}

VkCommandBuffer CommandRunner::allocate() {
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool_;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  VK_CHECK(vkAllocateCommandBuffers(ctx_.device(), &ai, &cmd));
  return cmd;
}

void CommandRunner::begin(VkCommandBuffer cmd) {
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
}
void CommandRunner::end(VkCommandBuffer cmd) { VK_CHECK(vkEndCommandBuffer(cmd)); }

double CommandRunner::submitAndWait(VkCommandBuffer cmd) {
  VK_CHECK(vkResetFences(ctx_.device(), 1, &fence_));
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  auto t0 = std::chrono::high_resolution_clock::now();
  VK_CHECK(vkQueueSubmit(ctx_.computeQueue(), 1, &si, fence_));
  VK_CHECK(vkWaitForFences(ctx_.device(), 1, &fence_, VK_TRUE, UINT64_MAX));
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void CommandRunner::oneShot(const std::function<void(VkCommandBuffer)>& fn) {
  VkCommandBuffer cmd = allocate();
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
  fn(cmd);
  VK_CHECK(vkEndCommandBuffer(cmd));
  submitAndWait(cmd);
  vkFreeCommandBuffers(ctx_.device(), pool_, 1, &cmd);
}

}  // namespace vk
}  // namespace vx
