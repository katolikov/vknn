// Vulkan instance/device context and capability discovery.
#pragma once
#include <set>
#include <string>
#include <vector>

#include "vk_common.h"

namespace vknn {
namespace vk {

/// Performance-relevant capabilities probed from the physical device at runtime, so the
/// engine adapts to whatever GPU it actually runs on.
struct VulkanCaps {
  std::string deviceName;
  std::string driverName;
  std::string driverInfo;
  uint32_t apiVersion = 0;
  uint32_t driverVersion = 0;
  uint32_t vendorID = 0, deviceID = 0;
  uint32_t driverID = 0;

  // Compute limits
  uint32_t subgroupSize = 0;
  uint32_t maxWorkGroupInvocations = 0;
  uint32_t maxWorkGroupSize[3] = {0, 0, 0};
  uint32_t maxSharedMemory = 0;
  float timestampPeriod = 0.f;
  bool timestampSupported = false;

  // Feature flags we exploit
  bool shaderFloat16 = false;
  bool shaderInt8 = false;
  bool storage16bit = false;
  bool storage8bit = false;
  bool int8DotProduct = false;
  bool cooperativeMatrix = false;
  bool timelineSemaphore = false;
  bool pushDescriptor = false;
  bool dedicatedAllocation = false;
  bool externalMemoryFd = false;
  bool externalMemoryDmaBuf = false;
  bool externalMemoryAhb = false;
  bool memoryBudget = false;
  bool subgroupArithmetic = false;
  bool subgroupShuffle = false;

  std::set<std::string> deviceExtensions;
  bool has(const std::string& ext) const { return deviceExtensions.count(ext) > 0; }

  std::string summary() const;
};

/// Owns the VkInstance/VkDevice/queue and exposes caps. One per process is typical.
class VulkanContext {
public:
  VulkanContext();
  ~VulkanContext();
  VulkanContext(const VulkanContext&) = delete;
  VulkanContext& operator=(const VulkanContext&) = delete;

  bool initialized() const { return device_ != VK_NULL_HANDLE; }
  const VulkanCaps& caps() const { return caps_; }

  VkInstance instance() const { return instance_; }
  VkPhysicalDevice physicalDevice() const { return phys_; }
  VkDevice device() const { return device_; }
  VkQueue computeQueue() const { return queue_; }
  uint32_t computeQueueFamily() const { return queueFamily_; }
  const VkPhysicalDeviceMemoryProperties& memProps() const { return memProps_; }

  // Extension function pointers (loaded if available).
  PFN_vkCmdPushDescriptorSetKHR cmdPushDescriptorSet = nullptr;
  PFN_vkGetMemoryFdKHR getMemoryFd = nullptr;

private:
  void createInstance();
  void selectPhysicalDevice();
  void queryCaps();
  void createDevice();

  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice phys_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t queueFamily_ = 0;
  VkPhysicalDeviceMemoryProperties memProps_{};
  VulkanCaps caps_;
  std::vector<const char*> enabledDeviceExts_;
};

}  // namespace vk
}  // namespace vknn
