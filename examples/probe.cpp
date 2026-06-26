// vx_probe - enumerate the device's Vulkan compute capabilities (M0 hello-vulkan).
#include <cstdio>

#include "vknn/logging.h"
#if defined(VKNN_ENABLE_VULKAN)
#include "backends/vulkan/vk_context.h"
#endif

int main() {
#if defined(VKNN_ENABLE_VULKAN)
  using namespace vknn;
  vk::VulkanContext ctx;
  if (!ctx.initialized()) {
    fprintf(stderr, "Vulkan context failed to initialize\n");
    return 1;
  }
  const auto& c = ctx.caps();
  printf("==== vknn Vulkan probe ====\n");
  printf("device         : %s\n", c.deviceName.c_str());
  printf("driver         : %s | %s\n", c.driverName.c_str(), c.driverInfo.c_str());
  printf("driverID       : %u\n", c.driverID);
  printf("apiVersion     : %u.%u.%u\n", VK_VERSION_MAJOR(c.apiVersion),
         VK_VERSION_MINOR(c.apiVersion), VK_VERSION_PATCH(c.apiVersion));
  printf("vendor/device  : 0x%x / 0x%x\n", c.vendorID, c.deviceID);
  printf("subgroupSize   : %u (arith=%d shuffle=%d)\n", c.subgroupSize, c.subgroupArithmetic,
         c.subgroupShuffle);
  printf("maxWGInvoc     : %u\n", c.maxWorkGroupInvocations);
  printf("maxWGSize      : [%u,%u,%u]\n", c.maxWorkGroupSize[0], c.maxWorkGroupSize[1],
         c.maxWorkGroupSize[2]);
  printf("sharedMemory   : %u KiB\n", c.maxSharedMemory / 1024);
  printf("timestampPeriod: %.4f ns (supported=%d)\n", c.timestampPeriod, c.timestampSupported);
  printf("---- features ----\n");
  printf("  shaderFloat16=%d shaderInt8=%d storage16=%d storage8=%d\n", c.shaderFloat16,
         c.shaderInt8, c.storage16bit, c.storage8bit);
  printf("  int8DotProduct=%d cooperativeMatrix=%d\n", c.int8DotProduct, c.cooperativeMatrix);
  printf("  timelineSemaphore=%d pushDescriptor=%d dedicatedAllocation=%d\n", c.timelineSemaphore,
         c.pushDescriptor, c.dedicatedAllocation);
  printf("  externalMemoryFd=%d dmaBuf=%d ahb=%d memoryBudget=%d\n", c.externalMemoryFd,
         c.externalMemoryDmaBuf, c.externalMemoryAhb, c.memoryBudget);
  printf("---- memory types ----\n");
  const auto& mp = ctx.memProps();
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
    auto f = mp.memoryTypes[i].propertyFlags;
    printf("  type %2u heap=%u flags=0x%03x %s%s%s%s%s\n", i, mp.memoryTypes[i].heapIndex, f,
           (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "DEVICE_LOCAL " : "",
           (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? "HOST_VISIBLE " : "",
           (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? "HOST_COHERENT " : "",
           (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "HOST_CACHED " : "",
           (f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) ? "LAZY " : "");
  }
  for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
    printf("  heap %2u size=%.2f GiB flags=0x%x\n", i,
           mp.memoryHeaps[i].size / (1024.0 * 1024 * 1024), mp.memoryHeaps[i].flags);
  }
  printf("==== probe OK ====\n");
  return 0;
#else
  fprintf(stderr, "Built without Vulkan support\n");
  return 2;
#endif
}
