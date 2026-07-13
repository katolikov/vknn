// vknn_probe - print what the GPU under VKNN can and cannot do.
//
// A standalone diagnostic. It brings up VKNN's Vulkan context (the same instance + device + compute
// queue the engine uses to run a model) and prints the physical device's identity, driver, API
// version, the compute limits and feature flags VKNN's kernels rely on (subgroup width, shared memory,
// fp16/int8 storage and arithmetic, external-memory interop, ...), and the full memory-type / heap
// table. Run it first on a new device to see which fast paths are available before loading a model.
//
// The VKNN API used here (this tool touches the Vulkan context directly, not the Model/Session API):
//   - vk::VulkanContext context;    // construct the engine's Vulkan context: instance + device + queue
//   - context.initialized()         // false when no usable compute device could be created
//   - context.caps()                // the probed VulkanCaps: device identity, limits, and feature flags
//   - context.memProps()            // the raw Vulkan memory-type / memory-heap table the allocator uses
//
// Takes no arguments. Exit status:
//   - 0 : probe completed and all capabilities were printed.
//   - 1 : Vulkan context failed to initialize (no usable compute device).
//   - 2 : binary was built without Vulkan support (VKNN_ENABLE_VULKAN unset).
//
//   vknn_probe
#include "vknn/logging.h"
#include <cstdio>
#if defined(VKNN_ENABLE_VULKAN)
#include "backend/vulkan/vk_context.h"
#endif

int main() {
#if defined(VKNN_ENABLE_VULKAN)
    using namespace vknn;

    // Step 1 - bring up the Vulkan context.
    // This is the one object the engine builds per process: it picks a physical device, creates the
    // logical device and a compute queue, and probes the device's capabilities. Everything below just
    // reads back what this construction discovered.
    vk::VulkanContext context;
    if (!context.initialized())
    {
        // The constructor reports failure by leaving initialized() false rather than throwing, so a
        // machine with no usable compute device fails cleanly here instead of mid-run.
        fprintf(stderr, "Vulkan context failed to initialize\n");
        return 1;
    }

    // Step 2 - print the device identity, driver, and API version.
    // caps() hands back the VulkanCaps struct filled in at construction. This first block is the "who
    // am I" section: which GPU, which driver, and which Vulkan version the engine is talking to.
    const auto &caps = context.caps();
    printf("==== vknn Vulkan probe ====\n");
    printf("device         : %s\n", caps.deviceName.c_str());
    printf("driver         : %s | %s\n", caps.driverName.c_str(), caps.driverInfo.c_str());
    printf("driverID       : %u\n", caps.driverID);
    printf("apiVersion     : %u.%u.%u\n", VK_VERSION_MAJOR(caps.apiVersion), VK_VERSION_MINOR(caps.apiVersion), VK_VERSION_PATCH(caps.apiVersion));
    printf("vendor/device  : 0x%x / 0x%x\n", caps.vendorID, caps.deviceID);

    // Step 3 - print the compute limits.
    // These bound how VKNN shapes its dispatches: the subgroup width and shared-memory size drive the
    // tiled-GEMM and reduction kernels, and the workgroup-size limits cap how wide a single dispatch
    // can be. The timestamp period is the clock resolution used for on-device timing.
    printf("subgroupSize   : %u (arith=%d shuffle=%d)\n", caps.subgroupSize, caps.subgroupArithmetic, caps.subgroupShuffle);
    printf("maxWGInvoc     : %u\n", caps.maxWorkGroupInvocations);
    printf("maxWGSize      : [%u,%u,%u]\n", caps.maxWorkGroupSize[0], caps.maxWorkGroupSize[1], caps.maxWorkGroupSize[2]);
    printf("sharedMemory   : %u KiB\n", caps.maxSharedMemory / 1024);
    printf("timestampPeriod: %.4f ns (supported=%d)\n", caps.timestampPeriod, caps.timestampSupported);

    // Step 4 - print the feature flags the kernels exploit.
    // Each flag is a fast path the engine turns on only when the device advertises it: fp16/int8
    // storage and arithmetic, the int8 dot-product and cooperative-matrix matmul accelerators, and the
    // external-memory extensions that make zero-copy dma-buf / AHardwareBuffer I/O possible.
    printf("---- features ----\n");
    printf("  shaderFloat16=%d shaderInt8=%d storage16=%d storage8=%d\n", caps.shaderFloat16, caps.shaderInt8, caps.storage16bit, caps.storage8bit);
    printf("  int8DotProduct=%d cooperativeMatrix=%d\n", caps.int8DotProduct, caps.cooperativeMatrix);
    printf("  timelineSemaphore=%d pushDescriptor=%d dedicatedAllocation=%d\n", caps.timelineSemaphore, caps.pushDescriptor, caps.dedicatedAllocation);
    printf("  externalMemoryFd=%d dmaBuf=%d ahb=%d memoryBudget=%d\n", caps.externalMemoryFd, caps.externalMemoryDmaBuf, caps.externalMemoryAhb, caps.memoryBudget);

    // Step 5 - print the memory-type and memory-heap table.
    // memProps() is the raw Vulkan table the engine's allocator draws from. Each memory TYPE indexes a
    // heap and carries the property flags (device-local, host-visible, ...) that decide where an
    // allocation lands and whether the host can map it; each HEAP reports its total size.
    printf("---- memory types ----\n");
    const auto &memoryProperties = context.memProps();
    for (uint32_t typeIndex = 0; typeIndex < memoryProperties.memoryTypeCount; ++typeIndex)
    {
        // Decoding the bit flags into readable names is plain formatting, not a VKNN call.
        auto flags = memoryProperties.memoryTypes[typeIndex].propertyFlags;
        printf("  type %2u heap=%u flags=0x%03x %s%s%s%s%s\n", typeIndex, memoryProperties.memoryTypes[typeIndex].heapIndex, flags, (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "DEVICE_LOCAL " : "", (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? "HOST_VISIBLE " : "", (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? "HOST_COHERENT " : "", (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "HOST_CACHED " : "", (flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) ? "LAZY " : "");
    }
    for (uint32_t heapIndex = 0; heapIndex < memoryProperties.memoryHeapCount; ++heapIndex)
    {
        printf("  heap %2u size=%.2f GiB flags=0x%x\n", heapIndex, memoryProperties.memoryHeaps[heapIndex].size / (1024.0 * 1024 * 1024), memoryProperties.memoryHeaps[heapIndex].flags);
    }
    printf("==== probe OK ====\n");
    return 0;
#else
    // The whole tool is Vulkan-only; without the backend compiled in there is nothing to probe.
    fprintf(stderr, "Built without Vulkan support\n");
    return 2;
#endif
}
