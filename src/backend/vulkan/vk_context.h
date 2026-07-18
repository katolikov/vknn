// Vulkan instance/device context and capability discovery.
#pragma once
#include "vk_common.h"
#include "vknn/priority.h"
#include <set>
#include <string>
#include <vector>

namespace vknn { namespace vk {

    /// Performance-relevant capabilities probed from the physical device at runtime, so the
    /// engine adapts to whatever GPU it actually runs on.
    struct VulkanCaps {
        std::string deviceName;
        std::string driverName;
        std::string driverInfo;
        uint32_t    apiVersion    = 0;
        uint32_t    driverVersion = 0;
        uint32_t    vendorID = 0, deviceID = 0;
        uint32_t    driverID = 0;
        uint8_t     pipelineCacheUUID[16] = {}; // device+driver pipeline-cache identity (cache guard)

        // Compute limits
        uint32_t subgroupSize            = 0;
        uint32_t maxWorkGroupInvocations = 0;
        uint32_t maxWorkGroupSize[3]     = {0, 0, 0};
        // Max workgroups per dispatch dimension. A 1D dispatch with gx > maxWorkGroupCount[0]
        // is illegal and silently does nothing on this driver, so large flat ops must split
        // the overflow into the y dimension (see ComputePipeline::dispatch).
        uint32_t maxWorkGroupCount[3] = {0, 0, 0};
        uint32_t maxSharedMemory      = 0;
        // Largest push-constant block the device accepts; the Vulkan-guaranteed floor is 128 B
        // and some kernel PC blocks exceed it, so ComputePipeline validates against this cap.
        uint32_t maxPushConstantsSize = 0;
        float    timestampPeriod      = 0.f;
        bool     timestampSupported   = false;

        // Feature flags we exploit
        bool shaderFloat16        = false;
        bool shaderInt8           = false;
        // Core VkPhysicalDeviceFeatures::shaderInt64 — 64-bit ints in shader code. The Cast-from-int64
        // path does not require it (int64 shape/index tensors decode to compute-precision float at the
        // pack boundary, exact for their small magnitudes), so this is reported for diagnostics only.
        bool shaderInt64          = false;
        bool storage16bit         = false;
        bool storage8bit          = false;
        bool int8DotProduct       = false;
        bool cooperativeMatrix    = false;
        bool timelineSemaphore    = false;
        bool pushDescriptor       = false;
        bool dedicatedAllocation  = false;
        bool externalMemoryFd     = false;
        bool externalMemoryDmaBuf = false;
        bool externalMemoryAhb    = false;
        bool memoryBudget         = false;
        bool subgroupArithmetic   = false;
        bool subgroupShuffle      = false;
        // VK_KHR/EXT_global_priority: the queue scheduling-priority tier the Config::priority knob drives.
        bool globalPriority       = false;

        // VK_KHR_synchronization2 (feature enabled at device creation): scoped VkMemoryBarrier2
        // between compute dispatches instead of the coarser sync1 access classes.
        bool synchronization2 = false;
        // VK_EXT_subgroup_size_control: a pipeline may pin its subgroup size (cooperative-matrix
        // kernels require an exact width on drivers that vary it per pipeline).
        bool     subgroupSizeControl         = false;
        bool     computeFullSubgroups        = false;
        uint32_t minSubgroupSize             = 0;
        uint32_t maxSubgroupSize             = 0;
        bool     requiredSubgroupSizeCompute = false; // requiredSubgroupSizeStages includes COMPUTE
        // SPIR-V Vulkan memory model (core 1.2 feature). Cooperative-matrix GLSL compiles to
        // "OpMemoryModel Logical Vulkan", so coopmat pipelines are gated on it.
        bool vulkanMemoryModel            = false;
        bool vulkanMemoryModelDeviceScope = false;
        // VK_KHR_cooperative_matrix FEATURE bit (cooperativeMatrix above is extension presence).
        bool cooperativeMatrixFeature = false;
        // VK_EXT_shader_float8: fp8 (e4m3/e5m2) conversions, and fp8 coopmat mul-add when the
        // second bit is set. Opt-in low-precision path; never a default.
        bool shaderFloat8        = false;
        bool shaderFloat8CoopMat = false;
        // VkPhysicalDeviceShaderIntegerDotProductProperties acceleration bits. The feature bit
        // (int8DotProduct) alone only promises the OpSDot* opcodes EXIST — a driver may emulate
        // them slower than plain FMA, so int8-dot kernels gate on these instead.
        bool int8DotAccel8Bit     = false; // integerDotProduct8BitSignedAccelerated
        bool int8DotAccel4x8Packed = false; // integerDotProduct4x8BitPackedSignedAccelerated

        /// One supported cooperative-matrix configuration row, as enumerated from the driver.
        /// Types are VkComponentTypeKHR values; scope is a VkScopeKHR value.
        struct CoopmatShape {
            uint32_t M = 0, N = 0, K = 0;
            uint32_t aType = 0, bType = 0, cType = 0, resultType = 0;
            uint32_t scope                  = 0;
            bool     saturatingAccumulation = false;
        };
        std::vector<CoopmatShape> coopmatShapes; // empty on a device without the extension

        /// True when the driver enumerates a subgroup-scope coopmat row with exactly these
        /// dimensions and component types (A and B share `abType`; C and Result share `accType`).
        bool hasCoopmatShape(uint32_t m, uint32_t n, uint32_t k, uint32_t abType, uint32_t accType) const noexcept {
            for (const auto &s: coopmatShapes)
            {
                if (s.M == m && s.N == n && s.K == k && s.aType == abType && s.bType == abType && s.cType == accType && s.resultType == accType && s.scope == VK_SCOPE_SUBGROUP_KHR)
                {
                    return true;
                }
            }
            return false;
        }

        std::set<std::string> deviceExtensions;
        /// True when the device advertises `ext` (queried once at startup).
        bool has(const std::string &ext) const noexcept {
            return deviceExtensions.count(ext) > 0;
        }

        std::string summary() const;
    };

    /// Owns the VkInstance/VkDevice/queue and exposes caps. One per process is typical.
    class VulkanContext {
      public:
        // priority selects the queue global-priority tier requested at device creation.
        // Priority::Normal reproduces the default creation path exactly.
        explicit VulkanContext(Priority priority = Priority::Normal);
        ~VulkanContext();
        VulkanContext(const VulkanContext &)            = delete;
        VulkanContext &operator=(const VulkanContext &) = delete;

        /// True once a device was successfully created; on failure the constructor leaves this false
        /// rather than throwing, so callers gate on it before using the context.
        bool initialized() const noexcept {
            return device_ != VK_NULL_HANDLE;
        }
        const VulkanCaps &caps() const noexcept {
            return caps_;
        }

        VkInstance instance() const noexcept {
            return instance_;
        }
        VkPhysicalDevice physicalDevice() const noexcept {
            return phys_;
        }
        VkDevice device() const noexcept {
            return device_;
        }
        VkQueue computeQueue() const noexcept {
            return queue_;
        }
        uint32_t computeQueueFamily() const noexcept {
            return queueFamily_;
        }
        const VkPhysicalDeviceMemoryProperties &memProps() const noexcept {
            return memProps_;
        }

        // Extension function pointers (loaded if available).
        PFN_vkCmdPushDescriptorSetKHR cmdPushDescriptorSet = nullptr;
        PFN_vkGetMemoryFdKHR          getMemoryFd          = nullptr;
        // Non-null exactly when the synchronization2 feature is enabled on the device; barrier
        // helpers fall back to the sync1 path when null (see vk_command.h).
        PFN_vkCmdPipelineBarrier2KHR cmdPipelineBarrier2 = nullptr;

      private:
        void createInstance();
        void selectPhysicalDevice();
        void queryCaps();
        void createDevice();

        VkInstance                       instance_    = VK_NULL_HANDLE;
        VkPhysicalDevice                 phys_        = VK_NULL_HANDLE;
        VkDevice                         device_      = VK_NULL_HANDLE;
        VkQueue                          queue_       = VK_NULL_HANDLE;
        uint32_t                         queueFamily_ = 0;
        VkPhysicalDeviceMemoryProperties memProps_ {};
        VulkanCaps                       caps_;
        std::vector<const char *>        enabledDeviceExts_;
        Priority                         priority_ = Priority::Normal;
    };

}} // namespace vknn::vk
