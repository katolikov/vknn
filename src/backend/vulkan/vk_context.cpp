#include "vk_context.h"
#include <cstring>
#include <sstream>
#include <vector>

namespace vknn { namespace vk {

    std::string VulkanCaps::summary() const {
        std::ostringstream os;
        os << deviceName << " | " << driverName << " (" << driverInfo << ")"
           << " | Vulkan " << VK_VERSION_MAJOR(apiVersion) << "." << VK_VERSION_MINOR(apiVersion) << "." << VK_VERSION_PATCH(apiVersion) << " | subgroup=" << subgroupSize << " maxWG=" << maxWorkGroupInvocations << " shared=" << (maxSharedMemory / 1024) << "KB"
           << " tsPeriod=" << timestampPeriod << "ns\n"
           << "  fp16=" << shaderFloat16 << " int8=" << shaderInt8 << " storage16=" << storage16bit << " storage8=" << storage8bit << " int8dot=" << int8DotProduct << " coopmat=" << cooperativeMatrix << "\n"
           << "  timeline=" << timelineSemaphore << " pushDesc=" << pushDescriptor << " dedicated=" << dedicatedAllocation << " extMemFd=" << externalMemoryFd << " dmabuf=" << externalMemoryDmaBuf << " ahb=" << externalMemoryAhb << " memBudget=" << memoryBudget << " subgroupArith=" << subgroupArithmetic << " shuffle=" << subgroupShuffle;
        return os.str();
    }

    VulkanContext::VulkanContext() {
        try
        {
            createInstance();
            selectPhysicalDevice();
            queryCaps();
            createDevice();
            VKNN_INFO << "Vulkan ready: " << caps_.summary();
        } catch (const std::exception &e)
        {
            VKNN_ERROR << "VulkanContext init failed: " << e.what();
            // Leave device_ == null; callers check initialized().
        }
    }

    VulkanContext::~VulkanContext() {
        if (device_)
        {
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_)
        {
            vkDestroyInstance(instance_, nullptr);
        }
    }

    void VulkanContext::createInstance() {
        VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName   = "vknn";
        app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        app.pEngineName        = "vknn";
        app.apiVersion         = VK_API_VERSION_1_3;

        VkInstanceCreateInfo ci {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &app;
        // Needs only core + KHR get-physical-device-properties2 (core in 1.1).
        VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
    }

    void VulkanContext::selectPhysicalDevice() {
        uint32_t n = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(instance_, &n, nullptr));
        if (n == 0)
        {
            throw Error(Status::kNotFound, "no Vulkan physical devices");
        }
        std::vector<VkPhysicalDevice> devs(n);
        VK_CHECK(vkEnumeratePhysicalDevices(instance_, &n, devs.data()));
        // Prefer an integrated/discrete GPU with a compute queue. On phones there is one.
        phys_ = devs[0];
        for (auto d: devs)
        {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(d, &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            {
                phys_ = d;
                break;
            }
        }
    }

    void VulkanContext::queryCaps() {
        // --- extensions ---
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(phys_, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(phys_, nullptr, &extCount, exts.data());
        for (auto &e: exts)
        {
            caps_.deviceExtensions.insert(e.extensionName);
        }

        // --- properties (+ driver, subgroup) via pNext chain ---
        VkPhysicalDeviceSubgroupProperties subgroup {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceDriverProperties   driver {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        driver.pNext = &subgroup;
        VkPhysicalDeviceProperties2 props2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &driver;
        vkGetPhysicalDeviceProperties2(phys_, &props2);

        const auto &p                 = props2.properties;
        caps_.deviceName              = p.deviceName;
        caps_.apiVersion              = p.apiVersion;
        caps_.driverVersion           = p.driverVersion;
        caps_.vendorID                = p.vendorID;
        caps_.deviceID                = p.deviceID;
        caps_.driverID                = driver.driverID;
        caps_.driverName              = driver.driverName;
        caps_.driverInfo              = driver.driverInfo;
        caps_.subgroupSize            = subgroup.subgroupSize;
        caps_.subgroupArithmetic      = (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
        caps_.subgroupShuffle         = (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0;
        caps_.maxWorkGroupInvocations = p.limits.maxComputeWorkGroupInvocations;
        caps_.maxWorkGroupSize[0]     = p.limits.maxComputeWorkGroupSize[0];
        caps_.maxWorkGroupSize[1]     = p.limits.maxComputeWorkGroupSize[1];
        caps_.maxWorkGroupSize[2]     = p.limits.maxComputeWorkGroupSize[2];
        caps_.maxSharedMemory         = p.limits.maxComputeSharedMemorySize;
        caps_.timestampPeriod         = p.limits.timestampPeriod;
        caps_.timestampSupported      = p.limits.timestampComputeAndGraphics;

        // --- features via pNext chain ---
        VkPhysicalDeviceShaderFloat16Int8Features f16i8 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDevice16BitStorageFeatures      s16 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
        f16i8.pNext = &s16;
        VkPhysicalDevice8BitStorageFeatures s8 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
        s16.pNext = &s8;
        VkPhysicalDeviceShaderIntegerDotProductFeatures dot {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
        s8.pNext = &dot;
        VkPhysicalDeviceTimelineSemaphoreFeatures tsem {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
        dot.pNext = &tsem;
        VkPhysicalDeviceFeatures2 feats2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        feats2.pNext = &f16i8;
        vkGetPhysicalDeviceProperties2(phys_, &props2); // refresh (harmless)
        vkGetPhysicalDeviceFeatures2(phys_, &feats2);

        caps_.shaderFloat16     = f16i8.shaderFloat16;
        caps_.shaderInt8        = f16i8.shaderInt8;
        caps_.storage16bit      = s16.storageBuffer16BitAccess;
        caps_.storage8bit       = s8.storageBuffer8BitAccess;
        caps_.int8DotProduct    = dot.shaderIntegerDotProduct;
        caps_.timelineSemaphore = tsem.timelineSemaphore;

        caps_.pushDescriptor       = caps_.has("VK_KHR_push_descriptor");
        caps_.dedicatedAllocation  = caps_.has("VK_KHR_dedicated_allocation");
        caps_.externalMemoryFd     = caps_.has("VK_KHR_external_memory_fd");
        caps_.externalMemoryDmaBuf = caps_.has("VK_EXT_external_memory_dma_buf");
        caps_.externalMemoryAhb    = caps_.has("VK_ANDROID_external_memory_android_hardware_buffer");
        caps_.memoryBudget         = caps_.has("VK_EXT_memory_budget");
        caps_.cooperativeMatrix    = caps_.has("VK_KHR_cooperative_matrix");

        vkGetPhysicalDeviceMemoryProperties(phys_, &memProps_);
    }

    void VulkanContext::createDevice() {
        // Pick a compute-capable queue family, preferring a dedicated compute queue
        // (COMPUTE without GRAPHICS) - on the target GPU that is family 1.
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, qfs.data());
        int chosen = -1, fallback = -1;
        for (uint32_t i = 0; i < qn; ++i)
        {
            if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                if (fallback < 0)
                {
                    fallback = (int) i;
                }
                if (!(qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
                {
                    chosen = (int) i;
                    break;
                }
            }
        }
        if (chosen < 0)
        {
            chosen = fallback;
        }
        if (chosen < 0)
        {
            throw Error(Status::kNotFound, "no compute queue family");
        }
        queueFamily_ = (uint32_t) chosen;
        VKNN_INFO << "Compute queue family = " << queueFamily_ << (qfs[chosen].queueFlags & VK_QUEUE_GRAPHICS_BIT ? " (shared w/ graphics)" : " (dedicated compute)");

        float                   prio = 1.0f;
        VkDeviceQueueCreateInfo qci {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = queueFamily_;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &prio;

        // Enable available perf extensions.
        auto addExt = [&](const char *e) {
            if (caps_.has(e))
            {
                enabledDeviceExts_.push_back(e);
            }
        };
        addExt("VK_KHR_push_descriptor");
        addExt("VK_KHR_dedicated_allocation");
        addExt("VK_KHR_get_memory_requirements2");
        addExt("VK_KHR_external_memory");
        addExt("VK_KHR_external_memory_fd");
        addExt("VK_EXT_external_memory_dma_buf");
        addExt("VK_EXT_memory_budget");
        addExt("VK_KHR_shader_float16_int8");
        addExt("VK_KHR_16bit_storage");
        addExt("VK_KHR_8bit_storage");
        addExt("VK_KHR_shader_integer_dot_product");

        // Feature chain to enable.
        VkPhysicalDeviceShaderFloat16Int8Features f16i8 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        f16i8.shaderFloat16 = caps_.shaderFloat16;
        f16i8.shaderInt8    = caps_.shaderInt8;
        VkPhysicalDevice16BitStorageFeatures s16 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
        s16.storageBuffer16BitAccess = caps_.storage16bit;
        f16i8.pNext                  = &s16;
        VkPhysicalDeviceTimelineSemaphoreFeatures tsem {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
        tsem.timelineSemaphore = caps_.timelineSemaphore;
        s16.pNext              = &tsem;

        VkDeviceCreateInfo dci {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext                   = &f16i8;
        dci.queueCreateInfoCount    = 1;
        dci.pQueueCreateInfos       = &qci;
        dci.enabledExtensionCount   = (uint32_t) enabledDeviceExts_.size();
        dci.ppEnabledExtensionNames = enabledDeviceExts_.data();
        VK_CHECK(vkCreateDevice(phys_, &dci, nullptr, &device_));
        vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);

        if (caps_.pushDescriptor)
        {
            cmdPushDescriptorSet = (PFN_vkCmdPushDescriptorSetKHR) vkGetDeviceProcAddr(device_, "vkCmdPushDescriptorSetKHR");
        }
        if (caps_.externalMemoryFd)
        {
            getMemoryFd = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR");
        }
    }

}} // namespace vknn::vk
