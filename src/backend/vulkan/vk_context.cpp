#include "vk_context.h"
#include <cstring>
#include <sstream>
#include <vector>

namespace vknn { namespace vk {

    bool VulkanCaps::hasCoopmatShape(uint32_t m, uint32_t n, uint32_t k, uint32_t abType, uint32_t accType) const noexcept {
        for (const auto &s: coopmatShapes)
        {
            if (s.M == m && s.N == n && s.K == k && s.aType == abType && s.bType == abType && s.cType == accType && s.resultType == accType && s.scope == VK_SCOPE_SUBGROUP_KHR)
            {
                return true;
            }
        }
        return false;
    }

    std::string VulkanCaps::summary() const {
        std::ostringstream os;
        os << deviceName << " | " << driverName << " (" << driverInfo << ")"
           << " | Vulkan " << VK_VERSION_MAJOR(apiVersion) << "." << VK_VERSION_MINOR(apiVersion) << "." << VK_VERSION_PATCH(apiVersion) << " | subgroup=" << subgroupSize << " maxWG=" << maxWorkGroupInvocations << " maxWGCount=" << maxWorkGroupCount[0] << " shared=" << (maxSharedMemory / 1024) << "KB pushConst=" << maxPushConstantsSize << "B"
           << " tsPeriod=" << timestampPeriod << "ns\n"
           << "  fp16=" << shaderFloat16 << " int8=" << shaderInt8 << " int64=" << shaderInt64 << " storage16=" << storage16bit << " storage8=" << storage8bit << " int8dot=" << int8DotProduct << " coopmat=" << cooperativeMatrix << "\n"
           << "  timeline=" << timelineSemaphore << " pushDesc=" << pushDescriptor << " dedicated=" << dedicatedAllocation << " extMemFd=" << externalMemoryFd << " dmabuf=" << externalMemoryDmaBuf << " ahb=" << externalMemoryAhb << " memBudget=" << memoryBudget << " subgroupArith=" << subgroupArithmetic << " shuffle=" << subgroupShuffle << "\n"
           << "  globalPriority=" << globalPriority << " sync2=" << synchronization2 << " sgCtl=" << subgroupSizeControl << " sgRange=[" << minSubgroupSize << "," << maxSubgroupSize << "]"
           << " vkMemModel=" << vulkanMemoryModel << " coopmatRows=" << coopmatShapes.size() << " fp8=" << shaderFloat8 << " int8dotAccel=" << int8DotAccel8Bit << "/" << int8DotAccel4x8Packed;
        return os.str();
    }

    VulkanContext::VulkanContext(Priority priority): priority_(priority) {
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
        uint32_t deviceCount = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr));
        if (deviceCount == 0)
        {
            throw Error(Status::NotFound, "no Vulkan physical devices");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        VK_CHECK(vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data()));
        // Prefer an integrated or discrete GPU; a phone typically exposes exactly one.
        phys_ = devices[0];
        for (auto dev: devices)
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(dev, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            {
                phys_ = dev;
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

        // --- properties (+ driver, subgroup, subgroup-size-control, int-dot acceleration) ---
        VkPhysicalDeviceSubgroupProperties subgroup {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceDriverProperties   driver {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        driver.pNext = &subgroup;
        VkPhysicalDeviceSubgroupSizeControlProperties     subgroupSizeProps {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceShaderIntegerDotProductProperties dotProps {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES};
        // Extension-specific property structs join the chain only when the extension is present,
        // so an older driver never sees an sType it does not know.
        if (caps_.has("VK_EXT_subgroup_size_control"))
        {
            subgroupSizeProps.pNext = subgroup.pNext;
            subgroup.pNext          = &subgroupSizeProps;
        }
        if (caps_.has("VK_KHR_shader_integer_dot_product"))
        {
            dotProps.pNext = subgroup.pNext;
            subgroup.pNext = &dotProps;
        }
        VkPhysicalDeviceProperties2 props2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &driver;
        vkGetPhysicalDeviceProperties2(phys_, &props2);

        const auto &p       = props2.properties;
        caps_.deviceName    = p.deviceName;
        caps_.apiVersion    = p.apiVersion;
        caps_.driverVersion = p.driverVersion;
        caps_.vendorID      = p.vendorID;
        caps_.deviceID      = p.deviceID;
        caps_.driverID      = driver.driverID;
        caps_.driverName    = driver.driverName;
        caps_.driverInfo    = driver.driverInfo;
        std::memcpy(caps_.pipelineCacheUUID, p.pipelineCacheUUID, sizeof(caps_.pipelineCacheUUID));
        caps_.subgroupSize            = subgroup.subgroupSize;
        caps_.subgroupArithmetic      = (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
        caps_.subgroupShuffle         = (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0;
        caps_.maxWorkGroupInvocations = p.limits.maxComputeWorkGroupInvocations;
        caps_.maxWorkGroupSize[0]     = p.limits.maxComputeWorkGroupSize[0];
        caps_.maxWorkGroupSize[1]     = p.limits.maxComputeWorkGroupSize[1];
        caps_.maxWorkGroupSize[2]     = p.limits.maxComputeWorkGroupSize[2];
        caps_.maxWorkGroupCount[0]    = p.limits.maxComputeWorkGroupCount[0];
        caps_.maxWorkGroupCount[1]    = p.limits.maxComputeWorkGroupCount[1];
        caps_.maxWorkGroupCount[2]    = p.limits.maxComputeWorkGroupCount[2];
        caps_.maxSharedMemory         = p.limits.maxComputeSharedMemorySize;
        caps_.maxPushConstantsSize    = p.limits.maxPushConstantsSize;
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
        // Extension-gated feature structs, appended to the chain only when the extension (or the
        // owning core version) is present so an older driver never sees an unknown sType.
        VkPhysicalDeviceSynchronization2Features     sync2Feat {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
        VkPhysicalDeviceSubgroupSizeControlFeatures  subgroupSizeFeat {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        VkPhysicalDeviceVulkanMemoryModelFeatures    memModelFeat {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmatFeat {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDeviceShaderFloat8FeaturesEXT      float8Feat {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
        VkBaseOutStructure                          *featureTail       = reinterpret_cast<VkBaseOutStructure *>(&tsem);
        auto                                         chainFeatureQuery = [&featureTail](bool present, void *featureStruct) {
            if (present)
            {
                featureTail->pNext = reinterpret_cast<VkBaseOutStructure *>(featureStruct);
                featureTail        = featureTail->pNext;
            }
        };
        chainFeatureQuery(caps_.has("VK_KHR_synchronization2"), &sync2Feat);
        chainFeatureQuery(caps_.has("VK_EXT_subgroup_size_control"), &subgroupSizeFeat);
        chainFeatureQuery(caps_.apiVersion >= VK_API_VERSION_1_2 || caps_.has("VK_KHR_vulkan_memory_model"), &memModelFeat);
        chainFeatureQuery(caps_.has("VK_KHR_cooperative_matrix"), &coopmatFeat);
        chainFeatureQuery(caps_.has(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME), &float8Feat);
        VkPhysicalDeviceFeatures2 feats2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        feats2.pNext = &f16i8;
        vkGetPhysicalDeviceProperties2(phys_, &props2); // refresh (harmless)
        vkGetPhysicalDeviceFeatures2(phys_, &feats2);

        caps_.shaderFloat16     = f16i8.shaderFloat16;
        caps_.shaderInt8        = f16i8.shaderInt8;
        caps_.shaderInt64       = feats2.features.shaderInt64;
        caps_.storage16bit      = s16.storageBuffer16BitAccess;
        caps_.storage8bit       = s8.storageBuffer8BitAccess;
        caps_.int8DotProduct    = dot.shaderIntegerDotProduct;
        caps_.timelineSemaphore = tsem.timelineSemaphore;

        caps_.synchronization2             = sync2Feat.synchronization2;
        caps_.subgroupSizeControl          = subgroupSizeFeat.subgroupSizeControl;
        caps_.computeFullSubgroups         = subgroupSizeFeat.computeFullSubgroups;
        caps_.minSubgroupSize              = subgroupSizeProps.minSubgroupSize;
        caps_.maxSubgroupSize              = subgroupSizeProps.maxSubgroupSize;
        caps_.requiredSubgroupSizeCompute  = (subgroupSizeProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
        caps_.vulkanMemoryModel            = memModelFeat.vulkanMemoryModel;
        caps_.vulkanMemoryModelDeviceScope = memModelFeat.vulkanMemoryModelDeviceScope;
        caps_.cooperativeMatrixFeature     = coopmatFeat.cooperativeMatrix;
        caps_.shaderFloat8                 = float8Feat.shaderFloat8;
        caps_.shaderFloat8CoopMat          = float8Feat.shaderFloat8CooperativeMatrix;
        caps_.int8DotAccel8Bit             = dotProps.integerDotProduct8BitSignedAccelerated;
        caps_.int8DotAccel4x8Packed        = dotProps.integerDotProduct4x8BitPackedSignedAccelerated;

        caps_.pushDescriptor       = caps_.has("VK_KHR_push_descriptor");
        caps_.dedicatedAllocation  = caps_.has("VK_KHR_dedicated_allocation");
        caps_.externalMemoryFd     = caps_.has("VK_KHR_external_memory_fd");
        caps_.externalMemoryDmaBuf = caps_.has("VK_EXT_external_memory_dma_buf");
        caps_.externalMemoryAhb    = caps_.has("VK_ANDROID_external_memory_android_hardware_buffer");
        caps_.memoryBudget         = caps_.has("VK_EXT_memory_budget");
        caps_.cooperativeMatrix    = caps_.has("VK_KHR_cooperative_matrix");
        caps_.globalPriority       = caps_.has("VK_KHR_global_priority") || caps_.has("VK_EXT_global_priority");

        // Cooperative-matrix configuration rows. The extension entry point resolves through the
        // instance; a null pointer or an error leaves the row list empty, which downstream gates
        // treat as "no coopmat" (the SSBO kernels remain the only path).
        if (caps_.cooperativeMatrix && caps_.cooperativeMatrixFeature)
        {
            auto enumerateCoopmat = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
                vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
            uint32_t rowCount = 0;
            if (enumerateCoopmat && enumerateCoopmat(phys_, &rowCount, nullptr) == VK_SUCCESS && rowCount > 0)
            {
                std::vector<VkCooperativeMatrixPropertiesKHR> rows(rowCount, VkCooperativeMatrixPropertiesKHR {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
                if (enumerateCoopmat(phys_, &rowCount, rows.data()) == VK_SUCCESS)
                {
                    caps_.coopmatShapes.reserve(rowCount);
                    for (uint32_t i = 0; i < rowCount; ++i)
                    {
                        const auto &r = rows[i];
                        caps_.coopmatShapes.push_back(
                            {r.MSize, r.NSize, r.KSize, (uint32_t) r.AType, (uint32_t) r.BType, (uint32_t) r.CType, (uint32_t) r.ResultType, (uint32_t) r.scope, r.saturatingAccumulation == VK_TRUE});
                    }
                }
            }
        }

        vkGetPhysicalDeviceMemoryProperties(phys_, &memProps_);
    }

    void VulkanContext::createDevice() {
        // Pick a compute-capable queue family, preferring a dedicated compute queue
        // (COMPUTE without GRAPHICS) - on the target GPU that is family 1.
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> familyProps(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &familyCount, familyProps.data());
        int chosen = -1, fallback = -1;
        for (uint32_t i = 0; i < familyCount; ++i)
        {
            if (familyProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                if (fallback < 0)
                {
                    fallback = (int) i;
                }
                if (!(familyProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
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
            throw Error(Status::NotFound, "no compute queue family");
        }
        queueFamily_ = (uint32_t) chosen;
        VKNN_INFO << "Compute queue family = " << queueFamily_ << (familyProps[chosen].queueFlags & VK_QUEUE_GRAPHICS_BIT ? " (shared w/ graphics)" : " (dedicated compute)");

        float                   queuePriority = 1.0f;
        VkDeviceQueueCreateInfo qci {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = queueFamily_;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &queuePriority;

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
        if (caps_.synchronization2)
        {
            addExt("VK_KHR_synchronization2");
        }
        if (caps_.subgroupSizeControl)
        {
            addExt("VK_EXT_subgroup_size_control");
        }
        if (caps_.vulkanMemoryModel)
        {
            addExt("VK_KHR_vulkan_memory_model"); // no-op when the feature is core (>= 1.2)
        }
        if (caps_.cooperativeMatrixFeature)
        {
            addExt("VK_KHR_cooperative_matrix");
        }
        if (caps_.shaderFloat8)
        {
            addExt(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
        }

        // Queue scheduling priority (Config::priority). Priority::Normal leaves everything below exactly
        // as the default path; Low/High request the matching queue global-priority tier. Capability-gated,
        // so a device without VK_KHR/EXT_global_priority is an inert no-op. Scheduling only, never output.
        //   Largest allowed tier not exceeding the requested one (the driver reports the allowed set per
        //   family; the tier enum is monotonic LOW<MEDIUM<HIGH<REALTIME).
        auto clampGlobalPriority = [&](VkQueueGlobalPriorityKHR want) -> VkQueueGlobalPriorityKHR {
            uint32_t familyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties2(phys_, &familyCount, nullptr);
            std::vector<VkQueueFamilyGlobalPriorityPropertiesKHR> priorityProps(familyCount);
            std::vector<VkQueueFamilyProperties2>                 familyProps(familyCount);
            for (uint32_t i = 0; i < familyCount; ++i)
            {
                priorityProps[i]     = VkQueueFamilyGlobalPriorityPropertiesKHR {VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_KHR};
                familyProps[i]       = VkQueueFamilyProperties2 {VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2};
                familyProps[i].pNext = &priorityProps[i];
            }
            vkGetPhysicalDeviceQueueFamilyProperties2(phys_, &familyCount, familyProps.data());
            if (queueFamily_ >= familyCount || priorityProps[queueFamily_].priorityCount == 0)
            {
                return want; // family reports no set; request as-is (the create-time ladder handles refusal)
            }
            const auto              &allowed = priorityProps[queueFamily_];
            VkQueueGlobalPriorityKHR best {};
            bool                     haveBest = false;
            for (uint32_t k = 0; k < allowed.priorityCount; ++k)
            {
                VkQueueGlobalPriorityKHR tier = allowed.priorities[k];
                if (tier <= want && (!haveBest || tier > best))
                {
                    best     = tier;
                    haveBest = true;
                }
            }
            return haveBest ? best : want;
        };

        VkDeviceQueueGlobalPriorityCreateInfoKHR gpci {VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR};
        const bool                               useGlobalPriority = caps_.globalPriority && priority_ != Priority::Normal;
        if (useGlobalPriority)
        {
            gpci.globalPriority = clampGlobalPriority(priority_ == Priority::High ? VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR : VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR);
            qci.pNext           = &gpci;
            addExt(caps_.has("VK_KHR_global_priority") ? "VK_KHR_global_priority" : "VK_EXT_global_priority");
        }

        // Feature chain to enable.
        VkPhysicalDeviceShaderFloat16Int8Features f16i8 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        f16i8.shaderFloat16 = caps_.shaderFloat16;
        f16i8.shaderInt8    = caps_.shaderInt8;
        VkPhysicalDevice16BitStorageFeatures s16 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
        s16.storageBuffer16BitAccess = caps_.storage16bit;
        f16i8.pNext                  = &s16;
        // 8-bit storage: enables uint8 SSBOs for the boundary_convert image-I/O variants (u8 <-> device fp16).
        VkPhysicalDevice8BitStorageFeatures s8 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
        s8.storageBuffer8BitAccess = caps_.storage8bit;
        s16.pNext                  = &s8;
        // Integer dot product: required (beyond the extension) for OpSDotKHR-family SPIR-V in int8 kernels.
        VkPhysicalDeviceShaderIntegerDotProductFeatures dot {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
        dot.shaderIntegerDotProduct = caps_.int8DotProduct;
        s8.pNext                    = &dot;
        VkPhysicalDeviceTimelineSemaphoreFeatures tsem {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
        tsem.timelineSemaphore = caps_.timelineSemaphore;
        dot.pNext              = &tsem;
        // Feature structs below join the enable chain only when queryCaps() confirmed support, so
        // the created device state matches the caps flags exactly.
        VkPhysicalDeviceSynchronization2Features sync2Feat {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
        sync2Feat.synchronization2 = VK_TRUE;
        VkPhysicalDeviceSubgroupSizeControlFeatures subgroupSizeFeat {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        subgroupSizeFeat.subgroupSizeControl  = VK_TRUE;
        subgroupSizeFeat.computeFullSubgroups = caps_.computeFullSubgroups;
        VkPhysicalDeviceVulkanMemoryModelFeatures memModelFeat {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES};
        memModelFeat.vulkanMemoryModel            = VK_TRUE;
        memModelFeat.vulkanMemoryModelDeviceScope = caps_.vulkanMemoryModelDeviceScope;
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmatFeat {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        coopmatFeat.cooperativeMatrix = VK_TRUE;
        VkPhysicalDeviceShaderFloat8FeaturesEXT float8Feat {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
        float8Feat.shaderFloat8                  = VK_TRUE;
        float8Feat.shaderFloat8CooperativeMatrix = caps_.shaderFloat8CoopMat;
        VkBaseOutStructure *enableTail           = reinterpret_cast<VkBaseOutStructure *>(&tsem);
        auto                chainFeatureEnable   = [&enableTail](bool enable, void *featureStruct) {
            if (enable)
            {
                enableTail->pNext = reinterpret_cast<VkBaseOutStructure *>(featureStruct);
                enableTail        = enableTail->pNext;
            }
        };
        chainFeatureEnable(caps_.synchronization2, &sync2Feat);
        chainFeatureEnable(caps_.subgroupSizeControl, &subgroupSizeFeat);
        chainFeatureEnable(caps_.vulkanMemoryModel, &memModelFeat);
        chainFeatureEnable(caps_.cooperativeMatrixFeature, &coopmatFeat);
        chainFeatureEnable(caps_.shaderFloat8, &float8Feat);

        VkDeviceCreateInfo dci {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext                   = &f16i8;
        dci.queueCreateInfoCount    = 1;
        dci.pQueueCreateInfos       = &qci;
        dci.enabledExtensionCount   = (uint32_t) enabledDeviceExts_.size();
        dci.ppEnabledExtensionNames = enabledDeviceExts_.data();

        VkResult r = vkCreateDevice(phys_, &dci, nullptr, &device_);
        // A driver may gate HIGH/REALTIME behind a privilege and return VK_ERROR_NOT_PERMITTED_KHR. Step
        // the requested tier down (High->Medium->Low), then drop the priority request entirely, so the
        // hint degrades to the default path rather than failing device creation. (The target mobile
        // drivers permit every tier for an ordinary process, so this ladder is a portability safety net.)
        while (r == VK_ERROR_NOT_PERMITTED_KHR && qci.pNext)
        {
            if (gpci.globalPriority == VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR)
            {
                gpci.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;
            } else if (gpci.globalPriority == VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR)
            {
                gpci.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR;
            } else
            {
                qci.pNext = nullptr;
            }
            VKNN_WARN << "global-priority tier refused; retrying at a lower tier";
            r = vkCreateDevice(phys_, &dci, nullptr, &device_);
        }
        VK_CHECK(r);
        vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);
        if (priority_ != Priority::Normal)
        {
            VKNN_INFO << "Queue priority " << (priority_ == Priority::High ? "high" : "low") << ": globalPriority=" << (useGlobalPriority ? 1 : 0);
        }

        if (caps_.pushDescriptor)
        {
            cmdPushDescriptorSet = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(vkGetDeviceProcAddr(device_, "vkCmdPushDescriptorSetKHR"));
        }
        if (caps_.externalMemoryFd)
        {
            getMemoryFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR"));
        }
        if (caps_.synchronization2)
        {
            // The core 1.3 name resolves on newer drivers; the KHR alias covers 1.1/1.2 devices
            // exposing only the extension. Barrier helpers use the sync1 path when both are null.
            cmdPipelineBarrier2 = reinterpret_cast<PFN_vkCmdPipelineBarrier2KHR>(vkGetDeviceProcAddr(device_, "vkCmdPipelineBarrier2"));
            if (!cmdPipelineBarrier2)
            {
                cmdPipelineBarrier2 = reinterpret_cast<PFN_vkCmdPipelineBarrier2KHR>(vkGetDeviceProcAddr(device_, "vkCmdPipelineBarrier2KHR"));
            }
        }
    }

}} // namespace vknn::vk
