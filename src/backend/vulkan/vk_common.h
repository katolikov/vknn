// Vulkan common includes, error checking, function-pointer loading.
#pragma once
#include "vknn/common.h"
#include "vknn/logging.h"
#include <string>
#include <vulkan/vulkan.h>

// VK_EXT_shader_float8 ships in Vulkan headers newer than the NDK's (the probe compiles against
// any header, so the feature struct and the fp8 cooperative-matrix component types are defined
// here when absent). Values match the Khronos registry.
#ifndef VK_EXT_shader_float8
#define VK_EXT_shader_float8 1
#define VK_EXT_SHADER_FLOAT8_EXTENSION_NAME "VK_EXT_shader_float8"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT ((VkStructureType) 1000567000)
typedef struct VkPhysicalDeviceShaderFloat8FeaturesEXT {
    VkStructureType sType;
    void           *pNext;
    VkBool32        shaderFloat8;
    VkBool32        shaderFloat8CooperativeMatrix;
} VkPhysicalDeviceShaderFloat8FeaturesEXT;
#define VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT ((VkComponentTypeKHR) 1000491002)
#define VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT ((VkComponentTypeKHR) 1000491003)
#endif

namespace vknn { namespace vk {

    inline const char *vkResultStr(VkResult r) {
        switch (r)
        {
            case VK_SUCCESS:
                return "VK_SUCCESS";
            case VK_NOT_READY:
                return "VK_NOT_READY";
            case VK_TIMEOUT:
                return "VK_TIMEOUT";
            case VK_ERROR_OUT_OF_HOST_MEMORY:
                return "VK_ERROR_OUT_OF_HOST_MEMORY";
            case VK_ERROR_OUT_OF_DEVICE_MEMORY:
                return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
            case VK_ERROR_INITIALIZATION_FAILED:
                return "VK_ERROR_INITIALIZATION_FAILED";
            case VK_ERROR_DEVICE_LOST:
                return "VK_ERROR_DEVICE_LOST";
            case VK_ERROR_EXTENSION_NOT_PRESENT:
                return "VK_ERROR_EXTENSION_NOT_PRESENT";
            case VK_ERROR_FEATURE_NOT_PRESENT:
                return "VK_ERROR_FEATURE_NOT_PRESENT";
            case VK_ERROR_INCOMPATIBLE_DRIVER:
                return "VK_ERROR_INCOMPATIBLE_DRIVER";
            case VK_ERROR_MEMORY_MAP_FAILED:
                return "VK_ERROR_MEMORY_MAP_FAILED";
            case VK_ERROR_TOO_MANY_OBJECTS:
                return "VK_ERROR_TOO_MANY_OBJECTS"; // hit maxMemoryAllocationCount
            case VK_ERROR_FORMAT_NOT_SUPPORTED:
                return "VK_ERROR_FORMAT_NOT_SUPPORTED";
            case VK_ERROR_FRAGMENTED_POOL:
                return "VK_ERROR_FRAGMENTED_POOL";
            case VK_ERROR_OUT_OF_POOL_MEMORY:
                return "VK_ERROR_OUT_OF_POOL_MEMORY";
            case VK_ERROR_INVALID_EXTERNAL_HANDLE:
                return "VK_ERROR_INVALID_EXTERNAL_HANDLE"; // bad dma-buf fd on import
            case VK_ERROR_NOT_PERMITTED_KHR:
                return "VK_ERROR_NOT_PERMITTED_KHR"; // global-priority tier refused
            default:
                return "VK_ERROR_<other>";
        }
    }

#define VK_CHECK(expr)                                                                                                                              \
    do                                                                                                                                              \
    {                                                                                                                                               \
        VkResult _vr = (expr);                                                                                                                      \
        if (_vr != VK_SUCCESS)                                                                                                                      \
        {                                                                                                                                           \
            throw ::vknn::Error(::vknn::Status::RuntimeError, std::string("Vulkan call failed: ") + #expr + " -> " + ::vknn::vk::vkResultStr(_vr)); \
        }                                                                                                                                           \
    } while (0)

}} // namespace vknn::vk
