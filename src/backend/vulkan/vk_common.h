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
#define VK_EXT_shader_float8                                         1
#define VK_EXT_SHADER_FLOAT8_EXTENSION_NAME                          "VK_EXT_shader_float8"
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

    const char *vkResultStr(VkResult r);

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
