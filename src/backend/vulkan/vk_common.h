// Vulkan common includes, error checking, function-pointer loading.
#pragma once
#include "vknn/common.h"
#include "vknn/logging.h"
#include <string>
#include <vulkan/vulkan.h>

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
            default:
                return "VK_ERROR_<other>";
        }
    }

#define VK_CHECK(expr)                                                                                                                               \
    do                                                                                                                                               \
    {                                                                                                                                                \
        VkResult _vr = (expr);                                                                                                                       \
        if (_vr != VK_SUCCESS)                                                                                                                       \
        {                                                                                                                                            \
            throw ::vknn::Error(::vknn::Status::kRuntimeError, std::string("Vulkan call failed: ") + #expr + " -> " + ::vknn::vk::vkResultStr(_vr)); \
        }                                                                                                                                            \
    } while (0)

}} // namespace vknn::vk
