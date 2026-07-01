// Opaque forward declaration for backend-owned device storage.
#pragma once

namespace vknn {

    /// Forward declaration: the Vulkan backend attaches its device storage here as an
    /// opaque handle, keeping the core free of Vulkan types.
    struct DeviceStorage;

} // namespace vknn
