// Opaque forward declaration for backend-owned device storage.
#pragma once

namespace vknn {

    /// A backend's device-side residency for a tensor, held by the core only as an opaque handle.
    ///
    /// This is an intentional forward (incomplete-type) declaration: the concrete definition lives
    /// with the backend that owns it (the Vulkan backend defines it as a buffer holding an NC4HW4
    /// tensor). Core translation units reference the storage only through @c std::shared_ptr<DeviceStorage>
    /// (see @c RtTensor::device), for which the incomplete type is sufficient, so the core never needs
    /// the Vulkan definition and stays free of Vulkan types.
    struct DeviceStorage;

} // namespace vknn
