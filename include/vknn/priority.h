// GPU queue-priority tier enum and its string parser.
#pragma once
#include <string>

namespace vknn {

    // GPU queue scheduling-priority tiers (string tokens "low" / "normal" / "high"):
    //   Low    request the low global-priority tier.
    //   Normal driver default (no priority requested).
    //   High   request the high global-priority tier.
    // Applied on the Vulkan backend at queue creation via VK_KHR_global_priority / VK_EXT_global_priority,
    // clamped to the tiers the queue family allows. Scheduling only — never changes numerical output; an
    // inert no-op on a device without a global-priority extension. Normal reproduces the default
    // device-creation path exactly (no extension enabled, no priority requested).
    enum class Priority { Low = 0, Normal = 1, High = 2 };

    // Priority tier from a string: "low", "normal", "high" (unknown -> normal).
    Priority priorityFromStr(const std::string &s);

} // namespace vknn
