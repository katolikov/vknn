// GPU queue-priority tier enum and its string parser.
#pragma once
#include <string>

namespace vknn {

    /// GPU queue scheduling-priority tier requested for the Vulkan compute queue. Applied at queue
    /// creation via VK_KHR_global_priority / VK_EXT_global_priority and clamped to the tiers the queue
    /// family allows. Scheduling only: it never changes numerical output, and it is an inert no-op on a
    /// device that exposes no global-priority extension. Normal reproduces the default device-creation
    /// path exactly (no extension enabled, no priority requested). The string tokens are "low" / "normal"
    /// / "high"; the integer values are stable.
    enum class Priority {
        Low    = 0, ///< Request the low global-priority tier.
        Normal = 1, ///< Driver default; no priority requested.
        High   = 2, ///< Request the high global-priority tier.
    };

    /// Parse a priority tier from a string: "low", "normal", or "high".
    /// @returns The matching tier; Priority::Normal for any unrecognized string (including the empty one).
    Priority priorityFromStr(const std::string &s);

} // namespace vknn
