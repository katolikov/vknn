// GPU power-policy enum and its string parser.
#pragma once
#include <string>

namespace vknn {

    /// GPU power policy for the Vulkan compute queue. A mobile GPU that sees no submission for a few
    /// tens of milliseconds power-collapses; the next submission then starts at the bottom DVFS step
    /// and ramps up over tens of milliseconds, so a caller that infers intermittently runs every
    /// inference on the ramp. High closes that gap from the engine side: once the session is built,
    /// the backend submits one trivial keep-alive dispatch on the compute queue every
    /// GpuKeepAlive::kGpuKeepAliveIntervalMs while the queue would otherwise be idle, and skips it
    /// whenever real work is in flight or was submitted within the last interval. Policy only: it
    /// never changes numerical output, and it is inert on the CPU backend. Normal reproduces the
    /// default path exactly (no keep-alive thread, no extra submissions). The string tokens are
    /// "normal" / "high"; the integer values are stable.
    enum class Power {
        Normal = 0, ///< Driver default: the GPU idles and parks between runs.
        High   = 1, ///< Idle-time keep-alive dispatches hold the GPU clock up between runs.
    };

    /// Parse a power policy from a string: "normal" or "high".
    /// @returns The matching policy; Power::Normal for any unrecognized string (including the empty one).
    Power powerFromStr(const std::string &s);

} // namespace vknn
