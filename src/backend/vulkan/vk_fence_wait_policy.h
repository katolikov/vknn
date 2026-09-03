// How a submission's fence is awaited. A blocking vkWaitForFences puts the calling thread to
// sleep, and on a mobile SoC the wake-up after the GPU signals costs 0.3-1.2 ms: the core has
// dropped into a deep idle state and its clock has fallen, so the thread resumes late and slow
// (measured on the release device with an empty command buffer: blocking 0.24-0.55 ms per round
// trip against 0.20-0.22 ms polling). Polling the fence from the start hides that latency but
// burns a core for the whole GPU run, which on a 20 ms decode step is heat the GPU then throttles
// on. The pre-wake policy takes both: sleep on the fence with a timeout that expires shortly
// before the predicted completion (the previous submission of the same command buffers), then poll
// until it signals. The timeout wake pays the same latency, but it lands while the GPU is still
// busy, so the completion itself is observed within a poll iteration. A poll that outlives its
// budget (a slower-than-predicted run) falls back to the blocking wait rather than spinning on.
// Pure policy math lives here so the host test suite pins it without a device.
#pragma once
#include <cstdint>

namespace vknn { namespace vk {
    /// How long before the predicted completion the sleeping wait returns to polling. Covers the
    /// measured wake-up latency with margin; a prediction shorter than this polls from the start.
    constexpr double kFencePreWakeMarginMs = 1.0;
    /// How long the poll may run past the predicted completion before the wait falls back to a
    /// blocking vkWaitForFences. Bounds the CPU cost of a mispredicted (slower) run.
    constexpr double kFenceSpinBudgetMs = 4.0;

    /// The sleeping-wait timeout in milliseconds for a submission predicted to take `predictedMs`
    /// (0 = no prediction: poll from the start). Never negative.
    inline double fencePreWakeSleepMs(double predictedMs) noexcept {
        const double sleepMs = predictedMs - kFencePreWakeMarginMs;
        return sleepMs > 0.0 ? sleepMs : 0.0;
    }

    /// The poll budget in milliseconds after the sleeping wait returns: the remaining predicted time
    /// (the margin, or the whole prediction when it was shorter than the margin) plus the spin budget.
    inline double fencePollBudgetMs(double predictedMs) noexcept {
        const double remaining = predictedMs - fencePreWakeSleepMs(predictedMs);
        return (remaining > 0.0 ? remaining : 0.0) + kFenceSpinBudgetMs;
    }

    /// Milliseconds to nanoseconds for a Vulkan timeout (truncating; never negative).
    inline uint64_t fenceTimeoutNs(double ms) noexcept {
        return ms > 0.0 ? (uint64_t) (ms * 1e6) : 0;
    }
}} // namespace vknn::vk
