// Idle-time compute-queue heartbeat: keeps the GPU clock from parking between intermittent runs.
#pragma once
#include "vk_buffer.h"
#include "vk_context.h"
#include "vk_pipeline.h"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

namespace vknn { namespace vk {

    /// The Config::power == Power::High policy on the Vulkan backend. The target mobile GPU
    /// power-collapses after ~50-70 ms without a submission; the next submission then starts at the
    /// bottom DVFS step and ramps up over ~35 ms, so a caller that infers intermittently (camera
    /// frames 33-600 ms apart) runs every inference on the ramp — the first run after a cool-down
    /// measures ~3.5x slower than steady state. The heartbeat closes that gap from the engine side:
    /// a thread submits one dispatch of the trivial gpu_keep_alive kernel (one workgroup of the
    /// device lane width, each lane storing its index into a private scratch buffer) whenever the
    /// compute queue has been idle for kGpuKeepAliveIntervalMs, so no idle stretch reaches the
    /// parking threshold.
    ///
    /// Idle gate: a heartbeat is submitted only when no real submission is in flight AND the latest
    /// real submission or completion (VulkanContext::lastQueueWorkSteadyNs, stamped by the
    /// CommandRunner around every submit it makes) is at least one interval old; otherwise the thread
    /// sleeps until the queue will have been idle for a full interval and re-checks. The queue mutex
    /// (VulkanContext::queueMutex) spans the heartbeat's vkQueueSubmit call alone — its fence is
    /// awaited outside — so a real submission arriving meanwhile waits for one trivial submit call
    /// and no longer, and a heartbeat that finds a real submission already marked in flight under the
    /// mutex yields to it. The kernel touches only the keep-alive's own buffer, so numerical output is
    /// unaffected by construction.
    ///
    /// Lifecycle: the VulkanBackend constructs it (Vulkan objects created, the heartbeat command
    /// buffer recorded once) and starts it when the plan is built (flushNewCacheWork), pauses it
    /// while a segment compiles (compileSegment: autotune probes time the queue), and destroys it
    /// before its own context. Not copyable or movable.
    class GpuKeepAlive {
      public:
        /// Longest stretch the compute queue is left without a submission, and the sleep between idle
        /// checks. Below the ~50-70 ms idle threshold after which the target GPU parks its clock,
        /// with margin for scheduler latency; MNN's OpenCL backend uses the same 30 ms cadence for its
        /// Power_High mode.
        static constexpr int64_t kGpuKeepAliveIntervalMs = 30;

        /// Create the heartbeat's Vulkan objects on `ctx` and record its command buffer; the thread is
        /// not started. The pipeline is built without the model's VkPipelineCache so a power-policy
        /// choice never changes the model cache file.
        /// @throws Error when the embedded kernel or a Vulkan object cannot be created.
        explicit GpuKeepAlive(VulkanContext &ctx);
        /// stop(), then release every owned Vulkan object.
        ~GpuKeepAlive();
        GpuKeepAlive(const GpuKeepAlive &)            = delete;
        GpuKeepAlive &operator=(const GpuKeepAlive &) = delete;
        GpuKeepAlive(GpuKeepAlive &&)                 = delete;
        GpuKeepAlive &operator=(GpuKeepAlive &&)      = delete;

        /// Start the heartbeat thread; a no-op while it is already started.
        void start();
        /// Ask the thread to stop and join it; a no-op when it is not started. The owned Vulkan
        /// objects stay valid, so start() may follow.
        void stop();
        /// True between start() and stop().
        bool running() const noexcept;
        /// Heartbeats submitted over the object's lifetime (diagnostics).
        uint64_t heartbeats() const noexcept;
        /// Workgroup width the heartbeat dispatches: the device-resolved lane width.
        uint32_t laneWidth() const noexcept;

      private:
        void loop();
        void submitHeartbeat();
        void destroyObjects() noexcept; ///< Release every owned handle; safe from the destructor and a failing constructor.

        VulkanContext                   &ctx_;
        uint32_t                         laneWidth_ = 0;
        std::unique_ptr<ComputePipeline> pipeline_;
        std::unique_ptr<Buffer>          scratch_;
        VkCommandPool                    pool_  = VK_NULL_HANDLE;
        VkCommandBuffer                  cmd_   = VK_NULL_HANDLE;
        VkFence                          fence_ = VK_NULL_HANDLE;

        std::thread             thread_;
        std::mutex              wakeMutex_;
        std::condition_variable wake_;
        bool                    stopRequested_ = false; // guarded by wakeMutex_
        std::atomic<uint64_t>   heartbeats_ {0};
    };

}} // namespace vknn::vk
