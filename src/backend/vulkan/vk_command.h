// command pool, one-shot + recordable command buffers, barriers, timestamps.
#pragma once
#include "vk_context.h"
#include <functional>
#include <vector>

namespace vknn { namespace vk {

    // The common case: dispatch N+1 reads what dispatch N wrote. Compute->compute only, which is
    // what the linear CNN graph almost always needs - cheaper than dragging the transfer stage in.
    // With synchronization2 the access scopes narrow to STORAGE reads/writes (every kernel operand
    // is an SSBO), which spares the driver the sampled-image/uniform cache maintenance implied by
    // the sync1 SHADER_READ class; without it the sync1 form is emitted unchanged.
    void computeBarrier(const VulkanContext &ctx, VkCommandBuffer cmd);

    // Wider barrier for the boundary around a vkCmdCopyBuffer (Reshape): covers transfer too.
    void transferBarrier(const VulkanContext &ctx, VkCommandBuffer cmd);

    // Execution-only ordering for a write-after-read hazard: the later dispatch's write must wait
    // for the earlier dispatch's read, but no data moved, so no availability/visibility operation
    // (cache flush or invalidate) is needed and prior unflushed writes stay tracked by the caller.
    // The sync1 zero-memory-barrier form is spec-equivalent but the target mobile driver drops it
    // entirely (measured: reused-buffer outputs corrupt on branchy graphs), while the sync2 form
    // with explicit stage masks and VK_ACCESS_2_NONE is honored. Without synchronization2 this
    // therefore emits the full compute barrier - a sync1-only device keeps the pre-elision
    // behavior rather than trusting the empty-barrier form of an unknown driver.
    void executionBarrier(const VulkanContext &ctx, VkCommandBuffer cmd);

    /// Owns a command pool and a submission fence for the compute queue (RAII); not copyable or movable.
    /// All submissions here are synchronous (submit + wait on the fence).
    class CommandRunner {
      public:
        explicit CommandRunner(VulkanContext &ctx);
        ~CommandRunner();
        CommandRunner(const CommandRunner &)            = delete;
        CommandRunner &operator=(const CommandRunner &) = delete;
        CommandRunner(CommandRunner &&)                 = delete;
        CommandRunner &operator=(CommandRunner &&)      = delete;

        /// Record `fn` into a transient primary command buffer, submit, and wait.
        void oneShot(const std::function<void(VkCommandBuffer)> &fn);

        /// Allocate a reusable primary command buffer (for pre-recorded static graphs).
        VkCommandBuffer allocate();
        void            begin(VkCommandBuffer cmd);
        void            end(VkCommandBuffer cmd);
        /// Submit a pre-recorded buffer and wait on the fence. @returns wall time in ms.
        /// When `submitCallMs` is non-null it receives the vkQueueSubmit call's own wall share,
        /// so the caller can split queue-submission cost from the fence wait.
        double submitAndWait(VkCommandBuffer cmd, double *submitCallMs = nullptr);

        /// Submit `count` pre-recorded buffers as one vkQueueSubmit (one batch each, in order) and
        /// wait once on the fence. The GPU consumes the batches back-to-back with no host round
        /// trip between them; ordering/visibility across buffers is the CALLER's contract (each
        /// buffer must end with a barrier covering the next one's reads). @returns wall time in ms.
        /// Unused by the engine: a single submit spanning watchdog chunks can run long enough for
        /// the driver to reset it and zero the tail, so segments submit per-chunk via submitAndWait.
        double submitBatchAndWait(const VkCommandBuffer *cmds, uint32_t count, double *submitCallMs = nullptr);

        VkCommandPool pool() const noexcept {
            return pool_;
        }

      private:
        VulkanContext &ctx_;
        VkCommandPool  pool_  = VK_NULL_HANDLE;
        VkFence        fence_ = VK_NULL_HANDLE;
    };

}} // namespace vknn::vk
