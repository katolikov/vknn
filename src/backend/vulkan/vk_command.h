// command pool, one-shot + recordable command buffers, barriers, timestamps.
#pragma once
#include "vk_context.h"
#include <functional>
#include <vector>

namespace vknn { namespace vk {

    // The common case: dispatch N+1 reads what dispatch N wrote. Compute->compute only, which is
    // what the linear CNN graph almost always needs - cheaper than dragging the transfer stage in.
    inline void computeBarrier(VkCommandBuffer cmd) {
        VkMemoryBarrier b {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    }

    // Wider barrier for the boundary around a vkCmdCopyBuffer (Reshape): covers transfer too.
    inline void transferBarrier(VkCommandBuffer cmd) {
        VkMemoryBarrier b {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        b.srcAccessMask              = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask              = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        const VkPipelineStageFlags s = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        vkCmdPipelineBarrier(cmd, s, s, 0, 1, &b, 0, nullptr, 0, nullptr);
    }

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
