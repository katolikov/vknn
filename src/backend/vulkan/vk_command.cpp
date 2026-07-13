#include "vk_command.h"
#include <chrono>

namespace vknn { namespace vk {

    namespace {
        // Fence wait with no deadline: the compute submissions here are always awaited to completion,
        // and a genuinely hung GPU surfaces as a driver-side device-lost rather than a client timeout.
        constexpr uint64_t kFenceWaitForever = UINT64_MAX;
    } // namespace

    CommandRunner::CommandRunner(VulkanContext &ctx): ctx_(ctx) {
        VkCommandPoolCreateInfo pci {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = ctx_.computeQueueFamily();
        VK_CHECK(vkCreateCommandPool(ctx_.device(), &pci, nullptr, &pool_));
        VkFenceCreateInfo fci {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkResult          fr = vkCreateFence(ctx_.device(), &fci, nullptr, &fence_);
        if (fr != VK_SUCCESS)
        {
            // The pool is already created; a throwing constructor never runs the destructor, so free it.
            vkDestroyCommandPool(ctx_.device(), pool_, nullptr);
            pool_ = VK_NULL_HANDLE;
            throw Error(Status::RuntimeError, std::string("vkCreateFence -> ") + vkResultStr(fr));
        }
    }

    CommandRunner::~CommandRunner() {
        if (fence_)
        {
            vkDestroyFence(ctx_.device(), fence_, nullptr);
        }
        if (pool_)
        {
            vkDestroyCommandPool(ctx_.device(), pool_, nullptr);
        }
    }

    VkCommandBuffer CommandRunner::allocate() {
        VkCommandBufferAllocateInfo ai {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool        = pool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        VK_CHECK(vkAllocateCommandBuffers(ctx_.device(), &ai, &cmd));
        return cmd;
    }

    void CommandRunner::begin(VkCommandBuffer cmd) {
        VkCommandBufferBeginInfo bi {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    }
    void CommandRunner::end(VkCommandBuffer cmd) {
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    double CommandRunner::submitAndWait(VkCommandBuffer cmd, double *submitCallMs) {
        VK_CHECK(vkResetFences(ctx_.device(), 1, &fence_));
        VkSubmitInfo si {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd;
        auto t0               = std::chrono::high_resolution_clock::now();
        VK_CHECK(vkQueueSubmit(ctx_.computeQueue(), 1, &si, fence_));
        auto tSubmitted = std::chrono::high_resolution_clock::now();
        VK_CHECK(vkWaitForFences(ctx_.device(), 1, &fence_, VK_TRUE, kFenceWaitForever));
        auto t1 = std::chrono::high_resolution_clock::now();
        if (submitCallMs)
        {
            *submitCallMs = std::chrono::duration<double, std::milli>(tSubmitted - t0).count();
        }
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    double CommandRunner::submitBatchAndWait(const VkCommandBuffer *cmds, uint32_t count, double *submitCallMs) {
        VK_CHECK(vkResetFences(ctx_.device(), 1, &fence_));
        std::vector<VkSubmitInfo> infos(count, VkSubmitInfo {VK_STRUCTURE_TYPE_SUBMIT_INFO});
        for (uint32_t i = 0; i < count; ++i)
        {
            infos[i].commandBufferCount = 1;
            infos[i].pCommandBuffers    = &cmds[i];
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        VK_CHECK(vkQueueSubmit(ctx_.computeQueue(), count, infos.data(), fence_));
        auto tSubmitted = std::chrono::high_resolution_clock::now();
        VK_CHECK(vkWaitForFences(ctx_.device(), 1, &fence_, VK_TRUE, kFenceWaitForever));
        auto t1 = std::chrono::high_resolution_clock::now();
        if (submitCallMs)
        {
            *submitCallMs = std::chrono::duration<double, std::milli>(tSubmitted - t0).count();
        }
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    void CommandRunner::oneShot(const std::function<void(VkCommandBuffer)> &fn) {
        VkCommandBuffer cmd = allocate();
        try
        {
            VkCommandBufferBeginInfo bi {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
            fn(cmd);
            VK_CHECK(vkEndCommandBuffer(cmd));
            submitAndWait(cmd);
        } catch (...)
        {
            // begin/record/submit can throw (a VK_CHECK on device-lost/OOM, or fn itself). Free the
            // buffer back to the pool on the exception path too, or a caught-and-retried load-time
            // error leaks one buffer per call for the runner's life. Mirrors the ctor's cleanup.
            vkFreeCommandBuffers(ctx_.device(), pool_, 1, &cmd);
            throw;
        }
        vkFreeCommandBuffers(ctx_.device(), pool_, 1, &cmd);
    }

}} // namespace vknn::vk
