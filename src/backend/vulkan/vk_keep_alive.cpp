#include "vk_keep_alive.h"
#include "ops/flat_ops.h" // flat::laneWidthFor - the device-resolved per-thread family width
#include "vknn/logging.h"
#include <algorithm>
#include <chrono>
#include <exception>
#include <vector>

namespace vknn { namespace vk {

    namespace {
        // Embedded kernel the heartbeat dispatches (shaders/gpu_keep_alive.comp): binds its scratch
        // buffer alone, takes no push constants, and runs as a single workgroup.
        constexpr const char *kGpuKeepAliveShader         = "gpu_keep_alive";
        constexpr uint32_t    kGpuKeepAliveBufferCount    = 1;
        constexpr uint32_t    kGpuKeepAlivePushConstBytes = 0;
        constexpr uint32_t    kGpuKeepAliveWorkGroups     = 1;
        // Shortest sleep between idle checks, so a queue that went idle a moment ago is re-checked
        // once the interval has elapsed rather than polled.
        constexpr int64_t kGpuKeepAliveMinWaitMs = 1;
        constexpr int64_t kNsPerMs               = 1000000;
        // Fence wait with no deadline: the heartbeat is one trivial workgroup, and a hung GPU
        // surfaces as a device-lost error rather than a client timeout.
        constexpr uint64_t kFenceWaitForever = UINT64_MAX;
    } // namespace

    GpuKeepAlive::GpuKeepAlive(VulkanContext &ctx): ctx_(ctx) {
        // A throwing constructor does not run the destructor, so reclaim any handle already created
        // before letting the exception propagate.
        try
        {
            // The same caps-derived width the per-thread conv family dispatches with: whole subgroups
            // under a 64-lane ceiling, legal on any device, passed as the kernel's workgroup-size
            // specialization constant.
            laneWidth_ = flat::laneWidthFor(ctx_.caps(), flat::kConvFamilyLaneWidth);
            pipeline_ = std::make_unique<ComputePipeline>(ctx_, kGpuKeepAliveShader, kGpuKeepAliveBufferCount, kGpuKeepAlivePushConstBytes, std::vector<uint32_t> {laneWidth_});
            scratch_ = std::make_unique<Buffer>(ctx_, (size_t) laneWidth_ * sizeof(uint32_t), MemPref::kDeviceOnly);

            VkCommandPoolCreateInfo pci {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pci.queueFamilyIndex = ctx_.computeQueueFamily();
            VK_CHECK(vkCreateCommandPool(ctx_.device(), &pci, nullptr, &pool_));
            VkCommandBufferAllocateInfo ai {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool        = pool_;
            ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            VK_CHECK(vkAllocateCommandBuffers(ctx_.device(), &ai, &cmd_));
            VkFenceCreateInfo fci {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            VK_CHECK(vkCreateFence(ctx_.device(), &fci, nullptr, &fence_));

            // Recorded once and re-submitted for the object's lifetime; every submission is awaited on
            // the fence before the next, so the buffer needs no simultaneous-use flag. Recorded on the
            // load thread after the plan is built, so its dispatch-tally note lands outside every
            // segment's attribution span.
            VkCommandBufferBeginInfo bi {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VK_CHECK(vkBeginCommandBuffer(cmd_, &bi));
            pipeline_->dispatch(cmd_, {scratch_->handle()}, nullptr, kGpuKeepAlivePushConstBytes, kGpuKeepAliveWorkGroups);
            VK_CHECK(vkEndCommandBuffer(cmd_));
        } catch (...)
        {
            destroyObjects();
            throw;
        }
    }

    GpuKeepAlive::~GpuKeepAlive() {
        stop();
        destroyObjects();
    }

    void GpuKeepAlive::destroyObjects() noexcept {
        if (fence_ != VK_NULL_HANDLE)
        {
            vkDestroyFence(ctx_.device(), fence_, nullptr);
            fence_ = VK_NULL_HANDLE;
        }
        if (pool_ != VK_NULL_HANDLE)
        {
            // Destroying the pool frees the command buffer allocated from it.
            vkDestroyCommandPool(ctx_.device(), pool_, nullptr);
            pool_ = VK_NULL_HANDLE;
            cmd_  = VK_NULL_HANDLE;
        }
        scratch_.reset();
        pipeline_.reset();
    }

    void GpuKeepAlive::start() {
        if (thread_.joinable())
        {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(wakeMutex_);
            stopRequested_ = false;
        }
        thread_ = std::thread([this] {
            loop();
        });
    }

    void GpuKeepAlive::stop() {
        if (!thread_.joinable())
        {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(wakeMutex_);
            stopRequested_ = true;
        }
        wake_.notify_all();
        thread_.join();
    }

    bool GpuKeepAlive::running() const noexcept {
        return thread_.joinable();
    }

    uint64_t GpuKeepAlive::heartbeats() const noexcept {
        return heartbeats_.load(std::memory_order_relaxed);
    }

    uint32_t GpuKeepAlive::laneWidth() const noexcept {
        return laneWidth_;
    }

    void GpuKeepAlive::loop() {
        // A Vulkan error on the heartbeat path (device lost) ends the thread with a warning rather than
        // propagating out of std::thread; stop() still joins it, and inference is unaffected.
        try
        {
            constexpr int64_t            intervalNs = kGpuKeepAliveIntervalMs * kNsPerMs;
            std::unique_lock<std::mutex> lock(wakeMutex_);
            while (!stopRequested_)
            {
                const int64_t idleNs = steadyNowNs() - ctx_.lastQueueWorkSteadyNs();
                int64_t       waitNs = intervalNs;
                if (ctx_.queueWorkInFlight() == 0 && idleNs >= intervalNs)
                {
                    lock.unlock();
                    submitHeartbeat();
                    lock.lock();
                } else
                {
                    // Wake when the queue will have been idle for a full interval (a submission in
                    // flight re-checks after one interval), never sooner than the poll floor.
                    waitNs = std::min(std::max(intervalNs - idleNs, kGpuKeepAliveMinWaitMs * kNsPerMs), intervalNs);
                }
                wake_.wait_for(lock, std::chrono::nanoseconds(waitNs), [this] {
                    return stopRequested_;
                });
            }
        } catch (const std::exception &e)
        { VKNN_WARN << "GPU keep-alive stopped after " << heartbeats() << " heartbeat(s): " << e.what(); }
    }

    void GpuKeepAlive::submitHeartbeat() {
        VK_CHECK(vkResetFences(ctx_.device(), 1, &fence_));
        VkSubmitInfo si {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd_;
        {
            std::lock_guard<std::mutex> queueLock(ctx_.queueMutex());
            // A real submission that marked itself in flight between this thread's idle check and the
            // mutex takes precedence: yield rather than queue a heartbeat ahead of it.
            if (ctx_.queueWorkInFlight() > 0)
            {
                return;
            }
            VK_CHECK(vkQueueSubmit(ctx_.computeQueue(), 1, &si, fence_));
        }
        VK_CHECK(vkWaitForFences(ctx_.device(), 1, &fence_, VK_TRUE, kFenceWaitForever));
        heartbeats_.fetch_add(1, std::memory_order_relaxed);
    }

}} // namespace vknn::vk
