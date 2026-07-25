#include "vk_tune_race.h"
#include "vk_buffer.h"
#include "vk_command.h"
#include "vk_context.h"
#include "vk_op_env.h"
#include "vk_pipeline.h"
#include <algorithm>
#include <cstdio>

namespace vknn { namespace vk {

    namespace {
        // Middle sample of a copy of `values` (mean of the two middle ones for an even count).
        double medianOf(std::vector<double> values) {
            std::sort(values.begin(), values.end());
            const size_t mid = values.size() / 2;
            return values.size() % 2 == 1 ? values[mid] : (values[mid - 1] + values[mid]) * 0.5;
        }

        // Threads per workgroup in shaders/cache_evict.comp.
        constexpr uint32_t kEvictLocalSize = 256;
        // Bytes one eviction thread loads (one uvec4).
        constexpr size_t kEvictBytesPerThread = 16;
        // Queries per measurement: the pair bracketing the recorded work.
        constexpr uint32_t kTimestampsPerSample = 2;
    } // namespace

    TuneTimer::TuneTimer(VkOpEnv &env): ctx_(env.ctx), runner_(env.runner) {
        if (!ctx_ || !runner_)
        {
            return;
        }
        if (ctx_->caps().timestampSupported && ctx_->caps().timestampPeriod > 0.f)
        {
            VkQueryPoolCreateInfo qi {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
            qi.queryCount = kTimestampsPerSample;
            if (vkCreateQueryPool(ctx_->device(), &qi, nullptr, &pool_) != VK_SUCCESS)
            {
                pool_ = VK_NULL_HANDLE; // wall-time fallback; the race still runs
            }
            period_ = (double) ctx_->caps().timestampPeriod;
        }
        // The stream buffer and its kernel are best-effort too: on a device too tight to hold
        // kTuneEvictBytes the race degrades to warm operands rather than failing the model load.
        try
        {
            stream_                = std::make_shared<Buffer>(*ctx_, kTuneEvictBytes, MemPref::kDeviceOnly);
            evictPipe_             = env.pipeline("cache_evict", 1, sizeof(evictPc_));
            const size_t vec4Count = kTuneEvictBytes / kEvictBytesPerThread;
            evictGroups_           = (uint32_t) ((vec4Count + kEvictLocalSize - 1) / kEvictLocalSize);
            evictPc_[0]            = (uint32_t) vec4Count;
            evictPc_[1]            = kTuneEvictSentinel;
            // Zero the stream once: the kernel's store is then never taken at runtime, so repeated
            // evictions neither write the buffer nor race each other over element 0.
            runner_->oneShot([&](VkCommandBuffer cmd) {
                vkCmdFillBuffer(cmd, stream_->handle(), 0, kTuneEvictBytes, 0);
            });
        } catch (const std::exception &)
        {
            stream_.reset();
            evictPipe_.reset();
        }
    }

    TuneTimer::~TuneTimer() {
        if (pool_ && ctx_)
        {
            vkDestroyQueryPool(ctx_->device(), pool_, nullptr);
        }
    }

    double TuneTimer::time(const std::function<void(VkCommandBuffer)> &recordOnce) {
        VkCommandBuffer cmd = runner_->allocate();
        runner_->begin(cmd);
        if (pool_)
        {
            vkCmdResetQueryPool(cmd, pool_, 0, kTimestampsPerSample);
        }
        if (evictPipe_ && stream_)
        {
            evictPipe_->dispatch(cmd, {stream_->handle()}, evictPc_, sizeof(evictPc_), evictGroups_);
            // The candidate's loads must not be served out of what the stream left behind, and the
            // opening timestamp must not resolve until the stream has drained.
            computeBarrier(*ctx_, cmd);
        }
        // BOTTOM_OF_PIPE at both ends: the opening query resolves once everything recorded before
        // it has completed (the eviction stream), the closing one once the candidate has. The span
        // between them is the candidate's own execution, with no submit or fence latency in it.
        if (pool_)
        {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool_, 0);
        }
        recordOnce(cmd);
        if (pool_)
        {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool_, 1);
        }
        runner_->end(cmd);
        const double wallMs = runner_->submitAndWait(cmd);
        vkFreeCommandBuffers(ctx_->device(), runner_->pool(), 1, &cmd);
        if (!pool_)
        {
            return wallMs;
        }
        uint64_t ticks[kTimestampsPerSample] = {0, 0};
        const VkResult read = vkGetQueryPoolResults(ctx_->device(), pool_, 0, kTimestampsPerSample, sizeof(ticks), ticks, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (read != VK_SUCCESS || ticks[1] <= ticks[0])
        {
            return wallMs; // unavailable or wrapped: fall back rather than report a nonsense span
        }
        constexpr double kNanosPerMilli = 1e6;
        return (double) (ticks[1] - ticks[0]) * period_ / kNanosPerMilli;
    }

    std::string raceTimes(const std::vector<double> &estimates) {
        if (estimates.empty())
        {
            return {};
        }
        constexpr int kMsDecimals = 3;
        std::string   out         = " ms=[";
        char          cell[32];
        for (size_t i = 0; i < estimates.size(); ++i)
        {
            snprintf(cell, sizeof(cell), "%.*f", kMsDecimals, estimates[i]);
            out += (i ? " " : "");
            out += cell;
        }
        return out + "]";
    }

    std::vector<double> raceCandidates(int count, const std::function<double(int)> &submitOnce, int rounds) {
        if (count <= 0)
        {
            return {};
        }
        const int evenRounds = std::max(2, rounds + (rounds & 1));
        // Warm-up: one discarded submit per candidate, in race order. Nothing here is recorded,
        // so the first-use pipeline cost and the clock ramp land outside every estimate.
        for (int candidate = 0; candidate < count; ++candidate)
        {
            submitOnce(candidate);
        }
        std::vector<std::vector<double>> samples((size_t) count);
        for (int round = 0; round < evenRounds; ++round)
        {
            const bool reversed = (round % 2) == 1;
            for (int slot = 0; slot < count; ++slot)
            {
                const int candidate = reversed ? (count - 1 - slot) : slot;
                samples[(size_t) candidate].push_back(submitOnce(candidate));
            }
        }
        std::vector<double> estimates((size_t) count);
        for (int candidate = 0; candidate < count; ++candidate)
        {
            estimates[(size_t) candidate] = medianOf(samples[(size_t) candidate]);
        }
        return estimates;
    }

}} // namespace vknn::vk
