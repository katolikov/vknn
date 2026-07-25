// Shared timing discipline for the load-time kernel races (matmul.cpp pickTile, the conv.cpp
// tile/local-size races, conv_gemm.cpp pickVariant).
//
// Two independent problems, one per type below.
//
// ORDER (raceCandidates). Timing one candidate to completion and then the next hands every later
// candidate a GPU whose clocks the earlier candidates already moved: the device ramps up over the
// first submits and throttles back under sustained load, and that drift is comparable to the
// differences the races are trying to resolve. raceCandidates removes the order term instead of
// hoping it is small - the candidates alternate one submit at a time and the round order reverses
// every other round, so every candidate occupies the same mean position in the race.
//
// OBJECTIVE (TuneTimer). A race decides which kernel the graph should run, so it has to measure
// what the graph will experience. Repeating one dispatch back to back on dedicated scratch buffers
// does not: from the second repetition the operands are cache-resident and the clocks are ramped,
// so the estimate rewards per-thread reuse. In the graph the same kernel runs ONCE, with ~100 other
// ops between two of its executions, and its operands come from DRAM. The two orderings disagree in
// sign, not just in magnitude - candidates that cut the thread count to raise per-thread reuse
// (wider output tiles, more output channel-blocks per thread) measure faster repeated and slower in
// place, because in place the only thing hiding the memory latency is having more threads in
// flight. TuneTimer therefore measures one un-repeated dispatch, behind a cache-eviction stream, on
// the GPU clock rather than on the submit wall.
#pragma once

#include "vk_common.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vknn {

    struct VkOpEnv;

    namespace vk {

        class Buffer;
        class CommandRunner;
        class ComputePipeline;
        class VulkanContext;

        /// Recorded rounds behind one candidate's estimate. Rounds alternate forward and reversed
        /// candidate order, so a monotone clock drift over a forward/reverse pair contributes the
        /// same amount to every candidate and cancels to first order; the count must stay even for
        /// that cancellation to be exact. Four rounds leave two samples on each side of the median.
        constexpr int kTuneRaceRounds = 4;

        /// Fraction of the incumbent's time a challenger must beat to replace it. Every race keeps its
        /// deterministic default as the incumbent and measures challengers against THAT time, so the
        /// margin cannot compound and the outcome does not depend on candidate order. The device
        /// throttles several-fold under sustained load and the isolated race's noise on small shapes is
        /// wider than the differences it resolves, so an unmargined challenger displaces a proven pick
        /// on noise alone.
        constexpr double kTuneRaceMargin = 0.97;

        /// Bytes the eviction kernel streams before each measured dispatch. Vulkan exposes no
        /// last-level-cache size, so this is a fixed budget chosen to sit far above the few MB of
        /// GPU L2 + system-level cache on the mobile parts this engine targets, and far below the
        /// point where the stream itself dominates cold load. Traffic is one read pass, so the cost
        /// is this figure divided by achievable read bandwidth.
        constexpr size_t kTuneEvictBytes = 24u << 20;

        /// Value the eviction kernel's data-dependent store compares against. The stream buffer is
        /// zero-filled, so the compared xor is always 0 and the store never fires; a non-zero
        /// sentinel is what keeps the compiler from proving that and deleting the loads.
        constexpr uint32_t kTuneEvictSentinel = 0xa5a5a5a5u;

        /// Times a candidate the way the graph will run it: one dispatch, cold operands, GPU clock.
        ///
        /// Each call streams kTuneEvictBytes through a scratch buffer, barriers, and brackets the
        /// caller's recorded work in timestamp queries. Timestamps rather than the submit wall
        /// because a single small dispatch (a depthwise conv on a mid-size feature map runs ~0.06 ms)
        /// is an order of magnitude below the submit + fence latency around it, so the wall time of a
        /// one-dispatch submit measures the driver, not the kernel.
        ///
        /// Degrades instead of failing: a device without timestamp support, a query pool that cannot
        /// be created, or a stream buffer that cannot be allocated leaves the corresponding stage out
        /// and the estimate falls back to the submit wall. The race still runs; it just measures less
        /// well on a device that cannot support the better measurement.
        ///
        /// Owns its query pool and stream buffer (RAII); not copyable or movable. One instance per
        /// race, constructed after the race has decided it will actually sweep.
        class TuneTimer {
          public:
            explicit TuneTimer(VkOpEnv &env);
            ~TuneTimer();
            TuneTimer(const TuneTimer &)            = delete;
            TuneTimer &operator=(const TuneTimer &) = delete;
            TuneTimer(TuneTimer &&)                 = delete;
            TuneTimer &operator=(TuneTimer &&)      = delete;

            /// Record `recordOnce(cmd)` behind the eviction stream, submit, wait, and return the GPU
            /// time of the recorded work in milliseconds (the submit wall when timestamps are
            /// unavailable). `recordOnce` records the candidate's work ONCE - one dispatch for a
            /// single-kernel candidate, the whole sequence for a candidate the winner replays as
            /// several dispatches (an output-channel split, a Winograd 3-pass), because that
            /// sequence is what the graph schedules as one op.
            double time(const std::function<void(VkCommandBuffer)> &recordOnce);

            /// True when `time()` reports GPU timestamps rather than the submit wall.
            bool onGpuClock() const noexcept {
                return pool_ != VK_NULL_HANDLE;
            }

          private:
            VulkanContext                   *ctx_    = nullptr;
            CommandRunner                   *runner_ = nullptr;
            VkQueryPool                      pool_   = VK_NULL_HANDLE;
            double                           period_ = 0.0; // timestamp ticks -> nanoseconds
            std::shared_ptr<Buffer>          stream_;
            std::shared_ptr<ComputePipeline> evictPipe_;
            uint32_t                         evictGroups_ = 0;
            uint32_t                         evictPc_[2]  = {0, 0}; // { vec4 count, sentinel }
        };

        /// Race `count` candidates and return one millisecond estimate per candidate (index-aligned).
        /// `submitOnce(index)` records and submits that candidate's work once and returns its
        /// measured time (TuneTimer::time); the caller keeps ownership of pipelines, scratch buffers
        /// and geometry.
        ///
        /// Every candidate gets one discarded warm-up submit before the recorded rounds, so first-use
        /// pipeline costs and the initial clock ramp are paid before any candidate is on the clock.
        /// The reported estimate is the median of the candidate's per-round samples: unlike a min it
        /// does not reward whichever candidate happened to catch the one quiet moment, and unlike a
        /// mean a single contaminated sample (an OS scheduling hiccup, a throttle step) does not move
        /// it. `rounds` is rounded up to the next even count.
        std::vector<double> raceCandidates(int count, const std::function<double(int)> &submitOnce, int rounds = kTuneRaceRounds);

        /// The estimates as a debug-log field, e.g. " ms=[0.061 0.078 0.070]" (empty for no
        /// candidates). Each race prints its winner at Debug; printing the times behind it is what
        /// makes a surprising pick diagnosable without rebuilding the engine.
        std::string raceTimes(const std::vector<double> &estimates);

    } // namespace vk
} // namespace vknn
