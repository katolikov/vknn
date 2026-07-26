// Analytical cost model for the load-time kernel races (src/backend/vulkan/vk_tune_race.h).
//
// WHY. A race pays for every candidate it times: each losing tile compiles its own pipeline
// variant, and on the CNN suite that compilation - not the timing - is most of the cold-load cost.
// The model does not have to name the winner to remove that cost; it only has to say which
// candidates cannot win, so the race measures a shortlist instead of the whole zoo.
//
// WHAT IT MODELS. Every candidate in a race computes the same outputs with the same arithmetic, so
// FLOPs and the compulsory memory footprint are identical across the list and only two things
// differ: how many global loads the kernel ISSUES (a wider tile reuses an operand in registers
// instead of re-reading it) and how many waves it dispatches (a wider tile runs fewer threads).
// Those two pull against each other, and which one wins is set by whether the dispatch is large
// enough to hide memory latency:
//
//     eff   = min(1, (waves / wavesToSaturate) ^ latencyExponent)
//     t     = launchMs * dispatches
//           + max( streamBytes / (streamBytesPerMs * eff) + residentBytes / (residentBytesPerMs * eff),
//                  footprintBytes / streamBytesPerMs )
//
// The issued loads split by which operand they hit. A conv's activation window is read once per
// output and its map is far larger than any cache, so re-reading it costs DRAM; its weight set is
// small enough to stay resident, so re-reading THAT costs a cache hit. Two tiles can issue the same
// total and behave differently: raising the output-channel block halves the activation re-reads,
// raising the pixel tile halves the weight re-reads, and only the first buys anything. Modelling
// one lumped issued figure makes those two indistinguishable, which is exactly the class the
// pointwise races resolve in favour of the channel block.
//
// Below wavesToSaturate a candidate that cuts the thread count loses more to un-hidden latency
// than it wins back in traffic, which is the in-situ inversion the races measured: a tile that is
// faster timed alone is slower in a graph where the same op runs once among ~100 others. Above it
// the traffic term decides. The footprint term is a floor no tiling can go under; it is why the
// candidates of a saturated dispatch land far closer together than their issued traffic suggests -
// the redundant reads are cache hits.
//
// HOW IT IS USED. As a PRUNER, never as the decision. Measured over 197 raced decisions on the
// 10-model CNN suite plus the YoNoSplat encoder, the model alone picks the raced winner 76% of the
// time and leaves 4.9% mean time on the table; racing its shortlist leaves 1.05%, against 16.0%
// for taking the deterministic default and 0.09% for racing everything. So the model is not
// accurate enough to replace measurement, and it is accurate enough to halve it.
//
// The shortlist keeps five indices: the deterministic default (so a pruned race can never do worse
// than Tuning::None), the model's own best, and three ENDS of the trade-off it is interpolating -
// the candidate that dispatches the most waves, the one that issues the least traffic in total, and
// the one that issues the least STREAMING traffic. Keeping the ends is what bounds the damage when
// the model puts the crossover in the wrong place: it can misjudge the interior, but the winner is
// on a frontier. Each end earns its slot on a measured failure. Without the total-traffic pick the
// worst case is 323% (the YoNoSplat patch-embed conv, where the widest tile moves 5x less traffic
// than the default); without the streaming-traffic pick it is 34% and ResNet-50 loses a
// reproducible 4% of runtime to five pointwise convs whose winning tile is an interior point.
#pragma once

#include <functional>
#include <vector>

namespace vknn {

    struct VkOpEnv;

    namespace vk {

        /// Constants of the model, calibrated per GPU architecture (never per shape). Defaults are
        /// the fitted Xclipse-class values, used only when calibration cannot run.
        struct TuneModelCaps {
            /// 64-wide waves a dispatch needs before more of them stop buying throughput. Scales
            /// with the compute-unit count, which Vulkan does not expose - hence the probe.
            double wavesToSaturate = 400.0;
            /// How sharply throughput falls off below saturation (1 = proportional).
            double latencyExponent = 1.0;
            /// Achievable rate, in bytes per millisecond, for loads that hit a cache-resident
            /// operand (a conv's weight set, a GEMM's B panel).
            double residentBytesPerMs = 400.0e6;
            /// Achievable rate, in bytes per millisecond, for a cold streaming read (a conv's
            /// activation map, a GEMM's A panel and output).
            double streamBytesPerMs = 15.0e6;
            /// Per-dispatch submit-side overhead in milliseconds.
            double launchMs = 0.010;
            /// True once the probes have run on this device; false means the defaults are in use.
            bool calibrated = false;
        };

        /// One candidate's cost inputs, all derived from geometry the race already has.
        struct KernelCost {
            /// Global vec4 the kernel issues against the STREAMING operand - the activation map
            /// and the output - summed over every thread.
            double streamVec4 = 0.0;
            /// Global vec4 the kernel issues against the CACHE-RESIDENT operand - the weight set -
            /// summed over every thread.
            double residentVec4 = 0.0;
            /// Distinct vec4 the dispatch touches (inputs + weights + outputs): the compulsory
            /// traffic, identical for every candidate of one race.
            double footprintVec4 = 0.0;
            /// Dispatched 64-wide waves (workgroups * localSize / 64).
            double waves = 1.0;
            /// Back-to-back dispatches the winner replays as one op (an output-channel split).
            int dispatches = 1;
        };

        /// Bytes in one fp16 vec4 - the element the packed NC4HW4 kernels load and store.
        constexpr double kTuneModelVec4Bytes = 8.0;

        /// Modelled milliseconds for one candidate.
        double modelMs(const KernelCost &cost, const TuneModelCaps &caps);

        /// Indices worth measuring, ascending and duplicate-free. Always contains 0 (the race's
        /// deterministic incumbent). See the header comment for why the frontier picks are in it.
        std::vector<int> analyticShortlist(const std::vector<KernelCost> &costs, const TuneModelCaps &caps);

        /// This device's constants: probed once per process, then persisted in the tune table so
        /// later loads skip the probes. Falls back to the defaults (calibrated = false) on a device
        /// where the probe cannot run; the shortlist tolerates it, since a 2-3x error in any single
        /// constant does not change which candidates sit on the frontier.
        const TuneModelCaps &deviceTuneModel(VkOpEnv &env);

        /// Race only `analyticShortlist(costs, ...)` and return one estimate per CANDIDATE, with a
        /// pruned candidate's slot left at infinity. Callers keep their existing "must beat the
        /// incumbent's time" selection loop unchanged: an infinite estimate never wins.
        std::vector<double> racePruned(const std::vector<KernelCost> &costs, const TuneModelCaps &caps, const std::function<double(int)> &submitOnce);

    } // namespace vk
} // namespace vknn
