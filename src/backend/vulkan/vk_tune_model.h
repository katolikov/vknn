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
//           + max( streamBytes / (reReadRate(streamFootprint) * eff)
//                    + residentBytes / (reReadRate(residentFootprint) * eff),
//                  footprintBytes / streamBytesPerMs )
//
// The issued loads split by which operand they hit, and each side's re-reads are charged at a rate
// set by how much data that side spans. A conv reads its activation window once per output and its
// weight block once per output pixel, so raising the output-channel block halves the activation
// re-reads while raising the pixel tile halves the weight re-reads; two tiles can issue the same
// total and behave differently, which one lumped issued figure cannot express. What decides how
// much each side's re-reads cost is not WHICH operand it is but how big it is: an operand that fits
// in the last-level cache is re-read at the cache's rate, one that does not at DRAM's. Both cases
// occur in the same network - a 299x299 stem streams its activation past any cache while its weight
// set is a few KB, and a 17x17 tower with 768 channels holds its whole activation map in cache
// while carrying a weight set several times larger. Charging a side by its identity rather than by
// its size inverts the ranking on the second class.
//
// Below wavesToSaturate a candidate that cuts the thread count loses more to un-hidden latency
// than it wins back in traffic, which is the in-situ inversion the races measured: a tile that is
// faster timed alone is slower in a graph where the same op runs once among ~100 others. Above it
// the traffic term decides. The footprint term is a floor no tiling can go under; it is why the
// candidates of a saturated dispatch land far closer together than their issued traffic suggests -
// the redundant reads are cache hits.
//
// HOW IT IS USED. It removes candidates before the race, and it settles a comparison the race
// cannot; it never overrides a measurement that resolves. Standalone selection is refused by
// measurement - over the raced decisions of the 10-model CNN suite plus the YoNoSplat encoder the
// model alone is a worse chooser than racing a shortlist, so it is not allowed to replace a
// difference the race can see. The second role is narrow and named in vk_tune_race.h: a challenger
// the race puts ahead of the incumbent by less than kTuneRaceMargin is taken when the model also
// ranks it cheaper, because what that margin exists to discard is one noisy sample, and two
// independent signals pointing the same way are not that.
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
//
// The ends do not bound everything: an interior candidate the model ranks second sits on no
// frontier and is pruned, so the ranking has to be right about which of two neighbours is cheaper.
// That is what makes the per-side rate load-bearing. Inception-v3's factorized 1x7/7x1 towers hold
// their whole activation map in cache and carry a weight set several times larger; priced by
// operand identity, the sliding-window tile that wins them ranks second behind one that dispatches
// half the waves, is pruned, and the race never sees it.
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
            /// Achievable rate, in bytes per millisecond, for re-reads of a set that fits in the
            /// last-level cache - measured at kTuneModelResidentFootprintBytes.
            double residentBytesPerMs = 400.0e6;
            /// Achievable rate, in bytes per millisecond, for re-reads of a set far too large to
            /// cache - measured at kTuneModelStreamFootprintBytes.
            double streamBytesPerMs = 15.0e6;
            /// Per-dispatch submit-side overhead in milliseconds.
            double launchMs = 0.010;
            /// True once the probes have run on this device; false means the defaults are in use.
            bool calibrated = false;
        };

        /// One candidate's cost inputs, all derived from geometry the race already has. The two
        /// sides are the activation side (input map + output map) and the weight side; each carries
        /// what the kernel ISSUES against it and how much distinct data it SPANS, because the second
        /// is what sets the price of the first.
        struct KernelCost {
            /// Global vec4 the kernel issues against the activation side, summed over every thread.
            double streamVec4 = 0.0;
            /// Global vec4 the kernel issues against the weight side, summed over every thread.
            double residentVec4 = 0.0;
            /// Distinct vec4 the activation side spans: the input map plus the output map.
            double streamFootprintVec4 = 0.0;
            /// Distinct vec4 the weight side spans. Identical for every candidate of one race, as
            /// is streamFootprintVec4 - only the issued figures and the wave count differ.
            double residentFootprintVec4 = 0.0;
            /// Dispatched 64-wide waves (workgroups * localSize / 64).
            double waves = 1.0;
            /// Back-to-back dispatches the winner replays as one op (an output-channel split).
            int dispatches = 1;
        };

        /// Bytes in one fp16 vec4 - the element the packed NC4HW4 kernels load and store.
        constexpr double kTuneModelVec4Bytes = 8.0;

        /// Footprint residentBytesPerMs is measured at: small enough to stay in L1/L2 for a whole
        /// sweep, so its rate is what an already-fetched load costs.
        constexpr double kTuneModelResidentFootprintBytes = 128.0 * 1024.0;
        /// Footprint streamBytesPerMs is measured at: far above the few MB of GPU L2 plus
        /// system-level cache on the mobile parts this engine targets, so its rate is DRAM's.
        constexpr double kTuneModelStreamFootprintBytes = 24.0 * 1024.0 * 1024.0;

        /// Rate, in bytes per millisecond, at which a set of `footprintBytes` is re-read. The two
        /// probes bracket the range and anything between them interpolates: the model needs no
        /// cache-size constant, only the two points it already measures.
        double reReadBytesPerMs(double footprintBytes, const TuneModelCaps &caps);

        /// Modelled milliseconds for one candidate.
        double modelMs(const KernelCost &cost, const TuneModelCaps &caps);

        /// modelMs of every candidate, index-aligned with `costs`. A race consults these to settle
        /// a comparison its measurement cannot resolve - see kTuneRaceMargin in vk_tune_race.h.
        std::vector<double> modelEstimates(const std::vector<KernelCost> &costs, const TuneModelCaps &caps);

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
        std::vector<double> racePruned(const std::vector<KernelCost> &costs, const TuneModelCaps &caps, const std::function<double(int)> &submitOnce, const std::vector<int> &alwaysKeep = {});

    } // namespace vk
} // namespace vknn
