#include "vk_tune_model.h"
#include "vk_buffer.h"
#include "vk_command.h"
#include "vk_context.h"
#include "vk_op_env.h"
#include "vk_pipeline.h"
#include "vk_tune_race.h"
#include "vk_weight_cache.h"
#include "vknn/logging.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace vknn { namespace vk {

    namespace {
        /// Threads per workgroup in shaders/tune_probe.comp.
        constexpr uint32_t kProbeLocalSize = 64;
        /// Bytes one probe load moves (one uvec4).
        constexpr uint32_t kProbeBytesPerLoad = 16;
        /// Cold-stream probe footprint. Far above the few MB of GPU L2 + system-level cache on the
        /// mobile parts this engine targets, so its rate is DRAM's and not a cache's.
        constexpr size_t kProbeStreamBytes = 24u << 20;
        /// Cache-resident probe footprint: small enough to stay in L1/L2 for the whole sweep, so
        /// its rate is what a kernel's redundant (already-fetched) loads actually cost.
        constexpr size_t kProbeResidentBytes = 128u << 10;
        /// Loads each thread issues in the resident probes. Enough that the loop, not the launch,
        /// is what the measurement resolves.
        constexpr uint32_t kProbeLoadsPerThread = 64;
        /// Stride between one thread's successive loads, in uvec4. A whole workgroup's worth, so
        /// consecutive threads stay coalesced while successive loads land on different lines.
        constexpr uint32_t kProbeStride = kProbeLocalSize;
        /// Sentinel the probe's dead store compares against; see shaders/tune_probe.comp.
        constexpr uint32_t kProbeSentinel = 0xa5a5a5a5u;
        /// Workgroup counts of the saturation sweep. The knee of the throughput curve over this
        /// range is what wavesToSaturate reports; the range brackets the 74-800 waves the CNN
        /// suite's raced dispatches actually span.
        constexpr uint32_t kProbeSweepGroups[] = {16, 32, 64, 128, 256, 512, 1024, 2048};
        /// Fraction of the sweep's peak throughput that counts as saturated.
        constexpr double kProbeSaturationFraction = 0.90;

        /// Tune-table value scale: the constants are doubles and the table stores ints, so each is
        /// persisted as a fixed-point integer. Large enough that the rounding is far below the
        /// factor-of-two error the shortlist tolerates.
        constexpr double kPersistScale = 1000.0;
        /// Bumped whenever the constant set or its meaning changes, so a stale entry cannot decode.
        constexpr int kTuneModelVersion = 1;

        /// Slope of throughput against waves below saturation, on a log-log fit of the sweep.
        double fitExponent(const std::vector<double> &waves, const std::vector<double> &rate, double peak, double satWaves) {
            double sxx = 0.0, sxy = 0.0;
            int    n = 0;
            for (size_t i = 0; i < waves.size(); ++i)
            {
                if (waves[i] >= satWaves || rate[i] <= 0.0 || peak <= 0.0)
                {
                    continue;
                }
                // Both axes are relative to the saturation point, so the fitted line passes through
                // it by construction and the single free parameter is the slope.
                const double x = std::log(waves[i] / satWaves);
                const double y = std::log(rate[i] / peak);
                sxx += x * x;
                sxy += x * y;
                ++n;
            }
            if (n < 2 || sxx <= 0.0)
            {
                return 1.0;
            }
            // eff = (waves/sat)^exponent, so the throughput slope IS the exponent.
            return std::min(2.0, std::max(0.25, sxy / sxx));
        }

        /// Runs the probes. Returns false when the device cannot host them, leaving `out` untouched.
        bool calibrate(VkOpEnv &env, TuneModelCaps &out) {
            if (!env.ctx || !env.runner)
            {
                return false;
            }
            std::shared_ptr<Buffer>          stream, resident;
            std::shared_ptr<ComputePipeline> probe;
            try
            {
                stream   = std::make_shared<Buffer>(*env.ctx, kProbeStreamBytes, MemPref::kDeviceOnly);
                resident = std::make_shared<Buffer>(*env.ctx, kProbeResidentBytes, MemPref::kDeviceOnly);
                probe    = env.pipeline("tune_probe", 1, sizeof(uint32_t) * 4);
                env.runner->oneShot([&](VkCommandBuffer cmd) {
                    vkCmdFillBuffer(cmd, stream->handle(), 0, kProbeStreamBytes, 0);
                    vkCmdFillBuffer(cmd, resident->handle(), 0, kProbeResidentBytes, 0);
                });
            } catch (const std::exception &)
            { return false; }
            TuneTimer timer(env);
            auto      run = [&](Buffer &buf, size_t bytes, uint32_t perThread, uint32_t groups) {
                const uint32_t pc[4] = {(uint32_t) (bytes / kProbeBytesPerLoad), perThread, kProbeStride, kProbeSentinel};
                return timer.time([&](VkCommandBuffer cmd) {
                    probe->dispatch(cmd, {buf.handle()}, pc, sizeof(pc), groups);
                });
            };

            // Launch floor: one workgroup, one load each. Whatever is left is the submit-side cost
            // every candidate pays per dispatch.
            const double launchMs = run(*resident, kProbeResidentBytes, 1, 1);

            // Cold streaming rate: one pass over a footprint no cache holds, at a wave count far
            // above anything the sweep below finds saturating.
            const uint32_t streamGroups = (uint32_t) (kProbeStreamBytes / kProbeBytesPerLoad / kProbeLocalSize);
            const double   streamMs     = run(*stream, kProbeStreamBytes, 1, streamGroups);

            // Saturation sweep: fixed work per thread on a cache-resident footprint, so the only
            // thing changing across the sweep is how many waves are in flight.
            std::vector<double> waves, rate;
            double              peak = 0.0;
            for (uint32_t groups: kProbeSweepGroups)
            {
                const double ms = run(*resident, kProbeResidentBytes, kProbeLoadsPerThread, groups);
                if (ms <= 0.0)
                {
                    continue;
                }
                const double issued = (double) groups * kProbeLocalSize * kProbeLoadsPerThread * kProbeBytesPerLoad;
                waves.push_back((double) groups * kProbeLocalSize / 64.0);
                rate.push_back(issued / ms);
                peak = std::max(peak, rate.back());
            }
            if (waves.empty() || peak <= 0.0 || streamMs <= 0.0)
            {
                return false;
            }
            double satWaves = waves.back();
            for (size_t i = 0; i < waves.size(); ++i)
            {
                if (rate[i] >= peak * kProbeSaturationFraction)
                {
                    satWaves = waves[i];
                    break;
                }
            }
            out.wavesToSaturate = satWaves;
            out.latencyExponent = fitExponent(waves, rate, peak, satWaves);
            out.residentBytesPerMs = peak;
            out.streamBytesPerMs   = (double) kProbeStreamBytes / streamMs;
            out.launchMs        = std::max(0.0, launchMs);
            out.calibrated      = true;
            return true;
        }

        const char *kTuneModelSigStem = "/tunemodel";

        /// Persisted layout: one tune-table entry per constant, all under the device's own tag.
        struct PersistField {
            const char *name;
            double TuneModelCaps::*field;
        };
        constexpr PersistField kPersistFields[] = {
            {"w", &TuneModelCaps::wavesToSaturate}, {"e", &TuneModelCaps::latencyExponent}, {"i", &TuneModelCaps::residentBytesPerMs},
            {"d", &TuneModelCaps::streamBytesPerMs},  {"l", &TuneModelCaps::launchMs},
        };

        std::string fieldSig(const VkOpEnv &env, const char *name) {
            return env.gpuTag + kTuneModelSigStem + std::to_string(kTuneModelVersion) + "_" + name;
        }

        bool loadPersisted(VkOpEnv &env, TuneModelCaps &out) {
            if (!env.weights)
            {
                return false;
            }
            TuneModelCaps got;
            for (const PersistField &f: kPersistFields)
            {
                int       level  = -1;
                const int stored = env.weights->tuned(fieldSig(env, f.name), 0, &level);
                if (level < 0 || stored <= 0)
                {
                    return false;
                }
                got.*(f.field) = (double) stored / kPersistScale;
            }
            got.calibrated = true;
            out            = got;
            return true;
        }

        void persist(VkOpEnv &env, const TuneModelCaps &caps) {
            if (!env.weights)
            {
                return;
            }
            for (const PersistField &f: kPersistFields)
            {
                const double v = caps.*(f.field);
                // The table stores ints, so a constant below the fixed-point resolution would round
                // to 0 and read back as "absent" - clamp to the smallest representable value.
                const int stored = (int) std::max<long long>(1, std::llround(v * kPersistScale));
                env.weights->setTuned(fieldSig(env, f.name), stored, (int) env.tuning);
            }
        }
    } // namespace

    double modelMs(const KernelCost &cost, const TuneModelCaps &caps) {
        const double waves    = std::max(cost.waves, 1.0);
        const double eff      = std::max(1e-4, std::min(1.0, std::pow(waves / caps.wavesToSaturate, caps.latencyExponent)));
        const double issued   = cost.streamVec4 * kTuneModelVec4Bytes / (caps.streamBytesPerMs * eff) + cost.residentVec4 * kTuneModelVec4Bytes / (caps.residentBytesPerMs * eff);
        const double compulsory = cost.footprintVec4 * kTuneModelVec4Bytes / caps.streamBytesPerMs;
        return caps.launchMs * (double) std::max(1, cost.dispatches) + std::max(issued, compulsory);
    }

    std::vector<int> analyticShortlist(const std::vector<KernelCost> &costs, const TuneModelCaps &caps) {
        std::vector<int> keep;
        if (costs.empty())
        {
            return keep;
        }
        auto total    = [](const KernelCost &c) {
            return c.streamVec4 + c.residentVec4;
        };
        int cheapest = 0, leanest = 0, leanestStream = 0, widest = 0;
        for (int i = 1; i < (int) costs.size(); ++i)
        {
            if (modelMs(costs[(size_t) i], caps) < modelMs(costs[(size_t) cheapest], caps))
            {
                cheapest = i;
            }
            if (total(costs[(size_t) i]) < total(costs[(size_t) leanest]))
            {
                leanest = i;
            }
            if (costs[(size_t) i].streamVec4 < costs[(size_t) leanestStream].streamVec4)
            {
                leanestStream = i;
            }
            if (costs[(size_t) i].waves > costs[(size_t) widest].waves)
            {
                widest = i;
            }
        }
        // Index 0 is every race's deterministic incumbent and is never pruned: a pruned race can
        // then never resolve to something Tuning::None would not also dispatch.
        keep = {0, cheapest, leanest, leanestStream, widest};
        std::sort(keep.begin(), keep.end());
        keep.erase(std::unique(keep.begin(), keep.end()), keep.end());
        return keep;
    }

    const TuneModelCaps &deviceTuneModel(VkOpEnv &env) {
        // One calibration per process. The constants describe the device, not the model, so every
        // graph loaded in this process shares them.
        static TuneModelCaps caps;
        static bool          probed    = false;
        static bool          persisted = false;
        if (!probed)
        {
            probed = true;
            if (loadPersisted(env, caps))
            {
                persisted = true;
                VKNN_DEBUG << "tune model: reusing calibrated constants for " << env.gpuTag;
            } else
            {
                calibrate(env, caps);
                VKNN_DEBUG << "tune model: wavesToSaturate=" << caps.wavesToSaturate << " exponent=" << caps.latencyExponent << " resident=" << caps.residentBytesPerMs / 1e6 << " GB/s stream=" << caps.streamBytesPerMs / 1e6
                           << " GB/s launch=" << caps.launchMs * 1e3 << " us calibrated=" << (int) caps.calibrated;
            }
        }
        // The first model loaded in a process may have no writable cache (a --no-cache run, an
        // in-memory graph); persisting on the first load that DOES have one is what keeps the
        // probes off every later cold load.
        if (!persisted && caps.calibrated && env.weights)
        {
            persist(env, caps);
            persisted = true;
        }
        return caps;
    }

    std::vector<double> racePruned(const std::vector<KernelCost> &costs, const TuneModelCaps &caps, const std::function<double(int)> &submitOnce) {
        const std::vector<int> keep = analyticShortlist(costs, caps);
        // The race sees a dense list of survivors; the caller sees one slot per original candidate,
        // with a pruned slot at infinity so its existing selection loop skips it untouched.
        const std::vector<double> raced = raceCandidates((int) keep.size(), [&](int slot) {
            return submitOnce(keep[(size_t) slot]);
        });
        std::vector<double>       out(costs.size(), std::numeric_limits<double>::infinity());
        for (size_t slot = 0; slot < keep.size(); ++slot)
        {
            out[(size_t) keep[slot]] = raced[slot];
        }
        return out;
    }

}} // namespace vknn::vk
