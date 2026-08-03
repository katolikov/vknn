// Decision rules of the flat / vectorized-quad kernel family, factored out of the operators that
// apply them. Every Vulkan handle stays on the other side of this header, so each rule is provable
// on a host build (tests/test_flat_kernel_rules.cpp) and the operators — flat_ops.h,
// fused_pointwise.cpp, convert_layout.cpp — are thin mirrors that feed them device values read at
// load.
//
// Four rules live here:
//   * the vec4 base-alignment gate every quad pick applies to its re-typed bindings,
//   * the load-time race-scratch budget,
//   * the family workgroup width derived from device caps,
//   * the int32 element-count contract the flat push constants assume,
// plus the process-lifetime verdict memo that keeps a cache-less load from re-racing one signature
// at every node that carries it.
#pragma once
#include "vknn/error.h"
#include "vknn/op_type.h"
#include "vknn/tuning.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace vknn {

    // ---- vec4 binding alignment ----

    // Elements one STORE_QUAD covers. Every _v4 twin of the family re-types its vectorized SSBO
    // bindings to a 4-element vector (vec4 in fp32, f16vec4 in fp16); mirrors FLAT_GATHER_QUAD in
    // shaders/flat_gather_v4.comp, FLAT_MOVE_QUAD in the broadcast/binary/pad twins, PW_FLAT_QUAD in
    // shaders/fused_pw_flat_v4.comp and CLAYOUT_QUAD in shaders/convert_layout_v4.comp.
    constexpr int kQuadBindingElems = 4;
    // Storage width of one element at each running precision.
    constexpr size_t kFp16StorageBytes = sizeof(uint16_t);
    constexpr size_t kFp32StorageBytes = sizeof(float);

    /// std430 base alignment of a re-typed binding: the whole vector's byte width — 16 B in fp32,
    /// 8 B in fp16.
    inline constexpr size_t quadBindingBytes(bool fp16) {
        return (size_t) kQuadBindingElems * (fp16 ? kFp16StorageBytes : kFp32StorageBytes);
    }

    /// True when a buffer whose device base sits `baseOffsetBytes` into its allocation may be bound
    /// under a `quadBytes`-wide vector SSBO declaration.
    ///
    /// A scalar kernel addresses its SSBO one element at a time and tolerates any element-aligned
    /// base; its _v4 twin addresses the same memory as STORE_QUAD, whose base must be a whole
    /// vector. Zero-copy Concat/Split views bind into arena memory at their member's accumulated
    /// byte offset, derived from raw element counts with no vector rounding and validated only
    /// against the driver-reported VkMemoryRequirements::alignment — so a view base of, say, 12 B is
    /// legal for the scalar kernel and illegal for the quad twin. vk::Buffer::baseAlignedTo mirrors
    /// this rule over a live buffer's offset.
    inline constexpr bool quadBaseAligned(size_t baseOffsetBytes, size_t quadBytes) {
        return quadBytes == 0 || baseOffsetBytes % quadBytes == 0;
    }

    // ---- load-time race-scratch budget ----

    // Share of the device memory still free that a load-time kernel race may hold as dedicated
    // scratch: the race allocates buffers shaped like the node's REAL input and output tensors, so a
    // large movement node would otherwise transiently double or triple its own footprint while the
    // model's working set is already resident.
    constexpr size_t kQuadRaceScratchBudgetDivisor = 4;

    /// True when a race needing `scratchBytes` of dedicated scratch may run against `freeDeviceBytes`
    /// of reported free device memory. `freeDeviceBytes` 0 means the device reports no budget, which
    /// leaves the race unconstrained (the allocation itself still throws if it cannot be served).
    /// Declining only costs a measurement: every quad pick is placement-only, so the deterministic
    /// scalar incumbent stands.
    inline constexpr bool quadRaceScratchAffordable(size_t scratchBytes, size_t freeDeviceBytes) {
        return freeDeviceBytes == 0 || scratchBytes <= freeDeviceBytes / kQuadRaceScratchBudgetDivisor;
    }

    // ---- family workgroup width ----

    // Wave width assumed when the device reports no subgroup size.
    constexpr uint32_t kDefaultSubgroupWidth = 64;
    // Narrowest workgroup the family ever dispatches with.
    constexpr uint32_t kMinLaneWidth = 1;

    /// Workgroup width for the element-parallel family on a device with these caps: the family's
    /// `ceiling` clamped to both workgroup limits, then rounded down to whole subgroups. A cap of 0
    /// reads as "not reported" and does not clamp.
    ///
    /// When the clamped width cannot host one whole subgroup the clamped width itself is the answer:
    /// a partial-subgroup workgroup still runs, whereas any width above the caps fails pipeline
    /// creation at load.
    inline constexpr uint32_t laneWidthFrom(uint32_t ceiling, uint32_t maxWorkGroupInvocations, uint32_t maxWorkGroupSizeX, uint32_t subgroupSize) {
        uint32_t width = ceiling;
        if (maxWorkGroupInvocations != 0u && maxWorkGroupInvocations < width)
        {
            width = maxWorkGroupInvocations;
        }
        if (maxWorkGroupSizeX != 0u && maxWorkGroupSizeX < width)
        {
            width = maxWorkGroupSizeX;
        }
        if (width < kMinLaneWidth)
        {
            return kMinLaneWidth;
        }
        const uint32_t subgroup = subgroupSize != 0u ? subgroupSize : kDefaultSubgroupWidth;
        const uint32_t whole    = width / subgroup * subgroup;
        return whole != 0u ? whole : width;
    }

    // ---- int32 element-count contract ----

    // Largest element count the flat family's push constants carry: every flat kernel guards its
    // grid with `PC { int total, items; }`, so an element count is an int32 by contract and a
    // silent narrowing would wrap the guard negative.
    constexpr int64_t kFlatElementCountMax = INT32_MAX;

    /// Narrow an element count to the int32 the flat push constants carry, refusing by name instead
    /// of wrapping. `what` names the tensor or unit in the refusal.
    inline int flatElementCount(int64_t elements, const std::string &what) {
        if (elements > kFlatElementCountMax)
        {
            throw Error(Status::Unsupported, what + " has " + std::to_string(elements) + " elements, above the flat kernels' int32 element-count contract of " + std::to_string(kFlatElementCountMax));
        }
        return (int) elements;
    }

    // ---- flat pointwise access class ----

    // Step-record field offsets read here, mirroring the record layout kPwStepInts names in
    // vknn/op_type.h (kind, code, srcA, srcB, srcC, dst, bcast, bcastSrc).
    constexpr int kPwStepKindField        = 0;
    constexpr int kPwStepCodeField        = 1;
    constexpr int kPwStepBcastSourceField = 7;

    /// The plan fields that set a flat pointwise unit's memory access pattern: per step the op kind,
    /// the op code, the broadcast class, the broadcast source field, and that step's per-axis
    /// broadcast strides. Two units whose class words match issue the same loads and stores at every
    /// element, so one raced scalar-vs-quad verdict answers for both; two units that differ here — a
    /// chain over full-size operands against one over per-channel broadcast operands — do not, and
    /// each races its own pattern.
    /// @param step   plan.step, kPwStepInts ints per step.
    /// @param stride plan.stride, kPwMaxRank ints per step.
    inline std::vector<int32_t> pwFlatAccessClass(const int32_t *step, const int32_t *stride, int numSteps) {
        std::vector<int32_t> words;
        if (step == nullptr || stride == nullptr || numSteps <= 0)
        {
            return words;
        }
        constexpr int kStepClassFields = 4; // kind, code, broadcast class, broadcast source field
        words.reserve((size_t) numSteps * (size_t) (kStepClassFields + kPwMaxRank));
        for (int s = 0; s < numSteps; ++s)
        {
            words.push_back(step[s * kPwStepInts + kPwStepKindField]);
            words.push_back(step[s * kPwStepInts + kPwStepCodeField]);
            words.push_back(step[s * kPwStepInts + kPwStepBcastField]);
            words.push_back(step[s * kPwStepInts + kPwStepBcastSourceField]);
            for (int k = 0; k < kPwMaxRank; ++k)
            {
                words.push_back(stride[s * kPwMaxRank + k]);
            }
        }
        return words;
    }

    // FNV-1a fold constants, and the byte width one fold step consumes.
    constexpr uint64_t kFnv1aOffsetBasis = 1469598103934665603ull;
    constexpr uint64_t kFnv1aPrime       = 1099511628211ull;
    constexpr unsigned kBitsPerByte      = 8;

    /// 64-bit FNV-1a fold over access-class words, so a signature stays a fixed-width token however
    /// long the step chain is.
    inline uint64_t foldAccessClass(const std::vector<int32_t> &words) {
        uint64_t hash = kFnv1aOffsetBasis;
        for (int32_t word: words)
        {
            for (unsigned byte = 0; byte < sizeof(int32_t); ++byte)
            {
                hash ^= (uint64_t) (uint8_t) ((uint32_t) word >> (byte * kBitsPerByte));
                hash *= kFnv1aPrime;
            }
        }
        return hash;
    }

    /// Tune-table signature of a flat pointwise unit's scalar-vs-quad pick. The element count, the
    /// operand count, the rounding discipline and the storage precision appear literally; the step
    /// chain and its broadcast geometry ride the folded access class, so two units share one cached
    /// verdict exactly when they run the same kernel over the same access pattern.
    inline std::string pwFlatKernelSignature(int total, int numOperands, bool relax, bool fp16, const std::vector<int32_t> &accessClass) {
        char        buf[128];
        const char *relaxTag = relax ? "rx" : "st";
        const char *precTag  = fp16 ? "f16" : "f32";
        snprintf(buf, sizeof(buf), "pwflat_%d_%d_%s_%s_%016llx", total, numOperands, relaxTag, precTag, (unsigned long long) foldAccessClass(accessClass));
        return std::string(buf);
    }

    // ---- load-time verdict memo ----

    /// Process-lifetime memo of scalar-vs-quad race verdicts, keyed by the same signature the
    /// persistent tune table uses.
    ///
    /// VkOpEnv::reuseTuned/WeightCache::setTuned answer a pick without racing only when the session
    /// HAS a weight cache. Without one, every node carrying a signature races that signature again:
    /// a full TuneTimer race per node instead of per signature, and two nodes of one graph can land
    /// on different kernels. This memo gives a cache-less load the same once-per-signature answer.
    /// The verdicts are placement-only — each _v4 twin is byte-identical to its scalar kernel at
    /// every element — so a retained entry can change timing and never bytes, and the signature
    /// carries VkOpEnv::gpuTag, so two devices never share one entry.
    class QuadVerdictMemo {
      public:
        /// Verdict stored for `sig`, when it was measured at an effort level this request may reuse:
        /// any level under Tuning::None (which runs no new race), otherwise a level at least the
        /// requested one — the rule VkOpEnv::reuseTuned applies to a cache entry.
        static bool lookup(const std::string &sig, Tuning requested, int &kernel) {
            std::lock_guard<std::mutex> held(tableLock());
            auto                        it = table().find(sig);
            if (it == table().end())
            {
                return false;
            }
            if (requested != Tuning::None && it->second.level < (int) requested)
            {
                return false;
            }
            kernel = it->second.kernel;
            return true;
        }
        /// Record a freshly raced verdict, replacing any entry measured at a lower effort level.
        static void store(const std::string &sig, Tuning measured, int kernel) {
            std::lock_guard<std::mutex> held(tableLock());
            auto                        it = table().find(sig);
            if (it != table().end() && it->second.level > (int) measured)
            {
                return;
            }
            table()[sig] = Entry {kernel, (int) measured};
        }
        /// Entries currently held.
        static size_t size() {
            std::lock_guard<std::mutex> held(tableLock());
            return table().size();
        }
        /// Drop every entry. Exists so a test starts from a known state; a session never calls it.
        static void clear() {
            std::lock_guard<std::mutex> held(tableLock());
            table().clear();
        }

      private:
        struct Entry {
            int kernel;
            int level;
        };
        static std::map<std::string, Entry> &table() {
            static std::map<std::string, Entry> entries;
            return entries;
        }
        static std::mutex &tableLock() {
            static std::mutex guard;
            return guard;
        }
    };

} // namespace vknn
