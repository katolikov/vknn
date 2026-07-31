// Host tests for the flat / vectorized-quad kernel family's decision rules
// (src/backend/vulkan/ops/flat_kernel_rules.h).
//
// Scope note: the rules are applied inside src/backend/vulkan, which is compiled only when
// VKNN_ENABLE_VULKAN is on, so no host test can prove that a real flat_binary_v4 dispatch reads a
// vec4-aligned base or that a race actually skipped its scratch allocation — those are device
// gates. What IS host-provable, and what this file covers, is every rule the operators consult:
// the vec4 base-alignment gate over a zero-copy view offset, the race-scratch budget, the workgroup
// width derived from device caps, the int32 element-count contract, the pointwise access class that
// keys a cached kernel verdict, and the cache-less verdict memo.
#include "backend/vulkan/ops/flat_kernel_rules.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {
    // Byte offset of the second member of a flat Concat/Split arena whose first member holds
    // `firstMemberElems` elements — the raw accumulated offset the segment planner assigns, with no
    // vector rounding.
    size_t arenaMemberOffsetBytes(int64_t firstMemberElems, bool fp16) {
        return (size_t) firstMemberElems * (fp16 ? kFp16StorageBytes : kFp32StorageBytes);
    }
} // namespace

// ---- vec4 base alignment ----

TEST(FlatKernelRules, QuadBindingBytesIsTheWholeVectorWidth) {
    EXPECT_EQ(quadBindingBytes(/*fp16=*/false), 16u);
    EXPECT_EQ(quadBindingBytes(/*fp16=*/true), 8u);
}

// A scalar kernel tolerates the offset; its quad twin does not. This is the offset a Concat whose
// first member holds 3 fp32 elements hands its second member.
TEST(FlatKernelRules, QuadBaseRejectsAnArenaOffsetThatIsNotAWholeVector) {
    const size_t quadBytes = quadBindingBytes(/*fp16=*/false);
    EXPECT_EQ(arenaMemberOffsetBytes(3, /*fp16=*/false), 12u);
    EXPECT_FALSE(quadBaseAligned(arenaMemberOffsetBytes(3, false), quadBytes));
    // The Split case: [1,16390] halved on axis 1 puts the second output at 8195 * 4 = 32780 B,
    // which is 4-byte aligned (so the view constructor accepts it) and 12 mod 16.
    EXPECT_EQ(arenaMemberOffsetBytes(8195, false), 32780u);
    EXPECT_EQ(arenaMemberOffsetBytes(8195, false) % sizeof(uint32_t), 0u);
    EXPECT_FALSE(quadBaseAligned(arenaMemberOffsetBytes(8195, false), quadBytes));
}

TEST(FlatKernelRules, QuadBaseAcceptsWholeVectorOffsets) {
    const size_t quadBytes = quadBindingBytes(/*fp16=*/false);
    EXPECT_TRUE(quadBaseAligned(0, quadBytes));
    EXPECT_TRUE(quadBaseAligned(arenaMemberOffsetBytes(4, false), quadBytes));
    EXPECT_TRUE(quadBaseAligned(arenaMemberOffsetBytes(8196, false), quadBytes));
}

// fp16 halves the vector width, so the failing element counts are the ones 2 mod 4.
TEST(FlatKernelRules, QuadBaseFollowsTheStoragePrecision) {
    const size_t quadBytes = quadBindingBytes(/*fp16=*/true);
    EXPECT_EQ(arenaMemberOffsetBytes(2, true), 4u);
    EXPECT_FALSE(quadBaseAligned(arenaMemberOffsetBytes(2, true), quadBytes));
    EXPECT_TRUE(quadBaseAligned(arenaMemberOffsetBytes(4, true), quadBytes));
    // An offset legal at one precision can be illegal at the other, so the gate must read the
    // running precision rather than a fixed width: 8 B is a whole f16vec4 and half a vec4.
    EXPECT_TRUE(quadBaseAligned(8u, quadBindingBytes(/*fp16=*/true)));
    EXPECT_FALSE(quadBaseAligned(8u, quadBindingBytes(/*fp16=*/false)));
}

// ---- race-scratch budget ----

TEST(FlatKernelRules, RaceScratchDeclinedWhenItWouldClaimTooMuchFreeMemory) {
    const size_t freeBytes = 1u << 20;
    EXPECT_TRUE(quadRaceScratchAffordable(freeBytes / kQuadRaceScratchBudgetDivisor, freeBytes));
    EXPECT_FALSE(quadRaceScratchAffordable(freeBytes / kQuadRaceScratchBudgetDivisor + 1, freeBytes));
    EXPECT_FALSE(quadRaceScratchAffordable(freeBytes, freeBytes));
}

TEST(FlatKernelRules, RaceScratchUnconstrainedWhenTheDeviceReportsNoBudget) {
    EXPECT_TRUE(quadRaceScratchAffordable(size_t(1) << 40, 0));
}

// ---- family workgroup width ----

TEST(FlatKernelRules, LaneWidthRoundsDownToWholeSubgroups) {
    EXPECT_EQ(laneWidthFrom(256, 1024, 1024, 64), 256u);
    EXPECT_EQ(laneWidthFrom(256, 128, 1024, 64), 128u);
    EXPECT_EQ(laneWidthFrom(256, 1024, 1024, 96), 192u);
    // A subgroup wider than the ceiling leaves no whole subgroup: the clamped width itself runs.
    EXPECT_EQ(laneWidthFrom(64, 1024, 1024, 128), 64u);
}

// A cap of 0 means "the device did not report this limit" and must not clamp to zero.
TEST(FlatKernelRules, LaneWidthTreatsAnUnreportedCapAsNoLimit) {
    EXPECT_EQ(laneWidthFrom(256, 0, 0, 0), 256u);
    EXPECT_EQ(laneWidthFrom(256, 0, 128, 64), 128u);
}

// The reported caps are a hard ceiling: a width above them fails pipeline creation at load, so the
// whole-subgroup rounding must never round UP past them.
TEST(FlatKernelRules, LaneWidthNeverExceedsTheReportedCaps) {
    const uint32_t maxInvocations = 32, subgroup = 64;
    const uint32_t width = laneWidthFrom(256, maxInvocations, 1024, subgroup);
    EXPECT_LE(width, maxInvocations);
    EXPECT_GE(width, kMinLaneWidth);
    EXPECT_EQ(width, maxInvocations);
    EXPECT_EQ(laneWidthFrom(256, 1024, 48, 64), 48u);
    EXPECT_EQ(laneWidthFrom(256, 8, 8, 64), 8u);
}

// ---- int32 element-count contract ----

TEST(FlatKernelRules, ElementCountPassesThroughUpToTheContract) {
    EXPECT_EQ(flatElementCount(0, "unit"), 0);
    EXPECT_EQ(flatElementCount(1234, "unit"), 1234);
    EXPECT_EQ(flatElementCount(kFlatElementCountMax, "unit"), (int) kFlatElementCountMax);
}

TEST(FlatKernelRules, ElementCountRefusesAboveTheContractInsteadOfWrapping) {
    EXPECT_THROW(flatElementCount(kFlatElementCountMax + 1, "unit"), Error);
    // The narrowing this replaces wrapped negative, which the shader's `total` grid guard reads as
    // an empty dispatch.
    EXPECT_THROW(flatElementCount(2400000000LL, "unit"), Error);
    EXPECT_LT((int) 2400000000LL, 0);
}

// ---- flat pointwise access class ----

namespace {
    // A two-step plan whose broadcast strides can be varied per step.
    struct PlanWords {
        int32_t step[2 * kPwStepInts] {};
        int32_t stride[2 * kPwMaxRank] {};
    };

    PlanWords makePlan(int kind0, int code0, int bcast0, int32_t innerStride0) {
        PlanWords p {};
        p.step[0 * kPwStepInts + kPwStepKindField]        = kind0;
        p.step[0 * kPwStepInts + kPwStepCodeField]        = code0;
        p.step[0 * kPwStepInts + kPwStepBcastField]       = bcast0;
        p.step[0 * kPwStepInts + kPwStepBcastSourceField] = 1;
        p.stride[0 * kPwMaxRank + kPwMaxRank - 1]         = innerStride0;
        p.step[1 * kPwStepInts + kPwStepKindField]        = kind0;
        p.step[1 * kPwStepInts + kPwStepCodeField]        = code0;
        p.step[1 * kPwStepInts + kPwStepBcastField]       = kPwBcastSame;
        return p;
    }
} // namespace

TEST(FlatKernelRules, AccessClassCarriesEveryStepsOpcodeAndBroadcastGeometry) {
    const PlanWords plan  = makePlan(1, 7, kPwBcastSame, 1);
    const auto      words = pwFlatAccessClass(plan.step, plan.stride, 2);
    ASSERT_EQ(words.size(), (size_t) 2 * (4 + kPwMaxRank));
    EXPECT_EQ(words[0], 1); // kind
    EXPECT_EQ(words[1], 7); // code
    EXPECT_EQ(words[2], kPwBcastSame);
    EXPECT_EQ(words[3], 1); // broadcast source field
}

// Two units equal in (total, step count, operand count, relax, precision) but differing in what
// their operands look like must not share one cached kernel verdict: a chain over full-size
// operands issues coalesced whole-vec4 loads, a per-channel broadcast chain issues scattered ones.
TEST(FlatKernelRules, SignatureSeparatesUnitsThatDifferOnlyInBroadcastClass) {
    const PlanWords   full      = makePlan(1, 7, kPwBcastSame, 1);
    const PlanWords   broadcast = makePlan(1, 7, kPwBcastChannel, 0);
    const int         total = 200704, operandCount = 2;
    const std::string fullSig      = pwFlatKernelSignature(total, operandCount, false, false, pwFlatAccessClass(full.step, full.stride, 2));
    const std::string broadcastSig = pwFlatKernelSignature(total, operandCount, false, false, pwFlatAccessClass(broadcast.step, broadcast.stride, 2));
    EXPECT_NE(fullSig, broadcastSig);
}

TEST(FlatKernelRules, SignatureSeparatesUnitsThatDifferOnlyInOpcode) {
    const PlanWords mulAdd = makePlan(1, 7, kPwBcastSame, 1);
    const PlanWords divSub = makePlan(1, 9, kPwBcastSame, 1);
    EXPECT_NE(pwFlatKernelSignature(1024, 2, false, false, pwFlatAccessClass(mulAdd.step, mulAdd.stride, 2)),
              pwFlatKernelSignature(1024, 2, false, false, pwFlatAccessClass(divSub.step, divSub.stride, 2)));
}

TEST(FlatKernelRules, SignatureIsStableForOneUnit) {
    const PlanWords plan = makePlan(1, 7, kPwBcastSame, 1);
    const auto      a    = pwFlatKernelSignature(1024, 2, false, true, pwFlatAccessClass(plan.step, plan.stride, 2));
    const auto      b    = pwFlatKernelSignature(1024, 2, false, true, pwFlatAccessClass(plan.step, plan.stride, 2));
    EXPECT_EQ(a, b);
}

TEST(FlatKernelRules, SignatureSeparatesRoundingDisciplineAndPrecision) {
    const PlanWords plan  = makePlan(1, 7, kPwBcastSame, 1);
    const auto      words = pwFlatAccessClass(plan.step, plan.stride, 2);
    EXPECT_NE(pwFlatKernelSignature(1024, 2, false, false, words), pwFlatKernelSignature(1024, 2, true, false, words));
    EXPECT_NE(pwFlatKernelSignature(1024, 2, false, false, words), pwFlatKernelSignature(1024, 2, false, true, words));
    EXPECT_NE(pwFlatKernelSignature(1024, 2, false, false, words), pwFlatKernelSignature(1024, 3, false, false, words));
}

// ---- cache-less verdict memo ----

TEST(FlatKernelRules, MemoAnswersASecondNodeCarryingTheSameSignature) {
    QuadVerdictMemo::clear();
    int kernel = -1;
    EXPECT_FALSE(QuadVerdictMemo::lookup("gpu/fbin_1", Tuning::Fast, kernel));
    QuadVerdictMemo::store("gpu/fbin_1", Tuning::Fast, 1);
    ASSERT_TRUE(QuadVerdictMemo::lookup("gpu/fbin_1", Tuning::Fast, kernel));
    EXPECT_EQ(kernel, 1);
    EXPECT_EQ(QuadVerdictMemo::size(), 1u);
    QuadVerdictMemo::clear();
}

// The level rule matches VkOpEnv::reuseTuned: a Fast entry does not answer a Heavy request, but
// Tuning::None reuses any measured entry.
TEST(FlatKernelRules, MemoHonorsTheMeasuredEffortLevel) {
    QuadVerdictMemo::clear();
    int kernel = -1;
    QuadVerdictMemo::store("gpu/fgather_1", Tuning::Fast, 1);
    EXPECT_FALSE(QuadVerdictMemo::lookup("gpu/fgather_1", Tuning::Heavy, kernel));
    EXPECT_TRUE(QuadVerdictMemo::lookup("gpu/fgather_1", Tuning::None, kernel));
    QuadVerdictMemo::store("gpu/fgather_1", Tuning::Heavy, 0);
    ASSERT_TRUE(QuadVerdictMemo::lookup("gpu/fgather_1", Tuning::Heavy, kernel));
    EXPECT_EQ(kernel, 0);
    // A later Fast measurement must not demote the heavier entry.
    QuadVerdictMemo::store("gpu/fgather_1", Tuning::Fast, 1);
    ASSERT_TRUE(QuadVerdictMemo::lookup("gpu/fgather_1", Tuning::Heavy, kernel));
    EXPECT_EQ(kernel, 0);
    QuadVerdictMemo::clear();
}

TEST(FlatKernelRules, MemoKeepsDeviceSignaturesApart) {
    QuadVerdictMemo::clear();
    int kernel = -1;
    QuadVerdictMemo::store("gpuA/fbin_1", Tuning::Fast, 1);
    EXPECT_FALSE(QuadVerdictMemo::lookup("gpuB/fbin_1", Tuning::Fast, kernel));
    QuadVerdictMemo::clear();
}
