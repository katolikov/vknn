// The pre-wake fence wait policy (vk_fence_wait_policy.h) is pure arithmetic; these pin the
// contract the device path relies on: the sleeping wait ends a margin before the prediction, a
// prediction inside the margin polls from the start, the poll budget always covers the remaining
// predicted time plus the spin allowance, and no timeout is ever negative.
#include "backend/vulkan/vk_fence_wait_policy.h"
#include <gtest/gtest.h>

using namespace vknn::vk;

TEST(FenceWaitPolicy, SleepEndsAMarginBeforeThePrediction) {
    EXPECT_DOUBLE_EQ(fencePreWakeSleepMs(8.0), 8.0 - kFencePreWakeMarginMs);
    EXPECT_DOUBLE_EQ(fencePreWakeSleepMs(kFencePreWakeMarginMs + 0.25), 0.25);
}

TEST(FenceWaitPolicy, ShortOrUnknownPredictionPollsFromTheStart) {
    EXPECT_DOUBLE_EQ(fencePreWakeSleepMs(0.0), 0.0);
    EXPECT_DOUBLE_EQ(fencePreWakeSleepMs(kFencePreWakeMarginMs), 0.0);
    EXPECT_DOUBLE_EQ(fencePreWakeSleepMs(0.5 * kFencePreWakeMarginMs), 0.0);
    EXPECT_DOUBLE_EQ(fencePreWakeSleepMs(-3.0), 0.0);
}

TEST(FenceWaitPolicy, PollBudgetCoversTheRemainingPredictionPlusTheSpinAllowance) {
    // A long prediction: the sleep leaves exactly the margin to poll through.
    EXPECT_DOUBLE_EQ(fencePollBudgetMs(8.0), kFencePreWakeMarginMs + kFenceSpinBudgetMs);
    // A prediction inside the margin: the whole prediction is polled.
    EXPECT_DOUBLE_EQ(fencePollBudgetMs(0.4), 0.4 + kFenceSpinBudgetMs);
    // No prediction: only the spin allowance.
    EXPECT_DOUBLE_EQ(fencePollBudgetMs(0.0), kFenceSpinBudgetMs);
    EXPECT_DOUBLE_EQ(fencePollBudgetMs(-1.0), kFenceSpinBudgetMs);
}

TEST(FenceWaitPolicy, TimeoutConversionIsNanosecondsAndNeverNegative) {
    EXPECT_EQ(fenceTimeoutNs(1.5), (uint64_t) 1500000);
    EXPECT_EQ(fenceTimeoutNs(0.0), (uint64_t) 0);
    EXPECT_EQ(fenceTimeoutNs(-2.0), (uint64_t) 0);
}
