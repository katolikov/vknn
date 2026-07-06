// Host tests for the device-weight pool bookkeeping (src/backend/vulkan/vk_weight_pool.h).
//
// The pool is the cross-plan-bucket sharing point for uploaded weight/bias/transformed-weight
// buffers: two op instances that reference the same initializer + pack-kind + precision share ONE
// device allocation, and the allocation is freed only when the last instance drops it. The template
// is deliberately free of any Vulkan handle, so its refcount/keying logic is exercised here on host
// with a stand-in buffer type; the Vulkan backend instantiates the same template with vk::Buffer.
#include "backend/vulkan/vk_weight_pool.h"
#include <gtest/gtest.h>
#include <memory>

using namespace vknn;

namespace {
    // Stand-in for vk::Buffer: counts live instances so a test can assert exactly one allocation.
    struct FakeBuf {
        int  tag;
        int *liveCounter;
        FakeBuf(int t, int *c): tag(t), liveCounter(c) {
            ++*liveCounter;
        }
        ~FakeBuf() {
            --*liveCounter;
        }
    };
} // namespace

// Two instances referencing the same (key, precision) share one buffer: the factory runs once and
// both handles point at the same object (refcount 2, one alloc).
TEST(WeightPool, SameKeySharesOneBuffer) {
    DeviceWeightPool<FakeBuf> pool;
    int                       live = 0, made = 0;
    auto                      make = [&] {
        ++made;
        return std::make_shared<FakeBuf>(1, &live);
    };

    auto a = pool.acquire("/conv1#w", /*fp16=*/true, make);
    auto b = pool.acquire("/conv1#w", /*fp16=*/true, make);

    EXPECT_EQ(made, 1);          // factory ran exactly once
    EXPECT_EQ(live, 1);          // one live allocation
    EXPECT_EQ(a.get(), b.get()); // same object
    // The pool holds the buffer weakly (frees with its last user, like uploadPooled), so the
    // only owners are the two acquired handles: a single-bucket model keeps today's live count.
    EXPECT_EQ(a.use_count(), 2);
}

// A different precision for the same key gets a distinct buffer (fp16 vs fp32 storage differ).
TEST(WeightPool, DistinctPrecisionDistinctBuffer) {
    DeviceWeightPool<FakeBuf> pool;
    int                       live = 0, made = 0;
    auto                      make = [&] {
        ++made;
        return std::make_shared<FakeBuf>(1, &live);
    };

    auto half = pool.acquire("/conv1#w", /*fp16=*/true, make);
    auto full = pool.acquire("/conv1#w", /*fp16=*/false, make);

    EXPECT_EQ(made, 2);
    EXPECT_EQ(live, 2);
    EXPECT_NE(half.get(), full.get());
}

// A different pack-kind (encoded in the key, e.g. "#w" vs "#wino2") gets a distinct buffer.
TEST(WeightPool, DistinctPackDistinctBuffer) {
    DeviceWeightPool<FakeBuf> pool;
    int                       live = 0, made = 0;
    auto                      make = [&] {
        ++made;
        return std::make_shared<FakeBuf>(1, &live);
    };

    auto dense = pool.acquire("/conv1#w", /*fp16=*/true, make);
    auto wino  = pool.acquire("/conv1#wino2", /*fp16=*/true, make);

    EXPECT_EQ(made, 2);
    EXPECT_EQ(live, 2);
    EXPECT_NE(dense.get(), wino.get());
}

// Freeing one instance keeps the buffer live for the other; freeing the last frees the buffer.
TEST(WeightPool, RefcountKeepsAliveUntilLastDrop) {
    DeviceWeightPool<FakeBuf> pool;
    int                       live = 0, made = 0;
    auto                      make = [&] {
        ++made;
        return std::make_shared<FakeBuf>(1, &live);
    };

    auto a = pool.acquire("/conv1#w", true, make);
    {
        auto b = pool.acquire("/conv1#w", true, make);
        EXPECT_EQ(live, 1);
    } // b (one instance) drops here
    EXPECT_EQ(live, 1); // still held by a
    a.reset();
    EXPECT_EQ(live, 0); // last instance dropped -> allocation freed
}

// After the last instance drops, a fresh acquire re-runs the factory (no stale handle returned).
TEST(WeightPool, ReacquireAfterFullReleaseRebuilds) {
    DeviceWeightPool<FakeBuf> pool;
    int                       live = 0, made = 0;
    auto                      make = [&] {
        ++made;
        return std::make_shared<FakeBuf>(1, &live);
    };

    pool.acquire("/conv1#w", true, make).reset(); // acquire and immediately drop
    EXPECT_EQ(live, 0);
    auto again = pool.acquire("/conv1#w", true, make);
    EXPECT_EQ(made, 2); // rebuilt, not a dangling handle
    EXPECT_EQ(live, 1);
}
