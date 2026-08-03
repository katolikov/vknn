// Two tensors that hold the SAME payload under different shapes must share one allocation.
//
// A pass that regroups a constant — the rank-collapse pass reshaping a weight so a high-rank
// pointwise region becomes fusable — needs the identical bytes under a second tensor id. Assigning
// the buffer duplicates every byte: once in host memory for the life of the compile, and again in
// the artifact when both tensors survive. Sharing costs one pointer, and copy-on-write keeps the
// two independent the moment either is written.
#include "vknn/host_buffer.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

using namespace vknn;

namespace {
    std::vector<uint8_t> rampBytes(size_t n) {
        std::vector<uint8_t> v(n);
        std::iota(v.begin(), v.end(), (uint8_t) 1);
        return v;
    }
} // namespace

TEST(HostBufferSharing, SharedBuffersReportTheSameBytesFromOneAllocation) {
    ByteStorage source;
    source = rampBytes(64);

    ByteStorage view;
    view.setSharedBytes(source.shareBytes());

    // Observation goes through const references: the MUTABLE data() materializes on purpose, so
    // reading through it would dissolve the very sharing under test.
    const ByteStorage &sourceRead = source;
    const ByteStorage &viewRead   = view;
    ASSERT_EQ(viewRead.size(), sourceRead.size());
    EXPECT_EQ(viewRead.data(), sourceRead.data()) << "sharing must hand out one allocation, not two";
    EXPECT_EQ(std::vector<uint8_t>(viewRead.begin(), viewRead.end()), rampBytes(64));
    EXPECT_TRUE(source.viewed()) << "the source now shares its bytes and must not free them alone";
    EXPECT_TRUE(view.viewed());
}

TEST(HostBufferSharing, WritingThroughOneSideLeavesTheOtherUntouched) {
    ByteStorage source;
    source = rampBytes(32);
    ByteStorage view;
    view.setSharedBytes(source.shareBytes());
    const ByteStorage &sourceRead    = source;
    const ByteStorage &viewRead      = view;
    const uint8_t     *sharedAddress = sourceRead.data();

    view.data()[0] = 0xEE; // mutable access materializes a private copy

    EXPECT_NE(viewRead.data(), sharedAddress) << "the writer must have taken its own copy";
    EXPECT_EQ(viewRead.data()[0], 0xEE);
    EXPECT_EQ(sourceRead.data()[0], 1) << "the other side keeps the original bytes";
    EXPECT_EQ(sourceRead.data(), sharedAddress) << "and keeps the shared allocation";
    EXPECT_FALSE(view.viewed()) << "a materialized buffer owns its bytes privately again";
}

TEST(HostBufferSharing, ResizeAndAssignAlsoDetachFromTheSharedBlock) {
    ByteStorage source;
    source = rampBytes(16);
    ByteStorage grown, assigned;
    grown.setSharedBytes(source.shareBytes());
    assigned.setSharedBytes(source.shareBytes());

    grown.resize(24);
    assigned = rampBytes(8);

    EXPECT_EQ(grown.size(), 24u);
    EXPECT_EQ(std::vector<uint8_t>(grown.begin(), grown.begin() + 16), rampBytes(16)) << "resize keeps the bytes it had before growing";
    EXPECT_EQ(assigned.size(), 8u);
    const ByteStorage &sourceRead = source;
    EXPECT_EQ(sourceRead.size(), 16u) << "neither operation may disturb the shared source";
    EXPECT_EQ(sourceRead.data()[0], 1);
}

TEST(HostBufferSharing, ThreeBuffersShareOneBlockAndItOutlivesItsOriginalOwner) {
    auto        block = std::shared_ptr<const std::vector<uint8_t>>();
    ByteStorage a, b;
    {
        ByteStorage source;
        source = rampBytes(48);
        block  = source.shareBytes();
        a.setSharedBytes(block);
        b.setSharedBytes(block);
    } // the original buffer goes away; the block must not

    const ByteStorage &aRead = a;
    const ByteStorage &bRead = b;
    ASSERT_EQ(aRead.size(), 48u);
    ASSERT_EQ(bRead.size(), 48u);
    EXPECT_EQ(aRead.data(), bRead.data());
    EXPECT_EQ(std::vector<uint8_t>(aRead.begin(), aRead.end()), rampBytes(48));
}

// A file-backed buffer is already free to copy — its handle is shared, never its bytes — so
// sharing declines and the caller keeps assigning the buffer.
TEST(HostBufferSharing, SharingDeclinesForEmptyBuffers) {
    ByteStorage empty;
    EXPECT_EQ(empty.shareBytes(), nullptr);

    ByteStorage target;
    target = rampBytes(4);
    target.setSharedBytes(nullptr);
    EXPECT_EQ(target.size(), 0u) << "a null block leaves the buffer empty, not dangling";
    EXPECT_FALSE(target.viewed());
}

// toVector() is what the artifact writer uses; it must see shared bytes like any other.
TEST(HostBufferSharing, SerializationReadsSharedBytes) {
    ByteStorage source;
    source = rampBytes(20);
    ByteStorage view;
    view.setSharedBytes(source.shareBytes());

    EXPECT_EQ(view.toVector(), rampBytes(20));
    EXPECT_EQ(view.toVector(), source.toVector());
}
