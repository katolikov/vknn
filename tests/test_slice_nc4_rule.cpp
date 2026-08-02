// A block-aligned channel Slice is a contiguous NC4HW4 block-range copy, not a flat gather.
//
// NC4HW4 packs four channels into one block, so a channel range maps to whole blocks exactly when
// both its start and its extent are multiples of kNC4Block. Anything else -- another axis, a step,
// a partial block, a folded epilogue -- would split a block across the seam, which no byte copy can
// express, and keeps the flat gather.
#include "../src/import/passes_internal.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    TensorId addTensor(Graph &g, const char *name, const Shape &s) {
        TensorDesc d;
        d.name  = name;
        d.shape = s;
        return g.addTensor(d);
    }

    void setInts(Node &n, const char *key, const std::vector<int64_t> &v) {
        Attr a;
        a.ints          = v;
        n.attr.map[key] = a;
    }

    /// A rank-4 channel Slice from `start` to `end` over an input with `cin` channels.
    Node channelSlice(Graph &g, int64_t cin, int64_t start, int64_t end, int64_t step = 1) {
        Node n;
        n.type = OpType::Slice;
        n.name = "slice";
        n.inputs.push_back(addTensor(g, "x", {1, cin, 8, 8}));
        n.outputs.push_back(addTensor(g, "y", {1, end - start, 8, 8}));
        setInts(n, "starts", {start});
        setInts(n, "ends", {end});
        setInts(n, "axes", {1});
        setInts(n, "steps", {step});
        return n;
    }

} // namespace

TEST(SliceNc4Rule, BlockAlignedChannelSliceIsPacked) {
    Graph g;
    EXPECT_TRUE(sliceIsNc4(g, channelSlice(g, 16, 0, 8)));
    EXPECT_TRUE(sliceIsNc4(g, channelSlice(g, 16, 4, 12)));
    EXPECT_TRUE(sliceIsNc4(g, channelSlice(g, 16, 8, 16)));
}

TEST(SliceNc4Rule, PartialBlockKeepsTheFlatGather) {
    Graph g;
    EXPECT_FALSE(sliceIsNc4(g, channelSlice(g, 16, 1, 9))); // unaligned start
    EXPECT_FALSE(sliceIsNc4(g, channelSlice(g, 16, 0, 6))); // unaligned extent
    EXPECT_FALSE(sliceIsNc4(g, channelSlice(g, 16, 2, 6))); // both
    EXPECT_FALSE(sliceIsNc4(g, channelSlice(g, 16, 0, 0))); // empty
}

TEST(SliceNc4Rule, OnlyTheChannelAxisWalkedForward) {
    Graph g;
    Node  strided = channelSlice(g, 16, 0, 8, /*step=*/2);
    EXPECT_FALSE(sliceIsNc4(g, strided));
    Node spatial = channelSlice(g, 16, 0, 8);
    setInts(spatial, "axes", {2}); // a spatial axis is not a block range
    EXPECT_FALSE(sliceIsNc4(g, spatial));
}

TEST(SliceNc4Rule, AFoldedEpilogueOrMovementChainNeedsAKernel) {
    Graph g;
    Node  epi = channelSlice(g, 16, 0, 8);
    setInts(epi, "pw_steps", {1});
    EXPECT_FALSE(sliceIsNc4(g, epi));
    Node folded = channelSlice(g, 16, 0, 8);
    setInts(folded, "view_stride", {1, 1, 1, 1});
    EXPECT_FALSE(sliceIsNc4(g, folded));
}

TEST(SliceNc4Rule, TwoSlicedAxesKeepTheFlatGather) {
    Graph g;
    Node  n = channelSlice(g, 16, 0, 8);
    setInts(n, "starts", {0, 0});
    setInts(n, "ends", {8, 4});
    setInts(n, "axes", {1, 2});
    setInts(n, "steps", {1, 1});
    EXPECT_FALSE(sliceIsNc4(g, n));
}
