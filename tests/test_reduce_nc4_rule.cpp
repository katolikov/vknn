// A spatial reduction runs blocked; every other axis set keeps the flat kernel.
//
// NC4HW4 packs four channels into one block, and a reduction over H and W is one independent
// reduction per channel -- exactly what the four lanes of a block carry, so the buffer is read as
// stored. Reducing the channel axis instead would have to cross lanes inside a block, which the
// blocked kernels cannot express.
#include "../src/import/passes_internal.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    TensorId value(Graph &g, const char *name, const Shape &s) {
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

    Node reduce(Graph &g, const Shape &in, const Shape &out, const std::vector<int64_t> &axes) {
        Node n;
        n.type = OpType::Reduce;
        n.name = "reduce";
        n.inputs.push_back(value(g, "x", in));
        n.outputs.push_back(value(g, "y", out));
        setInts(n, "axes", axes);
        return n;
    }

} // namespace

TEST(ReduceNc4Rule, SpatialAxesKeepingThemRunBlocked) {
    Graph g;
    EXPECT_TRUE(reduceIsNc4(g, reduce(g, {1, 8, 16, 12}, {1, 8, 1, 1}, {2, 3})));
    EXPECT_TRUE(reduceIsNc4(g, reduce(g, {2, 6, 5, 7}, {2, 6, 1, 1}, {2, 3})));
    EXPECT_TRUE(reduceIsNc4(g, reduce(g, {1, 8, 16, 12}, {1, 8, 1, 1}, {-2, -1})));
}

TEST(ReduceNc4Rule, AnyOtherAxisSetStaysFlat) {
    Graph g;
    EXPECT_FALSE(reduceIsNc4(g, reduce(g, {1, 8, 16, 12}, {1, 1, 16, 12}, {1})));     // channels cross lanes
    EXPECT_FALSE(reduceIsNc4(g, reduce(g, {1, 8, 16, 12}, {1, 8, 1, 12}, {2})));      // one spatial axis only
    EXPECT_FALSE(reduceIsNc4(g, reduce(g, {1, 8, 16, 12}, {1, 1, 1, 1}, {1, 2, 3}))); // three axes
    EXPECT_FALSE(reduceIsNc4(g, reduce(g, {1, 8, 16, 12}, {1, 8, 1, 1}, {0, 1})));    // batch + channels
}

TEST(ReduceNc4Rule, DroppingTheReducedAxesStaysFlat) {
    // keepdims=0 gives a rank-2 output, which is not the blocked [N,C,1,1] the kernels store.
    Graph g;
    EXPECT_FALSE(reduceIsNc4(g, reduce(g, {1, 8, 16, 12}, {1, 8}, {2, 3})));
}

TEST(ReduceNc4Rule, AFoldedMovementChainStaysFlat) {
    Graph g;
    Node  n = reduce(g, {1, 8, 16, 12}, {1, 8, 1, 1}, {2, 3});
    setInts(n, "view_stride", {1, 1, 1, 1});
    EXPECT_FALSE(reduceIsNc4(g, n));
}

TEST(ReduceNc4Rule, ARankOtherThanFourStaysFlat) {
    Graph g;
    EXPECT_FALSE(reduceIsNc4(g, reduce(g, {8, 16, 12}, {8, 1, 1}, {1, 2})));
}
