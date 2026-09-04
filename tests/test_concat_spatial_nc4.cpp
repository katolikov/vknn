// The layout pass keeps a spatial-axis Concat blocked (import/insert_layout_converts.cpp): the rows
// of a rank-3 [N,C,L] map (a detection head joining its per-scale maps) and the rows or columns of a
// rank-4 map stay NC4HW4 when every part shares the output's N, C and other spatial extent, so no
// layout convert sits between the producing convs and the concat or after it. Mismatched parts and
// other axes keep the flat path. The blocked kernel itself is gated on the device.
#include "import/passes.h"
#include "vknn/graph.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace vknn {
    bool gpuFlatNode(const Graph &g, const Node &n);
}

namespace {
    TensorId addAct(Graph &g, const std::string &name, Shape shape, bool isInput = false) {
        TensorDesc d;
        d.name    = name;
        d.shape   = std::move(shape);
        d.isInput = isInput;
        return g.addTensor(d);
    }
    Attr integer(int64_t v) {
        Attr a;
        a.kind = Attr::Int;
        a.i    = v;
        return a;
    }
    // Concat of `parts` (each a graph input) along `axis` into `out`.
    Graph concatGraph(const std::vector<Shape> &parts, const Shape &out, int64_t axis) {
        Graph g;
        Node  cat;
        cat.type = OpType::Concat;
        cat.name = "cat";
        for (size_t i = 0; i < parts.size(); ++i)
        {
            TensorId t = addAct(g, "p" + std::to_string(i), parts[i], true);
            g.inputs.push_back(t);
            cat.inputs.push_back(t);
        }
        TensorId y          = addAct(g, "y", out);
        cat.outputs         = {y};
        cat.attr.map["axis"] = integer(axis);
        g.nodes             = {cat};
        g.outputs           = {y};
        return g;
    }
} // namespace

TEST(ConcatSpatialNc4, Rank3LastAxisConcatOfSameChannelMapsStaysBlocked) {
    // The detection head's three scales: [1,64,6400] + [1,64,1600] + [1,64,400] -> [1,64,8400].
    Graph g = concatGraph({{1, 64, 6400}, {1, 64, 1600}, {1, 64, 400}}, {1, 64, 8400}, -1);
    EXPECT_FALSE(gpuFlatNode(g, g.nodes[0]));
    // Any channel count: 84 is not a multiple of 4, and that is not the concatenated axis.
    Graph odd = concatGraph({{1, 84, 100}, {1, 84, 25}}, {1, 84, 125}, 2);
    EXPECT_FALSE(gpuFlatNode(odd, odd.nodes[0]));
}

TEST(ConcatSpatialNc4, Rank4RowAndColumnConcatsStayBlocked) {
    Graph rows = concatGraph({{1, 8, 8, 10}, {1, 8, 5, 10}}, {1, 8, 13, 10}, 2);
    EXPECT_FALSE(gpuFlatNode(rows, rows.nodes[0]));
    Graph cols = concatGraph({{1, 8, 8, 10}, {1, 8, 8, 6}}, {1, 8, 8, 16}, 3);
    EXPECT_FALSE(gpuFlatNode(cols, cols.nodes[0]));
    Graph colsNeg = concatGraph({{2, 6, 4, 4}, {2, 6, 4, 4}}, {2, 6, 4, 8}, -1);
    EXPECT_FALSE(gpuFlatNode(colsNeg, colsNeg.nodes[0])) << "a negative axis resolves to the columns";
}

TEST(ConcatSpatialNc4, MismatchedPartsAndOtherAxesRunFlat) {
    Graph diffC = concatGraph({{1, 8, 8, 10}, {1, 4, 8, 10}}, {1, 8, 8, 20}, 3);
    diffC.tensors[diffC.nodes[0].outputs[0]].shape = {1, 8, 8, 20};
    EXPECT_TRUE(gpuFlatNode(diffC, diffC.nodes[0])) << "parts of different channel counts share no block structure";
    Graph diffW = concatGraph({{1, 8, 8, 10}, {1, 8, 5, 6}}, {1, 8, 13, 10}, 2);
    EXPECT_TRUE(gpuFlatNode(diffW, diffW.nodes[0])) << "a row concat needs the same column extent";
    Graph batch = concatGraph({{1, 8, 4, 4}, {1, 8, 4, 4}}, {2, 8, 4, 4}, 0);
    EXPECT_TRUE(gpuFlatNode(batch, batch.nodes[0])) << "the batch axis is not a blocked concat";
    Graph rank2 = concatGraph({{4, 8}, {4, 8}}, {4, 16}, 1);
    EXPECT_TRUE(gpuFlatNode(rank2, rank2.nodes[0]));
    // The channel-axis rule is unchanged: 4-aligned parts stay blocked, an unaligned part runs flat.
    Graph chan = concatGraph({{1, 8, 4, 4}, {1, 12, 4, 4}}, {1, 20, 4, 4}, 1);
    EXPECT_FALSE(gpuFlatNode(chan, chan.nodes[0]));
    Graph chanOdd = concatGraph({{1, 8, 4, 4}, {1, 6, 4, 4}}, {1, 14, 4, 4}, 1);
    EXPECT_TRUE(gpuFlatNode(chanOdd, chanOdd.nodes[0]));
}
