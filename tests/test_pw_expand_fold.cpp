// A pointwise unit reads a broadcast producer's SOURCE, not its materialised result.
//
// Expand exists only to write a broadcast out in full, and a unit already broadcasts its operands --
// so reading the source computes identical values over a fraction of the memory, and the Expand
// becomes dead code (it has only a flat kernel, so it also drags a layout convert in and out).
// The fold is refused when the source has no blocked index: forcing the whole unit onto the flat
// kernel costs far more than the Expand saved.
#include "../src/import/passes.h"
#include "vknn/op_type.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    TensorId value(Graph &g, const char *name, const Shape &s) {
        TensorDesc d;
        d.name  = name;
        d.shape = s;
        return g.addTensor(d);
    }

    /// Relu(x) * Broadcast(b -> full): two members, which is what makes a unit.
    Graph expandIntoMul(const Shape &full, const Shape &broadcastSource, OpType producer = OpType::Expand) {
        Graph    g;
        TensorId x = value(g, "x", full), b = value(g, "b", broadcastSource);
        TensorId r = value(g, "r", full), e = value(g, "e", full), m = value(g, "m", full);
        g.desc(x).isInput = g.desc(b).isInput = true;
        Node relu;
        relu.type    = OpType::Relu;
        relu.name    = "relu";
        relu.inputs  = {x};
        relu.outputs = {r};
        g.nodes.push_back(relu);
        Node expand;
        expand.type    = producer;
        expand.name    = "expand";
        expand.inputs  = {b};
        expand.outputs = {e};
        Node mul;
        mul.type    = OpType::Binary;
        mul.name    = "mul";
        mul.subOp   = (int32_t) BinaryType::Mul;
        mul.inputs  = {r, e};
        mul.outputs = {m};
        g.nodes.push_back(expand);
        g.nodes.push_back(mul);
        g.outputs.push_back(m);
        g.desc(m).isOutput = true;
        return g;
    }

    int countOf(const Graph &g, OpType t) {
        int n = 0;
        for (const Node &nd: g.nodes)
        {
            n += nd.type == t;
        }
        return n;
    }

} // namespace

TEST(PwExpandFold, AClassifiableSourceIsReadDirectly) {
    // Per-channel, per-pixel and per-row sources all have a blocked index.
    for (const Shape &src: {Shape {1, 8, 1, 1}, Shape {1, 1, 16, 12}, Shape {1, 1, 1, 12}})
    {
        Graph g = expandIntoMul({1, 8, 16, 12}, src);
        fusePointwiseChains(g, /*strictFuse=*/false);
        eliminateDeadNodes(g);
        EXPECT_EQ(countOf(g, OpType::Expand), 0) << "the Expand should be dead once the unit reads its source";
        EXPECT_EQ(countOf(g, OpType::FusedPointwise), 1);
    }
}

// A Tile whose repeated axes are size 1 at the source IS a broadcast, and folds the same way.
TEST(PwExpandFold, ABroadcastingTileFoldsToo) {
    for (const Shape &src: {Shape {1, 8, 1, 1}, Shape {1, 1, 16, 12}})
    {
        Graph g = expandIntoMul({1, 8, 16, 12}, src, OpType::Tile);
        fusePointwiseChains(g, /*strictFuse=*/false);
        eliminateDeadNodes(g);
        EXPECT_EQ(countOf(g, OpType::Tile), 0) << "a broadcasting Tile is dead once the unit reads its source";
        EXPECT_EQ(countOf(g, OpType::FusedPointwise), 1);
    }
}

// A Tile that genuinely repeats -- four source rows laid out twice -- has no broadcast class, so it
// stays and materialises its result.
TEST(PwExpandFold, ARepeatingTileIsLeftAlone) {
    Graph g = expandIntoMul({1, 8, 16, 12}, {1, 8, 8, 12}, OpType::Tile);
    fusePointwiseChains(g, /*strictFuse=*/false);
    eliminateDeadNodes(g);
    EXPECT_EQ(countOf(g, OpType::Tile), 1) << "repeating is not broadcasting; the unit cannot express it";
}

TEST(PwExpandFold, TheUnitStillProducesTheGraphOutput) {
    Graph g = expandIntoMul({1, 8, 16, 12}, {1, 8, 1, 1});
    fusePointwiseChains(g, /*strictFuse=*/false);
    eliminateDeadNodes(g);
    ASSERT_FALSE(g.outputs.empty());
    bool produced = false;
    for (const Node &nd: g.nodes)
    {
        for (TensorId o: nd.outputs)
        {
            produced = produced || o == g.outputs[0];
        }
    }
    EXPECT_TRUE(produced) << "folding must not orphan the graph output";
}
