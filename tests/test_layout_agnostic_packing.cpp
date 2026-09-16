// The NC4HW4 byte-copy rule of the layout pass for layout-agnostic ops (Reshape, Flatten, Squeeze,
// Unsqueeze, Cast, ChannelShuffle). Such an op keeps NC4HW4 only when its output shape stores every
// element at the buffer position its input shape stores it at; otherwise it runs flat and the layout
// pass converts its NC4HW4 input in front of it.
//
// The rule (import/nc4_packing.h) is checked against an explicit NC4HW4 storage map computed here
// straight from the lane-quad indexing the pack and convert shaders use, over every pair of small
// shapes with equal element counts: it keeps NC4HW4 exactly when the maps and footprints agree. The
// graph tests run the session's Vulkan load sequence (planFlatLayoutAndStorage) and the layout pass
// on graphs where a square matrix gains or loses a leading axis, the shape class whose NC4HW4 byte
// copy transposes the data while the channel count stays equal.
#include "import/nc4_packing.h"
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/nchw.h"
#include <gtest/gtest.h>
#include <map>
#include <vector>

using namespace vknn;

namespace {

    // --- explicit NC4HW4 storage map (the oracle) -----------------------------------------------------

    struct Nc4Storage {
        std::vector<int64_t> positionOfElement; // buffer slot of each row-major element index
        int64_t              footprint = 0;     // slots in the buffer, padding lanes included

        bool operator==(const Nc4Storage &other) const {
            return footprint == other.footprint && positionOfElement == other.positionOfElement;
        }
    };

    // Where each element of `shape` lives in an NC4HW4 buffer: the quad at (n, channel block, h, w) is
    // ((n * cBlocks(C) + block) * H + h) * W + w and the element occupies lane c % kNC4Block of it (the
    // indexing of shaders/convert_layout.comp and shaders/boundary_convert.comp over NCHW::from).
    Nc4Storage nc4Storage(const Shape &shape) {
        const NCHW    view   = NCHW::from(shape);
        const int64_t blocks = cBlocks(view.c);
        Nc4Storage    storage;
        storage.footprint = formatElems(TensorFormat::NC4HW4, view);
        for (int64_t n = 0; n < view.n; ++n)
        {
            for (int64_t c = 0; c < view.c; ++c)
            {
                for (int64_t h = 0; h < view.h; ++h)
                {
                    for (int64_t w = 0; w < view.w; ++w)
                    {
                        const int64_t quad = ((n * blocks + c / kNC4Block) * view.h + h) * view.w + w;
                        storage.positionOfElement.push_back(quad * kNC4Block + c % kNC4Block);
                    }
                }
            }
        }
        return storage;
    }

    // Every shape of rank `rank` whose dims come from `dims`, appended to `out`.
    void enumerateShapes(size_t rank, const std::vector<int64_t> &dims, Shape &prefix, std::vector<Shape> &out) {
        if (prefix.size() == rank)
        {
            out.push_back(prefix);
            return;
        }
        for (int64_t d: dims)
        {
            prefix.push_back(d);
            enumerateShapes(rank, dims, prefix, out);
            prefix.pop_back();
        }
    }

    // --- graph construction ---------------------------------------------------------------------------

    TensorId addTensor(Graph &g, const std::string &name, Shape shape, DType dtype = DType::Float32) {
        TensorDesc d;
        d.name  = name;
        d.shape = std::move(shape);
        d.dtype = dtype;
        return g.addTensor(d);
    }

    TensorId addInput(Graph &g, const std::string &name, Shape shape, DType dtype = DType::Float32) {
        TensorId id        = addTensor(g, name, std::move(shape), dtype);
        g.desc(id).isInput = true;
        g.inputs.push_back(id);
        return id;
    }

    TensorId addInt64Initializer(Graph &g, const std::string &name, Shape shape, const std::vector<int64_t> &values) {
        TensorId id              = addTensor(g, name, std::move(shape), DType::Int64);
        g.desc(id).isInitializer = true;
        HostBuffer hb;
        hb.resizeElems((int64_t) values.size(), DType::Int64);
        for (size_t k = 0; k < values.size(); ++k)
        {
            hb.i64()[k] = values[k];
        }
        g.initializers[id] = hb;
        return id;
    }

    TensorId addFloatInitializer(Graph &g, const std::string &name, Shape shape, const std::vector<float> &values) {
        TensorId id              = addTensor(g, name, std::move(shape), DType::Float32);
        g.desc(id).isInitializer = true;
        HostBuffer hb;
        hb.resizeElems((int64_t) values.size(), DType::Float32);
        for (size_t k = 0; k < values.size(); ++k)
        {
            hb.f32()[k] = values[k];
        }
        g.initializers[id] = hb;
        return id;
    }

    Node &addNode(Graph &g, OpType type, const std::string &name, std::vector<TensorId> inputs, std::vector<TensorId> outputs) {
        Node n;
        n.type    = type;
        n.name    = name;
        n.inputs  = std::move(inputs);
        n.outputs = std::move(outputs);
        g.nodes.push_back(std::move(n));
        return g.nodes.back();
    }

    void addOutput(Graph &g, TensorId id) {
        g.desc(id).isOutput = true;
        g.outputs.push_back(id);
    }

    Attr intAttr(int64_t value) {
        Attr a;
        a.kind = Attr::Int;
        a.i    = value;
        return a;
    }

    const Node *findNode(const Graph &g, const std::string &name) {
        for (const Node &n: g.nodes)
        {
            if (n.name == name)
            {
                return &n;
            }
        }
        return nullptr;
    }

    const Node *producerOf(const Graph &g, TensorId t) {
        for (const Node &n: g.nodes)
        {
            for (TensorId o: n.outputs)
            {
                if (o == t)
                {
                    return &n;
                }
            }
        }
        return nullptr;
    }

    int countNodes(const Graph &g, OpType type) {
        int count = 0;
        for (const Node &n: g.nodes)
        {
            count += n.type == type ? 1 : 0;
        }
        return count;
    }

    bool isLayoutAgnosticType(OpType type) {
        return type == OpType::Reshape || type == OpType::Flatten || type == OpType::Squeeze || type == OpType::Unsqueeze || type == OpType::Cast || type == OpType::ChannelShuffle;
    }

    // Every layout-agnostic node of a planned graph reads and writes one layout, and an NC4HW4 one only
    // when its input and output shapes share the explicit storage map: the buffer its kernel copies,
    // aliases or remaps never reinterprets an element.
    void expectAgnosticNodesKeepElementPositions(const Graph &g) {
        for (const Node &n: g.nodes)
        {
            if (!isLayoutAgnosticType(n.type) || n.inputs.empty() || n.inputs[0] == kNoTensor || g.isInitializer(n.inputs[0]))
            {
                continue;
            }
            const TensorDesc &in  = g.desc(n.inputs[0]);
            const TensorDesc &out = g.desc(n.outputs[0]);
            EXPECT_EQ(in.gpuFlat, out.gpuFlat) << n.name << ": an agnostic op reads and writes one layout";
            if (!out.gpuFlat)
            {
                EXPECT_TRUE(nc4Storage(in.shape) == nc4Storage(out.shape)) << n.name << ": NC4HW4 copy from " << shapeStr(in.shape) << " to " << shapeStr(out.shape) << " moves elements";
            }
        }
    }

} // namespace

// --- the packing rule against the explicit storage map --------------------------------------------------

TEST(LayoutAgnosticPacking, RuleKeepsNc4ExactlyWhenStorageMapsAgree) {
    // Ranks 0-5 over dims with ones, squares, multiples and non-multiples of the block width (zero dims
    // on the low ranks), capped so every map stays small. For each pair with an equal element count the
    // rule must keep NC4HW4 exactly when the storage maps and footprints agree: never when an element
    // would move (the wrong-answer direction), and always when none does (no convert the layouts do not
    // need).
    constexpr int64_t          kMaxElements       = 64;
    constexpr int64_t          kSingleChannel     = 1;
    constexpr size_t           kMaxRank           = 5;
    constexpr size_t           kMaxRankWithZeroes = 3;
    const std::vector<int64_t> dims               = {1, 2, 3, 4, 5, 8};
    const std::vector<int64_t> dimsWithZero       = {0, 1, 2, 4, 8};
    std::vector<Shape>         shapes             = {Shape {}};
    for (size_t rank = 1; rank <= kMaxRank; ++rank)
    {
        Shape prefix;
        enumerateShapes(rank, rank <= kMaxRankWithZeroes ? dimsWithZero : dims, prefix, shapes);
    }
    std::map<int64_t, std::vector<std::pair<Shape, Nc4Storage>>> byElementCount;
    for (const Shape &s: shapes)
    {
        const int64_t elements = NCHW::from(s).elems();
        if (elements > kMaxElements)
        {
            continue;
        }
        const Nc4PackingKey key = nc4PackingKey(s);
        if (elements > 0)
        {
            EXPECT_EQ(key.batch * key.channels * key.plane, elements) << shapeStr(s) << ": the key keeps the element count, so unequal counts never share a key";
        }
        byElementCount[elements].push_back({s, nc4Storage(s)});
    }
    int64_t keptSameView = 0, keptBatchFold = 0, keptSingleChannel = 0, flattened = 0;
    for (const auto &group: byElementCount)
    {
        const auto &members = group.second;
        for (size_t i = 0; i < members.size(); ++i)
        {
            for (size_t j = 0; j < members.size(); ++j)
            {
                const Shape &from           = members[i].first;
                const Shape &to             = members[j].first;
                const bool   storageAgrees  = members[i].second == members[j].second;
                const bool   ruleKeepsNc4   = nc4PackingIdentical(from, to);
                const NCHW   fromView       = NCHW::from(from);
                const NCHW   toView         = NCHW::from(to);
                const bool   sameNchwFields = fromView.n == toView.n && fromView.c == toView.c && fromView.h * fromView.w == toView.h * toView.w;
                ASSERT_EQ(ruleKeepsNc4, storageAgrees) << shapeStr(from) << " -> " << shapeStr(to);
                if (!ruleKeepsNc4)
                {
                    ++flattened;
                } else if (group.first > 0 && sameNchwFields)
                {
                    ++keptSameView;
                } else if (group.first > 0 && fromView.c == kSingleChannel && toView.c == kSingleChannel)
                {
                    ++keptSingleChannel;
                } else if (group.first > 0)
                { ++keptBatchFold; }
            }
        }
    }
    // Each class the key canonicalizes is exercised, and so is the flat outcome.
    EXPECT_GT(keptSameView, 0);
    EXPECT_GT(keptSingleChannel, 0);
    EXPECT_GT(keptBatchFold, 0);
    EXPECT_GT(flattened, 0);
}

TEST(LayoutAgnosticPacking, NamedShapePairs) {
    struct Case {
        Shape from;
        Shape to;
        bool  keepsNc4;
    };
    const Case cases[] = {
        // A square matrix gaining or losing a leading axis: equal channel counts, transposed storage.
        {{4, 4}, {1, 4, 4}, false},
        {{8, 8}, {1, 8, 8}, false},
        {{1, 4, 4}, {4, 4}, false},
        {{2, 4, 4}, {1, 2, 4, 4}, false},
        // A trailing unit axis keeps N, C and the plane.
        {{4, 4}, {4, 4, 1}, true},
        {{2, 4, 4}, {2, 4, 4, 1}, true},
        // Channel count changes.
        {{3, 5}, {1, 3, 5}, false},
        {{4, 6}, {1, 4, 6}, false},
        // Classifier and detection-head reshapes.
        {{1, 1280, 1, 1}, {1, 1280}, true},
        {{1, 64, 80, 80}, {1, 64, 6400}, true},
        {{1, 116, 28, 28}, {1, 2, 58, 28, 28}, false},
        // Batches of whole channel blocks store like one batch of all channels.
        {{2, 4, 3, 3}, {1, 8, 3, 3}, true},
        {{2, 3, 3, 3}, {1, 6, 3, 3}, false},
        // One channel stores one element per quad however N and the plane split the elements.
        {{2, 1, 3, 3}, {1, 1, 18}, true},
        // An unresolved dim proves nothing beyond identity.
        {{-1, 4}, {-1, 4}, true},
        {{-1, 4}, {1, -1, 4}, false},
    };
    for (const Case &c: cases)
    {
        EXPECT_EQ(nc4PackingIdentical(c.from, c.to), c.keepsNc4) << shapeStr(c.from) << " -> " << shapeStr(c.to);
    }
}

// --- the layout pass --------------------------------------------------------------------------------------

TEST(LayoutAgnosticPacking, UnsqueezeOfSquareInputRunsFlat) {
    // x [4,4] -> Unsqueeze(0) -> [1,4,4] -> Add(0.5). Read as NC4HW4, [4,4] is N=4 C=4 and [1,4,4] is
    // N=1 C=4 H=4: an NC4HW4 byte copy would transpose the matrix. The Unsqueeze runs flat, so the
    // graph input is packed flat for its only (flat) reader and no convert is needed at all.
    for (const Shape &square: {Shape {4, 4}, Shape {8, 8}})
    {
        Graph    g;
        TensorId x        = addInput(g, "x", square);
        TensorId axes     = addInt64Initializer(g, "axes", {1}, {0});
        TensorId expanded = addTensor(g, "expanded", {1, square[0], square[1]});
        TensorId half     = addFloatInitializer(g, "half", {1}, {0.5f});
        TensorId y        = addTensor(g, "y", {1, square[0], square[1]});
        addNode(g, OpType::Unsqueeze, "unsqueeze", {x, axes}, {expanded});
        addNode(g, OpType::Add, "add", {expanded, half}, {y});
        addOutput(g, y);

        planFlatLayoutAndStorage(g, "", nullptr);
        expectAgnosticNodesKeepElementPositions(g);
        EXPECT_TRUE(g.desc(expanded).gpuFlat) << shapeStr(square);
        EXPECT_TRUE(g.desc(x).gpuFlat) << shapeStr(square) << ": the input's only reader runs flat";
        EXPECT_EQ(countNodes(g, OpType::ConvertLayout), 0) << shapeStr(square);
    }
}

TEST(LayoutAgnosticPacking, ReshapeOfSquareInputRunsFlat) {
    Graph    g;
    TensorId x        = addInput(g, "x", {4, 4});
    TensorId target   = addInt64Initializer(g, "target", {3}, {1, 4, 4});
    TensorId reshaped = addTensor(g, "reshaped", {1, 4, 4});
    TensorId half     = addFloatInitializer(g, "half", {1}, {0.5f});
    TensorId y        = addTensor(g, "y", {1, 4, 4});
    addNode(g, OpType::Reshape, "reshape", {x, target}, {reshaped});
    addNode(g, OpType::Add, "add", {reshaped, half}, {y});
    addOutput(g, y);

    planFlatLayoutAndStorage(g, "", nullptr);
    expectAgnosticNodesKeepElementPositions(g);
    EXPECT_TRUE(g.desc(reshaped).gpuFlat);
    EXPECT_EQ(countNodes(g, OpType::ConvertLayout), 0);
}

TEST(LayoutAgnosticPacking, Nc4ReaderAfterSquareUnsqueezeReadsThroughConvert) {
    // x [4,4] -> Unsqueeze(0) -> Relu. Relu runs NC4HW4, so the flat [1,4,4] is converted to NC4HW4 with
    // its own shape in front of it, instead of the [4,4] NC4HW4 buffer being reread as [1,4,4].
    Graph    g;
    TensorId x        = addInput(g, "x", {4, 4});
    TensorId axes     = addInt64Initializer(g, "axes", {1}, {0});
    TensorId expanded = addTensor(g, "expanded", {1, 4, 4});
    TensorId y        = addTensor(g, "y", {1, 4, 4});
    addNode(g, OpType::Unsqueeze, "unsqueeze", {x, axes}, {expanded});
    addNode(g, OpType::Relu, "relu", {expanded}, {y});
    addOutput(g, y);

    planFlatLayoutAndStorage(g, "", nullptr);
    expectAgnosticNodesKeepElementPositions(g);
    EXPECT_TRUE(g.desc(expanded).gpuFlat);
    const Node *relu = findNode(g, "relu");
    ASSERT_NE(relu, nullptr);
    EXPECT_FALSE(g.desc(relu->inputs[0]).gpuFlat) << "Relu reads NC4HW4";
    const Node *toNc4 = producerOf(g, relu->inputs[0]);
    ASSERT_NE(toNc4, nullptr);
    ASSERT_EQ(toNc4->type, OpType::ConvertLayout);
    EXPECT_EQ(toNc4->inputs[0], expanded) << "the flat [1,4,4] itself is packed for Relu";
}

TEST(LayoutAgnosticPacking, SqueezeOfNc4ProducerConvertsBeforeTheSqueeze) {
    // Relu writes [1,4,4] in NC4HW4 (N=1 C=4 H=4); Squeeze to [4,4] (N=4 C=4) changes the packing, so
    // the Squeeze runs flat and reads the Relu output through a NC4HW4 -> flat convert of [1,4,4].
    Graph    g;
    TensorId x        = addInput(g, "x", {1, 4, 4});
    TensorId rect     = addTensor(g, "rect", {1, 4, 4});
    TensorId axes     = addInt64Initializer(g, "axes", {1}, {0});
    TensorId squeezed = addTensor(g, "squeezed", {4, 4});
    TensorId half     = addFloatInitializer(g, "half", {1}, {0.5f});
    TensorId y        = addTensor(g, "y", {4, 4});
    addNode(g, OpType::Relu, "relu", {x}, {rect});
    addNode(g, OpType::Squeeze, "squeeze", {rect, axes}, {squeezed});
    addNode(g, OpType::Add, "add", {squeezed, half}, {y});
    addOutput(g, y);

    planFlatLayoutAndStorage(g, "", nullptr);
    expectAgnosticNodesKeepElementPositions(g);
    EXPECT_FALSE(g.desc(rect).gpuFlat) << "Relu runs NC4HW4";
    EXPECT_TRUE(g.desc(squeezed).gpuFlat);
    const Node *squeeze = findNode(g, "squeeze");
    ASSERT_NE(squeeze, nullptr);
    const Node *toFlat = producerOf(g, squeeze->inputs[0]);
    ASSERT_NE(toFlat, nullptr);
    ASSERT_EQ(toFlat->type, OpType::ConvertLayout);
    EXPECT_EQ(toFlat->inputs[0], rect);
    EXPECT_EQ(g.desc(squeeze->inputs[0]).shape, (Shape {1, 4, 4})) << "the convert unpacks with the producer's shape";
}

TEST(LayoutAgnosticPacking, PackingPreservingReshapesStayNc4) {
    // Reshapes that keep the storage map keep NC4HW4 between NC4HW4 ops with no interior convert: the
    // detection-head flatten of the plane, and batches of whole channel blocks joined into one batch.
    struct Case {
        Shape from;
        Shape to;
    };
    const Case cases[] = {{{1, 64, 8, 8}, {1, 64, 64}}, {{2, 4, 3, 3}, {1, 8, 3, 3}}, {{1, 8, 1, 1}, {1, 8}}};
    for (const Case &c: cases)
    {
        Graph    g;
        TensorId x        = addInput(g, "x", c.from);
        TensorId rect     = addTensor(g, "rect", c.from);
        TensorId target   = addInt64Initializer(g, "target", {(int64_t) c.to.size()}, c.to);
        TensorId reshaped = addTensor(g, "reshaped", c.to);
        TensorId y        = addTensor(g, "y", c.to);
        addNode(g, OpType::Relu, "relu_before", {x}, {rect});
        addNode(g, OpType::Reshape, "reshape", {rect, target}, {reshaped});
        addNode(g, OpType::Relu, "relu_after", {reshaped}, {y});
        addOutput(g, y);

        planFlatLayoutAndStorage(g, "", nullptr);
        expectAgnosticNodesKeepElementPositions(g);
        EXPECT_FALSE(g.desc(reshaped).gpuFlat) << shapeStr(c.from) << " -> " << shapeStr(c.to);
        EXPECT_EQ(countNodes(g, OpType::ConvertLayout), 1) << shapeStr(c.from) << " -> " << shapeStr(c.to) << ": only the graph-output readback convert";
    }
}

TEST(LayoutAgnosticPacking, IntegerSquareInputUnsqueezeIntoBitwiseAnd) {
    // int32 [4,4] -> Unsqueeze(0) -> BitwiseAnd(mask): the integer pin keeps the chain fp32. The
    // Unsqueeze runs flat (no NC4HW4 reinterpretation), so the whole chain is flat fp32 and unconverted.
    Graph    g;
    TensorId x        = addInput(g, "x", {4, 4}, DType::Int32);
    TensorId axes     = addInt64Initializer(g, "axes", {1}, {0});
    TensorId expanded = addTensor(g, "expanded", {1, 4, 4});
    TensorId mask     = addInt64Initializer(g, "mask", {1}, {0xFFFF0});
    TensorId y        = addTensor(g, "y", {1, 4, 4});
    addNode(g, OpType::Unsqueeze, "unsqueeze", {x, axes}, {expanded});
    addNode(g, OpType::BitwiseAnd, "bitwise_and", {expanded, mask}, {y});
    addOutput(g, y);

    planFlatLayoutAndStorage(g, "", nullptr);
    expectAgnosticNodesKeepElementPositions(g);
    EXPECT_TRUE(g.desc(x).gpuFlat);
    EXPECT_TRUE(g.desc(expanded).gpuFlat);
    EXPECT_TRUE(g.desc(x).storeFp32);
    EXPECT_TRUE(g.desc(expanded).storeFp32);
    EXPECT_EQ(countNodes(g, OpType::ConvertLayout), 0);
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
}

TEST(LayoutAgnosticPacking, CastOfNc4TensorKeepsItsLayout) {
    // Cast keeps its shape, so it always adopts its input's layout: an NC4HW4 Relu output stays NC4HW4
    // through the Cast into the next NC4HW4 op.
    constexpr int64_t kOnnxFloat = 1;
    Graph             g;
    TensorId          x    = addInput(g, "x", {4, 4});
    TensorId          rect = addTensor(g, "rect", {4, 4});
    TensorId          cast = addTensor(g, "cast", {4, 4});
    TensorId          y    = addTensor(g, "y", {4, 4});
    addNode(g, OpType::Relu, "relu_before", {x}, {rect});
    Node &castNode          = addNode(g, OpType::Cast, "cast", {rect}, {cast});
    castNode.attr.map["to"] = intAttr(kOnnxFloat);
    addNode(g, OpType::Relu, "relu_after", {cast}, {y});
    addOutput(g, y);

    planFlatLayoutAndStorage(g, "", nullptr);
    expectAgnosticNodesKeepElementPositions(g);
    EXPECT_FALSE(g.desc(cast).gpuFlat);
    EXPECT_EQ(countNodes(g, OpType::ConvertLayout), 1) << "only the graph-output readback convert";
}

TEST(LayoutAgnosticPacking, FlexibleUnitCountsPackingChangingReaderAsFlat) {
    // A flexible fused pointwise unit (rank 4, expressible in both layouts) between a flat producer
    // (DepthToSpace) and two readers: a MaxPool (NC4HW4) and a Reshape to [4,4], whose packing differs
    // from [1,4,2,2], so it runs flat whatever the unit chooses. Flat input + flat reshape outweigh the
    // one NC4HW4 reader: the unit runs flat and the only interior convert is the one in front of MaxPool.
    Graph             g;
    TensorId          x          = addInput(g, "x", {1, 16, 1, 1});
    TensorId          spread     = addTensor(g, "spread", {1, 4, 2, 2});
    TensorId          rect       = addTensor(g, "rect", {1, 4, 2, 2});
    TensorId          rect2      = addTensor(g, "rect2", {1, 4, 2, 2});
    TensorId          target     = addInt64Initializer(g, "target", {2}, {4, 4});
    TensorId          reshaped   = addTensor(g, "reshaped", {4, 4});
    TensorId          pooled     = addTensor(g, "pooled", {1, 4, 2, 2});
    constexpr int64_t kBlockSize = 2;
    Node             &d2s        = addNode(g, OpType::DepthToSpace, "depth_to_space", {x}, {spread});
    d2s.attr.map["blocksize"]    = intAttr(kBlockSize);
    addNode(g, OpType::Relu, "relu_a", {spread}, {rect});
    addNode(g, OpType::Relu, "relu_b", {rect}, {rect2});
    addNode(g, OpType::Reshape, "reshape", {rect2, target}, {reshaped});
    addNode(g, OpType::MaxPool, "pool", {rect2}, {pooled});
    addOutput(g, reshaped);
    addOutput(g, pooled);
    fusePointwiseChains(g, true);
    int flexibleUnits = 0;
    for (const Node &n: g.nodes)
    {
        flexibleUnits += n.type == OpType::FusedPointwise && n.attr.has("pw_steps") ? 1 : 0;
    }
    ASSERT_EQ(flexibleUnits, 1) << "the Relu pair fuses into one standalone unit";

    insertLayoutConverts(g);
    expectAgnosticNodesKeepElementPositions(g);
    EXPECT_TRUE(g.desc(rect2).gpuFlat) << "the unit follows its flat neighbors";
    const Node *reshape = findNode(g, "reshape");
    ASSERT_NE(reshape, nullptr);
    EXPECT_EQ(reshape->inputs[0], rect2) << "the flat reshape reads the unit directly";
    const Node *pool = findNode(g, "pool");
    ASSERT_NE(pool, nullptr);
    const Node *toNc4 = producerOf(g, pool->inputs[0]);
    ASSERT_NE(toNc4, nullptr);
    EXPECT_EQ(toNc4->type, OpType::ConvertLayout);
    EXPECT_EQ(countNodes(g, OpType::ConvertLayout), 2) << "the MaxPool input convert and the MaxPool output readback convert";
}
