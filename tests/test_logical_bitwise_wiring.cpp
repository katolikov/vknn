// Import-side wiring of the logical (Or/Xor/Not), bitwise (BitwiseAnd/Or/Xor/Not, BitShift), Mod and
// ArgMax/ArgMin ops, independent of their kernels: the shape/dtype rules inferShapes applies, the
// dtype lattices that decide which Casts survive, constFold's handling of 1-byte INT8/UINT8
// initializers, and the load-time fp32 pins (pinIntegerResultsFp32 + the markFp32 frontier) that keep
// integer values exact on the GPU's float storage. The pin passes only run in a Vulkan session, so
// these tests call them directly on hand-built graphs with the layout flags the flat pass would set.
#include "import/passes.h"
#include "import/passes_internal.h" // eliminateFloatCast, foldIntRoundtripCast
#include "vknn/graph.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    Attr intAttr(int64_t value) {
        Attr a;
        a.kind = Attr::Int;
        a.i    = value;
        return a;
    }

    TensorId addTensor(Graph &g, const std::string &name, Shape shape, DType dtype = DType::Float32, bool flat = false) {
        TensorDesc d;
        d.name    = name;
        d.shape   = std::move(shape);
        d.dtype   = dtype;
        d.gpuFlat = flat;
        return g.addTensor(d);
    }

    TensorId addInput(Graph &g, const std::string &name, Shape shape, DType dtype = DType::Float32, bool flat = false) {
        TensorId id        = addTensor(g, name, std::move(shape), dtype, flat);
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

    // A native 1-byte INT8/UINT8 initializer, stored the way materializeInitializers keeps it.
    TensorId addByteInitializer(Graph &g, const std::string &name, Shape shape, DType dtype, const std::vector<uint8_t> &lanes) {
        TensorId id              = addTensor(g, name, std::move(shape), dtype);
        g.desc(id).isInitializer = true;
        HostBuffer hb;
        hb.bytes           = lanes;
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

    int countNodes(const Graph &g, OpType type) {
        int count = 0;
        for (const Node &n: g.nodes)
        {
            count += n.type == type ? 1 : 0;
        }
        return count;
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

} // namespace

// --- inferShapes ------------------------------------------------------------------------------------

TEST(OpWiring, ArgExtremeShapeAndInt64Dtype) {
    struct Case {
        OpType  type;
        Shape   input;
        int64_t axis;
        int64_t keepDims;
        Shape   expected;
    };
    const Case cases[] = {
        {OpType::ArgMax, {2, 3, 4}, 0, 1, {1, 3, 4}}, {OpType::ArgMax, {2, 3, 4}, 1, 0, {2, 4}}, {OpType::ArgMin, {2, 3, 4}, -1, 1, {2, 3, 1}},
        {OpType::ArgMin, {2, 3, 4}, -1, 0, {2, 3}},   {OpType::ArgMax, {5}, 0, 0, {1}},          {OpType::ArgMin, {5}, 0, 1, {1}},
        {OpType::ArgMax, {2, 0, 3}, 2, 1, {2, 0, 1}},
    };
    for (const Case &c: cases)
    {
        Graph    g;
        TensorId x             = addInput(g, "x", c.input);
        TensorId y             = addTensor(g, "y", {});
        Node    &n             = addNode(g, c.type, "arg_extreme", {x}, {y});
        n.attr.map["axis"]     = intAttr(c.axis);
        n.attr.map["keepdims"] = intAttr(c.keepDims);
        g.outputs              = {y};
        inferShapes(g, 1);
        EXPECT_EQ(g.desc(y).shape, c.expected) << opTypeName(c.type) << " axis=" << c.axis << " keepdims=" << c.keepDims;
        EXPECT_EQ(g.desc(y).dtype, DType::Int64) << opTypeName(c.type);
    }
}

TEST(OpWiring, ArgExtremeDefaultsAndInvalidAxisStayUnresolved) {
    // Defaults: axis 0, keepdims 1.
    {
        Graph    g;
        TensorId x = addInput(g, "x", {4, 6});
        TensorId y = addTensor(g, "y", {});
        addNode(g, OpType::ArgMax, "argmax_defaults", {x}, {y});
        g.outputs = {y};
        inferShapes(g, 1);
        EXPECT_EQ(g.desc(y).shape, (Shape {1, 6}));
    }
    // An out-of-range axis leaves the output unresolved (the kernels report it as InvalidArgument).
    {
        Graph    g;
        TensorId x         = addInput(g, "x", {4, 6});
        TensorId y         = addTensor(g, "y", {});
        Node    &n         = addNode(g, OpType::ArgMin, "argmin_bad_axis", {x}, {y});
        n.attr.map["axis"] = intAttr(2);
        g.outputs          = {y};
        inferShapes(g, 1);
        EXPECT_TRUE(g.desc(y).shape.empty());
    }
}

TEST(OpWiring, LogicalAndIntegerBroadcastKeepOutputDtype) {
    // Or/Xor/Mod/BitShift/BitwiseAnd/Or/Xor share the two-operand NumPy broadcast rule and leave the
    // output dtype untouched; a zero extent broadcasts to zero.
    for (OpType type: {OpType::Or, OpType::Xor, OpType::Mod, OpType::BitShift, OpType::BitwiseAnd, OpType::BitwiseOr, OpType::BitwiseXor})
    {
        Graph    g;
        TensorId a = addInput(g, "a", {2, 1, 4}, DType::UInt8);
        TensorId b = addInput(g, "b", {3, 1}, DType::Int64);
        TensorId y = addTensor(g, "y", {});
        TensorId z = addInput(g, "z", {0, 1});
        TensorId w = addTensor(g, "w", {});
        addNode(g, type, "broadcast", {a, b}, {y});
        addNode(g, type, "broadcast_empty", {z, b}, {w});
        g.outputs = {y, w};
        inferShapes(g, 1);
        EXPECT_EQ(g.desc(y).shape, (Shape {2, 3, 4})) << opTypeName(type);
        EXPECT_EQ(g.desc(y).dtype, DType::Float32) << opTypeName(type) << " must not stamp an operand dtype";
        EXPECT_EQ(g.desc(w).shape, (Shape {0, 1})) << opTypeName(type);
    }
}

TEST(OpWiring, NotKeepsFloatDtypeBitwiseNotCopiesInputDtype) {
    Graph    g;
    TensorId mask     = addInput(g, "mask", {2, 3}, DType::UInt8); // a BOOL graph input imports as UInt8
    TensorId inverted = addTensor(g, "inverted", {});
    TensorId ids      = addInput(g, "ids", {5}, DType::Int64);
    TensorId flipped  = addTensor(g, "flipped", {});
    addNode(g, OpType::Not, "not", {mask}, {inverted});
    addNode(g, OpType::BitwiseNot, "bitwise_not", {ids}, {flipped});
    g.outputs = {inverted, flipped};
    inferShapes(g, 1);
    EXPECT_EQ(g.desc(inverted).shape, (Shape {2, 3}));
    EXPECT_EQ(g.desc(inverted).dtype, DType::Float32) << "Not must not copy the UInt8 bool label (it would block fusion downstream)";
    EXPECT_EQ(g.desc(flipped).shape, (Shape {5}));
    EXPECT_EQ(g.desc(flipped).dtype, DType::Int64);
}

// --- constFold --------------------------------------------------------------------------------------

TEST(OpWiring, ConstFoldReadsByteInitializersAsIntegerValues) {
    // INT8/UINT8 initializers keep native 1-byte lanes, while the CPU kernels read non-int64 operands
    // through host.f32(). constFold widens such an operand to integer-valued fp32 before folding, for
    // label-preserving kernels (Reshape) and fp32 readers (Cast, Where) alike, and leaves the graph's
    // own 1-byte payload untouched.
    Graph    g;
    TensorId signedBytes = addByteInitializer(g, "q", {4}, DType::Int8, {0x80, 0xFF, 0x00, 0x7F}); // -128, -1, 0, 127
    TensorId mask        = addByteInitializer(g, "mask", {4}, DType::UInt8, {1, 0, 255, 0});
    TensorId target      = addInt64Initializer(g, "target", {2}, {2, 2});
    TensorId whenTrue    = addFloatInitializer(g, "when_true", {4}, {10, 20, 30, 40});
    TensorId whenFalse   = addFloatInitializer(g, "when_false", {4}, {-1, -2, -3, -4});
    TensorId reshaped    = addTensor(g, "reshaped", {});
    TensorId casted      = addTensor(g, "casted", {});
    TensorId selected    = addTensor(g, "selected", {});
    addNode(g, OpType::Reshape, "reshape", {signedBytes, target}, {reshaped});
    Node &cast          = addNode(g, OpType::Cast, "cast", {signedBytes}, {casted});
    cast.attr.map["to"] = intAttr(1); // FLOAT
    addNode(g, OpType::Where, "where", {mask, whenTrue, whenFalse}, {selected});
    g.outputs = {reshaped, casted, selected};

    inferShapes(g, 1);
    EXPECT_EQ(constFold(g), 3);
    EXPECT_TRUE(g.nodes.empty());

    const float expectedSigned[4] = {-128, -1, 0, 127};
    ASSERT_TRUE(g.isInitializer(reshaped));
    EXPECT_EQ(g.desc(reshaped).shape, (Shape {2, 2}));
    EXPECT_EQ(g.desc(reshaped).dtype, DType::Float32);
    ASSERT_EQ(g.initializers[reshaped].bytes.size(), 4 * sizeof(float));
    ASSERT_TRUE(g.isInitializer(casted));
    ASSERT_EQ(g.initializers[casted].bytes.size(), 4 * sizeof(float));
    for (int k = 0; k < 4; ++k)
    {
        EXPECT_EQ(g.initializers[reshaped].f32()[k], expectedSigned[k]) << "k=" << k;
        EXPECT_EQ(g.initializers[casted].f32()[k], expectedSigned[k]) << "k=" << k;
    }
    const float expectedSelected[4] = {10, -2, 30, -4};
    ASSERT_TRUE(g.isInitializer(selected));
    ASSERT_EQ(g.initializers[selected].bytes.size(), 4 * sizeof(float));
    for (int k = 0; k < 4; ++k)
    {
        EXPECT_EQ(g.initializers[selected].f32()[k], expectedSelected[k]) << "k=" << k;
    }
    // The source initializers keep their native 1-byte storage and label.
    EXPECT_EQ(g.desc(signedBytes).dtype, DType::Int8);
    EXPECT_EQ(g.initializers[signedBytes].bytes.size(), 4u);
    EXPECT_EQ(g.desc(mask).dtype, DType::UInt8);
    EXPECT_EQ(g.initializers[mask].bytes.size(), 4u);
}

// --- dtype lattices ---------------------------------------------------------------------------------

TEST(OpWiring, ArgExtremeAndBoolResultsKeepTheirCastToFloat) {
    // ArgMax/ArgMin indices are int64 and Or/Xor/Not results are bool, whatever the operand dtype: a
    // Cast of them to FLOAT is a genuine conversion that eliminateFloatCast must keep. A Cast of a
    // float activation is the control that proves the pass ran.
    Graph    g;
    TensorId x = addInput(g, "x", {2, 5});
    const std::pair<OpType, const char *> producers[] = {{OpType::ArgMax, "argmax"}, {OpType::ArgMin, "argmin"}, {OpType::Or, "or"}, {OpType::Xor, "xor"}, {OpType::Not, "not"}};
    for (const auto &[type, name]: producers)
    {
        const std::string stem   = name;
        TensorId          result = addTensor(g, stem + "_out", {});
        TensorId          casted = addTensor(g, stem + "_float", {});
        TensorId          relu   = addTensor(g, stem + "_relu", {});
        const bool        binary = type == OpType::Or || type == OpType::Xor;
        addNode(g, type, stem, binary ? std::vector<TensorId> {x, x} : std::vector<TensorId> {x}, {result});
        Node &cast          = addNode(g, OpType::Cast, stem + "_cast", {result}, {casted});
        cast.attr.map["to"] = intAttr(1); // FLOAT
        addNode(g, OpType::Relu, stem + "_consumer", {casted}, {relu});
        g.outputs.push_back(relu);
    }
    TensorId controlCast = addTensor(g, "control_float", {});
    TensorId controlRelu = addTensor(g, "control_relu", {});
    Node    &cast        = addNode(g, OpType::Cast, "control_cast", {x}, {controlCast});
    cast.attr.map["to"]  = intAttr(1);
    addNode(g, OpType::Relu, "control_consumer", {controlCast}, {controlRelu});
    g.outputs.push_back(controlRelu);

    eliminateFloatCast(g);
    for (const auto &[type, name]: producers)
    {
        EXPECT_NE(findNode(g, std::string(name) + "_cast"), nullptr) << opTypeName(type) << " result cast must survive";
    }
    EXPECT_EQ(findNode(g, "control_cast"), nullptr) << "a float->float cast is still dropped";
}

TEST(OpWiring, ArgExtremeAndBoolResultsNeverFoldToTrunc) {
    // foldIntRoundtripCast folds Cast(float -> INT64) -> Cast(-> FLOAT) into Unary(Trunc) only for a
    // proven float source; indices and bool masks are not float sources.
    Graph    g;
    TensorId x = addInput(g, "x", {2, 5});
    const std::pair<OpType, const char *> producers[] = {{OpType::ArgMax, "argmax"}, {OpType::ArgMin, "argmin"}, {OpType::Or, "or"}, {OpType::Xor, "xor"}, {OpType::Not, "not"}};
    for (const auto &[type, name]: producers)
    {
        const std::string stem   = name;
        TensorId          result = addTensor(g, stem + "_out", {});
        TensorId          wide   = addTensor(g, stem + "_int64", {});
        TensorId          narrow = addTensor(g, stem + "_float", {});
        TensorId          relu   = addTensor(g, stem + "_relu", {});
        const bool        binary = type == OpType::Or || type == OpType::Xor;
        addNode(g, type, stem, binary ? std::vector<TensorId> {x, x} : std::vector<TensorId> {x}, {result});
        Node &toInt            = addNode(g, OpType::Cast, stem + "_to_int", {result}, {wide});
        toInt.attr.map["to"]   = intAttr(7); // INT64
        Node &toFloat          = addNode(g, OpType::Cast, stem + "_to_float", {wide}, {narrow});
        toFloat.attr.map["to"] = intAttr(1); // FLOAT
        addNode(g, OpType::Relu, stem + "_consumer", {narrow}, {relu});
        g.outputs.push_back(relu);
    }
    foldIntRoundtripCast(g);
    EXPECT_EQ(countNodes(g, OpType::Unary), 0) << "no integer/bool round-trip may become Trunc";
    EXPECT_EQ(countNodes(g, OpType::Cast), 10);
}

// --- load-time fp32 pins ----------------------------------------------------------------------------

TEST(IntegerPins, ArgExtremePinsIndicesButNotData) {
    // The indices output is pinned fp32; the data keeps its fp16 storage and markFp32 inserts no
    // ConvertDtype in front of ArgMax/ArgMin input 0 (the kernel reads the data at its own precision).
    for (OpType type: {OpType::ArgMax, OpType::ArgMin})
    {
        Graph    g;
        TensorId logits  = addInput(g, "logits", {1, 4096}, DType::Float32, true);
        TensorId indices = addTensor(g, "indices", {1, 1}, DType::Int64, true);
        addNode(g, type, "select", {logits}, {indices});
        g.outputs                = {indices};
        g.desc(indices).isOutput = true;

        pinIntegerResultsFp32(g);
        EXPECT_TRUE(g.desc(indices).storeFp32) << opTypeName(type);
        EXPECT_FALSE(g.desc(logits).storeFp32) << opTypeName(type) << " data must keep its storage precision";

        markFp32(g, "");
        EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0) << opTypeName(type) << " data input must not be bridged";
        const Node *select = findNode(g, "select");
        ASSERT_NE(select, nullptr);
        EXPECT_EQ(select->inputs[0], logits);
    }
}

TEST(IntegerPins, ResultPinsOnlyIntegerValuedFlatOutputs) {
    // Integer results: ArgMax/ArgMin, Mod with fmod 0 (the default), BitShift, BitwiseAnd/Or/Xor/Not.
    // Bool results (1.0/0.0 exact in fp16), Mod with fmod 1, and non-flat outputs stay unpinned.
    struct Case {
        OpType  type;
        int64_t fmod; // -1 = attribute absent
        bool    flat;
        bool    expectPinned;
    };
    const Case cases[] = {
        {OpType::ArgMax, -1, true, true},     {OpType::ArgMin, -1, true, true},       {OpType::Mod, -1, true, true},        {OpType::Mod, 0, true, true},
        {OpType::Mod, 1, true, false},        {OpType::BitShift, -1, true, true},     {OpType::BitwiseAnd, -1, true, true}, {OpType::BitwiseOr, -1, true, true},
        {OpType::BitwiseXor, -1, true, true}, {OpType::BitwiseNot, -1, true, true},   {OpType::Or, -1, true, false},        {OpType::Xor, -1, true, false},
        {OpType::Not, -1, true, false},       {OpType::BitwiseAnd, -1, false, false},
    };
    for (const Case &c: cases)
    {
        Graph    g;
        TensorId a = addInput(g, "a", {8}, DType::Float32, c.flat);
        TensorId y = addTensor(g, "y", {8}, DType::Float32, c.flat);
        Node    &n = addNode(g, c.type, "op", {a}, {y});
        if (c.fmod >= 0)
        {
            n.attr.map["fmod"] = intAttr(c.fmod);
        }
        g.outputs = {y};
        pinIntegerResultsFp32(g);
        EXPECT_EQ(g.desc(y).storeFp32, c.expectPinned) << opTypeName(c.type) << " fmod=" << c.fmod << " flat=" << c.flat;
    }
}

TEST(IntegerPins, IntegerOperandsPinThroughLayoutHopsAndBridgeAtNc4) {
    // BitwiseAnd(ids#flat, mask const): the runtime operand chain ids (NC4 graph input) -> ConvertLayout
    // -> ids#flat is pinned up to the flat/NC4 frontier; markFp32 bridges the NC4 source with a
    // ConvertDtype in front of the (now fp32) ConvertLayout. BitwiseXor(bits, act): a flat graph input is
    // pinned directly, while an operand produced by an NC4 fp16 node (Relu) is not pinned and gets a
    // frontier convert in front of the integer op. The constant mask is never pinned or bridged.
    Graph    g;
    TensorId ids     = addInput(g, "ids", {1, 4, 2, 2}, DType::Int64, false);
    TensorId idsFlat = addTensor(g, "ids#flat", {1, 4, 2, 2}, DType::Int64, true);
    TensorId mask    = addInt64Initializer(g, "mask", {1}, {0xFFFF});
    TensorId masked  = addTensor(g, "masked", {1, 4, 2, 2}, DType::Float32, true);
    TensorId bits    = addInput(g, "bits", {1, 4, 2, 2}, DType::Float32, true);
    TensorId x       = addInput(g, "x", {1, 4, 2, 2}, DType::Float32, false);
    TensorId act     = addTensor(g, "act", {1, 4, 2, 2}, DType::Float32, false);
    TensorId mixed   = addTensor(g, "mixed", {1, 4, 2, 2}, DType::Float32, true);
    addNode(g, OpType::ConvertLayout, "to_flat", {ids}, {idsFlat});
    addNode(g, OpType::BitwiseAnd, "bitwise_and", {idsFlat, mask}, {masked});
    addNode(g, OpType::Relu, "relu", {x}, {act});
    addNode(g, OpType::BitwiseXor, "bitwise_xor", {bits, act}, {mixed});
    g.outputs = {masked, mixed};

    pinIntegerResultsFp32(g);
    EXPECT_TRUE(g.desc(masked).storeFp32);
    EXPECT_TRUE(g.desc(idsFlat).storeFp32) << "the flat ConvertLayout output is a pinned hop";
    EXPECT_FALSE(g.desc(ids).storeFp32) << "the walk stops before the NC4 source";
    EXPECT_FALSE(g.desc(mask).storeFp32) << "constants upload at the node's precision";
    EXPECT_TRUE(g.desc(mixed).storeFp32);
    EXPECT_TRUE(g.desc(bits).storeFp32) << "a flat integer graph input packs at fp32";
    EXPECT_FALSE(g.desc(act).storeFp32) << "an NC4 producer's output is never pinned";

    markFp32(g, "");
    const Node *toFlat = findNode(g, "to_flat");
    const Node *andOp  = findNode(g, "bitwise_and");
    const Node *xorOp  = findNode(g, "bitwise_xor");
    ASSERT_NE(toFlat, nullptr);
    ASSERT_NE(andOp, nullptr);
    ASSERT_NE(xorOp, nullptr);
    EXPECT_EQ(andOp->inputs[0], idsFlat) << "a pinned operand needs no bridge";
    EXPECT_EQ(andOp->inputs[1], mask);
    EXPECT_NE(toFlat->inputs[0], ids) << "the NC4 source is bridged in front of the fp32 ConvertLayout";
    EXPECT_TRUE(g.desc(toFlat->inputs[0]).storeFp32);
    EXPECT_EQ(xorOp->inputs[0], bits);
    EXPECT_NE(xorOp->inputs[1], act) << "the unpinned fp16 operand is bridged to fp32";
    EXPECT_TRUE(g.desc(xorOp->inputs[1]).storeFp32);
    EXPECT_NE(g.desc(xorOp->inputs[1]).name.find("#f32"), std::string::npos) << g.desc(xorOp->inputs[1]).name;
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 2);
}

TEST(IntegerPins, SecondaryOutputOperandPinsProducerPrimaryOutput) {
    // TopK indices (outputs[1]) feed a BitwiseAnd. markFp32 gives every output of a node the precision
    // of outputs[0], so pinning only the indices would be undone; the walk pins the flat values output
    // too, and the indices reach the integer op at fp32 with no bridge.
    Graph    g;
    TensorId scores    = addInput(g, "scores", {1, 64}, DType::Float32, true);
    TensorId values    = addTensor(g, "values", {1, 4}, DType::Float32, true);
    TensorId indices   = addTensor(g, "indices", {1, 4}, DType::Int64, true);
    TensorId mask      = addInt64Initializer(g, "mask", {1}, {3});
    TensorId low       = addTensor(g, "low", {1, 4}, DType::Float32, true);
    Node    &topk      = addNode(g, OpType::TopK, "topk", {scores}, {values, indices});
    topk.attr.map["k"] = intAttr(4);
    addNode(g, OpType::BitwiseAnd, "low_bits", {indices, mask}, {low});
    g.outputs = {values, low};

    pinIntegerResultsFp32(g);
    EXPECT_TRUE(g.desc(indices).storeFp32);
    EXPECT_TRUE(g.desc(values).storeFp32) << "the producer's primary output carries the node precision";

    markFp32(g, "");
    EXPECT_TRUE(g.desc(indices).storeFp32) << "output alignment keeps the pin";
    const Node *andOp = findNode(g, "low_bits");
    ASSERT_NE(andOp, nullptr);
    EXPECT_EQ(andOp->inputs[0], indices) << "no bridge in front of the integer op";
}
