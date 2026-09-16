// Import-side wiring of the logical (Or/Xor/Not), bitwise (BitwiseAnd/Or/Xor/Not, BitShift), Mod and
// ArgMax/ArgMin ops, independent of their kernels: the shape/dtype rules inferShapes applies, the
// dtype lattices that decide which Casts survive, constFold's handling of 1-byte INT8/UINT8
// initializers, and the load-time fp32 pins (pinIntegerResultsFp32 + the markFp32 frontier) that keep
// integer values exact on the GPU's float storage. The pin passes only run in a Vulkan session, so
// these tests run the session's load sequence (planFlatLayoutAndStorage) directly on hand-built graphs.
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

    // ONNX TensorProto.DataType codes the tests' Cast nodes target.
    constexpr int64_t kOnnxFloat = 1;
    constexpr int64_t kOnnxInt32 = 6;
    constexpr int64_t kOnnxInt64 = 7;

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

    // Mark a tensor as a declared graph output.
    void addOutput(Graph &g, TensorId id) {
        g.desc(id).isOutput = true;
        g.outputs.push_back(id);
    }

    // The node writing `t`, or null.
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
    // through host.f32(). constFold widens such an operand to integer-valued fp32 before a value-reading
    // fold (Cast, Where), leaves the graph's own 1-byte payload untouched, and a pure data-movement fold
    // (Reshape, Unsqueeze, Concat) keeps the native label and 1-byte payload.
    Graph    g;
    TensorId signedBytes = addByteInitializer(g, "q", {4}, DType::Int8, {0x80, 0xFF, 0x00, 0x7F}); // -128, -1, 0, 127
    TensorId mask        = addByteInitializer(g, "mask", {4}, DType::UInt8, {1, 0, 255, 0});
    TensorId target      = addInt64Initializer(g, "target", {2}, {2, 2});
    TensorId axes        = addInt64Initializer(g, "axes", {1}, {0});
    TensorId whenTrue    = addFloatInitializer(g, "when_true", {4}, {10, 20, 30, 40});
    TensorId whenFalse   = addFloatInitializer(g, "when_false", {4}, {-1, -2, -3, -4});
    TensorId reshaped    = addTensor(g, "reshaped", {});
    TensorId casted      = addTensor(g, "casted", {});
    TensorId selected    = addTensor(g, "selected", {});
    TensorId unsqueezed  = addTensor(g, "unsqueezed", {});
    TensorId joined      = addTensor(g, "joined", {});
    addNode(g, OpType::Reshape, "reshape", {signedBytes, target}, {reshaped});
    Node &cast          = addNode(g, OpType::Cast, "cast", {signedBytes}, {casted});
    cast.attr.map["to"] = intAttr(kOnnxFloat);
    addNode(g, OpType::Where, "where", {mask, whenTrue, whenFalse}, {selected});
    addNode(g, OpType::Unsqueeze, "unsqueeze", {mask, axes}, {unsqueezed});
    Node &concat            = addNode(g, OpType::Concat, "concat", {signedBytes, signedBytes}, {joined});
    concat.attr.map["axis"] = intAttr(0);
    g.outputs               = {reshaped, casted, selected, unsqueezed, joined};

    inferShapes(g, 1);
    EXPECT_EQ(constFold(g), 5);
    EXPECT_TRUE(g.nodes.empty());

    const std::vector<uint8_t> signedLanes       = {0x80, 0xFF, 0x00, 0x7F};
    const float                expectedSigned[4] = {-128, -1, 0, 127};
    ASSERT_TRUE(g.isInitializer(reshaped));
    EXPECT_EQ(g.desc(reshaped).shape, (Shape {2, 2}));
    EXPECT_EQ(g.desc(reshaped).dtype, DType::Int8) << "a data-movement fold keeps the INT8 label";
    ASSERT_EQ(g.initializers[reshaped].bytes.size(), 4u) << "... and the 1-byte payload";
    ASSERT_TRUE(g.isInitializer(casted));
    ASSERT_EQ(g.initializers[casted].bytes.size(), 4 * sizeof(float));
    for (int k = 0; k < 4; ++k)
    {
        EXPECT_EQ(g.initializers[reshaped].bytes.data()[k], signedLanes[(size_t) k]) << "k=" << k;
        EXPECT_EQ(g.initializers[casted].f32()[k], expectedSigned[k]) << "k=" << k;
    }
    const float expectedSelected[4] = {10, -2, 30, -4};
    ASSERT_TRUE(g.isInitializer(selected));
    ASSERT_EQ(g.initializers[selected].bytes.size(), 4 * sizeof(float));
    for (int k = 0; k < 4; ++k)
    {
        EXPECT_EQ(g.initializers[selected].f32()[k], expectedSelected[k]) << "k=" << k;
    }
    ASSERT_TRUE(g.isInitializer(unsqueezed));
    EXPECT_EQ(g.desc(unsqueezed).shape, (Shape {1, 4}));
    EXPECT_EQ(g.desc(unsqueezed).dtype, DType::UInt8);
    ASSERT_EQ(g.initializers[unsqueezed].bytes.size(), 4u);
    const std::vector<uint8_t> maskLanes = {1, 0, 255, 0};
    for (int k = 0; k < 4; ++k)
    {
        EXPECT_EQ(g.initializers[unsqueezed].bytes.data()[k], maskLanes[(size_t) k]) << "k=" << k;
    }
    ASSERT_TRUE(g.isInitializer(joined));
    EXPECT_EQ(g.desc(joined).shape, (Shape {8}));
    EXPECT_EQ(g.desc(joined).dtype, DType::Int8);
    ASSERT_EQ(g.initializers[joined].bytes.size(), 8u);
    for (int k = 0; k < 8; ++k)
    {
        EXPECT_EQ(g.initializers[joined].bytes.data()[k], signedLanes[(size_t) k % 4]) << "k=" << k;
    }
    // The source initializers keep their native 1-byte storage and label.
    EXPECT_EQ(g.desc(signedBytes).dtype, DType::Int8);
    EXPECT_EQ(g.initializers[signedBytes].bytes.size(), 4u);
    EXPECT_EQ(g.desc(mask).dtype, DType::UInt8);
    EXPECT_EQ(g.initializers[mask].bytes.size(), 4u);
}

TEST(OpWiring, ConstFoldedInt8ZeroPointKeepsQuantizeLinearSigned) {
    // QuantizeLinear takes its output type, and so its saturation range, from the zero_point's label.
    // A zero_point folded through Unsqueeze of a rank-0 INT8 initializer stays INT8, so the quantized
    // output is signed ([-128, 127]) rather than saturating negatives to 0.
    Graph    g;
    TensorId x         = addInput(g, "x", {1, 3});
    TensorId scale     = addFloatInitializer(g, "scale", {}, {0.5f});
    TensorId zeroPoint = addByteInitializer(g, "zero_point", {}, DType::Int8, {0xFE}); // -2
    TensorId axes      = addInt64Initializer(g, "axes", {1}, {0});
    TensorId zp1d      = addTensor(g, "zero_point_1d", {});
    TensorId quantized = addTensor(g, "quantized", {});
    addNode(g, OpType::Unsqueeze, "unsqueeze_zero_point", {zeroPoint, axes}, {zp1d});
    addNode(g, OpType::QuantizeLinear, "quantize", {x, scale, zp1d}, {quantized});
    g.outputs = {quantized};

    inferShapes(g, 1);
    EXPECT_EQ(constFold(g), 1);
    inferShapes(g, 1);
    ASSERT_TRUE(g.isInitializer(zp1d));
    EXPECT_EQ(g.desc(zp1d).dtype, DType::Int8);
    EXPECT_EQ(g.desc(zp1d).shape, (Shape {1}));
    ASSERT_EQ(g.initializers[zp1d].bytes.size(), 1u);
    EXPECT_EQ(g.initializers[zp1d].bytes.data()[0], 0xFE);
    EXPECT_EQ(g.desc(quantized).dtype, DType::Int8);
}

TEST(OpWiring, ConstFoldZeroElementByteInitializer) {
    // A zero-element INT8 initializer widens and folds to an empty INT8 initializer.
    Graph    g;
    TensorId empty    = addByteInitializer(g, "empty", {0}, DType::Int8, {});
    TensorId target   = addInt64Initializer(g, "target", {2}, {0, 3});
    TensorId reshaped = addTensor(g, "reshaped", {});
    TensorId casted   = addTensor(g, "casted", {});
    addNode(g, OpType::Reshape, "reshape", {empty, target}, {reshaped});
    Node &cast          = addNode(g, OpType::Cast, "cast", {empty}, {casted});
    cast.attr.map["to"] = intAttr(kOnnxFloat);
    g.outputs           = {reshaped, casted};

    inferShapes(g, 1);
    EXPECT_EQ(constFold(g), 2);
    ASSERT_TRUE(g.isInitializer(reshaped));
    EXPECT_EQ(g.desc(reshaped).dtype, DType::Int8);
    EXPECT_TRUE(g.initializers[reshaped].bytes.empty());
    ASSERT_TRUE(g.isInitializer(casted));
    EXPECT_TRUE(g.initializers[casted].bytes.empty());
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
        cast.attr.map["to"] = intAttr(kOnnxFloat);
        addNode(g, OpType::Relu, stem + "_consumer", {casted}, {relu});
        g.outputs.push_back(relu);
    }
    TensorId controlCast = addTensor(g, "control_float", {});
    TensorId controlRelu = addTensor(g, "control_relu", {});
    Node    &cast        = addNode(g, OpType::Cast, "control_cast", {x}, {controlCast});
    cast.attr.map["to"]  = intAttr(kOnnxFloat);
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
        toInt.attr.map["to"]   = intAttr(kOnnxInt64);
        Node &toFloat          = addNode(g, OpType::Cast, stem + "_to_float", {wide}, {narrow});
        toFloat.attr.map["to"] = intAttr(kOnnxFloat);
        addNode(g, OpType::Relu, stem + "_consumer", {narrow}, {relu});
        g.outputs.push_back(relu);
    }
    foldIntRoundtripCast(g);
    EXPECT_EQ(countNodes(g, OpType::Unary), 0) << "no integer/bool round-trip may become Trunc";
    EXPECT_EQ(countNodes(g, OpType::Cast), 10);
}

// --- load-time fp32 pins ----------------------------------------------------------------------------
// The pin tests run the Vulkan load sequence itself (planFlatLayoutAndStorage: insertLayoutConverts,
// the fp32 pins, markFp32) on graphs whose layouts start at the importer's defaults, so every layout,
// ConvertLayout and ConvertDtype is the one a session produces.

TEST(IntegerPins, ArgExtremePinsIndicesButNotData) {
    // The indices output is pinned fp32; float data keeps its fp16 storage and markFp32 inserts no
    // ConvertDtype in front of ArgMax/ArgMin input 0 (the kernel reads the data at its own precision).
    for (OpType type: {OpType::ArgMax, OpType::ArgMin})
    {
        Graph    g;
        TensorId logits  = addInput(g, "logits", {1, 4096});
        TensorId indices = addTensor(g, "indices", {1, 1}, DType::Int64);
        addNode(g, type, "select", {logits}, {indices});
        addOutput(g, indices);

        planFlatLayoutAndStorage(g, "", nullptr);
        EXPECT_TRUE(g.desc(logits).gpuFlat) << opTypeName(type);
        EXPECT_TRUE(g.desc(indices).storeFp32) << opTypeName(type);
        EXPECT_FALSE(g.desc(logits).storeFp32) << opTypeName(type) << " float data must keep its storage precision";
        EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0) << opTypeName(type) << " data input must not be bridged";
        const Node *select = findNode(g, "select");
        ASSERT_NE(select, nullptr);
        EXPECT_EQ(select->inputs[0], logits);
    }
}

TEST(IntegerPins, ArgExtremePinsInt64Data) {
    // Int64 data holds integers above fp16's exact range ([4096, 4097] would store as a tie), so the
    // data is pinned too and read at fp32.
    Graph    g;
    TensorId ids     = addInput(g, "ids", {1, 64}, DType::Int64);
    TensorId indices = addTensor(g, "indices", {1, 1}, DType::Int64);
    addNode(g, OpType::ArgMax, "select", {ids}, {indices});
    addOutput(g, indices);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_TRUE(g.desc(ids).storeFp32);
    EXPECT_TRUE(g.desc(indices).storeFp32);
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
}

TEST(IntegerPins, ResultPinsOnlyIntegerValuedResults) {
    // Integer results: ArgMax/ArgMin, Mod with fmod 0 (the default) or an Int32/Int64-typed operand,
    // BitShift, BitwiseAnd/Or/Xor/Not; the operands of the Mod and bitwise ops are pinned with them.
    // Bool results (1.0/0.0 exact in fp16) and Mod with fmod 1 on float operands stay unpinned.
    struct Case {
        OpType  type;
        int64_t fmod; // -1 = attribute absent
        DType   operandType;
        bool    expectPinned;
    };
    const Case cases[] = {
        {OpType::ArgMax, -1, DType::Float32, true},    {OpType::ArgMin, -1, DType::Float32, true},     {OpType::Mod, -1, DType::Float32, true},
        {OpType::Mod, 0, DType::Float32, true},        {OpType::Mod, 1, DType::Float32, false},        {OpType::Mod, 1, DType::Int64, true},
        {OpType::Mod, 1, DType::Int32, true},          {OpType::BitShift, -1, DType::Float32, true},   {OpType::BitwiseAnd, -1, DType::Float32, true},
        {OpType::BitwiseOr, -1, DType::Float32, true}, {OpType::BitwiseXor, -1, DType::Float32, true}, {OpType::BitwiseNot, -1, DType::Float32, true},
        {OpType::Or, -1, DType::Float32, false},       {OpType::Xor, -1, DType::Float32, false},       {OpType::Not, -1, DType::Float32, false},
    };
    for (const Case &c: cases)
    {
        Graph    g;
        TensorId a = addInput(g, "a", {8}, c.operandType, true);
        TensorId y = addTensor(g, "y", {8}, DType::Float32, true);
        Node    &n = addNode(g, c.type, "op", {a}, {y});
        if (c.fmod >= 0)
        {
            n.attr.map["fmod"] = intAttr(c.fmod);
        }
        g.outputs = {y};
        pinIntegerResultsFp32(g);
        EXPECT_EQ(g.desc(y).storeFp32, c.expectPinned) << opTypeName(c.type) << " fmod=" << c.fmod << " operand dtype=" << (int) c.operandType;
        const bool dataIsIntegerTyped = c.operandType == DType::Int64 || c.operandType == DType::Int32;
        const bool argExtreme         = c.type == OpType::ArgMax || c.type == OpType::ArgMin;
        const bool operandPinned      = argExtreme ? dataIsIntegerTyped : c.expectPinned;
        EXPECT_EQ(g.desc(a).storeFp32, operandPinned) << opTypeName(c.type) << " operand, fmod=" << c.fmod << " operand dtype=" << (int) c.operandType;
    }
}

TEST(IntegerPins, Int64GraphInputPinsThroughUnsqueeze) {
    // ids (int64 graph input) -> Unsqueeze -> BitwiseAnd: the graph input itself is pinned, so it packs
    // at fp32 (4097 stays 4097) and no ConvertDtype narrows or widens anything on the path.
    Graph    g;
    TensorId ids       = addInput(g, "ids", {1, 8}, DType::Int64);
    TensorId axes      = addInt64Initializer(g, "axes", {1}, {1});
    TensorId unsqueeze = addTensor(g, "ids_3d", {1, 1, 8});
    TensorId mask      = addInt64Initializer(g, "mask", {1}, {0xFFFF});
    TensorId masked    = addTensor(g, "masked", {1, 1, 8});
    addNode(g, OpType::Unsqueeze, "unsqueeze", {ids, axes}, {unsqueeze});
    addNode(g, OpType::BitwiseAnd, "bitwise_and", {unsqueeze, mask}, {masked});
    addOutput(g, masked);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_TRUE(g.desc(ids).storeFp32) << "the integer graph input packs at fp32";
    EXPECT_TRUE(g.desc(unsqueeze).storeFp32);
    EXPECT_TRUE(g.desc(masked).storeFp32);
    EXPECT_FALSE(g.desc(mask).storeFp32) << "constants upload at the node's precision";
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
}

TEST(IntegerPins, Int64GraphInputPinsThroughIntegerCastIntoMod) {
    // ids -> Cast(INT32) -> Mod(fmod 0): the Cast's operand is followed to the graph input. The Cast
    // keeps the channel count, so the layout pass leaves ids and the Cast NC4HW4 and converts to flat in
    // front of Mod; those NC4HW4 hops are written by no fp16-only kernel and are pinned too.
    Graph    g;
    TensorId ids        = addInput(g, "ids", {1, 8}, DType::Int64);
    TensorId narrow     = addTensor(g, "ids_int32", {1, 8});
    TensorId divisor    = addInt64Initializer(g, "divisor", {1}, {7});
    TensorId rem        = addTensor(g, "rem", {1, 8});
    Node    &cast       = addNode(g, OpType::Cast, "to_int32", {ids}, {narrow});
    cast.attr.map["to"] = intAttr(kOnnxInt32);
    addNode(g, OpType::Mod, "mod", {narrow, divisor}, {rem});
    addOutput(g, rem);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_FALSE(g.desc(narrow).gpuFlat) << "the case exercises NC4HW4 hops";
    EXPECT_TRUE(g.desc(ids).storeFp32) << "the integer graph input packs at fp32";
    EXPECT_TRUE(g.desc(narrow).storeFp32);
    EXPECT_TRUE(g.desc(rem).storeFp32);
    const Node *modOp = findNode(g, "mod");
    ASSERT_NE(modOp, nullptr);
    EXPECT_TRUE(g.desc(modOp->inputs[0]).storeFp32) << "the flat convert in front of Mod";
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
}

TEST(IntegerPins, ArgMaxIndicesStayFp32ThroughSqueezeToOutput) {
    // ArgMax(keepdims=1) -> Squeeze -> graph output: the Squeeze output (the declared output) is pinned
    // too, so the indices are never narrowed to fp16 before readback.
    Graph    g;
    TensorId logits             = addInput(g, "logits", {1, 32000});
    TensorId indices            = addTensor(g, "indices", {1, 1}, DType::Int64);
    TensorId squeezeAxes        = addInt64Initializer(g, "squeeze_axes", {1}, {1});
    TensorId token              = addTensor(g, "token", {1}, DType::Int64);
    Node    &argmax             = addNode(g, OpType::ArgMax, "argmax", {logits}, {indices});
    argmax.attr.map["axis"]     = intAttr(-1);
    argmax.attr.map["keepdims"] = intAttr(1);
    addNode(g, OpType::Squeeze, "squeeze", {indices, squeezeAxes}, {token});
    addOutput(g, token);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_TRUE(g.desc(indices).storeFp32);
    EXPECT_TRUE(g.desc(token).storeFp32) << "the graph output reads back at fp32";
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
}

TEST(IntegerPins, ArgMaxIndicesStayFp32ThroughIntegerCastAndLeaveAtFloatCast) {
    // ArgMax -> Cast(INT32) -> output stays in the integer region; ArgMax -> Cast(FLOAT) -> Unsqueeze ->
    // output leaves it: the float result keeps its storage precision and markFp32 bridges fp32 -> fp16
    // in front of that Cast only.
    Graph    g;
    TensorId logits    = addInput(g, "logits", {1, 32000});
    TensorId indices   = addTensor(g, "indices", {1, 1}, DType::Int64);
    TensorId asInt     = addTensor(g, "as_int32", {1, 1});
    TensorId asFloat   = addTensor(g, "as_float", {1, 1});
    TensorId axes      = addInt64Initializer(g, "axes", {1}, {0});
    TensorId asFloat3d = addTensor(g, "as_float_3d", {1, 1, 1});
    addNode(g, OpType::ArgMax, "argmax", {logits}, {indices});
    Node &toInt            = addNode(g, OpType::Cast, "to_int32", {indices}, {asInt});
    toInt.attr.map["to"]   = intAttr(kOnnxInt32);
    Node &toFloat          = addNode(g, OpType::Cast, "to_float", {indices}, {asFloat});
    toFloat.attr.map["to"] = intAttr(kOnnxFloat);
    addNode(g, OpType::Unsqueeze, "unsqueeze", {asFloat, axes}, {asFloat3d});
    addOutput(g, asInt);
    addOutput(g, asFloat3d);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_TRUE(g.desc(indices).storeFp32);
    EXPECT_TRUE(g.desc(asInt).storeFp32) << "an integer Cast keeps the indices in the region";
    EXPECT_FALSE(g.desc(asFloat).storeFp32) << "a float Cast leaves the region";
    EXPECT_FALSE(g.desc(asFloat3d).storeFp32) << "the region never spreads past a float Cast";
    const Node *toIntNode   = findNode(g, "to_int32");
    const Node *toFloatNode = findNode(g, "to_float");
    ASSERT_NE(toIntNode, nullptr);
    ASSERT_NE(toFloatNode, nullptr);
    EXPECT_EQ(toIntNode->inputs[0], indices) << "no bridge on the integer path";
    ASSERT_NE(toFloatNode->inputs[0], indices);
    EXPECT_NE(g.desc(toFloatNode->inputs[0]).name.find("#f16"), std::string::npos) << g.desc(toFloatNode->inputs[0]).name;
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 1);
}

TEST(IntegerPins, ConcatPartsJoinTheIntegerRegion) {
    // Concat(ids, ArgMax indices): ONNX Concat parts share one type, so the runtime int64 part is
    // pinned from the indices side and the joined result stays fp32.
    Graph    g;
    TensorId logits         = addInput(g, "logits", {1, 4096});
    TensorId ids            = addInput(g, "ids", {1, 4}, DType::Int64);
    TensorId indices        = addTensor(g, "indices", {1, 1}, DType::Int64);
    TensorId joined         = addTensor(g, "joined", {1, 5}, DType::Int64);
    Node    &argmax         = addNode(g, OpType::ArgMax, "argmax", {logits}, {indices});
    argmax.attr.map["axis"] = intAttr(1);
    Node &concat            = addNode(g, OpType::Concat, "concat", {ids, indices}, {joined});
    concat.attr.map["axis"] = intAttr(1);
    addOutput(g, joined);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_TRUE(g.desc(ids).storeFp32);
    EXPECT_TRUE(g.desc(indices).storeFp32);
    EXPECT_TRUE(g.desc(joined).storeFp32);
    EXPECT_FALSE(g.desc(logits).storeFp32);
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
}

TEST(IntegerPins, Nc4OperandBridgesInFrontOfItsLayoutConvert) {
    // BitwiseXor(bits, act): a flat integer graph input is pinned directly. The NC4HW4 Relu output
    // reaches the flat integer op through the ConvertLayout the layout pass splices in: that flat hop
    // is pinned, the fp16-only NC4 producer is not, and markFp32 bridges fp16 -> fp32 on the NC4 side,
    // in front of the (now fp32) ConvertLayout.
    Graph    g;
    TensorId bits  = addInput(g, "bits", {1, 4, 2, 2});
    TensorId x     = addInput(g, "x", {1, 4, 2, 2});
    TensorId act   = addTensor(g, "act", {1, 4, 2, 2});
    TensorId mixed = addTensor(g, "mixed", {1, 4, 2, 2});
    TensorId rect  = addTensor(g, "rect", {1, 4, 2, 2});
    addNode(g, OpType::Relu, "relu", {x}, {act});
    addNode(g, OpType::Relu, "relu_consumer", {act}, {rect});
    addNode(g, OpType::BitwiseXor, "bitwise_xor", {bits, act}, {mixed});
    addOutput(g, mixed);
    addOutput(g, rect);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_FALSE(g.desc(act).gpuFlat) << "Relu runs NC4HW4";
    EXPECT_FALSE(g.desc(act).storeFp32) << "an NC4 conv-family producer's output is never pinned";
    EXPECT_TRUE(g.desc(bits).storeFp32) << "a flat integer graph input packs at fp32";
    EXPECT_TRUE(g.desc(mixed).storeFp32);
    const Node *xorOp = findNode(g, "bitwise_xor");
    ASSERT_NE(xorOp, nullptr);
    EXPECT_EQ(xorOp->inputs[0], bits);
    const TensorId flatAct = xorOp->inputs[1];
    EXPECT_TRUE(g.desc(flatAct).gpuFlat);
    EXPECT_TRUE(g.desc(flatAct).storeFp32) << "the flat ConvertLayout output is a pinned hop";
    const Node *toFlat = producerOf(g, flatAct);
    ASSERT_NE(toFlat, nullptr);
    ASSERT_EQ(toFlat->type, OpType::ConvertLayout);
    const TensorId bridged = toFlat->inputs[0];
    ASSERT_NE(bridged, act) << "the NC4 source is bridged in front of the fp32 ConvertLayout";
    EXPECT_FALSE(g.desc(bridged).gpuFlat);
    EXPECT_TRUE(g.desc(bridged).storeFp32);
    EXPECT_NE(g.desc(bridged).name.find("#f32"), std::string::npos) << g.desc(bridged).name;
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 1);
}

TEST(IntegerPins, SecondaryOutputOperandPinsProducerPrimaryOutput) {
    // TopK indices (outputs[1]) feed a BitwiseAnd. markFp32 gives every output of a node the precision
    // of outputs[0], so pinning only the indices would be undone; the walk pins the flat values output
    // too, and the indices reach the integer op at fp32 with no bridge.
    Graph    g;
    TensorId scores    = addInput(g, "scores", {1, 64});
    TensorId values    = addTensor(g, "values", {1, 4});
    TensorId indices   = addTensor(g, "indices", {1, 4}, DType::Int64);
    TensorId mask      = addInt64Initializer(g, "mask", {1}, {3});
    TensorId low       = addTensor(g, "low", {1, 4});
    Node    &topk      = addNode(g, OpType::TopK, "topk", {scores}, {values, indices});
    topk.attr.map["k"] = intAttr(4);
    addNode(g, OpType::BitwiseAnd, "low_bits", {indices, mask}, {low});
    addOutput(g, values);
    addOutput(g, low);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_TRUE(g.desc(indices).storeFp32);
    EXPECT_TRUE(g.desc(values).storeFp32) << "the producer's primary output carries the node precision";
    const Node *andOp = findNode(g, "low_bits");
    ASSERT_NE(andOp, nullptr);
    EXPECT_EQ(andOp->inputs[0], indices) << "no bridge in front of the integer op";
}
