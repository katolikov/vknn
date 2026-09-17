// Load-time fp32 pins of integer arithmetic (pinIntegerResultsFp32). The GPU stores tensors as fp16 or
// fp32 float lanes: fp16 holds consecutive integers only up to 2^11 and saturates at 65504, so an int64
// Add/Sub/Mul/Max/Min (the CPU op computes it exactly in int64) must run on fp32 storage together with its
// operands, while float math keeps its precision tier. Every test runs the Vulkan load sequence
// (planFlatLayoutAndStorage) on a hand-built graph, or runStandardPasses first where the import lowering
// is part of the case, and checks the storeFp32 marks and the ConvertDtype bridges markFp32 places.
#include "core/segment_constant_operands.h"
#include "core/vk_gates.h"
#include "import/mod_integer_operands.h"
#include "import/passes.h"
#include "vknn/binary_type.h"
#include "vknn/graph.h"
#include "vknn/reduce_type.h"
#include "vknn/shape.h"
#include "vknn/unary_type.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    // ONNX TensorProto.DataType codes the tests' Cast nodes target.
    constexpr int64_t kOnnxFloat = 1;
    constexpr int64_t kOnnxInt32 = 6;
    constexpr int64_t kOnnxInt64 = 7;

    // An integer beyond fp16's range (it would store as 65504) and above fp16's exact-integer range.
    constexpr int64_t kBeyondHalfRange = 70000;

    // Parts of the two-part Concat the constant-part test builds.
    constexpr size_t kConcatPartCount = 2;

    Attr intAttr(int64_t value) {
        Attr attribute;
        attribute.kind = Attr::Int;
        attribute.i    = value;
        return attribute;
    }

    Attr intsAttr(std::vector<int64_t> values) {
        Attr attribute;
        attribute.kind = Attr::Ints;
        attribute.ints = std::move(values);
        return attribute;
    }

    TensorId addTensor(Graph &g, const std::string &name, Shape shape, DType dtype = DType::Float32) {
        TensorDesc desc;
        desc.name  = name;
        desc.shape = std::move(shape);
        desc.dtype = dtype;
        return g.addTensor(desc);
    }

    TensorId addInput(Graph &g, const std::string &name, Shape shape, DType dtype = DType::Float32) {
        TensorId id        = addTensor(g, name, std::move(shape), dtype);
        g.desc(id).isInput = true;
        g.inputs.push_back(id);
        return id;
    }

    void addOutput(Graph &g, TensorId id) {
        g.desc(id).isOutput = true;
        g.outputs.push_back(id);
    }

    TensorId addInt64Initializer(Graph &g, const std::string &name, Shape shape, const std::vector<int64_t> &values) {
        TensorId id              = addTensor(g, name, std::move(shape), DType::Int64);
        g.desc(id).isInitializer = true;
        HostBuffer buffer;
        buffer.resizeElems((int64_t) values.size(), DType::Int64);
        for (size_t k = 0; k < values.size(); ++k)
        {
            buffer.i64()[k] = values[k];
        }
        g.initializers[id] = buffer;
        return id;
    }

    // Every element of a tensor of `shape` holding `value`.
    template <typename Element> std::vector<Element> filledElements(const Shape &shape, Element value) {
        return std::vector<Element>((size_t) numElements(shape), value);
    }

    TensorId addFloatInitializer(Graph &g, const std::string &name, Shape shape, const std::vector<float> &values) {
        TensorId id              = addTensor(g, name, std::move(shape), DType::Float32);
        g.desc(id).isInitializer = true;
        HostBuffer buffer;
        buffer.resizeElems((int64_t) values.size(), DType::Float32);
        for (size_t k = 0; k < values.size(); ++k)
        {
            buffer.f32()[k] = values[k];
        }
        g.initializers[id] = buffer;
        return id;
    }

    void setAttr(Node &node, const char *key, Attr value) {
        node.attr.map[key] = std::move(value);
    }

    Node &addNode(Graph &g, OpType type, const std::string &name, std::vector<TensorId> inputs, std::vector<TensorId> outputs) {
        Node node;
        node.type    = type;
        node.name    = name;
        node.inputs  = std::move(inputs);
        node.outputs = std::move(outputs);
        g.nodes.push_back(std::move(node));
        return g.nodes.back();
    }

    Node &addBinary(Graph &g, BinaryType op, const std::string &name, std::vector<TensorId> inputs, TensorId output) {
        Node &node = addNode(g, OpType::Binary, name, std::move(inputs), {output});
        node.subOp = (int) op;
        return node;
    }

    Node &addCast(Graph &g, const std::string &name, TensorId input, TensorId output, int64_t onnxTarget) {
        Node &node          = addNode(g, OpType::Cast, name, {input}, {output});
        node.attr.map["to"] = intAttr(onnxTarget);
        return node;
    }

    int countNodes(const Graph &g, OpType type) {
        int count = 0;
        for (const Node &node: g.nodes)
        {
            count += node.type == type ? 1 : 0;
        }
        return count;
    }

    int countPinnedRuntimeTensors(const Graph &g) {
        int count = 0;
        for (const TensorDesc &desc: g.tensors)
        {
            count += (!desc.isInitializer && desc.storeFp32) ? 1 : 0;
        }
        return count;
    }

    const Node *findNode(const Graph &g, const std::string &name) {
        for (const Node &node: g.nodes)
        {
            if (node.name == name)
            {
                return &node;
            }
        }
        return nullptr;
    }

    const Node *producerOf(const Graph &g, TensorId tensor) {
        for (const Node &node: g.nodes)
        {
            for (TensorId output: node.outputs)
            {
                if (output == tensor)
                {
                    return &node;
                }
            }
        }
        return nullptr;
    }

} // namespace

TEST(IntegerArithmeticPins, Int64AddOfRuntimeOperandsPinsOperandsAndResult) {
    // int64 [2,5] + int64 [5] -> int64 output: both graph inputs pack at fp32 (70000 stays 70000 instead of
    // 65504), the Add runs its fp32 kernel and the result reads back from fp32 storage.
    Graph    g;
    TensorId lhs = addInput(g, "lhs", {2, 5}, DType::Int64);
    TensorId rhs = addInput(g, "rhs", {5}, DType::Int64);
    TensorId sum = addTensor(g, "sum", {2, 5}, DType::Int64);
    addNode(g, OpType::Add, "add", {lhs, rhs}, {sum});
    addOutput(g, sum);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_TRUE(g.desc(lhs).storeFp32);
    EXPECT_TRUE(g.desc(rhs).storeFp32);
    EXPECT_TRUE(g.desc(sum).storeFp32);
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
    const Node *add = findNode(g, "add");
    ASSERT_NE(add, nullptr);
    EXPECT_TRUE(g.desc(add->inputs[0]).storeFp32);
    EXPECT_TRUE(g.desc(add->inputs[1]).storeFp32);
}

TEST(IntegerArithmeticPins, Int64ArithmeticWithAConstantPinsTheRuntimeOperand) {
    // Add, Sub, Mul, Max and Min of a runtime int64 tensor and an int64 constant: the runtime operand and
    // the result are pinned; the constant is uploaded by the op at the node's (fp32) precision.
    struct Case {
        OpType     type;
        BinaryType op;
    };
    const Case cases[] = {
        {OpType::Add, BinaryType::Add},    {OpType::Binary, BinaryType::Sub}, {OpType::Binary, BinaryType::Mul},
        {OpType::Binary, BinaryType::Max}, {OpType::Binary, BinaryType::Min},
    };
    for (const Case &c: cases)
    {
        Graph    g;
        TensorId ids      = addInput(g, "ids", {1, 8}, DType::Int64);
        TensorId constant = addInt64Initializer(g, "constant", {1}, {kBeyondHalfRange});
        TensorId result   = addTensor(g, "result", {1, 8}, DType::Int64);
        Node    &node     = addNode(g, c.type, "arith", {ids, constant}, {result});
        node.subOp        = c.type == OpType::Binary ? (int) c.op : 0;
        addOutput(g, result);

        planFlatLayoutAndStorage(g, "", nullptr);
        EXPECT_TRUE(g.desc(ids).storeFp32) << (int) c.op;
        EXPECT_TRUE(g.desc(result).storeFp32) << (int) c.op;
        EXPECT_FALSE(g.desc(constant).storeFp32) << (int) c.op;
        EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0) << (int) c.op;
    }
}

TEST(IntegerArithmeticPins, LoweredVariadicMaxMinChainsPinEveryLink) {
    // A 3-input int64 Max/Min lowers to two Binary nodes; the partial result between them is typed only
    // when the node's output is, so the pin follows integer values through the chain, not dtype stamps.
    for (BinaryType op: {BinaryType::Max, BinaryType::Min})
    {
        for (DType resultType: {DType::Int64, DType::Float32})
        {
            Graph    g;
            TensorId a      = addInput(g, "a", {2, 5}, DType::Int64);
            TensorId b      = addInput(g, "b", {5}, DType::Int64);
            TensorId c      = addInput(g, "c", {2, 1}, DType::Int64);
            TensorId result = addTensor(g, "result", {}, resultType);
            addBinary(g, op, "variadic", {a, b, c}, result);
            addOutput(g, result);

            runStandardPasses(g);
            g.topoSort();
            ASSERT_EQ(countNodes(g, OpType::Binary), 2) << "the variadic node lowers to a two-node chain";
            planFlatLayoutAndStorage(g, "", nullptr);
            for (const Node &node: g.nodes)
            {
                ASSERT_EQ(node.type, OpType::Binary) << node.name;
                for (TensorId tensor: node.inputs)
                {
                    EXPECT_TRUE(g.desc(tensor).storeFp32) << node.name << " reads " << g.desc(tensor).name;
                }
                EXPECT_TRUE(g.desc(node.outputs[0]).storeFp32) << node.name << " writes " << g.desc(node.outputs[0]).name;
            }
            EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
        }
    }
}

TEST(IntegerArithmeticPins, Int64MulFeedingModPinsTheChain) {
    // ids -> Mul(int64 constant) -> Mod(fmod 0): the product is an integer operand of the Mod and an
    // integer result of the Mul; the whole chain is fp32 with no bridge.
    Graph    g;
    TensorId ids     = addInput(g, "ids", {1, 8}, DType::Int64);
    TensorId scale   = addInt64Initializer(g, "scale", {1}, {3});
    TensorId scaled  = addTensor(g, "scaled", {1, 8});
    TensorId divisor = addInt64Initializer(g, "divisor", {1}, {7});
    TensorId rem     = addTensor(g, "rem", {1, 8});
    addBinary(g, BinaryType::Mul, "mul", {ids, scale}, scaled);
    addNode(g, OpType::Mod, "mod", {scaled, divisor}, {rem});
    addOutput(g, rem);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_TRUE(g.desc(ids).storeFp32);
    EXPECT_TRUE(g.desc(scaled).storeFp32);
    EXPECT_TRUE(g.desc(rem).storeFp32);
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
}

TEST(IntegerArithmeticPins, MixedInt64AndFloatAddJoinsTheRegion) {
    // The CPU op computes int64 + fp32 in int64 (the float operand truncated toward zero), so the Add is
    // integer arithmetic: the float runtime operand is read at fp32 too.
    Graph    g;
    TensorId ids     = addInput(g, "ids", {1, 8}, DType::Int64);
    TensorId offsets = addInput(g, "offsets", {1, 8});
    TensorId sum     = addTensor(g, "sum", {1, 8});
    addNode(g, OpType::Add, "add", {ids, offsets}, {sum});
    addOutput(g, sum);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_TRUE(g.desc(ids).storeFp32);
    EXPECT_TRUE(g.desc(offsets).storeFp32);
    EXPECT_TRUE(g.desc(sum).storeFp32);
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
}

TEST(IntegerArithmeticPins, IntegerCastResultsJoinArithmetic) {
    // Cast(INT64) + Cast(INT64) of two float inputs: an untyped integer Cast result holds integers, so the
    // Add joins with no typed operand; each Cast's float operand is pinned toward its source (it may hold
    // values past fp16's range).
    Graph    g;
    TensorId x      = addInput(g, "x", {1, 8});
    TensorId y      = addInput(g, "y", {1, 8});
    TensorId xAsInt = addTensor(g, "x_as_int", {1, 8});
    TensorId yAsInt = addTensor(g, "y_as_int", {1, 8});
    TensorId sum    = addTensor(g, "sum", {1, 8});
    addCast(g, "x_to_int64", x, xAsInt, kOnnxInt64);
    addCast(g, "y_to_int64", y, yAsInt, kOnnxInt64);
    addNode(g, OpType::Add, "add", {xAsInt, yAsInt}, {sum});
    addOutput(g, sum);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_TRUE(g.desc(xAsInt).storeFp32);
    EXPECT_TRUE(g.desc(yAsInt).storeFp32);
    EXPECT_TRUE(g.desc(sum).storeFp32);
    EXPECT_TRUE(g.desc(x).storeFp32) << "an integer Cast reads its operand exactly";
    EXPECT_TRUE(g.desc(y).storeFp32) << "an integer Cast reads its operand exactly";
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
}

TEST(IntegerArithmeticPins, WhereSelectingIntegersJoinsTheRegion) {
    // Where(cond, ids, fallback) + offsets: the untyped selection holds integers, so the Add joins and the
    // selected operands are read at fp32; the float condition keeps its precision behind a bridge.
    Graph    g;
    TensorId cond     = addInput(g, "cond", {1, 8});
    TensorId ids      = addInput(g, "ids", {1, 8}, DType::Int64);
    TensorId fallback = addInput(g, "fallback", {1, 8}, DType::Int64);
    TensorId offsets  = addInput(g, "offsets", {1, 8});
    TensorId picked   = addTensor(g, "picked", {1, 8});
    TensorId sum      = addTensor(g, "sum", {1, 8});
    addNode(g, OpType::Where, "where", {cond, ids, fallback}, {picked});
    addNode(g, OpType::Add, "add", {picked, offsets}, {sum});
    addOutput(g, sum);

    planFlatLayoutAndStorage(g, "", nullptr);
    for (TensorId tensor: {ids, fallback, picked, offsets, sum})
    {
        EXPECT_TRUE(g.desc(tensor).storeFp32) << g.desc(tensor).name;
    }
    EXPECT_FALSE(g.desc(cond).storeFp32);
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 1) << "the fp16 condition is bridged into the fp32 Where";
}

TEST(IntegerArithmeticPins, IntegerResultLeavesTheRegionAtAFloatCast) {
    // ids -> Add(1) -> Cast(FLOAT) -> Mul(0.5): the integer part is fp32; the float part keeps its
    // precision and markFp32 bridges fp32 -> fp16 in front of the float Cast only.
    Graph    g;
    TensorId ids     = addInput(g, "ids", {1, 8}, DType::Int64);
    TensorId one     = addInt64Initializer(g, "one", {1}, {1});
    TensorId next    = addTensor(g, "next", {1, 8});
    TensorId asFloat = addTensor(g, "as_float", {1, 8});
    TensorId half    = addFloatInitializer(g, "half", {1}, {0.5f});
    TensorId scaled  = addTensor(g, "scaled", {1, 8});
    addNode(g, OpType::Add, "add", {ids, one}, {next});
    addCast(g, "to_float", next, asFloat, kOnnxFloat);
    addBinary(g, BinaryType::Mul, "mul", {asFloat, half}, scaled);
    addOutput(g, scaled);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_TRUE(g.desc(ids).storeFp32);
    EXPECT_TRUE(g.desc(next).storeFp32);
    EXPECT_FALSE(g.desc(asFloat).storeFp32);
    EXPECT_FALSE(g.desc(scaled).storeFp32);
    ASSERT_EQ(countNodes(g, OpType::ConvertDtype), 1);
    const Node *toFloat = findNode(g, "to_float");
    ASSERT_NE(toFloat, nullptr);
    ASSERT_NE(toFloat->inputs[0], next) << "the float Cast reads a bridged copy";
    EXPECT_NE(g.desc(toFloat->inputs[0]).name.find("#f16"), std::string::npos) << g.desc(toFloat->inputs[0]).name;
    const Node *bridge = producerOf(g, toFloat->inputs[0]);
    ASSERT_NE(bridge, nullptr);
    EXPECT_EQ(bridge->type, OpType::ConvertDtype);
    EXPECT_EQ(bridge->inputs[0], next);
}

TEST(IntegerArithmeticPins, IntegerInputReadOnlyThroughMovesIntoAFloatCastStaysUnpinned) {
    // An attention-mask style path: int64 mask -> Unsqueeze -> Unsqueeze -> Cast(FLOAT) -> Mul. No node
    // computes on the integers, so nothing is pinned and no bridge is inserted.
    Graph    g;
    TensorId mask      = addInput(g, "mask", {1, 8}, DType::Int64);
    TensorId axisOne   = addInt64Initializer(g, "axis_one", {1}, {1});
    TensorId axisTwo   = addInt64Initializer(g, "axis_two", {1}, {2});
    TensorId mask3d    = addTensor(g, "mask_3d", {1, 1, 8});
    TensorId mask4d    = addTensor(g, "mask_4d", {1, 1, 1, 8});
    TensorId maskFloat = addTensor(g, "mask_float", {1, 1, 1, 8});
    TensorId scale     = addFloatInitializer(g, "scale", {1}, {-10000.0f});
    TensorId bias      = addTensor(g, "bias", {1, 1, 1, 8});
    addNode(g, OpType::Unsqueeze, "unsqueeze_1", {mask, axisOne}, {mask3d});
    addNode(g, OpType::Unsqueeze, "unsqueeze_2", {mask3d, axisTwo}, {mask4d});
    addCast(g, "to_float", mask4d, maskFloat, kOnnxFloat);
    addBinary(g, BinaryType::Mul, "mul", {maskFloat, scale}, bias);
    addOutput(g, bias);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_EQ(countPinnedRuntimeTensors(g), 0);
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
}

TEST(IntegerArithmeticPins, FloatGraphsPinNothing) {
    // Float math of every op family the integer region extends through: nothing is pinned and no bridge
    // is inserted, including a float Pow with an int64 constant exponent and a rank-4 NC4HW4 Add.
    std::vector<Graph> graphs(4);
    {
        // Add/Mul/Sub, a lowered 3-input Max, a comparison selecting through Where, a float Cast.
        Graph   &g      = graphs[0];
        TensorId x      = addInput(g, "x", {2, 5});
        TensorId y      = addInput(g, "y", {5});
        TensorId z      = addInput(g, "z", {2, 1});
        TensorId c      = addFloatInitializer(g, "c", {1}, {1.5f});
        TensorId sum    = addTensor(g, "sum", {});
        TensorId prod   = addTensor(g, "prod", {});
        TensorId diff   = addTensor(g, "diff", {});
        TensorId peak   = addTensor(g, "peak", {});
        TensorId below  = addTensor(g, "below", {});
        TensorId picked = addTensor(g, "picked", {});
        TensorId copy   = addTensor(g, "copy", {});
        addNode(g, OpType::Add, "add", {x, c}, {sum});
        addBinary(g, BinaryType::Mul, "mul", {sum, y}, prod);
        addBinary(g, BinaryType::Sub, "sub", {prod, c}, diff);
        addBinary(g, BinaryType::Max, "max3", {diff, y, z}, peak);
        addNode(g, OpType::Less, "less", {peak, c}, {below});
        addNode(g, OpType::Where, "where", {below, peak, x}, {picked});
        addCast(g, "to_float", picked, copy, kOnnxFloat);
        addOutput(g, copy);
    }
    {
        // A same-shape rank-4 Add of runtime operands runs NC4HW4.
        Graph   &g   = graphs[1];
        TensorId a   = addInput(g, "a", {1, 4, 2, 2});
        TensorId b   = addInput(g, "b", {1, 4, 2, 2});
        TensorId sum = addTensor(g, "sum", {});
        TensorId act = addTensor(g, "act", {});
        addNode(g, OpType::Add, "add", {a, b}, {sum});
        addNode(g, OpType::Relu, "relu", {sum}, {act});
        addOutput(g, act);
    }
    {
        // ReduceSum, Clip and Neg on float data.
        Graph   &g       = graphs[2];
        TensorId x       = addInput(g, "x", {1, 8});
        TensorId lo      = addFloatInitializer(g, "lo", {}, {0.0f});
        TensorId hi      = addFloatInitializer(g, "hi", {}, {6.0f});
        TensorId clipped = addTensor(g, "clipped", {});
        TensorId negated = addTensor(g, "negated", {});
        TensorId total   = addTensor(g, "total", {});
        addNode(g, OpType::Clip, "clip", {x, lo, hi}, {clipped});
        Node &neg                   = addNode(g, OpType::Unary, "neg", {clipped}, {negated});
        neg.subOp                   = (int) UnaryType::Neg;
        Node &reduce                = addNode(g, OpType::Reduce, "sum", {negated}, {total});
        reduce.subOp                = (int) ReduceType::Sum;
        reduce.attr.map["axes"]     = intsAttr({1});
        reduce.attr.map["keepdims"] = intAttr(1);
        addOutput(g, total);
    }
    {
        // A float base raised to an int64 constant exponent is a float power.
        Graph   &g        = graphs[3];
        TensorId base     = addInput(g, "base", {1, 8});
        TensorId exponent = addInt64Initializer(g, "exponent", {1}, {3});
        TensorId power    = addTensor(g, "power", {});
        addBinary(g, BinaryType::Pow, "pow", {base, exponent}, power);
        addOutput(g, power);
    }
    for (size_t index = 0; index < graphs.size(); ++index)
    {
        Graph &g = graphs[index];
        runStandardPasses(g);
        g.topoSort();
        planFlatLayoutAndStorage(g, "", nullptr);
        EXPECT_EQ(countPinnedRuntimeTensors(g), 0) << "float graph " << index;
        EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0) << "float graph " << index;
    }
}

TEST(IntegerArithmeticPins, Nc4SameShapeInt64AddPinsItsOutput) {
    // A same-shape rank-4 Add of runtime operands runs NC4HW4; the NC4HW4 Add kernel has an fp32 variant,
    // so its integer output is pinned like a flat one.
    Graph    g;
    TensorId a   = addInput(g, "a", {1, 4, 2, 2}, DType::Int64);
    TensorId b   = addInput(g, "b", {1, 4, 2, 2}, DType::Int64);
    TensorId sum = addTensor(g, "sum", {1, 4, 2, 2}, DType::Int64);
    addNode(g, OpType::Add, "add", {a, b}, {sum});
    addOutput(g, sum);

    planFlatLayoutAndStorage(g, "", nullptr);
    const Node *add = findNode(g, "add");
    ASSERT_NE(add, nullptr);
    EXPECT_FALSE(g.desc(add->outputs[0]).gpuFlat) << "the case exercises the NC4HW4 Add";
    EXPECT_TRUE(g.desc(add->outputs[0]).storeFp32);
    EXPECT_TRUE(g.desc(a).storeFp32);
    EXPECT_TRUE(g.desc(b).storeFp32);
    EXPECT_TRUE(g.desc(g.outputs[0]).storeFp32);
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
}

TEST(IntegerArithmeticPins, IntegerComparisonComparesAtFp32) {
    // ids == 70000 -> Unsqueeze -> Where condition selecting between float tensors: the int64 operand is
    // read at fp32 and the 0/1 mask is pinned so the comparison kernel runs fp32 (at fp16 every id past
    // 65504 would equal 70000). The mask does not spread: the Unsqueeze and the float Where keep their
    // precision, with one fp32 -> fp16 bridge in front of the Unsqueeze.
    Graph    g;
    TensorId ids     = addInput(g, "ids", {1, 8}, DType::Int64);
    TensorId padId   = addInt64Initializer(g, "pad_id", {1}, {kBeyondHalfRange});
    TensorId isPad   = addTensor(g, "is_pad", {1, 8});
    TensorId axisOne = addInt64Initializer(g, "axis_one", {1}, {1});
    TensorId isPad3d = addTensor(g, "is_pad_3d", {1, 1, 8});
    TensorId onPad   = addInput(g, "on_pad", {1, 1, 8});
    TensorId onToken = addInput(g, "on_token", {1, 1, 8});
    TensorId picked  = addTensor(g, "picked", {1, 1, 8});
    addNode(g, OpType::Equal, "equal", {ids, padId}, {isPad});
    addNode(g, OpType::Unsqueeze, "unsqueeze", {isPad, axisOne}, {isPad3d});
    addNode(g, OpType::Where, "where", {isPad3d, onPad, onToken}, {picked});
    addOutput(g, picked);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_TRUE(g.desc(ids).storeFp32);
    EXPECT_TRUE(g.desc(isPad).storeFp32);
    EXPECT_FALSE(g.desc(isPad3d).storeFp32) << "a 0/1 mask is exact at fp16 and is not spread";
    EXPECT_FALSE(g.desc(picked).storeFp32);
    EXPECT_FALSE(g.desc(onPad).storeFp32);
    EXPECT_FALSE(g.desc(onToken).storeFp32);
    ASSERT_EQ(countNodes(g, OpType::ConvertDtype), 1);
    const Node *unsqueeze = findNode(g, "unsqueeze");
    ASSERT_NE(unsqueeze, nullptr);
    EXPECT_NE(g.desc(unsqueeze->inputs[0]).name.find("#f16"), std::string::npos) << g.desc(unsqueeze->inputs[0]).name;
    const Node *where = findNode(g, "where");
    ASSERT_NE(where, nullptr);
    EXPECT_EQ(where->inputs[0], isPad3d);
}

TEST(IntegerArithmeticPins, IntegerPowReadsItsExponentExactly) {
    // An int64 base makes an integer power; its runtime exponent is read at fp32 too (fp16 stores 2049 as
    // 2048, flipping the sign of (-1)^2049). A float base with the same int64 exponent is a float power and
    // pins nothing.
    {
        Graph    g;
        TensorId base     = addInput(g, "base", {1, 8}, DType::Int64);
        TensorId exponent = addInput(g, "exponent", {1, 8}, DType::Int64);
        TensorId power    = addTensor(g, "power", {1, 8}, DType::Int64);
        addBinary(g, BinaryType::Pow, "pow", {base, exponent}, power);
        addOutput(g, power);

        planFlatLayoutAndStorage(g, "", nullptr);
        EXPECT_TRUE(g.desc(base).storeFp32);
        EXPECT_TRUE(g.desc(exponent).storeFp32);
        EXPECT_TRUE(g.desc(power).storeFp32);
        EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
    }
    {
        Graph    g;
        TensorId base     = addInput(g, "base", {1, 8});
        TensorId exponent = addInput(g, "exponent", {1, 8}, DType::Int64);
        TensorId power    = addTensor(g, "power", {1, 8});
        addBinary(g, BinaryType::Pow, "pow", {base, exponent}, power);
        addOutput(g, power);

        planFlatLayoutAndStorage(g, "", nullptr);
        EXPECT_EQ(countPinnedRuntimeTensors(g), 0);
        EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
    }
}

TEST(IntegerArithmeticPins, Int64DivOnTheCpuKeepsTheRegionAroundIt) {
    // ids -> Add(1) -> Div(2) -> Mul(3): the Div stays on the CPU op (vkNodeGate refuses an int64 Div), and
    // the region runs through it, so the GPU Add hands it exact values and the GPU Mul reads its int64
    // quotient at fp32.
    Graph    g;
    TensorId ids      = addInput(g, "ids", {1, 8}, DType::Int64);
    TensorId one      = addInt64Initializer(g, "one", {1}, {1});
    TensorId next     = addTensor(g, "next", {1, 8}, DType::Int64);
    TensorId two      = addInt64Initializer(g, "two", {1}, {2});
    TensorId quotient = addTensor(g, "quotient", {1, 8});
    TensorId three    = addInt64Initializer(g, "three", {1}, {3});
    TensorId product  = addTensor(g, "product", {1, 8}, DType::Int64);
    addNode(g, OpType::Add, "add", {ids, one}, {next});
    addBinary(g, BinaryType::Div, "div", {next, two}, quotient);
    addBinary(g, BinaryType::Mul, "mul", {quotient, three}, product);
    addOutput(g, product);

    planFlatLayoutAndStorage(g, "", nullptr);
    const Node *div = findNode(g, "div");
    ASSERT_NE(div, nullptr);
    std::string reason;
    EXPECT_FALSE(vkNodeGate(g, *div, &reason));
    EXPECT_NE(reason.find("integer Div"), std::string::npos) << reason;
    for (TensorId tensor: {ids, next, quotient, product})
    {
        EXPECT_TRUE(g.desc(tensor).storeFp32) << g.desc(tensor).name;
    }
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
}

TEST(IntegerArithmeticPins, IntegerDataOpsPinOnlyIntegerResults) {
    // Int32 data (bound as fp32 lanes): ReduceSum/Max/Min/Prod, Clip, Neg and Abs keep integers and are
    // pinned with their operand; ReduceMean/L2 and Sqrt are float results and pin nothing.
    struct Case {
        OpType      type;
        int         subOp;
        bool        expectPinned;
        const char *label;
    };
    const Case cases[] = {
        {OpType::Reduce, (int) ReduceType::Sum, true, "ReduceSum"},
        {OpType::Reduce, (int) ReduceType::Max, true, "ReduceMax"},
        {OpType::Reduce, (int) ReduceType::Min, true, "ReduceMin"},
        {OpType::Reduce, (int) ReduceType::Prod, true, "ReduceProd"},
        {OpType::Reduce, (int) ReduceType::Mean, false, "ReduceMean"},
        {OpType::Reduce, (int) ReduceType::L2, false, "ReduceL2"},
        {OpType::Clip, 0, true, "Clip"},
        {OpType::Unary, (int) UnaryType::Neg, true, "Neg"},
        {OpType::Unary, (int) UnaryType::Abs, true, "Abs"},
        {OpType::Unary, (int) UnaryType::Sqrt, false, "Sqrt"},
    };
    for (const Case &c: cases)
    {
        Graph    g;
        TensorId counts = addInput(g, "counts", {1, 8}, DType::Int32);
        TensorId result = addTensor(g, "result", c.type == OpType::Reduce ? Shape {1, 1} : Shape {1, 8});
        Node    &node   = addNode(g, c.type, "op", {counts}, {result});
        node.subOp      = c.subOp;
        if (c.type == OpType::Reduce)
        {
            node.attr.map["axes"]     = intsAttr({1});
            node.attr.map["keepdims"] = intAttr(1);
        }
        addOutput(g, result);

        planFlatLayoutAndStorage(g, "", nullptr);
        EXPECT_EQ(g.desc(result).storeFp32, c.expectPinned) << c.label;
        EXPECT_EQ(g.desc(counts).storeFp32, c.expectPinned) << c.label;
        EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0) << c.label;
    }
}

TEST(IntegerArithmeticPins, ArgMaxOfIntegerCastDataReadsTheDataExactly) {
    // x -> Cast(INT32) -> ArgMax: the untyped Cast result holds integers, so ArgMax scans it at fp32
    // (fp16 would store 4096 and 4097 as a tie) and the Cast reads its operand exactly.
    Graph    g;
    TensorId x       = addInput(g, "x", {1, 64});
    TensorId asInt   = addTensor(g, "as_int", {1, 64});
    TensorId indices = addTensor(g, "indices", {1, 1}, DType::Int64);
    addCast(g, "to_int32", x, asInt, kOnnxInt32);
    Node &argmax            = addNode(g, OpType::ArgMax, "argmax", {asInt}, {indices});
    argmax.attr.map["axis"] = intAttr(1);
    addOutput(g, indices);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_TRUE(g.desc(asInt).storeFp32);
    EXPECT_TRUE(g.desc(x).storeFp32);
    EXPECT_TRUE(g.desc(indices).storeFp32);
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
}

namespace {

    // Builders of one movement op reading the int64 graph input `ids`: each adds the op and returns the
    // tensor holding the moved values, appending every other runtime integer operand it creates.
    using MovementBuilder = TensorId (*)(Graph &, TensorId, std::vector<TensorId> &);

    TensorId buildReshape(Graph &g, TensorId ids, std::vector<TensorId> &) {
        TensorId shape = addInt64Initializer(g, "shape", {2}, {2, 4});
        TensorId moved = addTensor(g, "moved", {2, 4});
        addNode(g, OpType::Reshape, "op", {ids, shape}, {moved});
        return moved;
    }

    TensorId buildFlatten(Graph &g, TensorId ids, std::vector<TensorId> &) {
        TensorId moved        = addTensor(g, "moved", {1, 8});
        Node    &node         = addNode(g, OpType::Flatten, "op", {ids}, {moved});
        node.attr.map["axis"] = intAttr(1);
        return moved;
    }

    TensorId buildSqueeze(Graph &g, TensorId ids, std::vector<TensorId> &) {
        TensorId axes  = addInt64Initializer(g, "axes", {1}, {0});
        TensorId moved = addTensor(g, "moved", {8});
        addNode(g, OpType::Squeeze, "op", {ids, axes}, {moved});
        return moved;
    }

    TensorId buildUnsqueeze(Graph &g, TensorId ids, std::vector<TensorId> &) {
        TensorId axes  = addInt64Initializer(g, "axes", {1}, {0});
        TensorId moved = addTensor(g, "moved", {1, 1, 8});
        addNode(g, OpType::Unsqueeze, "op", {ids, axes}, {moved});
        return moved;
    }

    TensorId buildSlice(Graph &g, TensorId ids, std::vector<TensorId> &) {
        TensorId starts = addInt64Initializer(g, "starts", {1}, {1});
        TensorId ends   = addInt64Initializer(g, "ends", {1}, {5});
        TensorId axes   = addInt64Initializer(g, "axes", {1}, {1});
        TensorId moved  = addTensor(g, "moved", {1, 4});
        addNode(g, OpType::Slice, "op", {ids, starts, ends, axes}, {moved});
        return moved;
    }

    TensorId buildTranspose(Graph &g, TensorId ids, std::vector<TensorId> &) {
        TensorId moved        = addTensor(g, "moved", {4, 2});
        Node    &node         = addNode(g, OpType::Transpose, "op", {ids}, {moved});
        node.attr.map["perm"] = intsAttr({1, 0});
        return moved;
    }

    TensorId buildExpand(Graph &g, TensorId ids, std::vector<TensorId> &) {
        TensorId shape = addInt64Initializer(g, "shape", {2}, {2, 8});
        TensorId moved = addTensor(g, "moved", {2, 8});
        addNode(g, OpType::Expand, "op", {ids, shape}, {moved});
        return moved;
    }

    TensorId buildTile(Graph &g, TensorId ids, std::vector<TensorId> &) {
        TensorId repeats = addInt64Initializer(g, "repeats", {2}, {2, 1});
        TensorId moved   = addTensor(g, "moved", {2, 8});
        addNode(g, OpType::Tile, "op", {ids, repeats}, {moved});
        return moved;
    }

    TensorId buildSplitHalves(Graph &g, TensorId ids, Shape halfShape) {
        TensorId sizes        = addInt64Initializer(g, "sizes", {2}, {4, 4});
        TensorId moved        = addTensor(g, "moved", halfShape);
        TensorId rest         = addTensor(g, "rest", halfShape);
        Node    &node         = addNode(g, OpType::Split, "op", {ids, sizes}, {moved, rest});
        node.attr.map["axis"] = intAttr(1);
        return moved;
    }

    TensorId buildFlatSplit(Graph &g, TensorId ids, std::vector<TensorId> &) {
        return buildSplitHalves(g, ids, {1, 4});
    }

    TensorId buildChannelSplit(Graph &g, TensorId ids, std::vector<TensorId> &) {
        return buildSplitHalves(g, ids, {1, 4, 2, 2});
    }

    TensorId buildGather(Graph &g, TensorId ids, std::vector<TensorId> &) {
        TensorId index        = addInt64Initializer(g, "index", {2}, {0, 3});
        TensorId moved        = addTensor(g, "moved", {1, 2});
        Node    &node         = addNode(g, OpType::Gather, "op", {ids, index}, {moved});
        node.attr.map["axis"] = intAttr(1);
        return moved;
    }

    TensorId buildPad(Graph &g, TensorId ids, std::vector<TensorId> &) {
        TensorId pads  = addInt64Initializer(g, "pads", {4}, {0, 1, 0, 1});
        TensorId moved = addTensor(g, "moved", {1, 10});
        addNode(g, OpType::Pad, "op", {ids, pads}, {moved});
        return moved;
    }

    TensorId buildDepthToSpace(Graph &g, TensorId ids, std::vector<TensorId> &) {
        TensorId moved             = addTensor(g, "moved", {1, 2, 4, 4});
        Node    &node              = addNode(g, OpType::DepthToSpace, "op", {ids}, {moved});
        node.attr.map["blocksize"] = intAttr(2);
        return moved;
    }

    TensorId buildScatterND(Graph &g, TensorId ids, std::vector<TensorId> &integerOperands) {
        TensorId indices = addInt64Initializer(g, "indices", {1, 1, 2}, {0, 3});
        TensorId updates = addInput(g, "updates", {1}, DType::Int64);
        TensorId moved   = addTensor(g, "moved", {1, 8});
        addNode(g, OpType::ScatterND, "op", {ids, indices, updates}, {moved});
        integerOperands.push_back(updates);
        return moved;
    }

    TensorId buildTopKValues(Graph &g, TensorId ids, std::vector<TensorId> &) {
        TensorId k            = addInt64Initializer(g, "k", {1}, {3});
        TensorId moved        = addTensor(g, "moved", {1, 3});
        TensorId indices      = addTensor(g, "indices", {1, 3}, DType::Int64);
        Node    &node         = addNode(g, OpType::TopK, "op", {ids, k}, {moved, indices});
        node.attr.map["axis"] = intAttr(1);
        return moved;
    }

    TensorId buildConcatWithInput(Graph &g, TensorId ids, std::vector<TensorId> &integerOperands) {
        Shape joinedShape = g.desc(ids).shape;
        joinedShape[1] *= 2;
        TensorId more         = addInput(g, "more", g.desc(ids).shape, DType::Int64);
        TensorId moved        = addTensor(g, "moved", joinedShape);
        Node    &node         = addNode(g, OpType::Concat, "op", {ids, more}, {moved});
        node.attr.map["axis"] = intAttr(1);
        integerOperands.push_back(more);
        return moved;
    }

    TensorId buildWhereValues(Graph &g, TensorId ids, std::vector<TensorId> &integerOperands) {
        TensorId condition = addInput(g, "condition", {1, 8});
        TensorId more      = addInput(g, "more", {1, 8}, DType::Int64);
        TensorId moved     = addTensor(g, "moved", {1, 8});
        addNode(g, OpType::Where, "op", {condition, ids, more}, {moved});
        integerOperands.push_back(more);
        return moved;
    }

    TensorId buildChannelShuffle(Graph &g, TensorId ids, std::vector<TensorId> &) {
        TensorId moved          = addTensor(g, "moved", {1, 8, 2, 2});
        Node    &node           = addNode(g, OpType::ChannelShuffle, "op", {ids}, {moved});
        node.attr.map["groups"] = intAttr(2);
        return moved;
    }

    struct MovementCase {
        const char     *label;
        Shape           idsShape;
        bool            expectNc4;           // the case exercises the op's NC4HW4 kernel
        bool            sharedByModResolver; // modOperandsAreInteger resolves the result's element type from ids
        int             floatOperandBridges; // float operands (Where's condition) reach the fp32 op through a bridge
        MovementBuilder build;
    };

    const MovementCase kMovementCases[] = {
        {"Reshape", {1, 8}, false, true, 0, buildReshape},
        {"Flatten", {1, 2, 4}, false, true, 0, buildFlatten},
        {"Squeeze", {1, 8}, false, true, 0, buildSqueeze},
        {"Unsqueeze", {1, 8}, false, true, 0, buildUnsqueeze},
        {"Slice", {1, 8}, false, true, 0, buildSlice},
        {"Transpose", {2, 4}, false, true, 0, buildTranspose},
        {"Expand", {1, 8}, false, true, 0, buildExpand},
        {"Tile", {1, 8}, false, true, 0, buildTile},
        {"Split (flat)", {1, 8}, false, true, 0, buildFlatSplit},
        {"Split (NC4HW4 channel)", {1, 8, 2, 2}, true, true, 0, buildChannelSplit},
        {"Gather", {1, 8}, false, true, 0, buildGather},
        {"Pad", {1, 8}, false, true, 0, buildPad},
        {"DepthToSpace", {1, 8, 2, 2}, false, true, 0, buildDepthToSpace},
        {"ScatterND", {1, 8}, false, true, 0, buildScatterND},
        {"TopK values", {1, 8}, false, true, 0, buildTopKValues},
        {"Concat (flat)", {1, 8}, false, true, 0, buildConcatWithInput},
        {"Concat (NC4HW4 channel)", {1, 4, 2, 2}, true, true, 0, buildConcatWithInput},
        {"Where values", {1, 8}, false, true, 1, buildWhereValues},
        {"ChannelShuffle", {1, 8, 2, 2}, false, false, 0, buildChannelShuffle},
    };

} // namespace

TEST(IntegerArithmeticPins, MovementOpsCarryTheRegionToTheirIntegerSources) {
    // ids (int64) -> movement op -> Add(1): the op copies element values at its storage precision, so the
    // int64 sources it reads (the data, ScatterND's updates, the other Concat part, Where's other value)
    // are pinned with the Add, and no ConvertDtype widens an already-saturated fp16 value (a float
    // operand, Where's condition, is the only bridge). A Mod reading the same untyped result resolves it as
    // integer (modOperandsAreInteger), so the two resolvers agree on every op that shares its data's
    // element type.
    for (const MovementCase &c: kMovementCases)
    {
        {
            Graph                 g;
            std::vector<TensorId> integerOperands;
            TensorId              ids   = addInput(g, "ids", c.idsShape, DType::Int64);
            TensorId              moved = c.build(g, ids, integerOperands);
            TensorId              one   = addInt64Initializer(g, "one", {1}, {1});
            TensorId              sum   = addTensor(g, "sum", g.desc(moved).shape, DType::Int64);
            addNode(g, OpType::Add, "add", {moved, one}, {sum});
            addOutput(g, sum);

            planFlatLayoutAndStorage(g, "", nullptr);
            if (c.expectNc4)
            {
                EXPECT_FALSE(g.desc(moved).gpuFlat) << c.label << ": the case exercises the NC4HW4 kernel";
            }
            EXPECT_TRUE(g.desc(ids).storeFp32) << c.label;
            for (TensorId operand: integerOperands)
            {
                EXPECT_TRUE(g.desc(operand).storeFp32) << c.label << " " << g.desc(operand).name;
            }
            EXPECT_TRUE(g.desc(moved).storeFp32) << c.label;
            EXPECT_TRUE(g.desc(sum).storeFp32) << c.label;
            EXPECT_EQ(countNodes(g, OpType::ConvertDtype), c.floatOperandBridges) << c.label;
        }
        if (c.sharedByModResolver)
        {
            Graph                 g;
            std::vector<TensorId> integerOperands;
            TensorId              ids       = addInput(g, "ids", c.idsShape, DType::Int64);
            TensorId              moved     = c.build(g, ids, integerOperands);
            TensorId              divisor   = addInput(g, "divisor", {1});
            TensorId              remainder = addTensor(g, "remainder", g.desc(moved).shape);
            Node                 &mod       = addNode(g, OpType::Mod, "mod", {moved, divisor}, {remainder});
            mod.attr.map["fmod"]            = intAttr(1);
            addOutput(g, remainder);
            EXPECT_TRUE(modOperandsAreInteger(g, g.nodes.back())) << c.label;
        }
    }
}

TEST(IntegerArithmeticPins, Nc4ConcatReadingAConstantPartKeepsTheSegmentPrecision) {
    // An NC4HW4 Concat reads each part through the segment's activation buffer, which the segment fills
    // from a constant in any slot at its own storage precision (segmentFilledConstantOperandEnd covers every
    // part, so a constant after the first part has a buffer too); an fp32 Concat would read that fp16
    // buffer as fp32. So the Concat keeps its precision, together with the runtime part it reads, and the
    // integer Add reading it gets one fp16 -> fp32 bridge. A flat Concat uploads its constant parts at its
    // own precision and joins the region; the segment fills only its operand 0.
    for (size_t constantSlot = 0; constantSlot < kConcatPartCount; ++constantSlot)
    {
        Graph                 g;
        TensorId              ids = addInput(g, "ids", {1, 4, 2, 2}, DType::Int64);
        const Shape           partShape {1, 4, 2, 2};
        TensorId              base   = addInt64Initializer(g, "base", partShape, filledElements(partShape, kBeyondHalfRange));
        TensorId              other  = addInput(g, "other", {1, 8, 2, 2}, DType::Int64);
        TensorId              joined = addTensor(g, "joined", {1, 8, 2, 2});
        TensorId              sum    = addTensor(g, "sum", {1, 8, 2, 2}, DType::Int64);
        std::vector<TensorId> parts  = constantSlot == 0 ? std::vector<TensorId> {base, ids} : std::vector<TensorId> {ids, base};
        setAttr(addNode(g, OpType::Concat, "concat", parts, {joined}), "axis", intAttr(1));
        addNode(g, OpType::Add, "add", {joined, other}, {sum});
        addOutput(g, sum);

        planFlatLayoutAndStorage(g, "", nullptr);
        EXPECT_FALSE(g.desc(joined).gpuFlat) << "the case exercises the NC4HW4 Concat";
        const Node *concat = findNode(g, "concat");
        ASSERT_NE(concat, nullptr);
        EXPECT_EQ(segmentFilledConstantOperandEnd(g, *concat), kConcatPartCount) << "the segment fills every part";
        EXPECT_FALSE(g.desc(joined).storeFp32) << "constant part in slot " << constantSlot;
        EXPECT_FALSE(g.desc(ids).storeFp32) << "constant part in slot " << constantSlot;
        EXPECT_TRUE(g.desc(other).storeFp32) << "constant part in slot " << constantSlot;
        EXPECT_TRUE(g.desc(sum).storeFp32) << "constant part in slot " << constantSlot;
        ASSERT_EQ(countNodes(g, OpType::ConvertDtype), 1) << "constant part in slot " << constantSlot;
        const Node *add = findNode(g, "add");
        ASSERT_NE(add, nullptr);
        const Node *bridge = producerOf(g, add->inputs[0]);
        ASSERT_NE(bridge, nullptr);
        EXPECT_EQ(bridge->type, OpType::ConvertDtype);
        EXPECT_EQ(bridge->inputs[0], joined);
    }
    {
        Graph    g;
        TensorId ids    = addInput(g, "ids", {1, 8}, DType::Int64);
        TensorId base   = addInt64Initializer(g, "base", {1, 8}, filledElements(Shape {1, 8}, kBeyondHalfRange));
        TensorId joined = addTensor(g, "joined", {1, 16});
        TensorId one    = addInt64Initializer(g, "one", {1}, {1});
        TensorId sum    = addTensor(g, "sum", {1, 16}, DType::Int64);
        setAttr(addNode(g, OpType::Concat, "concat", {base, ids}, {joined}), "axis", intAttr(1));
        addNode(g, OpType::Add, "add", {joined, one}, {sum});
        addOutput(g, sum);

        planFlatLayoutAndStorage(g, "", nullptr);
        EXPECT_TRUE(g.desc(joined).gpuFlat);
        const Node *concat = findNode(g, "concat");
        ASSERT_NE(concat, nullptr);
        EXPECT_EQ(segmentFilledConstantOperandEnd(g, *concat), kSegmentFilledLeadingOperandEnd) << "a flat Concat uploads its constant parts itself";
        EXPECT_TRUE(g.desc(joined).storeFp32);
        EXPECT_TRUE(g.desc(ids).storeFp32);
        EXPECT_TRUE(g.desc(sum).storeFp32);
        EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
    }
}

TEST(IntegerArithmeticPins, KernelsReadingAConstantThroughTheSegmentBufferKeepTheSegmentPrecision) {
    // An NC4HW4 Split and a ReduceSum read a constant operand 0 through the segment's fp16-filled
    // activation buffer, so neither is pinned even when an integer Add reads its result: the Add reads it
    // through one fp16 -> fp32 bridge, and the Split's other part, an integer graph output, reaches the
    // output's layout convert through another. (Const folding removes such all-constant nodes on the load
    // path; the rule keeps a node that survives it from reading fp16 bytes as fp32.)
    {
        Graph    g;
        TensorId source = addInt64Initializer(g, "source", {1, 8, 2, 2}, filledElements(Shape {1, 8, 2, 2}, kBeyondHalfRange));
        TensorId sizes  = addInt64Initializer(g, "sizes", {2}, {4, 4});
        TensorId head   = addTensor(g, "head", {1, 4, 2, 2});
        TensorId tail   = addTensor(g, "tail", {1, 4, 2, 2});
        TensorId other  = addInput(g, "other", {1, 4, 2, 2}, DType::Int64);
        TensorId sum    = addTensor(g, "sum", {1, 4, 2, 2}, DType::Int64);
        setAttr(addNode(g, OpType::Split, "split", {source, sizes}, {head, tail}), "axis", intAttr(1));
        addNode(g, OpType::Add, "add", {head, other}, {sum});
        addOutput(g, sum);
        addOutput(g, tail);

        planFlatLayoutAndStorage(g, "", nullptr);
        EXPECT_FALSE(g.desc(head).gpuFlat) << "the case exercises the NC4HW4 Split";
        EXPECT_FALSE(g.desc(head).storeFp32);
        EXPECT_FALSE(g.desc(tail).storeFp32);
        EXPECT_TRUE(g.desc(sum).storeFp32);
        EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 2);
    }
    {
        Graph    g;
        TensorId source             = addInt64Initializer(g, "source", {1, 8}, filledElements(Shape {1, 8}, kBeyondHalfRange));
        TensorId total              = addTensor(g, "total", {1, 1});
        TensorId other              = addInput(g, "other", {1, 1}, DType::Int64);
        TensorId sum                = addTensor(g, "sum", {1, 1}, DType::Int64);
        Node    &reduce             = addNode(g, OpType::Reduce, "reduce", {source}, {total});
        reduce.subOp                = (int) ReduceType::Sum;
        reduce.attr.map["axes"]     = intsAttr({1});
        reduce.attr.map["keepdims"] = intAttr(1);
        addNode(g, OpType::Add, "add", {total, other}, {sum});
        addOutput(g, sum);

        planFlatLayoutAndStorage(g, "", nullptr);
        EXPECT_FALSE(g.desc(total).storeFp32);
        EXPECT_TRUE(g.desc(other).storeFp32);
        EXPECT_TRUE(g.desc(sum).storeFp32);
        EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 1);
    }
}

TEST(IntegerArithmeticPins, TopKOfIntegerDataRanksAtFp32) {
    // TopK over int64 data with only its indices consumed: fp16 would store 70000 and 70001 as one value
    // and rank them as a tie, so the data is read exactly and both outputs run fp32.
    Graph    g;
    TensorId ids     = addInput(g, "ids", {1, 8}, DType::Int64);
    TensorId k       = addInt64Initializer(g, "k", {1}, {3});
    TensorId values  = addTensor(g, "values", {1, 3});
    TensorId indices = addTensor(g, "indices", {1, 3}, DType::Int64);
    setAttr(addNode(g, OpType::TopK, "topk", {ids, k}, {values, indices}), "axis", intAttr(1));
    addOutput(g, indices);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_TRUE(g.desc(ids).storeFp32);
    EXPECT_TRUE(g.desc(values).storeFp32);
    EXPECT_TRUE(g.desc(indices).storeFp32);
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
}

TEST(IntegerArithmeticPins, TopKIndicesInArithmeticLeaveTheFloatScoresAlone) {
    // Float scores -> TopK -> indices + 1: the indices are integers whatever the scores hold, so the Add
    // pins them (and the values, which share the node's precision) without pulling the float scores into
    // the region; the scores reach the fp32 TopK through one bridge.
    Graph    g;
    TensorId scores  = addInput(g, "scores", {1, 64});
    TensorId k       = addInt64Initializer(g, "k", {1}, {3});
    TensorId values  = addTensor(g, "values", {1, 3});
    TensorId indices = addTensor(g, "indices", {1, 3}, DType::Int64);
    TensorId one     = addInt64Initializer(g, "one", {1}, {1});
    TensorId next    = addTensor(g, "next", {1, 3}, DType::Int64);
    TensorId scaled  = addTensor(g, "scaled", {1, 64});
    TensorId half    = addFloatInitializer(g, "half", {1}, {0.5f});
    setAttr(addNode(g, OpType::TopK, "topk", {scores, k}, {values, indices}), "axis", intAttr(1));
    addNode(g, OpType::Add, "add", {indices, one}, {next});
    addBinary(g, BinaryType::Mul, "float_reader", {scores, half}, scaled);
    addOutput(g, next);
    addOutput(g, scaled);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_TRUE(g.desc(indices).storeFp32);
    EXPECT_TRUE(g.desc(values).storeFp32);
    EXPECT_TRUE(g.desc(next).storeFp32);
    EXPECT_FALSE(g.desc(scores).storeFp32);
    EXPECT_FALSE(g.desc(scaled).storeFp32);
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 1);
}

TEST(IntegerArithmeticPins, TopKIndicesIndexingAGatherStayFp32) {
    // Float scores -> TopK -> indices -> Gather index. The Gather reads its index as fp32 whatever the
    // index's storage (gather.comp binding 1 is float), and markFp32 gives every TopK output the values'
    // precision, so the pinned indices keep fp32 only because the values are pinned with them; the float
    // scores reach the fp32 TopK through one fp16 -> fp32 bridge.
    Graph    g;
    TensorId scores  = addInput(g, "scores", {1, 64});
    TensorId k       = addInt64Initializer(g, "k", {1}, {3});
    TensorId values  = addTensor(g, "values", {1, 3});
    TensorId indices = addTensor(g, "indices", {1, 3}, DType::Int64);
    TensorId table   = addFloatInitializer(g, "table", {64, 1}, filledElements(Shape {64, 1}, 0.5f));
    TensorId rows    = addTensor(g, "rows", {1, 3, 1});
    setAttr(addNode(g, OpType::TopK, "topk", {scores, k}, {values, indices}), "axis", intAttr(1));
    setAttr(addNode(g, OpType::Gather, "gather", {table, indices}, {rows}), "axis", intAttr(0));
    addOutput(g, rows);

    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_TRUE(g.desc(indices).storeFp32);
    EXPECT_TRUE(g.desc(values).storeFp32);
    EXPECT_FALSE(g.desc(scores).storeFp32);
    EXPECT_FALSE(g.desc(rows).storeFp32);
    ASSERT_EQ(countNodes(g, OpType::ConvertDtype), 1) << "the scores reach the fp32 TopK through a bridge";
    const Node *topk = findNode(g, "topk");
    ASSERT_NE(topk, nullptr);
    const Node *bridge = producerOf(g, topk->inputs[0]);
    ASSERT_NE(bridge, nullptr);
    EXPECT_EQ(bridge->type, OpType::ConvertDtype);
}

TEST(IntegerArithmeticPins, FloatMovementGraphsPinNothing) {
    // Float data through TopK, ScatterND, Pad, an NC4HW4 channel Split and DepthToSpace, with int64 index,
    // pad and count parameters: parameters never seed the region, so nothing is pinned or bridged.
    Graph    g;
    TensorId scores  = addInput(g, "scores", {1, 8});
    TensorId k       = addInt64Initializer(g, "k", {1}, {3});
    TensorId best    = addTensor(g, "best", {});
    TensorId indices = addTensor(g, "indices", {}, DType::Int64);
    TensorId bias    = addFloatInitializer(g, "bias", {1}, {0.25f});
    TensorId shifted = addTensor(g, "shifted", {});
    setAttr(addNode(g, OpType::TopK, "topk", {scores, k}, {best, indices}), "axis", intAttr(1));
    addNode(g, OpType::Add, "add", {best, bias}, {shifted});
    addOutput(g, shifted);

    TensorId data      = addInput(g, "data", {1, 8});
    TensorId where     = addInt64Initializer(g, "where", {1, 1, 2}, {0, 3});
    TensorId updates   = addInput(g, "updates", {1});
    TensorId scattered = addTensor(g, "scattered", {});
    TensorId pads      = addInt64Initializer(g, "pads", {4}, {0, 1, 0, 1});
    TensorId padded    = addTensor(g, "padded", {});
    addNode(g, OpType::ScatterND, "scatter", {data, where, updates}, {scattered});
    addNode(g, OpType::Pad, "pad", {scattered, pads}, {padded});
    addOutput(g, padded);

    TensorId features = addInput(g, "features", {1, 8, 2, 2});
    TensorId sizes    = addInt64Initializer(g, "sizes", {2}, {4, 4});
    TensorId head     = addTensor(g, "head", {});
    TensorId tail     = addTensor(g, "tail", {});
    TensorId merged   = addTensor(g, "merged", {});
    TensorId upscaled = addTensor(g, "upscaled", {});
    setAttr(addNode(g, OpType::Split, "split", {features, sizes}, {head, tail}), "axis", intAttr(1));
    addNode(g, OpType::Add, "merge", {head, tail}, {merged});
    setAttr(addNode(g, OpType::DepthToSpace, "d2s", {merged}, {upscaled}), "blocksize", intAttr(2));
    addOutput(g, upscaled);

    runStandardPasses(g);
    g.topoSort();
    planFlatLayoutAndStorage(g, "", nullptr);
    EXPECT_EQ(countPinnedRuntimeTensors(g), 0);
    EXPECT_EQ(countNodes(g, OpType::ConvertDtype), 0);
}
