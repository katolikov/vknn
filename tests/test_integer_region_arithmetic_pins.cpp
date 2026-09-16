// Load-time fp32 pins of integer arithmetic (pinIntegerResultsFp32). The GPU stores tensors as fp16 or
// fp32 float lanes: fp16 holds consecutive integers only up to 2^11 and saturates at 65504, so an int64
// Add/Sub/Mul/Max/Min (the CPU op computes it exactly in int64) must run on fp32 storage together with its
// operands, while float math keeps its precision tier. Every test runs the Vulkan load sequence
// (planFlatLayoutAndStorage) on a hand-built graph, or runStandardPasses first where the import lowering
// is part of the case, and checks the storeFp32 marks and the ConvertDtype bridges markFp32 places.
#include "core/vk_gates.h"
#include "import/passes.h"
#include "vknn/binary_type.h"
#include "vknn/graph.h"
#include "vknn/reduce_type.h"
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
