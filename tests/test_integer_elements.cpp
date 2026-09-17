// Integer element values and int64 storage resolved per tensor from the graph IR
// (import/integer_elements.h), and the three rules that read them:
//
// - Pointwise fusion keeps a member holding integer values or int64 storage out of a unit: the CPU fused
//   kernel reads every operand as fp32 lanes, and a unit's float steps cut the integer fp32 pin. Integer
//   chains whose intermediates carry no Int64 label (a Cast to INT64, an integer Mod, a BitwiseAnd, an
//   int64 graph input) run through CPU Sessions with the default passes and give the exact integers, and
//   the Vulkan load sequence pins such a chain to fp32 from its graph input.
// - vkNodeGate keeps a Div with an int64 operand and a Pow with an int64 base on the CPU op when the
//   operand is a computed int64 tensor (an Add, a Reshape, a bitwise result, a Cast), not only a labeled
//   graph input or initializer.
// - pinIntegerResultsFp32 pins a graph output holding integer values and the movement chain feeding it (a
//   Gather from an int64 table, a Where or a Concat of int64 inputs).
#include "core/vk_gates.h"
#include "import/integer_elements.h"
#include "import/onnx/onnx_types.h"
#include "import/passes.h"
#include "vknn/binary_type.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include "vknn/unary_type.h"
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    constexpr int64_t kOnnxInt64 = (int64_t) onnx::OnnxType::Int64;
    constexpr int64_t kOnnxBool  = (int64_t) onnx::OnnxType::Bool;
    constexpr int64_t kOnnxFloat = (int64_t) onnx::OnnxType::Float;

    // An integer beyond fp16's range: an fp16 store would saturate it to 65504.
    constexpr int64_t kBeyondHalfRange = 70000;

    Attr intAttr(int64_t value) {
        Attr attribute;
        attribute.kind = Attr::Int;
        attribute.i    = value;
        return attribute;
    }

    TensorId addTensor(Graph &g, const std::string &name, Shape shape = {}, DType dtype = DType::Float32) {
        TensorDesc desc;
        desc.name  = name;
        desc.shape = std::move(shape);
        desc.dtype = dtype;
        return g.addTensor(desc);
    }

    TensorId addInput(Graph &g, const std::string &name, Shape shape, DType dtype) {
        TensorId id        = addTensor(g, name, std::move(shape), dtype);
        g.desc(id).isInput = true;
        g.inputs.push_back(id);
        return id;
    }

    void addOutput(Graph &g, TensorId id) {
        g.desc(id).isOutput = true;
        g.outputs.push_back(id);
    }

    TensorId addInt64Constant(Graph &g, const std::string &name, Shape shape, const std::vector<int64_t> &values) {
        TensorId id              = addTensor(g, name, std::move(shape), DType::Int64);
        g.desc(id).isInitializer = true;
        HostBuffer buffer;
        buffer.resizeElems((int64_t) values.size(), DType::Int64);
        std::memcpy(buffer.bytes.data(), values.data(), values.size() * sizeof(int64_t));
        g.initializers[id] = buffer;
        return id;
    }

    TensorId addFloatConstant(Graph &g, const std::string &name, Shape shape, const std::vector<float> &values) {
        TensorId id              = addTensor(g, name, std::move(shape), DType::Float32);
        g.desc(id).isInitializer = true;
        HostBuffer buffer;
        buffer.resizeElems((int64_t) values.size(), DType::Float32);
        std::memcpy(buffer.bytes.data(), values.data(), values.size() * sizeof(float));
        g.initializers[id] = buffer;
        return id;
    }

    Node &addNode(Graph &g, OpType type, const std::string &name, std::vector<TensorId> inputs, std::vector<TensorId> outputs, int subOp = 0) {
        Node node;
        node.type    = type;
        node.name    = name;
        node.inputs  = std::move(inputs);
        node.outputs = std::move(outputs);
        node.subOp   = subOp;
        g.nodes.push_back(std::move(node));
        return g.nodes.back();
    }

    TensorId addBinary(Graph &g, BinaryType op, const std::string &name, TensorId lhs, TensorId rhs, DType outputDtype = DType::Float32) {
        TensorId result = addTensor(g, name, {}, outputDtype);
        addNode(g, OpType::Binary, name, {lhs, rhs}, {result}, (int) op);
        return result;
    }

    TensorId addCast(Graph &g, const std::string &name, TensorId operand, int64_t onnxTarget) {
        TensorId result     = addTensor(g, name);
        Node    &cast       = addNode(g, OpType::Cast, name, {operand}, {result});
        cast.attr.map["to"] = intAttr(onnxTarget);
        return result;
    }

    IOTensor int64Feed(const std::string &name, const Shape &shape, const std::vector<int64_t> &values) {
        IOTensor feed;
        feed.name  = name;
        feed.shape = shape;
        feed.dtype = DType::Int64;
        feed.data.resize(values.size() * sizeof(int64_t));
        std::memcpy(feed.data.data(), values.data(), feed.data.size());
        return feed;
    }

    IOTensor floatFeed(const std::string &name, const Shape &shape, const std::vector<float> &values) {
        IOTensor feed;
        feed.name  = name;
        feed.shape = shape;
        feed.dtype = DType::Float32;
        feed.data.resize(values.size() * sizeof(float));
        std::memcpy(feed.data.data(), values.data(), feed.data.size());
        return feed;
    }

    // Run `g` on the CPU backend (the Session applies the default passes, pointwise fusion included) and
    // return its single Int64 output; empty when the session fails (reported through an expectation).
    std::vector<int64_t> runInt64OnCpu(Graph &&g, const std::vector<IOTensor> &feeds) {
        Config cfg;
        cfg.backend    = BackendKind::Cpu;
        cfg.cpuThreads = 1;
        auto session   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(session);
        if (!session)
        {
            return {};
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(session->run(feeds, outs), Status::Ok);
        if (outs.size() != 1 || outs[0].dtype != DType::Int64)
        {
            ADD_FAILURE() << "expected one Int64 output";
            return {};
        }
        std::vector<int64_t> values(outs[0].data.size() / sizeof(int64_t));
        std::memcpy(values.data(), outs[0].data.data(), values.size() * sizeof(int64_t));
        return values;
    }

    // The nodes the default passes leave hosting a fused pointwise unit (a FusedPointwise node, or any node
    // carrying pw_steps).
    int countFusedUnits(Graph g) {
        PassOptions options;
        runStandardPasses(g, options);
        int units = 0;
        for (const Node &node: g.nodes)
        {
            units += (node.type == OpType::FusedPointwise || node.attr.has("pw_steps")) ? 1 : 0;
        }
        return units;
    }

    // The pointwise chain a -> Add(a, one) -> Mul(sum, sum) -> Sub(square, a) -> Mul(difference, two) with
    // unlabeled intermediates, appended after `source`; writes the result to an Int64 graph output "y".
    void appendIntegerChain(Graph &g, TensorId source) {
        TensorId one        = addInt64Constant(g, "one", {1}, {1});
        TensorId two        = addInt64Constant(g, "two", {1}, {2});
        TensorId sum        = addTensor(g, "sum");
        TensorId square     = addBinary(g, BinaryType::Mul, "square", sum, sum);
        TensorId difference = addBinary(g, BinaryType::Sub, "difference", square, source);
        addNode(g, OpType::Add, "add_one", {source, one}, {sum});
        TensorId y = addBinary(g, BinaryType::Mul, "y", difference, two, DType::Int64);
        addOutput(g, y);
    }

    int64_t integerChainReference(int64_t value) {
        const int64_t sum = value + 1;
        return (sum * sum - value) * 2;
    }

} // namespace

TEST(IntegerElements, ResolverSeparatesIntegerValuesFromInt64Storage) {
    Graph    g;
    TensorId int32Input  = addInput(g, "int32_input", {4}, DType::Int32);
    TensorId int64Input  = addInput(g, "int64_input", {4}, DType::Int64);
    TensorId floatInput  = addInput(g, "float_input", {4}, DType::Float32);
    TensorId negated     = addTensor(g, "negated");
    TensorId boolCast    = addCast(g, "bool_cast", floatInput, kOnnxBool);
    TensorId int64Cast   = addCast(g, "int64_cast", floatInput, kOnnxInt64);
    TensorId floatCast   = addCast(g, "float_cast", int64Input, kOnnxFloat);
    TensorId mixedRange  = addTensor(g, "mixed_range");
    TensorId int64Range  = addTensor(g, "int64_range");
    TensorId floatPower  = addBinary(g, BinaryType::Pow, "float_power", floatInput, int64Input);
    TensorId int64Power  = addBinary(g, BinaryType::Pow, "int64_power", int64Input, floatInput);
    TensorId joined      = addTensor(g, "joined");
    TensorId anded       = addTensor(g, "anded");
    TensorId fusedResult = addTensor(g, "fused_result");
    addNode(g, OpType::Unary, "negate", {int32Input}, {negated}, (int) UnaryType::Neg);
    addNode(g, OpType::Range, "mixed_range", {int64Input, addFloatConstant(g, "limit", {}, {8.0f}), int64Input}, {mixedRange});
    addNode(g, OpType::Range, "int64_range", {int64Input, int64Input, int64Input}, {int64Range});
    addNode(g, OpType::Concat, "joined", {floatInput, int64Cast}, {joined}).attr.map["axis"] = intAttr(0);
    addNode(g, OpType::BitwiseAnd, "anded", {int32Input, int32Input}, {anded});
    addNode(g, OpType::Add, "fused", {int64Cast, floatInput}, {fusedResult}).attr.map["pw_steps"] = intAttr(1);

    const std::vector<char> integerValues = resolveElementFact(g, ElementFact::IntegerValues);
    const std::vector<char> int64Storage  = resolveElementFact(g, ElementFact::Int64Storage);
    struct Expectation {
        TensorId    tensor;
        bool        integer;
        bool        int64;
        const char *what;
    };
    const std::vector<Expectation> expectations {
        {int32Input, true, false, "an INT32 graph input binds as fp32 lanes"},
        {int64Input, true, true, "an INT64 graph input"},
        {floatInput, false, false, "a float graph input"},
        {negated, true, false, "Neg of INT32 lanes stays integer, stored fp32"},
        {boolCast, false, true, "a Cast to BOOL is stored int64 but holds no integer element type"},
        {int64Cast, true, true, "a Cast to INT64"},
        {floatCast, false, false, "a Cast to FLOAT"},
        {mixedRange, true, false, "Range stores int64 only when all three operands are"},
        {int64Range, true, true, "Range of int64 operands"},
        {floatPower, false, false, "a float base raised to an int64 exponent"},
        {int64Power, true, true, "an int64 base raised to a float exponent"},
        {joined, true, true, "a Concat with an int64 part"},
        {anded, true, false, "BitwiseAnd of INT32 lanes"},
        {fusedResult, false, false, "a producer hosting a fused pointwise chain"},
    };
    ElementFactResolver lazyStorage(g, ElementFact::Int64Storage);
    for (const Expectation &expectation: expectations)
    {
        EXPECT_EQ(integerValues[(size_t) expectation.tensor] != 0, expectation.integer) << expectation.what;
        EXPECT_EQ(int64Storage[(size_t) expectation.tensor] != 0, expectation.int64) << expectation.what;
        EXPECT_EQ(lazyStorage.holds(expectation.tensor), expectation.int64) << expectation.what << " (lazy resolver)";
    }
    EXPECT_FALSE(lazyStorage.holds(kNoTensor));
}

TEST(IntegerElements, LongProducerChainsResolveWithoutRecursion) {
    // The walk keeps its own stack: a chain far deeper than a call stack allows still resolves.
    constexpr int kChainLength = 100000;
    Graph         g;
    TensorId      current = addInput(g, "ids", {1}, DType::Int64);
    for (int step = 0; step < kChainLength; ++step)
    {
        TensorId next = addTensor(g, "hop" + std::to_string(step));
        addNode(g, OpType::Identity, "hop" + std::to_string(step), {current}, {next});
        current = next;
    }
    ElementFactResolver integerValues(g, ElementFact::IntegerValues);
    EXPECT_TRUE(integerValues.holds(current));
}

TEST(IntegerElements, FusionKeepsIntegerChainsUnfusedAndExactOnTheCpu) {
    // Every chain's intermediates are unlabeled, so only the producers say they hold integers. Fused, the
    // CPU kernel would read their int64 lanes as fp32 bits (zeros and garbage); unfused, each op keeps its
    // int64 or integer-valued storage.
    const Shape                shape {8};
    const std::vector<int64_t> values {1, 2, 3, 4, 5, 6, 7, 100};
    enum class Source { CastToInt64, Int64Input, IntegerMod, BitwiseAnd };
    for (Source source: {Source::CastToInt64, Source::Int64Input, Source::IntegerMod, Source::BitwiseAnd})
    {
        auto build = [&](std::vector<IOTensor> &feeds, std::function<int64_t(int64_t)> &sourceValue) {
            Graph    g;
            TensorId chainSource = kNoTensor;
            switch (source)
            {
                case Source::CastToInt64: {
                    TensorId x  = addInput(g, "x", shape, DType::Float32);
                    chainSource = addCast(g, "cast", x, kOnnxInt64);
                    feeds       = {floatFeed("x", shape, std::vector<float>(values.begin(), values.end()))};
                    sourceValue = [](int64_t value) {
                        return value;
                    };
                    break;
                }
                case Source::Int64Input:
                    chainSource = addInput(g, "x", shape, DType::Int64);
                    feeds       = {int64Feed("x", shape, values)};
                    sourceValue = [](int64_t value) {
                        return value;
                    };
                    break;
                case Source::IntegerMod: {
                    TensorId x  = addInput(g, "x", shape, DType::Int64);
                    chainSource = addTensor(g, "remainder");
                    addNode(g, OpType::Mod, "mod", {x, addInt64Constant(g, "five", {1}, {5})}, {chainSource});
                    feeds       = {int64Feed("x", shape, values)};
                    sourceValue = [](int64_t value) {
                        return value % 5;
                    };
                    break;
                }
                case Source::BitwiseAnd: {
                    TensorId x  = addInput(g, "x", shape, DType::Int64);
                    chainSource = addTensor(g, "masked");
                    addNode(g, OpType::BitwiseAnd, "mask", {x, addInt64Constant(g, "six", {1}, {6})}, {chainSource});
                    feeds       = {int64Feed("x", shape, values)};
                    sourceValue = [](int64_t value) {
                        return value & 6;
                    };
                    break;
                }
            }
            appendIntegerChain(g, chainSource);
            return g;
        };
        std::vector<IOTensor>           feeds;
        std::function<int64_t(int64_t)> sourceValue;
        Graph                           g = build(feeds, sourceValue);
        EXPECT_EQ(countFusedUnits(g), 0) << "source " << (int) source;
        std::vector<int64_t> expected;
        for (int64_t value: values)
        {
            expected.push_back(integerChainReference(sourceValue(value)));
        }
        EXPECT_EQ(runInt64OnCpu(std::move(g), feeds), expected) << "source " << (int) source;
    }
}

TEST(IntegerElements, FloatChainsStillFuse) {
    // The same chain over float values fuses as before.
    Graph    g;
    TensorId x          = addInput(g, "x", {8}, DType::Float32);
    TensorId one        = addFloatConstant(g, "one", {1}, {1.0f});
    TensorId sum        = addTensor(g, "sum");
    TensorId square     = addBinary(g, BinaryType::Mul, "square", sum, sum);
    TensorId difference = addBinary(g, BinaryType::Sub, "difference", square, x);
    addNode(g, OpType::Add, "add_one", {x, one}, {sum});
    addOutput(g, difference);
    EXPECT_GT(countFusedUnits(g), 0);
}

TEST(IntegerElements, IntegerChainThroughAReshapePinsFp32FromItsGraphInput) {
    // int64 x -> Reshape -> Add(r, r) -> Mul(a, a) -> BitwiseAnd(m, 0xFFFFF): no member fuses, so the pin
    // reaches x through the Reshape and the int64 graph input packs at fp32 (70000 is not saturated to
    // 65504 in front of the GPU kernels), and the CPU answer is exact.
    constexpr int64_t kLowBitsMask = 0xFFFFF;
    const Shape       shape {1, 6};
    auto              build = [&]() {
        Graph    g;
        TensorId x        = addInput(g, "x", shape, DType::Int64);
        TensorId reshaped = addTensor(g, "reshaped");
        addNode(g, OpType::Reshape, "reshape", {x, addInt64Constant(g, "target", {2}, {1, 6})}, {reshaped});
        TensorId doubled = addTensor(g, "doubled");
        addNode(g, OpType::Add, "double", {reshaped, reshaped}, {doubled});
        TensorId squared = addBinary(g, BinaryType::Mul, "square", doubled, doubled);
        TensorId masked  = addTensor(g, "masked", {}, DType::Int64);
        addNode(g, OpType::BitwiseAnd, "mask", {squared, addInt64Constant(g, "low_bits", {1}, {kLowBitsMask})}, {masked});
        addOutput(g, masked);
        return g;
    };
    {
        Graph       g = build();
        PassOptions options;
        runStandardPasses(g, options);
        g.topoSort();
        planFlatLayoutAndStorage(g, "", nullptr);
        for (const Node &node: g.nodes)
        {
            EXPECT_NE(node.type, OpType::FusedPointwise) << node.name;
            EXPECT_FALSE(node.attr.has("pw_steps")) << node.name;
        }
        const TensorId x = g.inputs.at(0);
        EXPECT_TRUE(g.desc(x).storeFp32) << "the int64 graph input packs at fp32";
        for (const Node &node: g.nodes)
        {
            EXPECT_NE(node.type, OpType::ConvertDtype) << "no bridge widens an already narrowed integer: " << node.name;
        }
    }
    const std::vector<int64_t> values {1, 2, 3, kBeyondHalfRange, -4, 5};
    std::vector<int64_t>       expected;
    for (int64_t value: values)
    {
        expected.push_back((4 * value * value) & kLowBitsMask);
    }
    EXPECT_EQ(runInt64OnCpu(build(), {int64Feed("x", shape, values)}), expected);
}

TEST(IntegerElements, GateKeepsDivAndPowOfComputedInt64OperandsOnTheCpuOp) {
    struct GateCase {
        std::string                                                                     name;
        std::function<void(Graph &, TensorId, TensorId, TensorId, const std::string &)> build; // (graph, x, y, f, name)
        bool                                                                            cpu;
        const char                                                                     *reason;
    };
    const char                 *divReason = "Binary: integer Div on an int64 operand";
    const char                 *powReason = "Binary: integer Pow on an int64 base";
    const std::vector<GateCase> cases {
        {"div_sum_by_difference",
         [](Graph &g, TensorId x, TensorId y, TensorId, const std::string &name) {
             TensorId sum = addTensor(g, name + "_sum");
             addNode(g, OpType::Add, name + "_sum", {x, y}, {sum});
             addBinary(g, BinaryType::Div, name, sum, addBinary(g, BinaryType::Sub, name + "_difference", x, y));
         },
         true, divReason},
        {"div_or_by_and",
         [](Graph &g, TensorId x, TensorId y, TensorId, const std::string &name) {
             TensorId ored  = addTensor(g, name + "_or");
             TensorId anded = addTensor(g, name + "_and");
             addNode(g, OpType::BitwiseOr, name + "_or", {x, y}, {ored});
             addNode(g, OpType::BitwiseAnd, name + "_and", {x, y}, {anded});
             addBinary(g, BinaryType::Div, name, ored, anded);
         },
         true, divReason},
        {"div_reshaped_by_float",
         [](Graph &g, TensorId x, TensorId, TensorId f, const std::string &name) {
             TensorId reshaped = addTensor(g, name + "_reshaped");
             addNode(g, OpType::Reshape, name + "_reshape", {x, addInt64Constant(g, name + "_target", {1}, {4})}, {reshaped});
             addBinary(g, BinaryType::Div, name, f, reshaped);
         },
         true, divReason},
        {"pow_of_cast_to_int64",
         [](Graph &g, TensorId, TensorId, TensorId f, const std::string &name) {
             addBinary(g, BinaryType::Pow, name, addCast(g, name + "_cast", f, kOnnxInt64), f);
         },
         true, powReason},
        {"div_of_cast_to_float",
         [](Graph &g, TensorId x, TensorId, TensorId f, const std::string &name) {
             addBinary(g, BinaryType::Div, name, addCast(g, name + "_cast", x, kOnnxFloat), f);
         },
         false, ""},
        {"pow_float_base_computed_int64_exponent",
         [](Graph &g, TensorId x, TensorId y, TensorId f, const std::string &name) {
             TensorId sum = addTensor(g, name + "_sum");
             addNode(g, OpType::Add, name + "_sum", {x, y}, {sum});
             addBinary(g, BinaryType::Pow, name, f, sum);
         },
         false, ""},
    };
    for (const GateCase &gateCase: cases)
    {
        Graph    g;
        TensorId x = addInput(g, "x", {4}, DType::Int64);
        TensorId y = addInput(g, "y", {4}, DType::Int64);
        TensorId f = addInput(g, "f", {4}, DType::Float32);
        gateCase.build(g, x, y, f, gateCase.name);
        const std::vector<NodeSupport> rows  = vkSupportSurvey(g);
        bool                           found = false;
        for (const NodeSupport &row: rows)
        {
            if (row.node != gateCase.name)
            {
                continue;
            }
            found = true;
            EXPECT_EQ(row.backend, gateCase.cpu ? "cpu" : "vulkan") << gateCase.name;
            EXPECT_EQ(row.reason, gateCase.reason) << gateCase.name;
        }
        EXPECT_TRUE(found) << gateCase.name;
    }
}

TEST(IntegerElements, IntegerGraphOutputsPinTheMovementChainFeedingThem) {
    // Only movement ops lie between each Int64 output and its integer sources, so no computing node seeds
    // the region; the output does, and the pin reaches the int64 inputs (and the Gather's table uploads at
    // its pinned node's fp32 precision).
    {
        Graph    g;
        TensorId table  = addInt64Constant(g, "table", {3}, {kBeyondHalfRange, 128000, 5});
        TensorId index  = addInput(g, "index", {4}, DType::Int64);
        TensorId tokens = addTensor(g, "tokens", {4}, DType::Int64);
        addNode(g, OpType::Gather, "remap", {table, index}, {tokens});
        addOutput(g, tokens);
        planFlatLayoutAndStorage(g, "", nullptr);
        EXPECT_TRUE(g.desc(tokens).storeFp32) << "Gather from an int64 table";
        EXPECT_TRUE(g.desc(index).storeFp32) << "Gather index";
    }
    {
        Graph    g;
        TensorId condition = addInput(g, "condition", {4}, DType::UInt8);
        TensorId first     = addInput(g, "first", {4}, DType::Int64);
        TensorId second    = addInput(g, "second", {4}, DType::Int64);
        TensorId selected  = addTensor(g, "selected", {4}, DType::Int64);
        addNode(g, OpType::Where, "select", {condition, first, second}, {selected});
        addOutput(g, selected);
        planFlatLayoutAndStorage(g, "", nullptr);
        EXPECT_TRUE(g.desc(selected).storeFp32) << "Where output";
        EXPECT_TRUE(g.desc(first).storeFp32) << "Where value";
        EXPECT_TRUE(g.desc(second).storeFp32) << "Where value";
    }
    {
        Graph    g;
        TensorId first                                                                 = addInput(g, "first", {4}, DType::Int64);
        TensorId second                                                                = addInput(g, "second", {4}, DType::Int64);
        TensorId joined                                                                = addTensor(g, "joined", {8}, DType::Int64);
        addNode(g, OpType::Concat, "join", {first, second}, {joined}).attr.map["axis"] = intAttr(0);
        addOutput(g, joined);
        planFlatLayoutAndStorage(g, "", nullptr);
        EXPECT_TRUE(g.desc(joined).storeFp32) << "Concat output";
        EXPECT_TRUE(g.desc(first).storeFp32) << "Concat part";
        EXPECT_TRUE(g.desc(second).storeFp32) << "Concat part";
    }
    {
        // A float output of the same movement ops stays at its storage precision.
        Graph    g;
        TensorId first                                                                 = addInput(g, "first", {4}, DType::Float32);
        TensorId second                                                                = addInput(g, "second", {4}, DType::Float32);
        TensorId joined                                                                = addTensor(g, "joined", {8}, DType::Float32);
        addNode(g, OpType::Concat, "join", {first, second}, {joined}).attr.map["axis"] = intAttr(0);
        addOutput(g, joined);
        planFlatLayoutAndStorage(g, "", nullptr);
        EXPECT_FALSE(g.desc(joined).storeFp32);
        EXPECT_FALSE(g.desc(first).storeFp32);
    }
}
