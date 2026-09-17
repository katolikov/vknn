// Variadic Sum / Mean / Max / Min (lowerVariadicElementwise) and the ONNX name map of the logical,
// bitwise, Mod, ArgMax/ArgMin and variadic ops. The variadic ops accept 1..N operands; the pass
// rewrites them into the 2-input Add/Binary nodes every kernel implements before shape inference.
// Values run end to end through a CPU Session (which applies runStandardPasses) and are checked
// against a NumPy-broadcast reference fold computed here in the same fp32 left-to-right order.
#include "import/passes.h"
#include "vknn/error.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    // One constant operand of a variadic node (fp32 initializer).
    struct Operand {
        Shape              shape;
        std::vector<float> data;
    };

    struct Result {
        Shape              shape;
        std::vector<float> data;
    };

    // NumPy-broadcast elementwise application of `fn` to two row-major operands.
    template <typename Value, typename Fn>
    std::vector<Value> broadcastApply(const Shape &lhsShape, const std::vector<Value> &lhs, const Shape &rhsShape, const std::vector<Value> &rhs, Shape &outShape, Fn fn) {
        const size_t rank = std::max(lhsShape.size(), rhsShape.size());
        auto         dim  = [&](const Shape &s, size_t axis) -> int64_t {
            const size_t offset = rank - s.size();
            return axis < offset ? 1 : s[axis - offset];
        };
        outShape.assign(rank, 1);
        for (size_t axis = 0; axis < rank; ++axis)
        {
            const int64_t a = dim(lhsShape, axis), b = dim(rhsShape, axis);
            outShape[axis] = (a == 0 || b == 0) ? 0 : std::max(a, b);
        }
        const int64_t      count = numElements(outShape);
        std::vector<Value> out((size_t) std::max<int64_t>(count, 0));
        for (int64_t flat = 0; flat < count; ++flat)
        {
            int64_t remainder = flat, lhsIndex = 0, rhsIndex = 0, lhsStride = 1, rhsStride = 1;
            for (size_t step = 0; step < rank; ++step)
            {
                const size_t  axis  = rank - 1 - step;
                const int64_t coord = remainder % outShape[axis];
                remainder /= outShape[axis];
                const int64_t a = dim(lhsShape, axis), b = dim(rhsShape, axis);
                lhsIndex += (a == 1 ? 0 : coord) * lhsStride;
                rhsIndex += (b == 1 ? 0 : coord) * rhsStride;
                lhsStride *= a;
                rhsStride *= b;
            }
            out[(size_t) flat] = fn(lhs[(size_t) lhsIndex], rhs[(size_t) rhsIndex]);
        }
        return out;
    }

    // Reference: left fold of `fn` over x followed by the operands, in the lowered pass's order.
    template <typename Fn> Result referenceFold(const Shape &xShape, const std::vector<float> &x, const std::vector<Operand> &operands, Fn fn) {
        Result acc {xShape, x};
        for (const Operand &operand: operands)
        {
            Shape next;
            acc.data  = broadcastApply(acc.shape, acc.data, operand.shape, operand.data, next, fn);
            acc.shape = next;
        }
        return acc;
    }

    // Build a graph with one fp32 graph input "x" and N fp32 initializer operands feeding one node of
    // `type`/`subOp` (inputs {x, operands...}), run it on a CPU Session, return output shape + values.
    Result runVariadic(OpType type, int subOp, const Shape &xShape, const std::vector<float> &x, const std::vector<Operand> &operands) {
        Graph      g;
        TensorDesc xd;
        xd.name     = "x";
        xd.shape    = xShape;
        xd.isInput  = true;
        TensorId xi = g.addTensor(xd);
        g.inputs    = {xi};
        Node n;
        n.type   = type;
        n.subOp  = subOp;
        n.name   = "variadic";
        n.inputs = {xi};
        for (size_t k = 0; k < operands.size(); ++k)
        {
            TensorDesc d;
            d.name          = "operand" + std::to_string(k);
            d.shape         = operands[k].shape;
            d.isInitializer = true;
            TensorId   id   = g.addTensor(d);
            HostBuffer hb;
            hb.resizeElems((int64_t) operands[k].data.size(), DType::Float32);
            std::memcpy(hb.bytes.data(), operands[k].data.data(), operands[k].data.size() * sizeof(float));
            g.initializers[id] = hb;
            n.inputs.push_back(id);
        }
        TensorDesc yd;
        yd.name     = "y";
        yd.isOutput = true;
        TensorId y  = g.addTensor(yd);
        n.outputs   = {y};
        g.nodes     = {n};
        g.outputs   = {y};

        Config cfg;
        cfg.backend  = BackendKind::Cpu;
        auto session = Session::create(std::move(g), cfg);
        EXPECT_TRUE(session);
        if (!session)
        {
            return {};
        }
        IOTensor in;
        in.name  = "x";
        in.shape = xShape;
        in.dtype = DType::Float32;
        in.data.resize(x.size() * sizeof(float));
        std::memcpy(in.data.data(), x.data(), in.data.size());
        std::vector<IOTensor> outs;
        EXPECT_EQ(session->run({in}, outs), Status::Ok);
        if (outs.empty())
        {
            return {};
        }
        const float *values = outs[0].f32();
        return {outs[0].shape, std::vector<float>(values, values + numElements(outs[0].shape))};
    }

    // Bit-level equality: -0.0 vs +0.0 and NaN payloads count, so a reordered accumulation is caught.
    void expectBitIdentical(const std::vector<float> &got, const std::vector<float> &want) {
        ASSERT_EQ(got.size(), want.size());
        for (size_t k = 0; k < want.size(); ++k)
        {
            uint32_t gotBits = 0, wantBits = 0;
            std::memcpy(&gotBits, &got[k], sizeof(float));
            std::memcpy(&wantBits, &want[k], sizeof(float));
            EXPECT_EQ(gotBits, wantBits) << "k=" << k << " got " << got[k] << " want " << want[k];
        }
    }

    float maxOf(float a, float b) {
        return std::max(a, b);
    }
    float minOf(float a, float b) {
        return std::min(a, b);
    }
    float sumOf(float a, float b) {
        return a + b;
    }

    // Values with ties, negatives and fractional parts that do not sum exactly in fp32, so an
    // accumulation-order or scale-order change is visible bit for bit.
    const std::vector<float> kInputX = {0.1f, -2.5f, 3.3f, 7.0f, -0.7f, 1.0f / 3.0f};
    const Shape              kShapeX = {2, 3};

    // Eight constant operands (nine inputs with x): row and column broadcasts, a [1]-shaped scalar,
    // and a LATER operand ([4,1,1]) that broadcasts the running result to a larger rank.
    std::vector<Operand> eightOperands() {
        return {
            {{3}, {0.2f, -3.0f, 3.3f}},
            {{2, 1}, {1.7f, -0.9f}},
            {{2, 3}, {0.05f, 9.5f, -4.4f, 2.2f, 0.0f, 0.3f}},
            {{1}, {0.6f}},
            {{4, 1, 1}, {-1.0f, 2.5f, 0.1f, 8.8f}},
            {{3}, {1.1f, 1.2f, -1.3f}},
            {{2, 1}, {0.7f, 0.33f}},
            {{1, 3}, {-0.26f, 5.7f, 0.127f}},
        };
    }

} // namespace

// --- ONNX name map ----------------------------------------------------------------------------------

TEST(OnnxOpMap, LogicalBitwiseModArgExtremeVariadicNamesRoundTrip) {
    // Each logical, bitwise, Mod, ArgMax/ArgMin and Mean op maps to its own OpType and spells back to
    // its ONNX name.
    const std::pair<const char *, OpType> want[] = {
        {"Or", OpType::Or},
        {"Xor", OpType::Xor},
        {"Not", OpType::Not},
        {"ArgMax", OpType::ArgMax},
        {"ArgMin", OpType::ArgMin},
        {"Mod", OpType::Mod},
        {"BitShift", OpType::BitShift},
        {"BitwiseAnd", OpType::BitwiseAnd},
        {"BitwiseOr", OpType::BitwiseOr},
        {"BitwiseXor", OpType::BitwiseXor},
        {"BitwiseNot", OpType::BitwiseNot},
        {"Mean", OpType::Mean},
    };
    for (const auto &[name, type]: want)
    {
        EXPECT_EQ(opTypeFromOnnx(name), type) << name;
        EXPECT_STREQ(opTypeName(type), name);
    }
    // Sum is Add with any operand count; Max/Min stay in the Binary family.
    EXPECT_EQ(opTypeFromOnnx("Sum"), OpType::Add);
    EXPECT_EQ(opTypeFromOnnx("Max"), OpType::Binary);
    EXPECT_EQ(binaryFromOnnx("Max"), BinaryType::Max);
    EXPECT_EQ(opTypeFromOnnx("Min"), OpType::Binary);
    EXPECT_EQ(binaryFromOnnx("Min"), BinaryType::Min);
}

// --- values through a CPU Session ---------------------------------------------------------------------

TEST(VariadicElementwise, MaxMinOneThreeAndNineOperands) {
    for (const auto &[subOp, fn]: {std::pair<BinaryType, float (*)(float, float)> {BinaryType::Max, maxOf}, {BinaryType::Min, minOf}})
    {
        const char *label = subOp == BinaryType::Max ? "Max" : "Min";
        // 1 operand: the value itself.
        Result one = runVariadic(OpType::Binary, (int) subOp, kShapeX, kInputX, {});
        EXPECT_EQ(one.shape, kShapeX) << label;
        expectBitIdentical(one.data, kInputX);

        // 3 operands, the third broadcasting the result to a larger shape.
        const std::vector<Operand> three = {{{3}, {0.2f, -3.0f, 3.3f}}, {{2, 2, 1}, {1.7f, -0.9f, 0.0f, 4.0f}}};
        Result                     got3  = runVariadic(OpType::Binary, (int) subOp, kShapeX, kInputX, three);
        Result                     want3 = referenceFold(kShapeX, kInputX, three, fn);
        EXPECT_EQ(want3.shape, (Shape {2, 2, 3}));
        EXPECT_EQ(got3.shape, want3.shape) << label;
        expectBitIdentical(got3.data, want3.data);

        // 9 operands (x plus eight constants), a later one broadcasting the result to rank 3.
        const std::vector<Operand> eight = eightOperands();
        Result                     got9  = runVariadic(OpType::Binary, (int) subOp, kShapeX, kInputX, eight);
        Result                     want9 = referenceFold(kShapeX, kInputX, eight, fn);
        EXPECT_EQ(want9.shape, (Shape {4, 2, 3}));
        EXPECT_EQ(got9.shape, want9.shape) << label;
        expectBitIdentical(got9.data, want9.data);
    }
}

TEST(VariadicElementwise, SumOneTwoAndThreeOperands) {
    Result one = runVariadic(OpType::Add, 0, kShapeX, kInputX, {});
    EXPECT_EQ(one.shape, kShapeX);
    expectBitIdentical(one.data, kInputX);

    const std::vector<Operand> single = {{{3}, {0.2f, -3.0f, 3.3f}}};
    Result                     got2   = runVariadic(OpType::Add, 0, kShapeX, kInputX, single);
    Result                     want2  = referenceFold(kShapeX, kInputX, single, sumOf);
    EXPECT_EQ(got2.shape, want2.shape);
    expectBitIdentical(got2.data, want2.data);

    const std::vector<Operand> pair  = {{{3}, {0.2f, -3.0f, 3.3f}}, {{4, 1, 1}, {-1.0f, 2.5f, 0.1f, 8.8f}}};
    Result                     got3  = runVariadic(OpType::Add, 0, kShapeX, kInputX, pair);
    Result                     want3 = referenceFold(kShapeX, kInputX, pair, sumOf);
    EXPECT_EQ(want3.shape, (Shape {4, 2, 3}));
    EXPECT_EQ(got3.shape, want3.shape);
    expectBitIdentical(got3.data, want3.data);
}

TEST(VariadicElementwise, MeanIsSumTimesFp32ReciprocalBitExact) {
    // ONNX Runtime's CPU Mean sums its operands left to right and scales the sum by the fp32
    // reciprocal 1.0f / N; the lowering reproduces that bit for bit. A divide by N rounds differently
    // for an N that is not a power of two, and the data here is chosen so the two visibly disagree.
    Result one = runVariadic(OpType::Mean, 0, kShapeX, kInputX, {});
    EXPECT_EQ(one.shape, kShapeX);
    expectBitIdentical(one.data, kInputX);

    auto scaleByReciprocal = [](std::vector<float> values, float count) {
        const float reciprocal = 1.0f / count;
        for (float &v: values)
        {
            v = v * reciprocal;
        }
        return values;
    };
    auto divideBy = [](std::vector<float> values, float count) {
        for (float &v: values)
        {
            v = v / count;
        }
        return values;
    };
    auto anyBitDifference = [](const std::vector<float> &lhs, const std::vector<float> &rhs) {
        return lhs.size() != rhs.size() || std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) != 0;
    };

    // Three operands, the third broadcasting the result to a larger rank.
    const std::vector<Operand> pair      = {{{3}, {0.2f, -3.0f, 3.3f}}, {{4, 1, 1}, {-1.0f, 2.5f, 0.1f, 8.8f}}};
    static constexpr float     kThreeOps = 3.0f;
    Result                     got3      = runVariadic(OpType::Mean, 0, kShapeX, kInputX, pair);
    Result                     sum3      = referenceFold(kShapeX, kInputX, pair, sumOf);
    const std::vector<float>   want3     = scaleByReciprocal(sum3.data, kThreeOps);
    EXPECT_EQ(got3.shape, (Shape {4, 2, 3}));
    expectBitIdentical(got3.data, want3);
    EXPECT_TRUE(anyBitDifference(want3, divideBy(sum3.data, kThreeOps))) << "the data must separate reciprocal-scale from divide";

    // Nine operands: still one scale, after the whole sum.
    const std::vector<Operand> eight    = eightOperands();
    static constexpr float     kNineOps = 9.0f;
    Result                     got9     = runVariadic(OpType::Mean, 0, kShapeX, kInputX, eight);
    Result                     sum9     = referenceFold(kShapeX, kInputX, eight, sumOf);
    const std::vector<float>   want9    = scaleByReciprocal(sum9.data, kNineOps);
    EXPECT_EQ(got9.shape, (Shape {4, 2, 3}));
    EXPECT_EQ(got9.shape, sum9.shape);
    expectBitIdentical(got9.data, want9);
    EXPECT_TRUE(anyBitDifference(want9, divideBy(sum9.data, kNineOps))) << "the data must separate reciprocal-scale from divide";
}

TEST(VariadicElementwise, Int64MaxOfThreeRuntimeAndConstantOperands) {
    // An int64 graph input and two int64 initializers: the lowered 2-input chain stays on the exact
    // int64 path, so values past fp32's 24-bit mantissa keep their order.
    const int64_t              big = int64_t(1) << 40;
    const std::vector<int64_t> ids = {big + 1, -7, 0};
    Graph                      g;
    TensorDesc                 xd;
    xd.name     = "ids";
    xd.shape    = {3};
    xd.dtype    = DType::Int64;
    xd.isInput  = true;
    TensorId x  = g.addTensor(xd);
    g.inputs    = {x};
    auto addI64 = [&](const char *name, Shape shape, const std::vector<int64_t> &values) {
        TensorDesc d;
        d.name          = name;
        d.shape         = std::move(shape);
        d.dtype         = DType::Int64;
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) values.size(), DType::Int64);
        std::memcpy(hb.bytes.data(), values.data(), values.size() * sizeof(int64_t));
        g.initializers[id] = hb;
        return id;
    };
    TensorId   a = addI64("a", {3}, {big, -3, big + 2});
    TensorId   b = addI64("b", {1}, {-5});
    TensorDesc yd;
    yd.name     = "y";
    yd.dtype    = DType::Int64;
    yd.isOutput = true;
    TensorId y  = g.addTensor(yd);
    Node     n;
    n.type    = OpType::Binary;
    n.subOp   = (int) BinaryType::Max;
    n.name    = "max3";
    n.inputs  = {x, a, b};
    n.outputs = {y};
    g.nodes   = {n};
    g.outputs = {y};

    Config cfg;
    cfg.backend  = BackendKind::Cpu;
    auto session = Session::create(std::move(g), cfg);
    ASSERT_TRUE(session);
    IOTensor in;
    in.name  = "ids";
    in.shape = {3};
    in.dtype = DType::Int64;
    in.data.resize(ids.size() * sizeof(int64_t));
    std::memcpy(in.data.data(), ids.data(), in.data.size());
    std::vector<IOTensor> outs;
    ASSERT_EQ(session->run({in}, outs), Status::Ok);
    ASSERT_EQ(outs.size(), 1u);
    ASSERT_EQ(outs[0].dtype, DType::Int64);
    ASSERT_EQ(outs[0].data.size(), 3 * sizeof(int64_t));
    std::vector<int64_t> got(3);
    std::memcpy(got.data(), outs[0].data.data(), got.size() * sizeof(int64_t));
    EXPECT_EQ(got, (std::vector<int64_t> {big + 1, -3, big + 2}));
}

TEST(VariadicElementwise, Int64MaxOfThreeInitializersFolds) {
    // All-constant int64 operands: the lowered chain folds to an exact int64 initializer.
    const int64_t big = int64_t(1) << 40;
    Graph         g;
    auto          addI64 = [&](const char *name, Shape shape, const std::vector<int64_t> &values) {
        TensorDesc d;
        d.name          = name;
        d.shape         = std::move(shape);
        d.dtype         = DType::Int64;
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) values.size(), DType::Int64);
        std::memcpy(hb.bytes.data(), values.data(), values.size() * sizeof(int64_t));
        g.initializers[id] = hb;
        return id;
    };
    TensorId a = addI64("a", {3}, {big + 1, -7, 0});
    TensorId b = addI64("b", {3}, {big, -3, big + 2});
    TensorId c = addI64("c", {1}, {-5});
    TensorId y = g.addTensor({"y"});
    Node     n;
    n.type    = OpType::Binary;
    n.subOp   = (int) BinaryType::Max;
    n.name    = "max3";
    n.inputs  = {a, b, c};
    n.outputs = {y};
    g.nodes   = {n};
    g.outputs = {y};

    lowerVariadicElementwise(g);
    ASSERT_EQ(g.nodes.size(), 2u);
    inferShapes(g, 1);
    EXPECT_EQ(g.desc(y).shape, (Shape {3}));
    constFold(g);
    EXPECT_TRUE(g.nodes.empty()) << "the all-constant chain must fold away";
    ASSERT_TRUE(g.isInitializer(y));
    ASSERT_EQ(g.desc(y).dtype, DType::Int64);
    ASSERT_EQ(g.initializers[y].bytes.size(), 3 * sizeof(int64_t));
    EXPECT_EQ(g.initializers[y].i64()[0], big + 1);
    EXPECT_EQ(g.initializers[y].i64()[1], -3);
    EXPECT_EQ(g.initializers[y].i64()[2], big + 2);
}

// --- structure ------------------------------------------------------------------------------------------

TEST(VariadicElementwise, ZeroOperandsIsInvalidArgument) {
    for (OpType type: {OpType::Add, OpType::Binary, OpType::Mean})
    {
        Graph    g;
        TensorId y = g.addTensor({"y"});
        Node     n;
        n.type    = type;
        n.subOp   = type == OpType::Binary ? (int) BinaryType::Min : 0;
        n.name    = "empty";
        n.outputs = {y};
        g.nodes   = {n};
        g.outputs = {y};
        try
        {
            lowerVariadicElementwise(g);
            ADD_FAILURE() << opTypeName(type) << ": a zero-operand node must throw";
        } catch (const Error &e)
        {
            EXPECT_EQ(e.status(), Status::InvalidArgument) << e.what();
            EXPECT_NE(std::string(e.what()).find("'empty'"), std::string::npos) << "the error names the node: " << e.what();
        }
    }
}

TEST(VariadicElementwise, TwoInputBinaryAndOtherOpsAreUntouched) {
    // Only variadic forms are rewritten: a 2-input Add/Max/Sub, and a 1-input Sub (not a variadic op),
    // keep their nodes exactly.
    Graph    g;
    TensorId a = g.addTensor({"a"});
    TensorId b = g.addTensor({"b"});
    TensorId s = g.addTensor({"s"});
    TensorId m = g.addTensor({"m"});
    TensorId d = g.addTensor({"d"});
    Node     add;
    add.type     = OpType::Add;
    add.name     = "add2";
    add.inputs   = {a, b};
    add.outputs  = {s};
    Node max2    = add;
    max2.type    = OpType::Binary;
    max2.subOp   = (int) BinaryType::Max;
    max2.name    = "max2";
    max2.outputs = {m};
    Node sub1    = add;
    sub1.type    = OpType::Binary;
    sub1.subOp   = (int) BinaryType::Sub;
    sub1.name    = "sub1";
    sub1.inputs  = {a};
    sub1.outputs = {d};
    g.nodes      = {add, max2, sub1};
    lowerVariadicElementwise(g);
    ASSERT_EQ(g.nodes.size(), 3u);
    EXPECT_EQ(g.nodes[0].name, "add2");
    EXPECT_EQ(g.nodes[1].name, "max2");
    EXPECT_EQ(g.nodes[2].name, "sub1");
    EXPECT_EQ(g.nodes[2].inputs.size(), 1u);
}

TEST(VariadicElementwise, LoweredNodesAreTwoInputAfterStandardPasses) {
    // Runtime operands (graph inputs) keep the chains unfolded; with and without pointwise fusion, no
    // Mean survives and every Add/Binary reads exactly two operands.
    for (bool fuse: {false, true})
    {
        Graph                 g;
        std::vector<TensorId> xs;
        for (int k = 0; k < 5; ++k)
        {
            TensorDesc d;
            d.name    = "x" + std::to_string(k);
            d.shape   = {2, 3};
            d.isInput = true;
            xs.push_back(g.addTensor(d));
            g.inputs.push_back(xs.back());
        }
        auto addVariadic = [&](OpType type, int subOp, const char *name, std::vector<TensorId> inputs) {
            TensorDesc od;
            od.name     = std::string(name) + "_out";
            od.isOutput = true;
            TensorId o  = g.addTensor(od);
            Node     n;
            n.type    = type;
            n.subOp   = subOp;
            n.name    = name;
            n.inputs  = std::move(inputs);
            n.outputs = {o};
            g.nodes.push_back(n);
            g.outputs.push_back(o);
        };
        addVariadic(OpType::Add, 0, "sum5", xs);
        addVariadic(OpType::Mean, 0, "mean3", {xs[0], xs[1], xs[2]});
        addVariadic(OpType::Mean, 0, "mean2", {xs[3], xs[4]});
        addVariadic(OpType::Binary, (int) BinaryType::Max, "max4", {xs[1], xs[2], xs[3], xs[4]});
        addVariadic(OpType::Binary, (int) BinaryType::Min, "min3", {xs[4], xs[0], xs[2]});

        PassOptions opt;
        opt.fusePointwiseChains = fuse;
        runStandardPasses(g, opt);
        for (const Node &n: g.nodes)
        {
            EXPECT_NE(n.type, OpType::Mean) << n.name;
            if (n.type == OpType::Add || n.type == OpType::Binary)
            {
                EXPECT_EQ(n.inputs.size(), 2u) << n.name << " (fuse=" << fuse << ")";
            }
        }
        for (TensorId o: g.outputs)
        {
            EXPECT_EQ(g.desc(o).shape, (Shape {2, 3})) << g.desc(o).name << " (fuse=" << fuse << ")";
        }
    }
}

TEST(VariadicElementwise, MeanLowersToAddChainAndOneRankZeroScale) {
    static constexpr int  kOperandCount = 5;
    Graph                 g;
    std::vector<TensorId> xs;
    for (int k = 0; k < kOperandCount; ++k)
    {
        TensorDesc d;
        d.name    = "x" + std::to_string(k);
        d.shape   = {1, 2, 2, 2};
        d.isInput = true;
        xs.push_back(g.addTensor(d));
        g.inputs.push_back(xs.back());
    }
    TensorDesc od;
    od.name      = "avg";
    od.isOutput  = true;
    TensorId out = g.addTensor(od);
    Node     mean;
    mean.type    = OpType::Mean;
    mean.name    = "avg_node";
    mean.inputs  = xs;
    mean.outputs = {out};
    g.nodes      = {mean};
    g.outputs    = {out};

    lowerVariadicElementwise(g);
    ASSERT_EQ(g.nodes.size(), (size_t) kOperandCount) << "four Adds and one Mul";
    int adds = 0, scales = 0;
    for (const Node &n: g.nodes)
    {
        ASSERT_EQ(n.inputs.size(), 2u) << n.name;
        if (n.type == OpType::Add)
        {
            ++adds;
            EXPECT_NE(n.outputs[0], out) << "the sum never writes the Mean output directly";
        } else
        {
            ASSERT_EQ(n.type, OpType::Binary) << n.name;
            ASSERT_EQ(n.subOp, (int) BinaryType::Mul) << n.name;
            ++scales;
            EXPECT_EQ(n.outputs[0], out) << "the scale writes the original output";
            TensorId reciprocal = n.inputs[1];
            ASSERT_TRUE(g.isInitializer(reciprocal));
            EXPECT_TRUE(g.desc(reciprocal).shape.empty()) << "a rank-0 scale keeps the output rank";
            EXPECT_EQ(g.desc(reciprocal).dtype, DType::Float32);
            ASSERT_EQ(g.initializers[reciprocal].bytes.size(), sizeof(float));
            EXPECT_EQ(g.initializers[reciprocal].f32()[0], 1.0f / (float) kOperandCount);
        }
    }
    EXPECT_EQ(adds, kOperandCount - 1);
    EXPECT_EQ(scales, 1);
    EXPECT_EQ(g.nodes.back().type, OpType::Binary) << "topologically the scale runs last";
    inferShapes(g, 1);
    EXPECT_EQ(g.desc(out).shape, (Shape {1, 2, 2, 2}));
}

TEST(VariadicElementwise, OneOperandOutputsKeepTheirNamesAndBuffers) {
    // A one-operand Sum/Max/Min/Mean becomes an Identity. Every declared output keeps its own name and
    // readback buffer, whether the operand is a graph input (y1..y3 all read x) or an internal tensor
    // (z1, z2 both read relu), across consecutive runs.
    static constexpr int kRuns = 2;
    Graph                g;
    TensorDesc           xd;
    xd.name    = "x";
    xd.shape   = kShapeX;
    xd.isInput = true;
    TensorId x = g.addTensor(xd);
    g.inputs   = {x};
    TensorDesc rd;
    rd.name            = "relu";
    TensorId relu      = g.addTensor(rd);
    auto     addOutput = [&](const char *name) {
        TensorDesc od;
        od.name     = name;
        od.isOutput = true;
        TensorId o  = g.addTensor(od);
        g.outputs.push_back(o);
        return o;
    };
    auto addNode = [&](OpType type, int subOp, const char *name, TensorId input, TensorId output) {
        Node n;
        n.type    = type;
        n.subOp   = subOp;
        n.name    = name;
        n.inputs  = {input};
        n.outputs = {output};
        g.nodes.push_back(n);
    };
    const char *names[] = {"y1", "y2", "y3", "z1", "z2"};
    TensorId    y1 = addOutput(names[0]), y2 = addOutput(names[1]), y3 = addOutput(names[2]);
    TensorId    z1 = addOutput(names[3]), z2 = addOutput(names[4]);
    addNode(OpType::Relu, 0, "rectify", x, relu);
    addNode(OpType::Add, 0, "sum_x", x, y1);
    addNode(OpType::Binary, (int) BinaryType::Max, "max_x", x, y2);
    addNode(OpType::Mean, 0, "mean_x", x, y3);
    addNode(OpType::Add, 0, "sum_relu", relu, z1);
    addNode(OpType::Binary, (int) BinaryType::Min, "min_relu", relu, z2);

    Config cfg;
    cfg.backend  = BackendKind::Cpu;
    auto session = Session::create(std::move(g), cfg);
    ASSERT_TRUE(session);
    for (int run = 0; run < kRuns; ++run)
    {
        std::vector<float> values = kInputX;
        for (float &v: values)
        {
            v = v * (float) (run + 1) - (float) run;
        }
        std::vector<float> rectified = values;
        for (float &v: rectified)
        {
            v = std::max(v, 0.0f);
        }
        IOTensor in;
        in.name  = "x";
        in.shape = kShapeX;
        in.dtype = DType::Float32;
        in.data.resize(values.size() * sizeof(float));
        std::memcpy(in.data.data(), values.data(), in.data.size());
        std::vector<IOTensor> outs;
        ASSERT_EQ(session->run({in}, outs), Status::Ok) << "run " << run;
        ASSERT_EQ(outs.size(), std::size(names));
        for (size_t k = 0; k < outs.size(); ++k)
        {
            EXPECT_EQ(outs[k].name, names[k]) << "run " << run;
            EXPECT_EQ(outs[k].shape, kShapeX) << names[k] << " run " << run;
            ASSERT_EQ(outs[k].data.size(), values.size() * sizeof(float)) << names[k] << " run " << run;
            const float *got = outs[k].f32();
            expectBitIdentical(std::vector<float>(got, got + values.size()), k < 3 ? values : rectified);
        }
    }
}

TEST(VariadicElementwise, IdentityEliminationKeepsDeclaredOutputTensors) {
    // Relu -> Identity -> y: the Relu writes the declared output directly. x -> Identity -> w (a graph
    // input source) and y -> Identity -> v (another graph output) keep their Identity copies. Relu ->
    // Identity -> hidden -> Tanh: the internal result is rewired to the Relu output and removed.
    Graph      g;
    TensorDesc xd;
    xd.name    = "x";
    xd.shape   = {4};
    xd.isInput = true;
    TensorId x = g.addTensor(xd);
    g.inputs   = {x};
    auto add   = [&](const char *name, bool output) {
        TensorDesc d;
        d.name     = name;
        d.shape    = {4};
        d.isOutput = output;
        TensorId t = g.addTensor(d);
        if (output)
        {
            g.outputs.push_back(t);
        }
        return t;
    };
    TensorId activated = add("activated", false);
    TensorId y         = add("y", true);
    TensorId w         = add("w", true);
    TensorId v         = add("v", true);
    TensorId hidden    = add("hidden", false);
    TensorId squashed  = add("squashed", true);
    auto     addNode   = [&](OpType type, const char *name, TensorId input, TensorId output) {
        Node n;
        n.type    = type;
        n.name    = name;
        n.inputs  = {input};
        n.outputs = {output};
        g.nodes.push_back(n);
    };
    addNode(OpType::Relu, "relu", x, activated);
    addNode(OpType::Identity, "to_y", activated, y);
    addNode(OpType::Identity, "to_w", x, w);
    addNode(OpType::Identity, "to_v", y, v);
    addNode(OpType::Identity, "to_hidden", activated, hidden);
    addNode(OpType::Relu, "squash", hidden, squashed);

    eliminateIdentity(g);
    EXPECT_EQ(g.outputs, (std::vector<TensorId> {y, w, v, squashed}));
    const Node *relu = nullptr, *toW = nullptr, *toV = nullptr, *squash = nullptr;
    int         identities = 0;
    for (const Node &n: g.nodes)
    {
        identities += n.type == OpType::Identity ? 1 : 0;
        relu   = n.name == "relu" ? &n : relu;
        toW    = n.name == "to_w" ? &n : toW;
        toV    = n.name == "to_v" ? &n : toV;
        squash = n.name == "squash" ? &n : squash;
    }
    EXPECT_EQ(identities, 2);
    ASSERT_NE(relu, nullptr);
    ASSERT_NE(toW, nullptr);
    ASSERT_NE(toV, nullptr);
    ASSERT_NE(squash, nullptr);
    EXPECT_EQ(relu->outputs[0], y) << "the producer writes the declared output";
    EXPECT_EQ(toW->inputs[0], x);
    EXPECT_EQ(toV->inputs[0], y);
    EXPECT_EQ(squash->inputs[0], y) << "every other reader of the producer's result follows it";
}

TEST(VariadicElementwise, UnloweredVxmIsRejectedAtLoad) {
    // A .vxm whose graph still carries a 3-operand Max (compiled before the lowering existed) skips the
    // import passes at load; the Binary kernel would read two operands and drop the third, so the load
    // fails by name instead.
    Graph      g;
    TensorDesc xd;
    xd.name             = "x";
    xd.shape            = {3};
    xd.isInput          = true;
    TensorId x          = g.addTensor(xd);
    g.inputs            = {x};
    auto addInitializer = [&](const char *name, const std::vector<float> &values) {
        TensorDesc d;
        d.name          = name;
        d.shape         = {3};
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) values.size(), DType::Float32);
        std::memcpy(hb.bytes.data(), values.data(), values.size() * sizeof(float));
        g.initializers[id] = hb;
        return id;
    };
    TensorId   a = addInitializer("a", {1, 2, 3});
    TensorId   b = addInitializer("b", {9, 9, 9});
    TensorDesc yd;
    yd.name     = "y";
    yd.shape    = {3};
    yd.isOutput = true;
    TensorId y  = g.addTensor(yd);
    Node     n;
    n.type    = OpType::Binary;
    n.subOp   = (int) BinaryType::Max;
    n.name    = "max3";
    n.inputs  = {x, a, b};
    n.outputs = {y};
    g.nodes   = {n};
    g.outputs = {y};

    EXPECT_THROW(requireLoweredVariadicElementwise(g), Error);
    const std::string path = testing::TempDir() + "variadic_unlowered.vxm";
    ASSERT_TRUE(saveGraphBin(g, path));
    Config cfg;
    cfg.backend = BackendKind::Cpu;
    try
    {
        auto session = Session::createFromVxm(path, cfg);
        ADD_FAILURE() << "an unlowered 3-operand Max must not load";
    } catch (const Error &e)
    {
        EXPECT_EQ(e.status(), Status::InvalidArgument);
        EXPECT_NE(std::string(e.what()).find("max3"), std::string::npos) << e.what();
        EXPECT_NE(std::string(e.what()).find("recompile the .vxm"), std::string::npos) << e.what();
    }
    std::remove(path.c_str());

    lowerVariadicElementwise(g);
    EXPECT_NO_THROW(requireLoweredVariadicElementwise(g)) << "the lowered graph loads";
}
