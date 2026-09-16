// Integer Binary arithmetic, dtype-preserving ConvertDtype, Cast to BOOL, and the flat initializer
// upload count.
//
// - Binary and Add on int64 operands (backend/cpu/int64_arithmetic.h): Pow is an exact integer power
//   whose products wrap modulo 2^64, with the negative-exponent rules (1 -> 1, -1 -> +-1 by parity,
//   every other base -> 0); INT64_MIN / -1 wraps to INT64_MIN; Mul/Sub/Add wrap; an fp32 operand
//   truncates toward zero with NaN reading 0 and out-of-range values saturating. A float base raised to
//   an int64 exponent stays a float power.
// - ConvertDtype on the CPU copies an int64 tensor as int64 (values past 2^53 survive).
// - Cast to BOOL is a truth test: nonzero (negative, fractional, infinite, NaN) -> 1, +0/-0 -> 0, on
//   the CPU oracle and in shaders/cast.comp's kCastModeBool. GLSL does not run on the host, so the
//   shader's truth test is transcribed below and swept against the CPU oracle over every fp16 bit
//   pattern and a stratified set of fp32 bit patterns.
// - uploadInit's element count and payload guard (upload_init_rule.h): a rank-0 Int64/Int8/UInt8
//   initializer counts its payload lanes at the stored width. The upload itself is Vulkan-only; the
//   device probe covering it is a flat Mul of an int64 graph input by a rank-0 int64 initializer,
//   whose output must equal the CPU oracle's.
//
// Values run end to end through CPU Sessions (runStandardPasses included) or through constFold.
#include "backend/cpu/int64_arithmetic.h"
#include "backend/cpu/parallel.h"
#include "backend/vulkan/ops/upload_init_rule.h"
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace vknn;

namespace {

    // ONNX TensorProto.DataType codes the Cast nodes below target.
    constexpr int64_t kOnnxInt32 = 6;
    constexpr int64_t kOnnxInt64 = 7;
    constexpr int64_t kOnnxBool  = 9;

    constexpr int64_t kInt64Min = std::numeric_limits<int64_t>::min();
    constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();

    Attr integerAttr(int64_t value) {
        Attr a;
        a.kind = Attr::Int;
        a.i    = value;
        return a;
    }

    TensorId addGraphInput(Graph &g, const std::string &name, const Shape &shape, DType dtype) {
        TensorDesc d;
        d.name      = name;
        d.shape     = shape;
        d.dtype     = dtype;
        d.isInput   = true;
        TensorId id = g.addTensor(d);
        g.inputs.push_back(id);
        return id;
    }

    TensorId addGraphOutput(Graph &g, const std::string &name, DType dtype) {
        TensorDesc d;
        d.name      = name;
        d.dtype     = dtype;
        d.isOutput  = true;
        TensorId id = g.addTensor(d);
        g.outputs.push_back(id);
        return id;
    }

    TensorId addInt64Constant(Graph &g, const std::string &name, const Shape &shape, const std::vector<int64_t> &values) {
        TensorDesc d;
        d.name          = name;
        d.shape         = shape;
        d.dtype         = DType::Int64;
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) values.size(), DType::Int64);
        std::memcpy(hb.bytes.data(), values.data(), values.size() * sizeof(int64_t));
        g.initializers[id] = hb;
        return id;
    }

    TensorId addFloatConstant(Graph &g, const std::string &name, const Shape &shape, const std::vector<float> &values) {
        TensorDesc d;
        d.name          = name;
        d.shape         = shape;
        d.dtype         = DType::Float32;
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) values.size(), DType::Float32);
        std::memcpy(hb.bytes.data(), values.data(), values.size() * sizeof(float));
        g.initializers[id] = hb;
        return id;
    }

    Node &addNode(Graph &g, OpType type, const std::string &name, std::vector<TensorId> inputs, std::vector<TensorId> outputs) {
        Node n;
        n.type    = type;
        n.name    = name;
        n.inputs  = std::move(inputs);
        n.outputs = std::move(outputs);
        g.nodes.push_back(n);
        return g.nodes.back();
    }

    void addCast(Graph &g, const std::string &name, TensorId input, TensorId output, int64_t onnxTarget) {
        Node &n          = addNode(g, OpType::Cast, name, {input}, {output});
        n.attr.map["to"] = integerAttr(onnxTarget);
    }

    Node &addBinary(Graph &g, BinaryType op, const std::string &name, TensorId lhs, TensorId rhs, TensorId out) {
        Node &n = addNode(g, OpType::Binary, name, {lhs, rhs}, {out});
        n.subOp = (int) op;
        return n;
    }

    IOTensor int64Tensor(const std::string &name, const Shape &shape, const std::vector<int64_t> &values) {
        IOTensor t;
        t.name  = name;
        t.shape = shape;
        t.dtype = DType::Int64;
        t.data.resize(values.size() * sizeof(int64_t));
        std::memcpy(t.data.data(), values.data(), t.data.size());
        return t;
    }

    IOTensor floatTensor(const std::string &name, const Shape &shape, const std::vector<float> &values) {
        IOTensor t;
        t.name  = name;
        t.shape = shape;
        t.dtype = DType::Float32;
        t.data.resize(values.size() * sizeof(float));
        std::memcpy(t.data.data(), values.data(), t.data.size());
        return t;
    }

    std::vector<int64_t> int64Values(const IOTensor &t) {
        EXPECT_EQ(t.dtype, DType::Int64) << t.name;
        std::vector<int64_t> values(t.data.size() / sizeof(int64_t));
        std::memcpy(values.data(), t.data.data(), values.size() * sizeof(int64_t));
        return values;
    }

    std::vector<float> floatValues(const IOTensor &t) {
        EXPECT_EQ(t.dtype, DType::Float32) << t.name;
        std::vector<float> values(t.data.size() / sizeof(float));
        std::memcpy(values.data(), t.data.data(), values.size() * sizeof(float));
        return values;
    }

    // Run `g` on the CPU backend. Returns the outputs in graph-output order; empty when the session
    // fails to build or run (the failure is reported through a non-fatal expectation).
    std::vector<IOTensor> runOnCpu(Graph &&g, const std::vector<IOTensor> &inputs, int threads = 1) {
        Config cfg;
        cfg.backend    = BackendKind::Cpu;
        cfg.cpuThreads = threads;
        auto session   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(session);
        if (!session)
        {
            return {};
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(session->run(inputs, outs), Status::Ok);
        return outs;
    }

    // Independent integer-power reference: repeated wrapping multiplication for a non-negative
    // exponent (callers keep it small), the truncated reciprocal rules for a negative one.
    int64_t referencePow(int64_t base, int64_t exponent) {
        if (exponent < 0)
        {
            if (base == 1)
            {
                return 1;
            }
            if (base == -1)
            {
                return exponent % 2 != 0 ? -1 : 1;
            }
            return 0;
        }
        uint64_t product = 1;
        for (int64_t step = 0; step < exponent; ++step)
        {
            product *= (uint64_t) base;
        }
        return (int64_t) product;
    }

    // One aligned [count] pair of int64 runtime inputs "lhs"/"rhs" through Binary `op` on the CPU.
    std::vector<int64_t> runInt64Binary(BinaryType op, const std::vector<int64_t> &lhs, const std::vector<int64_t> &rhs) {
        const Shape shape {(int64_t) lhs.size()};
        Graph       g;
        TensorId    a = addGraphInput(g, "lhs", shape, DType::Int64);
        TensorId    b = addGraphInput(g, "rhs", shape, DType::Int64);
        TensorId    y = addGraphOutput(g, "y", DType::Int64);
        addBinary(g, op, "binary", a, b, y);
        std::vector<IOTensor> outs = runOnCpu(std::move(g), {int64Tensor("lhs", shape, lhs), int64Tensor("rhs", shape, rhs)});
        return outs.size() == 1 ? int64Values(outs[0]) : std::vector<int64_t> {};
    }

    // Deterministic 64-bit generator (splitmix64) for the randomized sweeps.
    uint64_t nextRandom(uint64_t &state) {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t mixed = state;
        mixed          = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
        mixed          = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
        return mixed ^ (mixed >> 31);
    }

    // --- transcription of shaders/cast.comp -------------------------------------------------------------
    // A line-for-line C++ transcription of the kCastModeBool arm of shaders/cast.comp, kept textually
    // parallel to the shader so the two diff cleanly. floatBitsToUint is the GLSL builtin (the IEEE bit
    // pattern of an fp32 value).

    uint32_t floatBitsToUint(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    // const uint kCastFloatMagnitudeBits = 0x7FFFFFFFu;
    constexpr uint32_t kCastFloatMagnitudeBits = 0x7FFFFFFFu;

    // float castBoolValue(float source) {
    //   return (floatBitsToUint(source) & kCastFloatMagnitudeBits) != 0u ? 1.0 : 0.0;
    // }
    float castBoolValue(float source) {
        return (floatBitsToUint(source) & kCastFloatMagnitudeBits) != 0u ? 1.0f : 0.0f;
    }

    // --- end of transcription -------------------------------------------------------------------------

    // Cast(to=BOOL) of a float runtime input on the CPU oracle, read back through a BOOL (UInt8) output.
    std::vector<uint8_t> castFloatsToBoolOnCpu(const std::vector<float> &values) {
        const Shape shape {(int64_t) values.size()};
        Graph       g;
        TensorId    x = addGraphInput(g, "x", shape, DType::Float32);
        TensorId    y = addGraphOutput(g, "y", DType::UInt8);
        addCast(g, "to_bool", x, y, kOnnxBool);
        std::vector<IOTensor> outs = runOnCpu(std::move(g), {floatTensor("x", shape, values)});
        if (outs.size() != 1)
        {
            return {};
        }
        EXPECT_EQ(outs[0].dtype, DType::UInt8);
        return outs[0].data;
    }

    float floatFromBits(uint32_t bits) {
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

} // namespace

// --- Binary on int64 ------------------------------------------------------------------------------------

TEST(IntegerBinaryFixes, Int64PowMatchesWrappingReference) {
    // Every base against every exponent: zero/one/minus-one, small even and odd bases, a base past fp32's
    // 24-bit mantissa, and both int64 extremes, raised to negative, zero, small and wrapping exponents.
    const std::vector<int64_t> bases {0, 1, -1, 2, -2, 3, -3, 7, int64_t(1) << 31, kInt64Max, kInt64Min};
    const std::vector<int64_t> exponents {-65, -64, -63, -3, -2, -1, 0, 1, 2, 3, 5, 31, 32, 39, 40, 62, 63, 64, 65};
    std::vector<int64_t>       lhs, rhs;
    for (int64_t base: bases)
    {
        for (int64_t exponent: exponents)
        {
            lhs.push_back(base);
            rhs.push_back(exponent);
        }
    }
    const std::vector<int64_t> got = runInt64Binary(BinaryType::Pow, lhs, rhs);
    ASSERT_EQ(got.size(), lhs.size());
    for (size_t k = 0; k < lhs.size(); ++k)
    {
        EXPECT_EQ(got[k], referencePow(lhs[k], rhs[k])) << lhs[k] << " ^ " << rhs[k];
    }
}

TEST(IntegerBinaryFixes, Int64PowAnchorsAndHugeExponents) {
    struct Case {
        int64_t base, exponent, expected;
    };
    const std::vector<Case> cases {
        {0, 0, 1},                               // 0^0 is 1
        {0, 5, 0},                               // zero base
        {0, -1, 0},                              // a zero base with a negative exponent is a defined 0
        {1, -5, 1},                              // base 1 survives any negative exponent
        {-1, -3, -1},                            // base -1, odd negative exponent
        {-1, -2, 1},                             // base -1, even negative exponent
        {2, -1, 0},                              // |base| >= 2: the reciprocal truncates to 0
        {-5, -2, 0},                             // |base| >= 2: the reciprocal truncates to 0
        {3, 39, 4052555153018976267LL},          // exact, far past 2^24
        {int64_t(1) << 31, 2, int64_t(1) << 62}, // exact
        {2, 62, int64_t(1) << 62},               // exact
        {2, 63, kInt64Min},                      // wraps onto the sign bit
        {2, 64, 0},                              // wraps out entirely
        {-2, 63, kInt64Min},                     // (-2)^63 is exactly INT64_MIN
        {-1, kInt64Max, -1},                     // odd huge exponent
        {-1, kInt64Min, 1},                      // even huge negative exponent
        {1, kInt64Min, 1},                       // base 1, huge negative exponent
        {0, kInt64Max, 0},                       // zero base, huge exponent
        {2, kInt64Max, 0},                       // every product past 2^64 wraps to 0
        {-2, kInt64Max, 0},                      // every product past 2^64 wraps to 0
        {kInt64Min, 2, 0},                       // (2^63)^2 wraps to 0
        {kInt64Max, 2, 1},                       // (2^63 - 1)^2 = 2^126 - 2^64 + 1 == 1 mod 2^64
    };
    std::vector<int64_t> lhs, rhs;
    for (const Case &c: cases)
    {
        lhs.push_back(c.base);
        rhs.push_back(c.exponent);
    }
    const std::vector<int64_t> got = runInt64Binary(BinaryType::Pow, lhs, rhs);
    ASSERT_EQ(got.size(), cases.size());
    for (size_t k = 0; k < cases.size(); ++k)
    {
        EXPECT_EQ(got[k], cases[k].expected) << cases[k].base << " ^ " << cases[k].exponent;
    }
}

TEST(IntegerBinaryFixes, Int64PowBroadcastsAndTakesARankZeroExponent) {
    // Broadcast [2,1] ^ [3] -> [2,3] through two runtime inputs.
    {
        Graph    g;
        TensorId a = addGraphInput(g, "base", {2, 1}, DType::Int64);
        TensorId b = addGraphInput(g, "exponent", {3}, DType::Int64);
        TensorId y = addGraphOutput(g, "y", DType::Int64);
        addBinary(g, BinaryType::Pow, "pow", a, b, y);
        std::vector<IOTensor> outs = runOnCpu(std::move(g), {int64Tensor("base", {2, 1}, {2, -3}), int64Tensor("exponent", {3}, {0, 1, 2})});
        ASSERT_EQ(outs.size(), 1u);
        EXPECT_EQ(outs[0].shape, (Shape {2, 3}));
        EXPECT_EQ(int64Values(outs[0]), (std::vector<int64_t> {1, 2, 4, 1, -3, 9}));
    }
    // A rank-0 int64 constant exponent against a runtime base.
    {
        Graph    g;
        TensorId a = addGraphInput(g, "base", {4}, DType::Int64);
        TensorId b = addInt64Constant(g, "cube", {}, {3});
        TensorId y = addGraphOutput(g, "y", DType::Int64);
        addBinary(g, BinaryType::Pow, "pow", a, b, y);
        std::vector<IOTensor> outs = runOnCpu(std::move(g), {int64Tensor("base", {4}, {2, 3, -2, 1000000})});
        ASSERT_EQ(outs.size(), 1u);
        EXPECT_EQ(int64Values(outs[0]), (std::vector<int64_t> {8, 27, -8, 1000000000000000000LL}));
    }
}

TEST(IntegerBinaryFixes, Int64PowConstantFoldsExactly) {
    Graph    g;
    TensorId base     = addInt64Constant(g, "base", {5}, {3, -1, 2, 0, 7});
    TensorId exponent = addInt64Constant(g, "exponent", {5}, {39, -7, 64, 0, -1});
    TensorId y        = g.addTensor({"y"});
    addBinary(g, BinaryType::Pow, "pow", base, exponent, y);
    g.outputs = {y};

    inferShapes(g, 1);
    constFold(g);
    EXPECT_TRUE(g.nodes.empty()) << "an all-constant int64 Pow must fold away";
    ASSERT_TRUE(g.isInitializer(y));
    ASSERT_EQ(g.desc(y).dtype, DType::Int64);
    ASSERT_EQ(g.initializers[y].bytes.size(), 5 * sizeof(int64_t));
    const int64_t *folded = g.initializers[y].i64();
    EXPECT_EQ(folded[0], 4052555153018976267LL);
    EXPECT_EQ(folded[1], -1);
    EXPECT_EQ(folded[2], 0);
    EXPECT_EQ(folded[3], 1);
    EXPECT_EQ(folded[4], 0);
}

TEST(IntegerBinaryFixes, Int64DivisionOfMinByMinusOneWraps) {
    // INT64_MIN / -1 has no int64 quotient: it wraps to INT64_MIN instead of trapping (x86 raises #DE on
    // the hardware division) or invoking undefined behavior. Every other quotient truncates toward zero,
    // and a zero divisor is a defined 0.
    const std::vector<int64_t> dividends {kInt64Min, kInt64Min, kInt64Max, 7, -7, 5, kInt64Min, (int64_t(1) << 40) + 1, -9};
    const std::vector<int64_t> divisors {-1, 1, -1, -2, 2, 0, 0, 1, -1};
    const std::vector<int64_t> expected {kInt64Min, kInt64Min, -kInt64Max, -3, -3, 0, 0, (int64_t(1) << 40) + 1, 9};
    EXPECT_EQ(runInt64Binary(BinaryType::Div, dividends, divisors), expected);
}

TEST(IntegerBinaryFixes, Int64MulAndSubWrapModuloTwoToTheSixtyFour) {
    // Overflowing products and differences wrap in two's complement (computed on uint64, so the
    // sanitizer build sees no signed overflow).
    EXPECT_EQ(runInt64Binary(BinaryType::Mul, {kInt64Max, kInt64Min, int64_t(1) << 62, -3}, {2, -1, 4, 5}), (std::vector<int64_t> {-2, kInt64Min, 0, -15}));
    EXPECT_EQ(runInt64Binary(BinaryType::Sub, {kInt64Min, kInt64Max, 0, 10}, {1, -1, kInt64Min, 3}), (std::vector<int64_t> {kInt64Max, kInt64Min, kInt64Min, 7}));
}

TEST(IntegerBinaryFixes, Int64AddWrapsOnBothAddPaths) {
    // The Binary default arm and the standalone Add op (ONNX Add / Sum) both wrap an overflowing sum.
    const std::vector<int64_t> lhs {kInt64Max, kInt64Min, kInt64Max, -5};
    const std::vector<int64_t> rhs {1, -1, kInt64Max, 3};
    const std::vector<int64_t> expected {kInt64Min, kInt64Max, -2, -2};
    EXPECT_EQ(runInt64Binary(BinaryType::Add, lhs, rhs), expected);

    const Shape shape {(int64_t) lhs.size()};
    Graph       g;
    TensorId    a = addGraphInput(g, "lhs", shape, DType::Int64);
    TensorId    b = addGraphInput(g, "rhs", shape, DType::Int64);
    TensorId    y = addGraphOutput(g, "y", DType::Int64);
    addNode(g, OpType::Add, "add", {a, b}, {y});
    std::vector<IOTensor> outs = runOnCpu(std::move(g), {int64Tensor("lhs", shape, lhs), int64Tensor("rhs", shape, rhs)});
    ASSERT_EQ(outs.size(), 1u);
    EXPECT_EQ(int64Values(outs[0]), expected);
}

TEST(IntegerBinaryFixes, FloatOperandOfInt64PathTruncatesWithDefinedSaturation) {
    // A float operand mixed with an int64 one truncates toward zero; NaN reads 0 and values outside the
    // int64 range saturate (a plain (int64_t) conversion of those is undefined). Both int64 paths agree.
    const float                quietNaN            = std::numeric_limits<float>::quiet_NaN();
    const float                infinity            = std::numeric_limits<float>::infinity();
    const float                largestBelowTwoTo63 = 9223371487098961920.0f; // 2^63 - 2^39
    const float                minusTwoTo63        = -9223372036854775808.0f;
    const std::vector<float>   floats {quietNaN, infinity, -infinity, 1e30f, -1e30f, -2.7f, 2.7f, largestBelowTwoTo63, minusTwoTo63, -0.0f};
    const std::vector<int64_t> expected {0, kInt64Max, kInt64Min, kInt64Max, kInt64Min, -2, 2, 9223371487098961920LL, kInt64Min, 0};
    const Shape                shape {(int64_t) floats.size()};
    const std::vector<int64_t> zeros(floats.size(), 0);
    for (bool standaloneAdd: {false, true})
    {
        Graph    g;
        TensorId x = addGraphInput(g, "x", shape, DType::Float32);
        TensorId k = addGraphInput(g, "k", shape, DType::Int64);
        TensorId y = addGraphOutput(g, "y", DType::Int64);
        if (standaloneAdd)
        {
            addNode(g, OpType::Add, "add", {x, k}, {y});
        } else
        {
            addBinary(g, BinaryType::Sub, "sub", x, k, y);
        }
        std::vector<IOTensor> outs = runOnCpu(std::move(g), {floatTensor("x", shape, floats), int64Tensor("k", shape, zeros)});
        ASSERT_EQ(outs.size(), 1u);
        EXPECT_EQ(int64Values(outs[0]), expected) << (standaloneAdd ? "Add" : "Binary Sub");
    }
}

TEST(IntegerBinaryFixes, Int64ArithmeticHelpersMatchReferencesOnRandomOperands) {
    // Randomized operands across the whole int64 range and near the extremes: the wrapping helpers obey
    // the ring identities of arithmetic modulo 2^64 (subtraction undoes addition, multiplication
    // distributes over a wrapping increment), division equals the truncating quotient wherever C++
    // defines it, and the power equals repeated wrapping multiplication for exponents in [-70, 70].
    static constexpr int     kSamples           = 20000;
    static constexpr int64_t kExponentMagnitude = 70;
    uint64_t                 state              = 0x5EEDu;
    for (int sample = 0; sample < kSamples; ++sample)
    {
        const uint64_t aBits = nextRandom(state);
        const uint64_t bBits = nextRandom(state);
        // Every fourth sample pulls an operand next to an extreme so wrapping boundaries are hit.
        const int64_t a = sample % 4 == 0 ? kInt64Min + (int64_t) (aBits % 4) : cpu::int64FromWrappedBits(aBits);
        const int64_t b = sample % 4 == 1 ? kInt64Max - (int64_t) (bBits % 4) : cpu::int64FromWrappedBits(bBits);
        EXPECT_EQ(cpu::wrappingSubInt64(cpu::wrappingAddInt64(a, b), b), a) << a << " + " << b;
        EXPECT_EQ(cpu::wrappingAddInt64(cpu::wrappingSubInt64(a, b), b), a) << a << " - " << b;
        EXPECT_EQ(cpu::wrappingMulInt64(a, cpu::wrappingAddInt64(b, 1)), cpu::wrappingAddInt64(cpu::wrappingMulInt64(a, b), a)) << a << " * " << b;
        if (b != 0 && !(a == kInt64Min && b == -1))
        {
            EXPECT_EQ(cpu::divideInt64(a, b), a / b) << a << " / " << b;
        }
        const int64_t smallBase = (int64_t) (aBits % 21) - 10;
        const int64_t exponent  = (int64_t) (bBits % (2 * kExponentMagnitude + 1)) - kExponentMagnitude;
        EXPECT_EQ(cpu::powInt64(smallBase, exponent), referencePow(smallBase, exponent)) << smallBase << " ^ " << exponent;
        EXPECT_EQ(cpu::powInt64(a, exponent), referencePow(a, exponent)) << a << " ^ " << exponent;
    }
    EXPECT_EQ(cpu::divideInt64(kInt64Min, -1), kInt64Min);
    EXPECT_EQ(cpu::divideInt64(kInt64Min, 0), 0);
}

TEST(IntegerBinaryFixes, FloatBaseWithInt64ExponentIsAFloatPower) {
    // ONNX Pow types its result by the base: a float base raised to an int64 exponent is a float power.
    // The base is never truncated and the output stays fp32 (a fractional base keeps its fraction).
    const std::vector<float>   base {1.5f, -2.0f, 0.5f, 2.0f, 3.0f, -0.5f};
    const std::vector<int64_t> exponent {2, 3, -1, 30, 0, -3};
    Graph                      g;
    TensorId                   x = addGraphInput(g, "x", {6}, DType::Float32);
    TensorId                   e = addInt64Constant(g, "e", {6}, exponent);
    TensorId                   y = addGraphOutput(g, "y", DType::Float32);
    addBinary(g, BinaryType::Pow, "pow", x, e, y);
    std::vector<IOTensor> outs = runOnCpu(std::move(g), {floatTensor("x", {6}, base)});
    ASSERT_EQ(outs.size(), 1u);
    const std::vector<float> got = floatValues(outs[0]);
    ASSERT_EQ(got.size(), base.size());
    for (size_t k = 0; k < base.size(); ++k)
    {
        const float want = std::pow(base[k], (float) exponent[k]);
        EXPECT_EQ(std::memcmp(&got[k], &want, sizeof(float)), 0) << base[k] << " ^ " << exponent[k] << " got " << got[k] << " want " << want;
    }
    EXPECT_EQ(got[0], 2.25f);
    EXPECT_EQ(got[1], -8.0f);
    EXPECT_EQ(got[2], 2.0f);
    EXPECT_EQ(got[3], 1073741824.0f);
    EXPECT_EQ(got[4], 1.0f);
    EXPECT_EQ(got[5], -8.0f);
}

TEST(IntegerBinaryFixes, FloatBaseWithInt64ExponentIsByteIdenticalAcrossThreads) {
    // The float-base power partitions across threads like the float path: every thread count must
    // produce the 1-thread bytes. 7151 x 7 = 50057 elements, past kMinChunkOps and not divisible by any
    // probed thread count; the [7] exponent broadcasts along the last axis so every chunk seeks its walker.
    static constexpr int64_t kRows = 7151;
    static constexpr int64_t kCols = 7;
    const Shape              shape {kRows, kCols};
    std::vector<float>       base((size_t) (kRows * kCols));
    uint32_t                 state = 12345u;
    for (float &v: base)
    {
        state = state * 1664525u + 1013904223u;
        v     = (float) ((int32_t) (state >> 8) % 2001 - 1000) * 0.002f;
    }
    auto build = [&] {
        Graph    g;
        TensorId x = addGraphInput(g, "x", shape, DType::Float32);
        TensorId e = addInt64Constant(g, "e", {kCols}, {-2, -1, 0, 1, 2, 3, 4});
        TensorId y = addGraphOutput(g, "y", DType::Float32);
        addBinary(g, BinaryType::Pow, "pow", x, e, y);
        return g;
    };
    const std::vector<IOTensor> reference = runOnCpu(build(), {floatTensor("x", shape, base)}, 1);
    ASSERT_EQ(reference.size(), 1u);
    const int64_t dispatchesBefore = cpu::detail::poolDispatches();
    for (int threads: {2, 3, 5, 8})
    {
        const std::vector<IOTensor> got = runOnCpu(build(), {floatTensor("x", shape, base)}, threads);
        ASSERT_EQ(got.size(), 1u);
        ASSERT_EQ(got[0].data.size(), reference[0].data.size()) << "threads=" << threads;
        EXPECT_EQ(std::memcmp(got[0].data.data(), reference[0].data.data(), reference[0].data.size()), 0) << "threads=" << threads;
    }
    EXPECT_GT(cpu::detail::poolDispatches(), dispatchesBefore) << "shape too small to partition: the byte comparison is vacuous";
}

// --- ConvertDtype on the CPU ----------------------------------------------------------------------------

TEST(ConvertDtypeCpu, Int64TensorKeepsItsValues) {
    // Values past fp32's and fp64's exact integer spans, and both extremes, survive bit for bit.
    const std::vector<int64_t> values {(int64_t(1) << 53) + 1, -1, kInt64Min, kInt64Max, 0};
    Graph                      g;
    TensorId                   x = addGraphInput(g, "x", {5}, DType::Int64);
    TensorId                   y = addGraphOutput(g, "y", DType::Int64);
    addNode(g, OpType::ConvertDtype, "convert", {x}, {y});
    std::vector<IOTensor> outs = runOnCpu(std::move(g), {int64Tensor("x", {5}, values)});
    ASSERT_EQ(outs.size(), 1u);
    EXPECT_EQ(outs[0].shape, (Shape {5}));
    EXPECT_EQ(int64Values(outs[0]), values);
}

TEST(ConvertDtypeCpu, FloatTensorCopiesEveryBit) {
    const std::vector<float> values {1.5f, std::numeric_limits<float>::quiet_NaN(), -0.0f, -std::numeric_limits<float>::infinity(), 16777217.0f};
    Graph                    g;
    TensorId                 x = addGraphInput(g, "x", {5}, DType::Float32);
    TensorId                 y = addGraphOutput(g, "y", DType::Float32);
    addNode(g, OpType::ConvertDtype, "convert", {x}, {y});
    std::vector<IOTensor> outs = runOnCpu(std::move(g), {floatTensor("x", {5}, values)});
    ASSERT_EQ(outs.size(), 1u);
    ASSERT_EQ(outs[0].data.size(), values.size() * sizeof(float));
    EXPECT_EQ(std::memcmp(outs[0].data.data(), values.data(), outs[0].data.size()), 0);
}

// --- Cast to BOOL ---------------------------------------------------------------------------------------

TEST(CastToBool, FloatAndInt64InputsThroughCpuSession) {
    const float                quietNaN          = std::numeric_limits<float>::quiet_NaN();
    const float                infinity          = std::numeric_limits<float>::infinity();
    const float                smallestSubnormal = std::numeric_limits<float>::denorm_min();
    const std::vector<float>   floats {0.5f, -3.0f, quietNaN, 0.0f, -0.0f, infinity, smallestSubnormal, -0.25f};
    const std::vector<int64_t> ints {5, -3, 0, kInt64Min, int64_t(1) << 40};
    Graph                      g;
    TensorId                   xf = addGraphInput(g, "xf", {8}, DType::Float32);
    TensorId                   xi = addGraphInput(g, "xi", {5}, DType::Int64);
    TensorId                   yf = addGraphOutput(g, "yf", DType::UInt8);
    TensorId                   yi = addGraphOutput(g, "yi", DType::UInt8);
    addCast(g, "float_to_bool", xf, yf, kOnnxBool);
    addCast(g, "int_to_bool", xi, yi, kOnnxBool);
    std::vector<IOTensor> outs = runOnCpu(std::move(g), {floatTensor("xf", {8}, floats), int64Tensor("xi", {5}, ints)});
    ASSERT_EQ(outs.size(), 2u);
    EXPECT_EQ(outs[0].dtype, DType::UInt8);
    EXPECT_EQ(outs[1].dtype, DType::UInt8);
    EXPECT_EQ(outs[0].data, (std::vector<uint8_t> {1, 1, 1, 0, 0, 1, 1, 1}));
    EXPECT_EQ(outs[1].data, (std::vector<uint8_t> {1, 1, 0, 1, 1}));
}

TEST(CastToBool, ConstantFoldsToTruthValues) {
    Graph    g;
    TensorId x = addFloatConstant(g, "x", {5}, {0.5f, -3.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f, -0.0f});
    TensorId y = g.addTensor({"y"});
    addCast(g, "to_bool", x, y, kOnnxBool);
    g.outputs = {y};

    inferShapes(g, 1);
    constFold(g);
    EXPECT_TRUE(g.nodes.empty()) << "a constant Cast must fold away";
    ASSERT_TRUE(g.isInitializer(y));
    ASSERT_EQ(g.desc(y).dtype, DType::Int64);
    ASSERT_EQ(g.initializers[y].bytes.size(), 5 * sizeof(int64_t));
    const int64_t *folded = g.initializers[y].i64();
    EXPECT_EQ(std::vector<int64_t>(folded, folded + 5), (std::vector<int64_t> {1, 1, 1, 0, 0}));
}

TEST(CastToBool, IntegerTargetsStillTruncate) {
    // Only BOOL tests truth; INT64 and INT32 keep truncating toward zero.
    const std::vector<float> values {2.7f, -2.7f, 0.5f, -0.5f, 16777216.0f};
    Graph                    g;
    TensorId                 x   = addGraphInput(g, "x", {5}, DType::Float32);
    TensorId                 y64 = addGraphOutput(g, "y64", DType::Int64);
    TensorId                 y32 = addGraphOutput(g, "y32", DType::Int32);
    addCast(g, "to_int64", x, y64, kOnnxInt64);
    addCast(g, "to_int32", x, y32, kOnnxInt32);
    std::vector<IOTensor> outs = runOnCpu(std::move(g), {floatTensor("x", {5}, values)});
    ASSERT_EQ(outs.size(), 2u);
    EXPECT_EQ(int64Values(outs[0]), (std::vector<int64_t> {2, -2, 0, 0, 16777216}));
    ASSERT_EQ(outs[1].dtype, DType::Int32);
    ASSERT_EQ(outs[1].data.size(), 5 * sizeof(int32_t));
    std::vector<int32_t> narrow(5);
    std::memcpy(narrow.data(), outs[1].data.data(), narrow.size() * sizeof(int32_t));
    EXPECT_EQ(narrow, (std::vector<int32_t> {2, -2, 0, 0, 16777216}));
}

TEST(CastToBool, ShaderBoolModeTranscriptionMatchesCpuOracleOnFp32Patterns) {
    // Stratified fp32 bit patterns: both signs x every biased exponent x mantissas covering zero, the
    // lowest bits, the quiet-NaN bit, all ones and a spread of mixed patterns. Exponent 0 yields +-0 and
    // subnormals; exponent 255 yields infinities and NaNs of both kinds.
    static constexpr uint32_t   kSignBit       = 0x80000000u;
    static constexpr int        kExponentShift = 23;
    static constexpr uint32_t   kExponentCount = 256u;
    const std::vector<uint32_t> mantissas {0u, 1u, 2u, 3u, 0x400000u, 0x3FFFFFu, 0x7FFFFFu, 0x2AAAAAu, 0x555555u, 0x000100u, 0x123456u, 0x7FFF00u};
    std::vector<float>          sources;
    for (uint32_t sign: {0u, kSignBit})
    {
        for (uint32_t exponent = 0; exponent < kExponentCount; ++exponent)
        {
            for (uint32_t mantissa: mantissas)
            {
                sources.push_back(floatFromBits(sign | (exponent << kExponentShift) | mantissa));
            }
        }
    }
    const std::vector<uint8_t> oracle = castFloatsToBoolOnCpu(sources);
    ASSERT_EQ(oracle.size(), sources.size());
    for (size_t k = 0; k < sources.size(); ++k)
    {
        const float shader = castBoolValue(sources[k]);
        EXPECT_EQ(shader, (float) oracle[k]) << "bits 0x" << std::hex << floatBitsToUint(sources[k]);
        EXPECT_EQ(shader, sources[k] != 0.0f ? 1.0f : 0.0f) << "bits 0x" << std::hex << floatBitsToUint(sources[k]);
    }
}

TEST(CastToBool, ShaderBoolModeTranscriptionMatchesCpuOracleOnEveryFp16Pattern) {
    // The fp16 variant reads each lane as float(s[g]): every one of the 65536 half patterns, widened the
    // way that conversion widens it, must test exactly as the CPU oracle tests the same fp32 value.
    static constexpr uint32_t kHalfPatternCount = 65536u;
    std::vector<float>        sources(kHalfPatternCount);
    for (uint32_t bits = 0; bits < kHalfPatternCount; ++bits)
    {
        sources[bits] = halfToFloat((fp16_t) bits);
    }
    const std::vector<uint8_t> oracle = castFloatsToBoolOnCpu(sources);
    ASSERT_EQ(oracle.size(), sources.size());
    int64_t zeros = 0;
    for (uint32_t bits = 0; bits < kHalfPatternCount; ++bits)
    {
        const float shader = castBoolValue(sources[bits]);
        EXPECT_EQ(shader, (float) oracle[bits]) << "half bits 0x" << std::hex << bits;
        zeros += shader == 0.0f ? 1 : 0;
    }
    EXPECT_EQ(zeros, 2) << "exactly +0 and -0 are false";
}

// --- uploadInit element count ---------------------------------------------------------------------------

TEST(UploadInitRule, RankZeroCountsPayloadLanesAtTheStoredWidth) {
    EXPECT_EQ(uploadInitElemCount({}, DType::Int64, sizeof(int64_t)), 1);
    EXPECT_EQ(uploadInitElemCount({}, DType::Int8, sizeof(int8_t)), 1);
    EXPECT_EQ(uploadInitElemCount({}, DType::UInt8, sizeof(uint8_t)), 1);
    EXPECT_EQ(uploadInitElemCount({}, DType::Float32, sizeof(float)), 1);
    EXPECT_EQ(uploadInitElemCount({}, DType::Float16, sizeof(fp16_t)), 1);
    EXPECT_EQ(uploadInitElemCount({}, DType::Int32, sizeof(float)), 1);
    // The payload of a rank-0 scalar always covers what its count needs.
    for (DType dtype: {DType::Int64, DType::Int8, DType::UInt8, DType::Float32, DType::Float16, DType::Int32})
    {
        const size_t  payloadBytes = dtypeSize(dtype);
        const int64_t count        = uploadInitElemCount({}, dtype, payloadBytes);
        EXPECT_EQ(uploadInitPayloadBytesNeeded(count, dtype), payloadBytes) << dtypeStr(dtype);
    }
}

TEST(UploadInitRule, ShapeCountWinsAndGuardNeedsNativeLanes) {
    EXPECT_EQ(uploadInitElemCount({2, 3}, DType::Int64, 6 * sizeof(int64_t)), 6);
    EXPECT_EQ(uploadInitElemCount({4}, DType::UInt8, 4), 4);
    EXPECT_EQ(uploadInitElemCount({0}, DType::Float32, 0), 0);
    EXPECT_EQ(uploadInitPayloadBytesNeeded(3, DType::Int8), 3u);
    EXPECT_EQ(uploadInitPayloadBytesNeeded(3, DType::Int64), 24u);
    EXPECT_EQ(uploadInitPayloadBytesNeeded(5, DType::Float16), 10u);
    EXPECT_EQ(uploadInitPayloadBytesNeeded(5, DType::Float32), 20u);
    EXPECT_EQ(uploadInitPayloadBytesNeeded(0, DType::Int64), 0u);
    // A packed int4 weight keeps its logical Float16 [4, 8] desc over a quarter-size payload: the guard
    // still refuses it.
    const int64_t packedCount = uploadInitElemCount({4, 8}, DType::Float16, 16);
    EXPECT_EQ(packedCount, 32);
    EXPECT_LT(16u, uploadInitPayloadBytesNeeded(packedCount, DType::Float16));
}

TEST(UploadInitRule, FloatStorageWidthsFollowTheFourAndTwoByteRule) {
    // Float32 and the Int32 label store 4-byte lanes and Float16 stores 2-byte lanes: for every shape
    // and payload size the count and the guard bytes equal the plain width rule for those dtypes.
    for (DType dtype: {DType::Float32, DType::Float16, DType::Int32})
    {
        const size_t laneBytes = dtype == DType::Float16 ? 2 : 4;
        for (const Shape &shape: {Shape {}, Shape {0}, Shape {1}, Shape {3, 5}})
        {
            for (size_t payloadBytes: {0u, 2u, 4u, 7u, 8u, 60u})
            {
                const int64_t shapeCount = numElements(shape);
                const int64_t widthRule  = shapeCount > 0 ? shapeCount : (int64_t) (payloadBytes / laneBytes);
                const int64_t count      = uploadInitElemCount(shape, dtype, payloadBytes);
                EXPECT_EQ(count, widthRule) << dtypeStr(dtype) << " shape " << shapeStr(shape) << " payload " << payloadBytes;
                EXPECT_EQ(uploadInitPayloadBytesNeeded(count, dtype), count > 0 ? (size_t) count * laneBytes : 0u) << dtypeStr(dtype);
            }
        }
    }
}
