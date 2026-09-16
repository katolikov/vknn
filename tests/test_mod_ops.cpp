// ONNX Mod: the CPU oracle (src/backend/cpu/ops/mod.cpp over backend/cpu/mod_remainder.h) run through
// a CPU Session, its constant folding, its thread-count byte invariance, and a host transcription of
// the GPU kernel's exact fmod (shaders/mod.comp modRemainder) compared bit for bit against std::fmod
// and the oracle. GLSL never runs on the host, so the transcription is what proves the shader's
// arithmetic; it is kept textually parallel to the .comp function.
#include "backend/cpu/mod_remainder.h"
#include "backend/cpu/parallel.h"
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    // ---- host transcription of shaders/mod.comp ---------------------------------------------------

    // Same names and values as the constants of shaders/mod.comp.
    constexpr int      kModFloorRemainder   = 0;
    constexpr uint32_t kQuietNaNBits        = 0x7FC00000u;
    constexpr uint32_t kSignBit             = 0x80000000u;
    constexpr uint32_t kFloatMaxHalfBits    = 0x7EFFFFFFu;
    constexpr int      kMaxDivisorDoublings = 276;

    // GLSL uintBitsToFloat.
    float uintBitsToFloat(uint32_t bits) {
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    // GLSL floatBitsToUint.
    uint32_t floatBitsToUint(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    // Transcription of modRemainder in shaders/mod.comp, statement for statement: GLSL abs/isnan/isinf
    // are std::fabs/std::isnan/std::isinf, the bit casts are the two helpers above, and float literals
    // carry the f suffix.
    float transcribedModRemainder(float dividend, float divisor, int fmodMode) {
        bool floorRemainder = fmodMode == kModFloorRemainder;
        if (divisor == 0.0f)
        {
            return floorRemainder ? 0.0f : uintBitsToFloat(kQuietNaNBits);
        }
        if (std::isnan(dividend) || std::isnan(divisor) || std::isinf(dividend))
        {
            return uintBitsToFloat(kQuietNaNBits);
        }
        if (dividend == 0.0f)
        {
            return dividend;
        }
        float remainder = dividend;
        if (!std::isinf(divisor))
        {
            float magnitude        = std::fabs(dividend);
            float divisorMagnitude = std::fabs(divisor);
            float step             = divisorMagnitude;
            float floatMaxHalf     = uintBitsToFloat(kFloatMaxHalfBits);
            for (int doubling = 0; doubling < kMaxDivisorDoublings && step <= floatMaxHalf && step * 2.0f <= magnitude; ++doubling)
            {
                step *= 2.0f;
            }
            for (int halving = 0; halving <= kMaxDivisorDoublings && step >= divisorMagnitude; ++halving)
            {
                if (magnitude >= step)
                {
                    magnitude -= step;
                }
                step *= 0.5f;
            }
            remainder = uintBitsToFloat(floatBitsToUint(magnitude) | (floatBitsToUint(dividend) & kSignBit));
        }
        if (floorRemainder && remainder != 0.0f && ((remainder < 0.0f) != (divisor < 0.0f)))
        {
            remainder += divisor;
        }
        return remainder;
    }

    // ---- single-node CPU session ------------------------------------------------------------------

    // fmod value that leaves the attribute off the node (the ONNX default, floor remainder, applies).
    constexpr int64_t kFmodAttributeAbsent = -1;

    Attr intAttr(int64_t value) {
        Attr a;
        a.kind = Attr::Int;
        a.i    = value;
        return a;
    }

    // One Mod operand: a runtime graph input or a constant initializer, carried as fp32 or int64.
    struct ModOperand {
        Shape                shape;
        DType                dtype    = DType::Float32;
        bool                 constant = false;
        std::vector<float>   floats; // payload when dtype is Float32
        std::vector<int64_t> ints;   // payload when dtype is Int64
    };

    ModOperand floatOperand(Shape shape, std::vector<float> values, bool constant) {
        ModOperand operand;
        operand.shape    = std::move(shape);
        operand.constant = constant;
        operand.floats   = std::move(values);
        return operand;
    }

    ModOperand int64Operand(Shape shape, std::vector<int64_t> values, bool constant) {
        ModOperand operand;
        operand.shape    = std::move(shape);
        operand.dtype    = DType::Int64;
        operand.constant = constant;
        operand.ints     = std::move(values);
        return operand;
    }

    struct ModRun {
        Status               status = Status::RuntimeError;
        Shape                shape;
        std::vector<uint8_t> bytes;

        std::vector<float> floats() const {
            std::vector<float> values(bytes.size() / sizeof(float));
            std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
            return values;
        }
        std::vector<int64_t> ints() const {
            std::vector<int64_t> values(bytes.size() / sizeof(int64_t));
            std::memcpy(values.data(), bytes.data(), values.size() * sizeof(int64_t));
            return values;
        }
    };

    // Build `y = Mod(dividend, divisor)` with the output declared `outputDtype` (the session reads an
    // output back in its declared dtype, so an Int64 declaration returns the int64 result exactly),
    // run it on the CPU backend with `threads` workers, and return the status, shape and raw bytes.
    ModRun runMod(int64_t fmodMode, const ModOperand &dividend, const ModOperand &divisor, DType outputDtype, int threads = 1) {
        Graph                 g;
        std::vector<IOTensor> feeds;
        auto                  addOperand = [&](const ModOperand &operand, const std::string &name) {
            TensorDesc d;
            d.name                 = name;
            d.shape                = operand.shape;
            d.dtype                = operand.dtype;
            d.isInitializer        = operand.constant;
            d.isInput              = !operand.constant;
            TensorId     id        = g.addTensor(d);
            const size_t count     = operand.dtype == DType::Int64 ? operand.ints.size() : operand.floats.size();
            const size_t byteWidth = operand.dtype == DType::Int64 ? sizeof(int64_t) : sizeof(float);
            const void  *payload   = operand.dtype == DType::Int64 ? (const void *) operand.ints.data() : (const void *) operand.floats.data();
            if (operand.constant)
            {
                HostBuffer hb;
                hb.resizeElems((int64_t) count, operand.dtype);
                if (count > 0)
                {
                    std::memcpy(hb.bytes.data(), payload, count * byteWidth);
                }
                g.initializers[id] = hb;
            } else
            {
                g.inputs.push_back(id);
                IOTensor feed;
                feed.name  = name;
                feed.shape = operand.shape;
                feed.dtype = operand.dtype;
                feed.data.resize(count * byteWidth);
                if (count > 0)
                {
                    std::memcpy(feed.data.data(), payload, count * byteWidth);
                }
                feeds.push_back(std::move(feed));
            }
            return id;
        };
        TensorId   a = addOperand(dividend, "dividend");
        TensorId   b = addOperand(divisor, "divisor");
        TensorDesc yo;
        yo.name     = "y";
        yo.dtype    = outputDtype;
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        g.outputs   = {y};
        Node n;
        n.type    = OpType::Mod;
        n.name    = "mod";
        n.inputs  = {a, b};
        n.outputs = {y};
        if (fmodMode != kFmodAttributeAbsent)
        {
            n.attr.map["fmod"] = intAttr(fmodMode);
        }
        g.nodes.push_back(n);

        Config cfg;
        cfg.backend    = BackendKind::Cpu;
        cfg.cpuThreads = threads;
        auto   sess    = Session::create(std::move(g), cfg);
        ModRun result;
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return result;
        }
        std::vector<IOTensor> outs;
        result.status = sess->run(feeds, outs);
        if (result.status == Status::Ok && !outs.empty())
        {
            result.shape = outs[0].shape;
            result.bytes = outs[0].data;
        }
        return result;
    }

    // A negative NaN with a payload: every NaN operand still yields the canonical positive quiet NaN.
    constexpr uint32_t kNegativeNaNBits = 0xFFC00001u;

    const float   kInf      = std::numeric_limits<float>::infinity();
    const float   kNaN      = std::numeric_limits<float>::quiet_NaN();
    const int64_t kInt64Min = std::numeric_limits<int64_t>::min();
    const int64_t kInt64Max = std::numeric_limits<int64_t>::max();

    // Every result is compared by bit pattern: -0 vs +0 and the canonical NaN are part of the contract.
    void expectFloatBits(const std::vector<float> &got, const std::vector<float> &expected, const std::string &what) {
        ASSERT_EQ(got.size(), expected.size()) << what;
        for (size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_EQ(floatBitsToUint(got[i]), floatBitsToUint(expected[i])) << what << " element " << i << ": got " << got[i] << ", expected " << expected[i];
        }
    }

} // namespace

// Both fmod modes over all four sign combinations of integer-valued and fractional operands. fmod 1
// takes the sign of the dividend (C fmod), fmod 0 the sign of the divisor; an absent attribute is fmod 0.
TEST(ModOps, FloatRemainderSignCombinationsBothModes) {
    const std::vector<float> dividends {7, 7, -7, -7, 7.5f, 7.5f, -7.5f, -7.5f};
    const std::vector<float> divisors {3, -3, 3, -3, 2, -2, 2, -2};
    const std::vector<float> truncExpected {1, 1, -1, -1, 1.5f, 1.5f, -1.5f, -1.5f};
    const std::vector<float> floorExpected {1, -2, 2, -1, 1.5f, -0.5f, 0.5f, -1.5f};
    const Shape              shape {8};
    for (bool constantDivisor: {false, true})
    {
        ModRun trunc = runMod(cpu::kModTruncRemainder, floatOperand(shape, dividends, false), floatOperand(shape, divisors, constantDivisor), DType::Float32);
        ASSERT_EQ(trunc.status, Status::Ok);
        EXPECT_EQ(trunc.shape, shape);
        expectFloatBits(trunc.floats(), truncExpected, "fmod 1");
        for (int64_t floorMode: {cpu::kModFloorRemainder, kFmodAttributeAbsent})
        {
            ModRun floor = runMod(floorMode, floatOperand(shape, dividends, false), floatOperand(shape, divisors, constantDivisor), DType::Float32);
            ASSERT_EQ(floor.status, Status::Ok);
            expectFloatBits(floor.floats(), floorExpected, floorMode == kFmodAttributeAbsent ? "fmod absent" : "fmod 0");
        }
    }
}

// The int64 path (both operands Int64) over the same sign combinations plus exact multiples.
TEST(ModOps, Int64RemainderSignCombinationsBothModes) {
    const std::vector<int64_t> dividends {7, 7, -7, -7, 6, -6, 0};
    const std::vector<int64_t> divisors {3, -3, 3, -3, -3, 3, -5};
    const std::vector<int64_t> truncExpected {1, 1, -1, -1, 0, 0, 0};
    const std::vector<int64_t> floorExpected {1, -2, 2, -1, 0, 0, 0};
    const Shape                shape {7};
    for (bool constantDivisor: {false, true})
    {
        ModRun trunc = runMod(cpu::kModTruncRemainder, int64Operand(shape, dividends, false), int64Operand(shape, divisors, constantDivisor), DType::Int64);
        ASSERT_EQ(trunc.status, Status::Ok);
        EXPECT_EQ(trunc.ints(), truncExpected);
        ModRun floor = runMod(cpu::kModFloorRemainder, int64Operand(shape, dividends, false), int64Operand(shape, divisors, constantDivisor), DType::Int64);
        ASSERT_EQ(floor.status, Status::Ok);
        EXPECT_EQ(floor.ints(), floorExpected);
    }
}

// A signed-zero dividend and a zero remainder keep the dividend's sign in both modes: fmod 0 applies
// its fix-up only to a NONZERO remainder.
TEST(ModOps, SignedZeroDividendAndExactMultiplesKeepTheDividendSign) {
    const std::vector<float> dividends {-0.0f, 0.0f, -0.0f, 0.0f, -6, 6, -6, 6};
    const std::vector<float> divisors {3, 3, -3, -3, 3, 3, -3, -3};
    const std::vector<float> expected {-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f};
    const Shape              shape {8};
    for (int64_t fmodMode: {cpu::kModFloorRemainder, cpu::kModTruncRemainder})
    {
        ModRun run = runMod(fmodMode, floatOperand(shape, dividends, false), floatOperand(shape, divisors, false), DType::Float32);
        ASSERT_EQ(run.status, Status::Ok);
        expectFloatBits(run.floats(), expected, "fmod " + std::to_string(fmodMode));
    }
}

// Zero divisors: the int64 path yields 0 in both modes; the float path yields the canonical NaN for
// fmod 1 (std::fmod) and +0 for fmod 0 (integer semantics), whatever the dividend, NaN and inf included.
TEST(ModOps, ZeroDivisor) {
    const Shape                floatShape {6};
    const std::vector<float>   floatDividends {5, -5, 0, kNaN, kInf, -0.0f};
    const std::vector<float>   floatDivisors {0.0f, -0.0f, 0.0f, 0.0f, 0.0f, -0.0f};
    const std::vector<float>   allNaN(floatShape[0], uintBitsToFloat(kQuietNaNBits));
    const std::vector<float>   allPositiveZero(floatShape[0], 0.0f);
    const Shape                intShape {5};
    const std::vector<int64_t> intDividends {5, -5, 0, kInt64Min, kInt64Max};
    const std::vector<int64_t> intDivisors(intShape[0], 0);
    for (int64_t fmodMode: {cpu::kModFloorRemainder, cpu::kModTruncRemainder})
    {
        ModRun floatRun = runMod(fmodMode, floatOperand(floatShape, floatDividends, false), floatOperand(floatShape, floatDivisors, true), DType::Float32);
        ASSERT_EQ(floatRun.status, Status::Ok);
        expectFloatBits(floatRun.floats(), fmodMode == cpu::kModTruncRemainder ? allNaN : allPositiveZero, "float fmod " + std::to_string(fmodMode));
        ModRun intRun = runMod(fmodMode, int64Operand(intShape, intDividends, false), int64Operand(intShape, intDivisors, true), DType::Int64);
        ASSERT_EQ(intRun.status, Status::Ok);
        EXPECT_EQ(intRun.ints(), std::vector<int64_t>(intShape[0], 0)) << "int64 fmod " << fmodMode;
    }
}

// Infinite and NaN operands: an infinite or NaN dividend or a NaN divisor gives the canonical positive
// quiet NaN (even from a negative NaN operand); a finite dividend over an infinite divisor is the
// dividend (fmod 1), which fmod 0 moves by the infinite divisor when their signs differ.
TEST(ModOps, InfinityAndNaNOperands) {
    const float              negativeNaN = uintBitsToFloat(kNegativeNaNBits);
    const float              nan         = uintBitsToFloat(kQuietNaNBits);
    const std::vector<float> dividends {kInf, -kInf, kNaN, 5, 5, -5, -5, 5, 0, -0.0f, negativeNaN, 5};
    const std::vector<float> divisors {3, 3, 3, kNaN, kInf, kInf, -kInf, -kInf, kInf, -kInf, 3, negativeNaN};
    const std::vector<float> truncExpected {nan, nan, nan, nan, 5, -5, -5, 5, 0, -0.0f, nan, nan};
    const std::vector<float> floorExpected {nan, nan, nan, nan, 5, kInf, -5, -kInf, 0, -0.0f, nan, nan};
    const Shape              shape {12};
    ModRun trunc = runMod(cpu::kModTruncRemainder, floatOperand(shape, dividends, false), floatOperand(shape, divisors, false), DType::Float32);
    ASSERT_EQ(trunc.status, Status::Ok);
    expectFloatBits(trunc.floats(), truncExpected, "fmod 1");
    ModRun floor = runMod(cpu::kModFloorRemainder, floatOperand(shape, dividends, false), floatOperand(shape, divisors, false), DType::Float32);
    ASSERT_EQ(floor.status, Status::Ok);
    expectFloatBits(floor.floats(), floorExpected, "fmod 0");
}

// INT64_MIN % -1 overflows in C++; the int64 path defines it as its exact value 0. The neighbouring
// extremes check the fix-up never leaves the int64 range.
TEST(ModOps, Int64MinByMinusOneIsZero) {
    const std::vector<int64_t> dividends {kInt64Min, kInt64Min, kInt64Min, kInt64Max, kInt64Min, kInt64Max};
    const std::vector<int64_t> divisors {-1, 1, 7, -1, kInt64Max, kInt64Min};
    // 2^63 = (2^3)^21 and 2^3 = 1 (mod 7), so -2^63 % 7 = -1; -2^63 = -(2^63 - 1) - 1, so INT64_MIN %
    // INT64_MAX = -1.
    const std::vector<int64_t> truncExpected {0, 0, -1, 0, -1, kInt64Max};
    const std::vector<int64_t> floorExpected {0, 0, 6, 0, kInt64Max - 1, -1};
    const Shape                shape {6};
    ModRun trunc = runMod(cpu::kModTruncRemainder, int64Operand(shape, dividends, false), int64Operand(shape, divisors, true), DType::Int64);
    ASSERT_EQ(trunc.status, Status::Ok);
    EXPECT_EQ(trunc.ints(), truncExpected);
    ModRun floor = runMod(cpu::kModFloorRemainder, int64Operand(shape, dividends, false), int64Operand(shape, divisors, true), DType::Int64);
    ASSERT_EQ(floor.status, Status::Ok);
    EXPECT_EQ(floor.ints(), floorExpected);
}

// Values past fp32's 24-bit mantissa stay exact on the int64 path (2^24 + 1 would read as 2^24 in
// fp32, and 2^24 + 1 mod 2 would become 0). Reference values from Python's % (floor) and a
// sign-of-dividend remainder.
TEST(ModOps, Int64ValuesAboveFloatMantissaStayExact) {
    const std::vector<int64_t> dividends {(1LL << 60) + 12345, -((1LL << 60) + 12345), (1LL << 53) + 1, (1LL << 24) + 1, -((1LL << 40) + 7), (1LL << 62) + 3};
    const std::vector<int64_t> divisors {(1LL << 31) + 11, (1LL << 31) + 11, 1LL << 26, 2, 1000003, -((1LL << 33) + 5)};
    const std::vector<int64_t> truncExpected {536883290, -536883290, 1, 1, -329259, 5905580040LL};
    const std::vector<int64_t> floorExpected {536883290, 1610600369, 1, 1, 670744, -2684354557LL};
    const Shape                shape {6};
    ModRun trunc = runMod(cpu::kModTruncRemainder, int64Operand(shape, dividends, false), int64Operand(shape, divisors, false), DType::Int64);
    ASSERT_EQ(trunc.status, Status::Ok);
    EXPECT_EQ(trunc.ints(), truncExpected);
    ModRun floor = runMod(cpu::kModFloorRemainder, int64Operand(shape, dividends, false), int64Operand(shape, divisors, false), DType::Int64);
    ASSERT_EQ(floor.status, Status::Ok);
    EXPECT_EQ(floor.ints(), floorExpected);
}

// One Int64 operand selects the int64 path: the fp32 operand truncates toward zero (NaN reads as 0),
// and the result is int64 in both operand orders.
TEST(ModOps, MixedInt64AndFloatOperandsTakeTheInt64Path) {
    const Shape shape {4};
    // Int64 dividend, fp32 divisor 2.9 -> 2 and -4.5 -> -4.
    ModRun intOverFloat = runMod(cpu::kModTruncRemainder, int64Operand(shape, {7, -7, 9, -9}, false), floatOperand(shape, {2.9f, 2.9f, -4.5f, -4.5f}, true), DType::Int64);
    ASSERT_EQ(intOverFloat.status, Status::Ok);
    EXPECT_EQ(intOverFloat.ints(), (std::vector<int64_t> {1, -1, 1, -1}));
    intOverFloat = runMod(cpu::kModFloorRemainder, int64Operand(shape, {7, -7, 9, -9}, false), floatOperand(shape, {2.9f, 2.9f, -4.5f, -4.5f}, true), DType::Int64);
    ASSERT_EQ(intOverFloat.status, Status::Ok);
    EXPECT_EQ(intOverFloat.ints(), (std::vector<int64_t> {1, 1, -3, -1}));
    // fp32 dividend -7.9 -> -7, 7.9 -> 7, NaN -> 0, 16777216 -> 16777216 over an Int64 divisor.
    ModRun floatOverInt = runMod(cpu::kModTruncRemainder, floatOperand(shape, {-7.9f, 7.9f, kNaN, 16777216.0f}, false), int64Operand(shape, {3, -3, 3, 5}, true), DType::Int64);
    ASSERT_EQ(floatOverInt.status, Status::Ok);
    EXPECT_EQ(floatOverInt.ints(), (std::vector<int64_t> {-1, 1, 0, 1}));
    floatOverInt = runMod(cpu::kModFloorRemainder, floatOperand(shape, {-7.9f, 7.9f, kNaN, 16777216.0f}, false), int64Operand(shape, {3, -3, 3, 5}, true), DType::Int64);
    ASSERT_EQ(floatOverInt.status, Status::Ok);
    EXPECT_EQ(floatOverInt.ints(), (std::vector<int64_t> {2, -2, 0, 1}));
}

// NumPy broadcasting: a row operand, operands broadcasting against each other on different axes, a
// rank-0 and a 1-element divisor, a rank-0 dividend, and a zero-extent axis. Expected values come
// from an independent per-element coordinate unravel.
TEST(ModOps, BroadcastShapes) {
    auto broadcastReference = [](const Shape &out, const Shape &sa, const std::vector<float> &a, const Shape &sb, const std::vector<float> &b, bool floorRemainder) {
        std::vector<float> expected;
        const int64_t      total = out.empty() ? 1 : numElements(out);
        for (int64_t flat = 0; flat < total; ++flat)
        {
            auto sourceIndex = [&](const Shape &s) {
                int64_t rem = flat, index = 0, stride = 1;
                for (int axis = (int) out.size() - 1; axis >= 0; --axis)
                {
                    const int64_t coord = rem % out[axis];
                    rem /= out[axis];
                    const int operandAxis = axis - ((int) out.size() - (int) s.size());
                    if (operandAxis >= 0)
                    {
                        index += (s[operandAxis] == 1 ? 0 : coord) * stride;
                        stride *= s[operandAxis];
                    }
                }
                return index;
            };
            expected.push_back(cpu::modRemainderFloat(a[sourceIndex(sa)], b[sourceIndex(sb)], floorRemainder));
        }
        return expected;
    };
    struct Case {
        const char        *what;
        Shape              dividendShape;
        bool               dividendConstant;
        Shape              divisorShape;
        bool               divisorConstant;
        Shape              outShape;
        std::vector<float> dividends, divisors;
    };
    const std::vector<Case> cases {
        {"row divisor", {2, 3}, false, {3}, true, {2, 3}, {1, 2, 3, -4, 5, -6}, {2, -3, 4}},
        {"mutual broadcast", {2, 1, 3}, false, {4, 1}, false, {2, 4, 3}, {7, -8, 9, 10.5f, -11, 12}, {3, -4, 5, -2.5f}},
        {"rank-0 divisor", {2, 2}, false, {}, true, {2, 2}, {9, -9, 10, -10}, {-4}},
        {"1-element divisor", {2, 3}, false, {1}, true, {2, 3}, {9, -9, 10, -10, 11, -11}, {4}},
        {"rank-0 dividend", {}, true, {4}, false, {4}, {-17}, {5, -5, 3, -3}},
    };
    for (const Case &c: cases)
    {
        for (int64_t fmodMode: {cpu::kModFloorRemainder, cpu::kModTruncRemainder})
        {
            ModRun run = runMod(fmodMode, floatOperand(c.dividendShape, c.dividends, c.dividendConstant), floatOperand(c.divisorShape, c.divisors, c.divisorConstant), DType::Float32);
            ASSERT_EQ(run.status, Status::Ok) << c.what;
            EXPECT_EQ(run.shape, c.outShape) << c.what;
            expectFloatBits(run.floats(), broadcastReference(c.outShape, c.dividendShape, c.dividends, c.divisorShape, c.divisors, fmodMode == cpu::kModFloorRemainder), std::string(c.what) + " fmod " + std::to_string(fmodMode));
        }
    }
    // A zero extent broadcasts against a size-1 axis to an empty result.
    ModRun empty = runMod(cpu::kModFloorRemainder, floatOperand({1, 3}, {1, 2, 3}, false), floatOperand({0, 1}, {}, true), DType::Float32);
    ASSERT_EQ(empty.status, Status::Ok);
    EXPECT_EQ(empty.shape, (Shape {0, 3}));
    EXPECT_TRUE(empty.bytes.empty());
}

// Invalid nodes fail the run with InvalidArgument rather than computing a value: an fmod outside
// {0, 1}, and operand shapes that do not broadcast.
TEST(ModOps, InvalidNodesFailTheRun) {
    const Shape shape {3};
    ModRun      badFmod = runMod(2, floatOperand(shape, {1, 2, 3}, false), floatOperand(shape, {2, 2, 2}, true), DType::Float32);
    EXPECT_NE(badFmod.status, Status::Ok);
    ModRun badShapes = runMod(cpu::kModTruncRemainder, floatOperand(shape, {1, 2, 3}, false), floatOperand({2}, {2, 2}, true), DType::Float32);
    EXPECT_NE(badShapes.status, Status::Ok);
}

// Mod over Int64 initializers folds at load: the node disappears and its output becomes an Int64
// initializer holding the exact remainders (values past 2^24 included), in both modes; two rank-0
// operands fold to a rank-0 result.
TEST(ModOps, Int64InitializersConstantFold) {
    Graph g;
    auto  addI64 = [&](const char *name, const Shape &shape, const std::vector<int64_t> &values) {
        TensorDesc d;
        d.name          = name;
        d.shape         = shape;
        d.isInitializer = true;
        d.dtype         = DType::Int64;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) values.size(), DType::Int64);
        for (size_t i = 0; i < values.size(); ++i)
        {
            hb.i64()[i] = values[i];
        }
        g.initializers[id] = hb;
        return id;
    };
    TensorId dividends = addI64("dividends", {6}, {(1LL << 40) + 7, -((1LL << 40) + 7), 17, kInt64Min, -17, 5});
    TensorId divisors  = addI64("divisors", {6}, {(1LL << 25) + 1, (1LL << 25) + 1, 0, -1, 5, -3});
    TensorId scalarA   = addI64("scalar_a", {}, {-9});
    TensorId scalarB   = addI64("scalar_b", {}, {4});
    auto     addMod    = [&](const char *name, TensorId a, TensorId b, int64_t fmodMode) {
        TensorId out = g.addTensor({name});
        Node     n;
        n.type             = OpType::Mod;
        n.name             = name;
        n.inputs           = {a, b};
        n.outputs          = {out};
        n.attr.map["fmod"] = intAttr(fmodMode);
        g.nodes.push_back(n);
        g.outputs.push_back(out);
        return out;
    };
    TensorId truncOut  = addMod("mod_trunc", dividends, divisors, cpu::kModTruncRemainder);
    TensorId floorOut  = addMod("mod_floor", dividends, divisors, cpu::kModFloorRemainder);
    TensorId scalarOut = addMod("mod_scalar", scalarA, scalarB, cpu::kModFloorRemainder);

    inferShapes(g, 1);
    ASSERT_EQ(g.desc(truncOut).shape, (Shape {6}));
    constFold(g);
    EXPECT_TRUE(g.nodes.empty()) << "all-constant Mod nodes must fold away";
    for (TensorId out: {truncOut, floorOut, scalarOut})
    {
        ASSERT_TRUE(g.isInitializer(out)) << g.desc(out).name;
        EXPECT_EQ(g.desc(out).dtype, DType::Int64) << g.desc(out).name;
    }
    // (2^40 + 7) mod (2^25 + 1): 2^40 = 2^15 * (2^25 + 1) - 2^15, so the C remainder is
    // (2^25 + 1) - 2^15 + 7 = 33521672, and the floor remainder of the negated dividend is 2^15 - 7.
    const std::vector<int64_t> truncExpected {33521672, -33521672, 0, 0, -2, 2};
    const std::vector<int64_t> floorExpected {33521672, 32761, 0, 0, 3, -1};
    for (size_t i = 0; i < truncExpected.size(); ++i)
    {
        EXPECT_EQ(g.initializers[truncOut].i64()[i], truncExpected[i]) << "fmod 1 element " << i;
        EXPECT_EQ(g.initializers[floorOut].i64()[i], floorExpected[i]) << "fmod 0 element " << i;
    }
    EXPECT_TRUE(g.desc(scalarOut).shape.empty()) << "two rank-0 operands fold to a rank-0 result";
    EXPECT_EQ(g.initializers[scalarOut].i64()[0], 3); // -9 mod 4, sign of the divisor
}

// The shader's exact fmod, transcribed, against std::fmod and the CPU oracle over more than a million
// randomized pairs: uniformly random bit patterns (every class: NaN, inf, subnormal, huge), log-uniform
// magnitudes over the whole fp32 exponent span, huge quotients (1e30 / 3e-30 up to FLT_MAX / 2^-149),
// subnormal pairs, integer-valued operands within 2^24, and a cross product of special values, all with
// random signs. A finite fmod 1 result must match std::fmod bit for bit, a NaN must be the canonical
// NaN, and both modes must match cpu::modRemainderFloat bit for bit (the fmod 0 fix-up included).
TEST(ModOps, ShaderExactFmodTranscriptionMatchesStdFmodBitwise) {
    static constexpr uint32_t kFuzzSeed = 20260916u; // a fixed seed keeps the sweep reproducible
    // Binary exponents of nonzero finite fp32 values: the smallest subnormal is 2^-149 and the largest
    // finite value lies below 2^128.
    static constexpr int kMinFloatExponent = -149;
    static constexpr int kMaxFloatExponent = 127;
    // Huge-quotient band: dividends from 2^99 (about 6e29) up, divisors up to 2^-98 (about 3e-30).
    static constexpr int kHugeDividendMinExponent = 99;
    static constexpr int kTinyDivisorMaxExponent  = -98;
    // Subnormal pairs: a dividend of up to 2^20 smallest-subnormal units, a divisor of 1 to 256 units.
    static constexpr uint32_t kSubnormalDividendUnitsMask = 0x000FFFFFu;
    static constexpr uint32_t kSubnormalDivisorUnitsMask  = 0x000000FFu;
    static constexpr int      kPairsPerCategory           = 220000;
    static constexpr int      kRandomPartnersPerSpecial   = 64;
    static constexpr size_t   kMinimumPairs               = 1000000;
    static constexpr int      kReportedMismatches         = 8;
    static constexpr int      kMantissaExactExponent      = 24; // |integer| <= 2^24 is exact in fp32
    static constexpr uint32_t kSignSelectBit              = 1u;

    std::mt19937                            rng(kFuzzSeed);
    std::uniform_int_distribution<uint32_t> anyBits;
    std::uniform_real_distribution<double>  unit(0.0, 1.0);
    auto                                    randomSign = [&](float magnitude) {
        return (anyBits(rng) & kSignSelectBit) ? -magnitude : magnitude;
    };
    auto logUniform = [&](int minExponent, int maxExponent) {
        std::uniform_int_distribution<int> exponent(minExponent, maxExponent);
        const float                        mantissa = (float) (1.0 + unit(rng)); // [1, 2)
        const float                        value    = std::ldexp(mantissa, exponent(rng));
        return std::isinf(value) ? std::numeric_limits<float>::max() : value;
    };
    std::uniform_int_distribution<int> smallInteger(-(1 << kMantissaExactExponent), 1 << kMantissaExactExponent);
    const float                        denormMin = std::numeric_limits<float>::denorm_min();
    const float                        floatMax  = std::numeric_limits<float>::max();
    const float                        floatMin  = std::numeric_limits<float>::min();
    const std::vector<float>           specials {0.0f,       -0.0f,     kInf,     -kInf,     kNaN,       uintBitsToFloat(kNegativeNaNBits),
                                                 floatMax,   -floatMax, floatMin, denormMin, -denormMin, 1.0f,
                                                 -1.0f,      3.0f,      0.5f,     1e30f,     3e-30f,     16777216.0f,
                                                 16777217.0f};

    std::vector<std::pair<float, float>> pairs;
    for (int i = 0; i < kPairsPerCategory; ++i)
    {
        pairs.emplace_back(uintBitsToFloat(anyBits(rng)), uintBitsToFloat(anyBits(rng)));
        pairs.emplace_back(randomSign(logUniform(kMinFloatExponent, kMaxFloatExponent)), randomSign(logUniform(kMinFloatExponent, kMaxFloatExponent)));
        pairs.emplace_back(randomSign(logUniform(kHugeDividendMinExponent, kMaxFloatExponent)), randomSign(logUniform(kMinFloatExponent, kTinyDivisorMaxExponent)));
        pairs.emplace_back(randomSign((float) (anyBits(rng) & kSubnormalDividendUnitsMask) * denormMin), randomSign((float) ((anyBits(rng) & kSubnormalDivisorUnitsMask) + 1u) * denormMin));
        pairs.emplace_back((float) smallInteger(rng), (float) smallInteger(rng));
    }
    for (float dividend: specials)
    {
        for (float divisor: specials)
        {
            pairs.emplace_back(dividend, divisor);
        }
        for (int i = 0; i < kRandomPartnersPerSpecial; ++i)
        {
            float other = uintBitsToFloat(anyBits(rng));
            pairs.emplace_back(dividend, other);
            pairs.emplace_back(other, dividend);
        }
    }
    pairs.emplace_back(floatMax, denormMin); // the most divisor doublings: kMaxDivisorDoublings
    pairs.emplace_back(-floatMax, -3.0f * denormMin);
    pairs.emplace_back(1e30f, 3e-30f);
    ASSERT_GE(pairs.size(), kMinimumPairs);

    int64_t     mismatches = 0;
    std::string firstMismatches;
    auto        report = [&](const char *what, float dividend, float divisor, float got, float expected) {
        if (++mismatches <= kReportedMismatches)
        {
            char line[256];
            std::snprintf(line, sizeof(line), "%s: fmod(%a, %a) got %a (0x%08x) expected %a (0x%08x)\n", what, dividend, divisor, got, floatBitsToUint(got), expected, floatBitsToUint(expected));
            firstMismatches += line;
        }
    };
    for (const auto &pair: pairs)
    {
        const float dividend = pair.first, divisor = pair.second;
        const float truncated = transcribedModRemainder(dividend, divisor, (int) cpu::kModTruncRemainder);
        const float floored   = transcribedModRemainder(dividend, divisor, (int) cpu::kModFloorRemainder);
        const float reference = std::fmod(dividend, divisor);
        if (std::isnan(reference) ? floatBitsToUint(truncated) != kQuietNaNBits : floatBitsToUint(truncated) != floatBitsToUint(reference))
        {
            report("transcription vs std::fmod", dividend, divisor, truncated, reference);
        }
        const float oracleTrunc = cpu::modRemainderFloat(dividend, divisor, false);
        const float oracleFloor = cpu::modRemainderFloat(dividend, divisor, true);
        if (floatBitsToUint(truncated) != floatBitsToUint(oracleTrunc))
        {
            report("fmod 1 transcription vs CPU oracle", dividend, divisor, truncated, oracleTrunc);
        }
        if (floatBitsToUint(floored) != floatBitsToUint(oracleFloor))
        {
            report("fmod 0 transcription vs CPU oracle", dividend, divisor, floored, oracleFloor);
        }
    }
    EXPECT_EQ(mismatches, 0) << firstMismatches;
}

// The doubling cap is exactly the fp32 exponent span, and the worst cases that need all of it (a
// FLT_MAX dividend over the smallest subnormal divisor) still reduce fully; the fmod 0 fix-up of a huge
// quotient remainder matches the oracle.
TEST(ModOps, ShaderExactFmodDoublingCapCoversTheExponentSpan) {
    const float denormMin = std::numeric_limits<float>::denorm_min();
    const float floatMax  = std::numeric_limits<float>::max();
    EXPECT_EQ(kMaxDivisorDoublings, std::ilogb(floatMax) - std::ilogb(denormMin));
    EXPECT_EQ(floatBitsToUint(uintBitsToFloat(kFloatMaxHalfBits) * 2.0f), floatBitsToUint(floatMax));
    struct Extreme {
        float dividend, divisor;
    };
    const std::vector<Extreme> extremes {{floatMax, denormMin}, {-floatMax, denormMin}, {floatMax, -3.0f * denormMin}, {floatMax, std::numeric_limits<float>::min()}, {floatMax, uintBitsToFloat(kFloatMaxHalfBits)}, {floatMax, floatMax}, {1e30f, 3e-30f}, {-1e30f, 3e-30f}, {std::nextafter(floatMax, 0.0f), 7.0f * denormMin}};
    for (const Extreme &e: extremes)
    {
        const float reference = std::fmod(e.dividend, e.divisor);
        EXPECT_EQ(floatBitsToUint(transcribedModRemainder(e.dividend, e.divisor, (int) cpu::kModTruncRemainder)), floatBitsToUint(reference)) << e.dividend << " fmod " << e.divisor;
        EXPECT_EQ(floatBitsToUint(transcribedModRemainder(e.dividend, e.divisor, (int) cpu::kModFloorRemainder)), floatBitsToUint(cpu::modRemainderFloat(e.dividend, e.divisor, true))) << e.dividend << " mod " << e.divisor;
    }
}

// Config::cpuThreads never changes a byte: both the float and the int64 path partition the broadcast
// sweep, and a shape past cpu::kMinChunkOps (7 * 13 * 1447 elements, indivisible by 2/3/5/8) must
// produce identical bytes at every thread count, with the partition proven engaged.
TEST(ModOps, ThreadCountByteInvariance) {
    const Shape          dividendShape {7, 13, 1447};
    const Shape          divisorShape {13, 1};
    const int64_t        dividendCount = numElements(dividendShape);
    const int64_t        divisorCount  = numElements(divisorShape);
    std::vector<float>   floatDividends((size_t) dividendCount), floatDivisors((size_t) divisorCount);
    std::vector<int64_t> intDividends((size_t) dividendCount), intDivisors((size_t) divisorCount);
    // A plain LCG (the CpuThreading suite's) so chunk boundaries land on differing data.
    static constexpr uint32_t kLcgSeed       = 12345u;
    static constexpr uint32_t kLcgMultiplier = 1664525u;
    static constexpr uint32_t kLcgIncrement  = 1013904223u;
    static constexpr int64_t  kLcgMidpoint   = 1LL << 31;
    // Operand spreads: signed dividends (fractional in fp32, past 2^24 in int64) and small signed
    // divisors that include 0, so every remainder rule runs in every chunk.
    static constexpr int64_t kFloatDividendHalfRange = 100000;
    static constexpr float   kFloatDividendScale     = 0.013f;
    static constexpr int64_t kIntDividendScale       = 4099;
    static constexpr int64_t kDivisorHalfRange       = 20;
    static constexpr float   kFloatDivisorScale      = 0.7f;
    uint32_t                 state                   = kLcgSeed;
    auto                     next                    = [&]() {
        state = state * kLcgMultiplier + kLcgIncrement;
        return (int64_t) state - kLcgMidpoint;
    };
    for (int64_t i = 0; i < dividendCount; ++i)
    {
        floatDividends[(size_t) i] = (float) (next() % (kFloatDividendHalfRange + 1)) * kFloatDividendScale;
        intDividends[(size_t) i]   = next() * kIntDividendScale;
    }
    for (int64_t i = 0; i < divisorCount; ++i)
    {
        const int64_t divisor     = next() % (kDivisorHalfRange + 1);
        floatDivisors[(size_t) i] = (float) divisor * kFloatDivisorScale;
        intDivisors[(size_t) i]   = divisor;
    }
    const std::vector<int> threadCounts {2, 3, 5, 8}; // counts that do not divide the element count
    for (int64_t fmodMode: {cpu::kModFloorRemainder, cpu::kModTruncRemainder})
    {
        for (bool int64Path: {false, true})
        {
            const ModOperand dividend  = int64Path ? int64Operand(dividendShape, intDividends, false) : floatOperand(dividendShape, floatDividends, false);
            const ModOperand divisor   = int64Path ? int64Operand(divisorShape, intDivisors, true) : floatOperand(divisorShape, floatDivisors, true);
            const DType      outDtype  = int64Path ? DType::Int64 : DType::Float32;
            const ModRun     reference = runMod(fmodMode, dividend, divisor, outDtype, 1);
            ASSERT_EQ(reference.status, Status::Ok);
            ASSERT_EQ(reference.bytes.size(), (size_t) dividendCount * (int64Path ? sizeof(int64_t) : sizeof(float)));
            for (int threads: threadCounts)
            {
                const int64_t dispatchesBefore = cpu::detail::poolDispatches();
                const ModRun  run              = runMod(fmodMode, dividend, divisor, outDtype, threads);
                ASSERT_EQ(run.status, Status::Ok);
                EXPECT_GT(cpu::detail::poolDispatches(), dispatchesBefore) << "threads=" << threads << ": the partition did not engage";
                EXPECT_TRUE(run.bytes == reference.bytes) << "fmod " << fmodMode << (int64Path ? " int64" : " float") << " threads=" << threads;
            }
        }
    }
}
