// Integer Binary arithmetic, int64-preserving ConvertDtype, Cast to BOOL, and the flat initializer
// upload count.
//
// - Binary and Add on int64 operands (backend/cpu/int64_arithmetic.h): Pow is an exact integer power
//   whose products wrap modulo 2^64, with the negative-exponent rules (1 -> 1, -1 -> +-1 by parity,
//   every other base -> 0); an int64 base with a fractional, infinite or NaN fp32 exponent takes the
//   fp64 power truncated to int64; INT64_MIN / -1 wraps to INT64_MIN; Mul/Sub/Add wrap; an fp32
//   operand truncates toward zero with NaN reading 0 and out-of-range values saturating. A float base
//   raised to an int64 exponent stays a float power. The Vulkan gate keeps int64 Div and int64-base Pow
//   on the CPU op, since the GPU kernels divide and raise in float.
// - ConvertDtype on the CPU copies an int64 tensor as int64 (values past 2^53 survive).
// - Cast to BOOL is a truth test: nonzero (negative, fractional, infinite, NaN) -> 1, +0/-0 -> 0, on
//   the CPU oracle and in shaders/cast.comp's kCastModeBool. GLSL does not run on the host, so the
//   per-element body of cast.comp's main() is transcribed below. A source check pins the shader's lines
//   and mode values to the transcription and to backend/vulkan/ops/cast_modes.h, and sweeps compare the
//   transcription against the CPU oracle: every fp16 bit pattern and a stratified set of fp32 bit
//   patterns for BOOL, a value grid for INT8 and UINT8.
// - uploadInit's element count and payload guard (upload_init_rule.h): a rank-0 Int64/Int8/UInt8
//   initializer counts its payload lanes at the stored width. The host tests cover only this count and
//   guard rule; uploadInit itself runs only on a Vulkan device.
//
// Values run end to end through CPU Sessions (runStandardPasses included) or through constFold.
#include "backend/cpu/int64_arithmetic.h"
#include "backend/cpu/parallel.h"
#include "backend/vulkan/ops/cast_modes.h"
#include "backend/vulkan/ops/upload_init_rule.h"
#include "core/vk_gates.h"
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    // ONNX TensorProto.DataType codes the Cast nodes below target.
    constexpr int64_t kOnnxUInt8 = 2;
    constexpr int64_t kOnnxInt8  = 3;
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
    // castBoolValue and castElement transcribe cast.comp line for line, each C++ line trailed by the
    // shader line it mirrors. The kCast*Lines tables below hold the shader's own source lines (surrounding
    // whitespace trimmed, internal whitespace runs collapsed, blank and comment lines dropped), and
    // CastShaderSource.TranscribedLinesAndModeValuesMatchCastComp checks cast.comp against them, so a
    // shader edit the transcription does not follow fails a test. The kCastMode* values come from
    // backend/vulkan/ops/cast_modes.h, the header cast.cpp dispatches with; the same test checks the
    // shader declares those values.

    // The push-constant block the transcription's `mode`, `lo` and `hi` parameters stand for.
    const std::string kCastPushConstantLine = "layout(push_constant) uniform PC { int total; float lo, hi; int mode; } pc;";

    // The lines between `float castBoolValue(float source) {` and its closing brace.
    const std::vector<std::string> kCastBoolValueBodyLines {
        "return (floatBitsToUint(source) & kCastFloatMagnitudeBits) != 0u ? 1.0 : 0.0;",
    };

    // The lines between `void main() {` and its closing brace.
    const std::vector<std::string> kCastMainBodyLines {
        "uint g = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x;",
        "if (g >= uint(pc.total)) return;",
        "float t = trunc(float(s[g]));",
        "float outv;",
        "if (pc.mode == kCastModeBool) {",
        "outv = castBoolValue(float(s[g]));",
        "} else if (pc.mode == kCastModeInt8Wrap) {",
        "int m = int(t) & 0xFF;",
        "if (m >= 128) m -= 256;",
        "outv = float(m);",
        "} else {",
        "outv = clamp(t, pc.lo, pc.hi);",
        "}",
        "d[g] = STORE(outv);",
    };

    // GLSL floatBitsToUint: the IEEE bit pattern of an fp32 value.
    uint32_t floatBitsToUint(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    // GLSL trunc.
    float glslTrunc(float value) {
        return std::trunc(value);
    }

    // GLSL int(float): truncation toward zero. Like the GLSL conversion it is defined only for values
    // inside the int range; the sweeps feed it nothing else.
    int glslInt(float value) {
        return (int) value;
    }

    // GLSL clamp(x, minVal, maxVal), specified as min(max(x, minVal), maxVal); std::max / std::min pick
    // the same operand as GLSL max / min for every ordered pair.
    float glslClamp(float value, float lo, float hi) {
        return std::min(std::max(value, lo), hi);
    }

    constexpr uint32_t kCastFloatMagnitudeBits = 0x7FFFFFFFu; // const uint kCastFloatMagnitudeBits = 0x7FFFFFFFu;

    // float castBoolValue(float source) {
    float castBoolValue(float source) {
        return (floatBitsToUint(source) & kCastFloatMagnitudeBits) != 0u ? 1.0f : 0.0f; // return (floatBitsToUint(source) & kCastFloatMagnitudeBits) != 0u ? 1.0 : 0.0;
    }

    // main()'s per-element body for one lane: `source` is float(s[g]) (an fp16 lane widens exactly),
    // `mode`, `lo` and `hi` are the push constants, and the return value is what d[g] stores. The g /
    // pc.total dispatch-bound lines have no per-element counterpart. The integer literals are the
    // shader's own.
    float castElement(float source, int mode, float lo, float hi) {
        float t = glslTrunc(source); // float t = trunc(float(s[g]));
        float outv;                  // float outv;
        if (mode == kCastModeBool)   // if (pc.mode == kCastModeBool) {
        {
            outv = castBoolValue(source);     // outv = castBoolValue(float(s[g]));
        } else if (mode == kCastModeInt8Wrap) // } else if (pc.mode == kCastModeInt8Wrap) {
        {
            int m = glslInt(t) & 0xFF; // int m = int(t) & 0xFF;
            if (m >= 128)              // if (m >= 128) m -= 256;
            {
                m -= 256;
            }
            outv = float(m); // outv = float(m);
        } else               // } else {
        {
            outv = glslClamp(t, lo, hi); // outv = clamp(t, pc.lo, pc.hi);
        }
        return outv; // d[g] = STORE(outv);
    }

    // --- end of transcription -------------------------------------------------------------------------

    // cast.cpp leaves the push constant's lo/hi zero for BOOL; the BOOL arm reads neither.
    constexpr float kBoolModeBound = 0.0f;
    // The [lo, hi] bounds cast.cpp passes for INT8 (read by no arm: INT8 wraps) and UINT8 (saturation).
    constexpr float kInt8ModeLo  = -128.0f;
    constexpr float kInt8ModeHi  = 127.0f;
    constexpr float kUInt8ModeLo = 0.0f;
    constexpr float kUInt8ModeHi = 255.0f;

    // shaders/cast.comp in the source tree. CMake globs tests/*.cpp as absolute paths, so __FILE__ names
    // this file inside the tree and the shader sits in the sibling shaders/ directory.
    std::string castShaderPath() {
        const std::string thisFile      = __FILE__;
        const size_t      testsDirEnd   = thisFile.find_last_of("/\\");
        const size_t      sourceRootEnd = testsDirEnd == std::string::npos ? std::string::npos : thisFile.find_last_of("/\\", testsDirEnd - 1);
        if (sourceRootEnd == std::string::npos)
        {
            return "shaders/cast.comp";
        }
        return thisFile.substr(0, sourceRootEnd + 1) + "shaders/cast.comp";
    }

    // A source line with surrounding whitespace trimmed and every internal whitespace run collapsed to
    // one space.
    std::string normalizedSourceLine(const std::string &line) {
        std::string out;
        bool        pendingSpace = false;
        for (char c: line)
        {
            if (c == ' ' || c == '\t' || c == '\r')
            {
                pendingSpace = !out.empty();
                continue;
            }
            if (pendingSpace)
            {
                out.push_back(' ');
                pendingSpace = false;
            }
            out.push_back(c);
        }
        return out;
    }

    // cast.comp's normalized lines without blank and whole-line comment lines; empty when the file
    // cannot be read.
    std::vector<std::string> castShaderLines() {
        std::ifstream            file(castShaderPath());
        std::vector<std::string> lines;
        for (std::string raw; std::getline(file, raw);)
        {
            std::string line = normalizedSourceLine(raw);
            if (!line.empty() && line.rfind("//", 0) != 0)
            {
                lines.push_back(line);
            }
        }
        return lines;
    }

    // The lines strictly between `header` (a line that opens a brace) and the line closing that brace.
    // Empty when `header` is absent or its block never closes.
    std::vector<std::string> blockBody(const std::vector<std::string> &lines, const std::string &header) {
        const auto opening = std::find(lines.begin(), lines.end(), header);
        if (opening == lines.end())
        {
            return {};
        }
        std::vector<std::string> body;
        int64_t                  depth = std::count(header.begin(), header.end(), '{') - std::count(header.begin(), header.end(), '}');
        for (auto it = opening + 1; it != lines.end(); ++it)
        {
            depth += std::count(it->begin(), it->end(), '{') - std::count(it->begin(), it->end(), '}');
            if (depth <= 0)
            {
                return body;
            }
            body.push_back(*it);
        }
        return {};
    }

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
    const float                twoTo63             = 9223372036854775808.0f; // the first positive value that saturates
    const float                minusTwoTo63        = -9223372036854775808.0f;
    const std::vector<float>   floats {quietNaN, infinity, -infinity, 1e30f, -1e30f, -2.7f, 2.7f, largestBelowTwoTo63, twoTo63, minusTwoTo63, -0.0f};
    const std::vector<int64_t> expected {0, kInt64Max, kInt64Min, kInt64Max, kInt64Min, -2, 2, 9223371487098961920LL, kInt64Max, kInt64Min, 0};
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

TEST(IntegerBinaryFixes, FloatToInt64ConversionsSaturateExactlyAtTwoToThe63) {
    // +2^63 is the one positive value where a `>=` and a `>` range test differ: it must saturate, since
    // converting it with (int64_t) is undefined. The largest value below it converts exactly, and -2^63
    // is INT64_MIN exactly.
    const float  twoTo63Fp32  = 9223372036854775808.0f;
    const double twoTo63Fp64  = 9223372036854775808.0;
    const double infinityFp64 = std::numeric_limits<double>::infinity();
    EXPECT_EQ(cpu::int64FromFp32Operand(twoTo63Fp32), kInt64Max);
    EXPECT_EQ(cpu::int64FromFp32Operand(std::nextafter(twoTo63Fp32, 0.0f)), 9223371487098961920LL);
    EXPECT_EQ(cpu::int64FromFp32Operand(-twoTo63Fp32), kInt64Min);
    EXPECT_EQ(cpu::int64FromFp64Result(twoTo63Fp64), kInt64Max);
    EXPECT_EQ(cpu::int64FromFp64Result(std::nextafter(twoTo63Fp64, 0.0)), 9223372036854774784LL);
    EXPECT_EQ(cpu::int64FromFp64Result(-twoTo63Fp64), kInt64Min);
    EXPECT_EQ(cpu::int64FromFp64Result(std::nextafter(-twoTo63Fp64, 0.0)), -9223372036854774784LL);
    EXPECT_EQ(cpu::int64FromFp64Result(std::numeric_limits<double>::quiet_NaN()), 0);
    EXPECT_EQ(cpu::int64FromFp64Result(infinityFp64), kInt64Max);
    EXPECT_EQ(cpu::int64FromFp64Result(-infinityFp64), kInt64Min);
    EXPECT_EQ(cpu::int64FromFp64Result(-2.7), -2);
    EXPECT_EQ(cpu::int64FromFp64Result(24.08), 24);
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

namespace {

    struct MixedPowCase {
        int64_t base;
        float   exponent;
        int64_t expected;
    };

    // An int64 base raised to an fp32 exponent. The first group is ONNX Runtime's output for the same
    // Pow (int64 X, float Y): the fp64 power truncated to int64, with NaN reading 0 and an out-of-range
    // result saturating. The second group has integral exponents, which take the exact integer power
    // (the value an int64 exponent yields): 3^39 keeps all its digits where the fp64 power rounds to
    // ...256, and 0^-1 is the defined 0 of the negative-exponent rule where the fp64 power is +inf.
    std::vector<MixedPowCase> mixedPowCases() {
        const float kQuietNaN = std::numeric_limits<float>::quiet_NaN();
        const float kInfinity = std::numeric_limits<float>::infinity();
        const float kTwoTo63  = 9223372036854775808.0f;
        return {
            {4, 0.5f, 2},                      // ONNX Runtime
            {64, -0.5f, 0},                    // ONNX Runtime: 0.125
            {2, 1.5f, 2},                      // ONNX Runtime: 2.83
            {9, 0.5f, 3},                      // ONNX Runtime
            {3, 2.9f, 24},                     // ONNX Runtime: 24.08
            {2, kQuietNaN, 0},                 // ONNX Runtime: NaN power reads 0
            {2, kInfinity, kInt64Max},         // ONNX Runtime: +inf saturates
            {2, -kInfinity, 0},                // ONNX Runtime
            {1, kQuietNaN, 1},                 // ONNX Runtime: pow(1, NaN) == 1
            {-8, 1.0f / 3.0f, 0},              // ONNX Runtime: a negative base to a fraction is NaN
            {16, 0.25f, 2},                    // ONNX Runtime
            {2, -2.5f, 0},                     // ONNX Runtime: 0.177
            {-2, 2.5f, 0},                     // ONNX Runtime: NaN
            {2, 63.5f, kInt64Max},             // ONNX Runtime: a result past 2^63 saturates
            {-2, 63.5f, 0},                    // ONNX Runtime: NaN
            {-1, kTwoTo63, 1},                 // ONNX Runtime: an even exponent past the int64 range
            {3, 39.0f, 4052555153018976267LL}, // integral: exact integer power
            {-1, -3.0f, -1},                   // integral: negative odd exponent
            {0, -1.0f, 0},                     // integral: zero base, negative exponent
            {7, -0.0f, 1},                     // integral: -0 is exponent 0
            {2, 63.0f, kInt64Min},             // integral: wraps onto the sign bit
            {-1, -kTwoTo63, 1},                // integral: INT64_MIN is even
        };
    }

} // namespace

TEST(IntegerBinaryFixes, Int64BaseWithFloatExponentKeepsTheFraction) {
    const std::vector<MixedPowCase> cases = mixedPowCases();
    std::vector<int64_t>            bases;
    std::vector<float>              exponents;
    for (const MixedPowCase &c: cases)
    {
        bases.push_back(c.base);
        exponents.push_back(c.exponent);
    }
    const Shape shape {(int64_t) cases.size()};
    // Runtime operands through a CPU Session.
    {
        Graph    g;
        TensorId base     = addGraphInput(g, "base", shape, DType::Int64);
        TensorId exponent = addGraphInput(g, "exponent", shape, DType::Float32);
        TensorId y        = addGraphOutput(g, "y", DType::Int64);
        addBinary(g, BinaryType::Pow, "pow", base, exponent, y);
        std::vector<IOTensor> outs = runOnCpu(std::move(g), {int64Tensor("base", shape, bases), floatTensor("exponent", shape, exponents)});
        ASSERT_EQ(outs.size(), 1u);
        const std::vector<int64_t> got = int64Values(outs[0]);
        ASSERT_EQ(got.size(), cases.size());
        for (size_t k = 0; k < cases.size(); ++k)
        {
            EXPECT_EQ(got[k], cases[k].expected) << "runtime " << cases[k].base << " ^ " << cases[k].exponent;
        }
    }
    // Constant operands through constFold, which runs the same kernel and bakes the result.
    {
        Graph    g;
        TensorId base     = addInt64Constant(g, "base", shape, bases);
        TensorId exponent = addFloatConstant(g, "exponent", shape, exponents);
        TensorId y        = g.addTensor({"y"});
        addBinary(g, BinaryType::Pow, "pow", base, exponent, y);
        g.outputs = {y};
        inferShapes(g, 1);
        constFold(g);
        EXPECT_TRUE(g.nodes.empty()) << "an all-constant int64-base Pow must fold away";
        ASSERT_TRUE(g.isInitializer(y));
        ASSERT_EQ(g.desc(y).dtype, DType::Int64);
        ASSERT_EQ(g.initializers[y].bytes.size(), cases.size() * sizeof(int64_t));
        const int64_t *folded = g.initializers[y].i64();
        for (size_t k = 0; k < cases.size(); ++k)
        {
            EXPECT_EQ(folded[k], cases[k].expected) << "folded " << cases[k].base << " ^ " << cases[k].exponent;
        }
    }
}

TEST(IntegerBinaryFixes, FloatBaseWithInt64ExponentBroadcastsAndTakesARankZeroExponent) {
    // Both operands broadcast: a float base [3] against a runtime int64 exponent [2,1] -> [2,3]. Every
    // expected power is exact in fp32.
    {
        Graph    g;
        TensorId x = addGraphInput(g, "x", {3}, DType::Float32);
        TensorId e = addGraphInput(g, "e", {2, 1}, DType::Int64);
        TensorId y = addGraphOutput(g, "y", DType::Float32);
        addBinary(g, BinaryType::Pow, "pow", x, e, y);
        std::vector<IOTensor> outs = runOnCpu(std::move(g), {floatTensor("x", {3}, {1.5f, -2.0f, 0.5f}), int64Tensor("e", {2, 1}, {2, 3})});
        ASSERT_EQ(outs.size(), 1u);
        EXPECT_EQ(outs[0].shape, (Shape {2, 3}));
        EXPECT_EQ(floatValues(outs[0]), (std::vector<float> {2.25f, 4.0f, 0.25f, 3.375f, -8.0f, 0.125f}));
    }
    // The int64 exponent broadcasts along the last axis of a [2,1] base -> [2,3], negative exponents
    // included.
    {
        Graph    g;
        TensorId x = addGraphInput(g, "x", {2, 1}, DType::Float32);
        TensorId e = addInt64Constant(g, "e", {3}, {-1, 0, 3});
        TensorId y = addGraphOutput(g, "y", DType::Float32);
        addBinary(g, BinaryType::Pow, "pow", x, e, y);
        std::vector<IOTensor> outs = runOnCpu(std::move(g), {floatTensor("x", {2, 1}, {2.0f, -4.0f})});
        ASSERT_EQ(outs.size(), 1u);
        EXPECT_EQ(outs[0].shape, (Shape {2, 3}));
        EXPECT_EQ(floatValues(outs[0]), (std::vector<float> {0.5f, 1.0f, 8.0f, -0.25f, 1.0f, -64.0f}));
    }
    // A rank-0 int64 constant exponent.
    {
        Graph    g;
        TensorId x = addGraphInput(g, "x", {3}, DType::Float32);
        TensorId e = addInt64Constant(g, "square", {}, {2});
        TensorId y = addGraphOutput(g, "y", DType::Float32);
        addBinary(g, BinaryType::Pow, "pow", x, e, y);
        std::vector<IOTensor> outs = runOnCpu(std::move(g), {floatTensor("x", {3}, {1.5f, -2.0f, 4.0f})});
        ASSERT_EQ(outs.size(), 1u);
        EXPECT_EQ(outs[0].shape, (Shape {3}));
        EXPECT_EQ(floatValues(outs[0]), (std::vector<float> {2.25f, 4.0f, 16.0f}));
    }
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
    const std::vector<int64_t> exponents {-2, -1, 0, 1, 2, 3, 4};
    auto                       build = [&] {
        Graph    g;
        TensorId x = addGraphInput(g, "x", shape, DType::Float32);
        TensorId e = addInt64Constant(g, "e", {kCols}, exponents);
        TensorId y = addGraphOutput(g, "y", DType::Float32);
        addBinary(g, BinaryType::Pow, "pow", x, e, y);
        return g;
    };
    const std::vector<IOTensor> reference = runOnCpu(build(), {floatTensor("x", shape, base)}, 1);
    ASSERT_EQ(reference.size(), 1u);
    // The 1-thread reference itself is the broadcast power: element [r, c] is base[r, c] ^ exponents[c].
    const std::vector<float> referenceValues = floatValues(reference[0]);
    ASSERT_EQ(referenceValues.size(), base.size());
    int64_t mismatches = 0, firstMismatch = -1;
    for (int64_t row = 0; row < kRows; ++row)
    {
        for (int64_t col = 0; col < kCols; ++col)
        {
            const int64_t at   = row * kCols + col;
            const float   want = std::pow(base[(size_t) at], (float) exponents[(size_t) col]);
            if (std::memcmp(&referenceValues[(size_t) at], &want, sizeof(float)) != 0)
            {
                firstMismatch = mismatches == 0 ? at : firstMismatch;
                ++mismatches;
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << "first mismatch at flat index " << firstMismatch;
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
        const float shader = castElement(sources[k], kCastModeBool, kBoolModeBound, kBoolModeBound);
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
        const float shader = castElement(sources[bits], kCastModeBool, kBoolModeBound, kBoolModeBound);
        EXPECT_EQ(shader, (float) oracle[bits]) << "half bits 0x" << std::hex << bits;
        zeros += shader == 0.0f ? 1 : 0;
    }
    EXPECT_EQ(zeros, 2) << "exactly +0 and -0 are false";
}

TEST(CastToBool, ShaderBoolArmReadsTheUntruncatedSource) {
    // Every value here truncates to 0 yet is nonzero, so the BOOL arm must test float(s[g]) and not the
    // truncated t: an arm reading t stores 0 for all of them and fails against the CPU oracle.
    const std::vector<float> fractional {
        0.5f,
        -0.5f,
        0.999f,
        -0.25f,
        std::numeric_limits<float>::denorm_min(),
        -std::numeric_limits<float>::denorm_min(),
        std::nextafter(std::numeric_limits<float>::min(), 0.0f),
        -std::nextafter(std::numeric_limits<float>::min(), 0.0f),
        std::numeric_limits<float>::min(),
    };
    const std::vector<uint8_t> oracle = castFloatsToBoolOnCpu(fractional);
    ASSERT_EQ(oracle.size(), fractional.size());
    for (size_t k = 0; k < fractional.size(); ++k)
    {
        const float source = fractional[k];
        ASSERT_EQ(glslTrunc(source), 0.0f) << source;
        EXPECT_EQ(oracle[k], 1u) << source;
        EXPECT_EQ(castElement(source, kCastModeBool, kBoolModeBound, kBoolModeBound), 1.0f) << source;
        EXPECT_EQ(castBoolValue(glslTrunc(source)), 0.0f) << "the truncated operand would read false for " << source;
    }
}

TEST(CastShaderTranscription, IntegerArmsMatchCpuOracleNarrowing) {
    // The INT8 wrap and UINT8 saturate arms of the transcribed body against the CPU Cast op followed by
    // the session's readback narrowing, over a quarter-step grid spanning several wraps and both
    // saturation edges plus large magnitudes inside the int range GLSL int() is defined on.
    static constexpr int kGridMagnitude = 1300; // the grid spans [-1300, 1300]: five INT8 wraps each way
    static constexpr int kStepsPerUnit  = 4;    // quarter steps: fractions of both signs at every integer
    std::vector<float>   values;
    for (int step = -kGridMagnitude * kStepsPerUnit; step <= kGridMagnitude * kStepsPerUnit; ++step)
    {
        values.push_back((float) step / (float) kStepsPerUnit);
    }
    for (float large: {65535.5f, 70000.75f, 1073741824.0f, 2147483520.0f, 16777217.0f})
    {
        values.push_back(large);
        values.push_back(-large);
    }
    values.push_back(-2147483648.0f);
    values.push_back(-0.0f);
    const Shape shape {(int64_t) values.size()};
    Graph       g;
    TensorId    x   = addGraphInput(g, "x", shape, DType::Float32);
    TensorId    yi8 = addGraphOutput(g, "yi8", DType::Int8);
    TensorId    yu8 = addGraphOutput(g, "yu8", DType::UInt8);
    addCast(g, "to_int8", x, yi8, kOnnxInt8);
    addCast(g, "to_uint8", x, yu8, kOnnxUInt8);
    std::vector<IOTensor> outs = runOnCpu(std::move(g), {floatTensor("x", shape, values)});
    ASSERT_EQ(outs.size(), 2u);
    ASSERT_EQ(outs[0].dtype, DType::Int8);
    ASSERT_EQ(outs[1].dtype, DType::UInt8);
    ASSERT_EQ(outs[0].data.size(), values.size());
    ASSERT_EQ(outs[1].data.size(), values.size());
    for (size_t k = 0; k < values.size(); ++k)
    {
        const float int8Oracle  = (float) (int8_t) outs[0].data[k];
        const float uint8Oracle = (float) outs[1].data[k];
        EXPECT_EQ(castElement(values[k], kCastModeInt8Wrap, kInt8ModeLo, kInt8ModeHi), int8Oracle) << values[k];
        EXPECT_EQ(castElement(values[k], kCastModeUInt8Saturate, kUInt8ModeLo, kUInt8ModeHi), uint8Oracle) << values[k];
    }
}

TEST(CastShaderSource, TranscribedLinesAndModeValuesMatchCastComp) {
    const std::vector<std::string> lines = castShaderLines();
#if defined(__ANDROID__)
    if (lines.empty())
    {
        GTEST_SKIP() << "the shader sources are not present on the device";
    }
#endif
    ASSERT_FALSE(lines.empty()) << "cannot read " << castShaderPath();

    // Every mode value cast.cpp dispatches with is the value the shader declares, and the shader declares
    // no other mode.
    struct ModeDeclaration {
        const char *name;
        int         value;
    };
    const std::vector<ModeDeclaration> modes {
        {"kCastModeWide", kCastModeWide},
        {"kCastModeInt8Wrap", kCastModeInt8Wrap},
        {"kCastModeUInt8Saturate", kCastModeUInt8Saturate},
        {"kCastModeBool", kCastModeBool},
    };
    for (const ModeDeclaration &mode: modes)
    {
        const std::string declaration = std::string("const int ") + mode.name + " = " + std::to_string(mode.value) + ";";
        EXPECT_NE(std::find(lines.begin(), lines.end(), declaration), lines.end()) << "cast.comp lacks `" << declaration << "`";
    }
    const std::string modePrefix   = "const int kCastMode";
    auto              declaresMode = [&](const std::string &line) {
        return line.rfind(modePrefix, 0) == 0;
    };
    EXPECT_EQ((size_t) std::count_if(lines.begin(), lines.end(), declaresMode), modes.size());

    // The magnitude mask the transcription uses, the push-constant block its parameters stand for, and
    // the transcribed function bodies.
    static constexpr size_t kLineCapacity = 96;
    char                    magnitudeDeclaration[kLineCapacity];
    std::snprintf(magnitudeDeclaration, sizeof(magnitudeDeclaration), "const uint kCastFloatMagnitudeBits = 0x%08Xu;", (unsigned) kCastFloatMagnitudeBits);
    EXPECT_NE(std::find(lines.begin(), lines.end(), std::string(magnitudeDeclaration)), lines.end()) << "cast.comp lacks `" << magnitudeDeclaration << "`";
    EXPECT_NE(std::find(lines.begin(), lines.end(), kCastPushConstantLine), lines.end()) << "cast.comp lacks `" << kCastPushConstantLine << "`";
    EXPECT_EQ(blockBody(lines, "float castBoolValue(float source) {"), kCastBoolValueBodyLines);
    EXPECT_EQ(blockBody(lines, "void main() {"), kCastMainBodyLines);
}

// --- Vulkan gate for integer Div and Pow ----------------------------------------------------------------

TEST(IntegerBinaryFixes, VulkanGateKeepsInt64DivAndInt64BasePowOnTheCpuOp) {
    // The CPU op computes int64 Div and int64-base Pow as integers (7 / 2 == 3, 2^-1 == 0); the GPU
    // kernels compute them in float (3.5, 0.5), so the gate routes exactly those nodes to the CPU. A
    // float base with an int64 exponent is a float power on both backends, and the other Binary ops
    // yield the same integers, so they stay on the GPU.
    struct GateCase {
        const char *name;
        OpType      type;
        BinaryType  op;
        DType       lhs, rhs;
        const char *backend;
        const char *reason;
    };
    const std::vector<GateCase> cases {
        {"div_i64_i64", OpType::Binary, BinaryType::Div, DType::Int64, DType::Int64, "cpu", "Binary: integer Div on an int64 operand"},
        {"div_f32_i64", OpType::Binary, BinaryType::Div, DType::Float32, DType::Int64, "cpu", "Binary: integer Div on an int64 operand"},
        {"div_i64_f32", OpType::Binary, BinaryType::Div, DType::Int64, DType::Float32, "cpu", "Binary: integer Div on an int64 operand"},
        {"pow_i64_i64", OpType::Binary, BinaryType::Pow, DType::Int64, DType::Int64, "cpu", "Binary: integer Pow on an int64 base"},
        {"pow_i64_f32", OpType::Binary, BinaryType::Pow, DType::Int64, DType::Float32, "cpu", "Binary: integer Pow on an int64 base"},
        {"pow_f32_i64", OpType::Binary, BinaryType::Pow, DType::Float32, DType::Int64, "vulkan", ""},
        {"pow_f32_f32", OpType::Binary, BinaryType::Pow, DType::Float32, DType::Float32, "vulkan", ""},
        {"div_f32_f32", OpType::Binary, BinaryType::Div, DType::Float32, DType::Float32, "vulkan", ""},
        {"mul_i64_i64", OpType::Binary, BinaryType::Mul, DType::Int64, DType::Int64, "vulkan", ""},
        {"sub_i64_i64", OpType::Binary, BinaryType::Sub, DType::Int64, DType::Int64, "vulkan", ""},
        {"max_i64_i64", OpType::Binary, BinaryType::Max, DType::Int64, DType::Int64, "vulkan", ""},
        {"add_i64_i64", OpType::Add, BinaryType::Add, DType::Int64, DType::Int64, "vulkan", ""},
    };
    Graph g;
    for (const GateCase &c: cases)
    {
        TensorDesc lhs, rhs, out;
        lhs.name  = std::string(c.name) + "_lhs";
        lhs.shape = {4};
        lhs.dtype = c.lhs;
        rhs.name  = std::string(c.name) + "_rhs";
        rhs.shape = {4};
        rhs.dtype = c.rhs;
        out.name  = std::string(c.name) + "_out";
        out.shape = {4};
        Node &n   = addNode(g, c.type, c.name, {g.addTensor(lhs), g.addTensor(rhs)}, {g.addTensor(out)});
        n.subOp   = (int) c.op;
    }
    // A rank-0 int64 constant base is an int64 operand too.
    TensorId scalarBase     = addInt64Constant(g, "scalar_base", {}, {2});
    TensorId scalarExponent = addGraphInput(g, "scalar_exponent", {4}, DType::Float32);
    TensorId scalarOut      = g.addTensor({"scalar_pow_out"});
    addBinary(g, BinaryType::Pow, "pow_scalar_i64_f32", scalarBase, scalarExponent, scalarOut);

    const std::vector<NodeSupport> rows = vkSupportSurvey(g);
    ASSERT_EQ(rows.size(), cases.size() + 1);
    for (size_t k = 0; k < cases.size(); ++k)
    {
        EXPECT_EQ(rows[k].node, cases[k].name);
        EXPECT_EQ(rows[k].backend, cases[k].backend) << cases[k].name;
        EXPECT_EQ(rows[k].reason, cases[k].reason) << cases[k].name;
    }
    EXPECT_EQ(rows.back().backend, "cpu");
    EXPECT_EQ(rows.back().reason, "Binary: integer Pow on an int64 base");
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
