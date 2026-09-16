// ONNX Mod: the CPU oracle (src/backend/cpu/ops/mod.cpp over backend/cpu/mod_remainder.h) run through
// a CPU Session, its constant folding, its thread-count byte invariance, the integer element-type
// resolver both kernels and the fp32 pin share (import/mod_integer_operands.h), the Vulkan kernel's
// operand broadcast geometry (backend/vulkan/ops/mod_operand_geometry.h), and a host transcription of
// the GPU kernel's per-element arithmetic (shaders/mod.comp modElement, modRemainder and
// modIntegerOperand) compared bit for bit against std::fmod and the oracle. GLSL never runs on the host,
// so the transcription is what proves the shader's arithmetic; ShaderTranscriptionMatchesCompSource
// checks that it stays token for token the .comp source.
#include "backend/cpu/mod_remainder.h"
#include "backend/cpu/parallel.h"
#include "backend/vulkan/ops/mod_operand_geometry.h"
#include "import/mod_integer_operands.h"
#include "import/passes.h"
#include "vknn/binary_type.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    // ---- host transcription of shaders/mod.comp ---------------------------------------------------

    // Same names and values as the constants of shaders/mod.comp (ShaderTranscriptionMatchesCompSource
    // compares the two sets).
    constexpr int      kModFloorRemainder   = 0;
    constexpr uint32_t kQuietNaNBits        = 0x7FC00000u;
    constexpr uint32_t kSignBit             = 0x80000000u;
    constexpr uint32_t kFloatMaxHalfBits    = 0x7EFFFFFFu;
    constexpr uint32_t kInt64RangeEndBits   = 0x5F000000u;
    constexpr int      kMaxDivisorDoublings = 276;
    constexpr int      kDividendStrideBlock = 1;
    constexpr int      kDivisorStrideBlock  = 2;

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

    // The transcriptions below follow shaders/mod.comp statement for statement: GLSL abs/isnan/isinf/
    // trunc/clamp are std::fabs/std::isnan/std::isinf/std::trunc/std::clamp, the bit casts are the two
    // helpers above, float literals carry the f suffix, and each shader function `name` is
    // `transcribedName` here.

    // Transcription of modIntegerOperand in shaders/mod.comp.
    float transcribedModIntegerOperand(float value) {
        if (std::isnan(value))
        {
            return 0.0f;
        }
        float int64RangeEnd = uintBitsToFloat(kInt64RangeEndBits);
        return std::trunc(std::clamp(value, -int64RangeEnd, int64RangeEnd));
    }

    // Transcription of modRemainder in shaders/mod.comp.
    float transcribedModRemainder(float dividend, float divisor, int fmodMode, bool integerOperands) {
        bool floorRemainder = fmodMode == kModFloorRemainder;
        if (divisor == 0.0f)
        {
            return (floorRemainder || integerOperands) ? 0.0f : uintBitsToFloat(kQuietNaNBits);
        }
        if (std::isnan(dividend) || std::isnan(divisor) || std::isinf(dividend))
        {
            return uintBitsToFloat(kQuietNaNBits);
        }
        if (dividend == 0.0f)
        {
            return integerOperands ? 0.0f : dividend;
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
        if (integerOperands && remainder == 0.0f)
        {
            return 0.0f;
        }
        return remainder;
    }

    // Transcription of modElement in shaders/mod.comp.
    float transcribedModElement(float dividend, float divisor, int fmodMode, bool integerOperands) {
        if (integerOperands)
        {
            return transcribedModRemainder(transcribedModIntegerOperand(dividend), transcribedModIntegerOperand(divisor), fmodMode, integerOperands);
        }
        return transcribedModRemainder(dividend, divisor, fmodMode, integerOperands);
    }

    // ---- token comparison of shaders/mod.comp with the transcription ------------------------------

    std::string readTextFile(const std::string &path) {
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            return {};
        }
        std::ostringstream text;
        text << in.rdbuf();
        return text.str();
    }

    // `source` with every line comment and block comment removed (newlines kept).
    std::string withoutComments(const std::string &source) {
        static constexpr size_t kCommentMarkerLength = 2;
        std::string             code;
        size_t                  at = 0;
        while (at < source.size())
        {
            if (source.compare(at, kCommentMarkerLength, "//") == 0)
            {
                at = source.find('\n', at);
                at = at == std::string::npos ? source.size() : at;
            } else if (source.compare(at, kCommentMarkerLength, "/*") == 0)
            {
                const size_t close = source.find("*/", at + kCommentMarkerLength);
                at                 = close == std::string::npos ? source.size() : close + kCommentMarkerLength;
            } else
            {
                code += source[at++];
            }
        }
        return code;
    }

    // The text of `float name(...) { ... }` in `source`, from the parameter list's '(' through the
    // body's closing '}'; empty when `source` defines no such function.
    std::string floatFunctionText(const std::string &source, const std::string &name) {
        const std::string signature = "float " + name + "(";
        const size_t      start     = source.find(signature);
        if (start == std::string::npos)
        {
            return {};
        }
        const size_t parameters = start + signature.size() - 1;
        const size_t bodyOpen   = source.find('{', parameters);
        int          depth      = 0;
        for (size_t at = bodyOpen; at < source.size(); ++at)
        {
            depth += source[at] == '{' ? 1 : 0;
            if (source[at] == '}' && --depth == 0)
            {
                return source.substr(parameters, at + 1 - parameters);
            }
        }
        return {};
    }

    // The tokens of C-family code, normalized so a GLSL function and its C++ transcription compare
    // equal: `precise` and `std::` are dropped, fabs reads as abs, a float literal loses its f suffix,
    // and `transcribedName` reads as `name`.
    std::vector<std::string> normalizedTokens(const std::string &code) {
        static constexpr const char *kTwoCharacterOperators[] = {
            "<=", ">=", "==", "!=", "&&", "||", "++", "--", "+=", "-=", "*=", "/=", "|=", "&=", "::", "<<", ">>"};
        static constexpr size_t  kTwoCharacters     = 2;
        static const std::string kTranscribedPrefix = "transcribed";
        auto                     wordCharacter      = [](char c) {
            return std::isalnum((unsigned char) c) || c == '_';
        };
        std::vector<std::string> tokens;
        size_t                   at = 0;
        while (at < code.size())
        {
            const char c = code[at];
            if (std::isspace((unsigned char) c))
            {
                ++at;
                continue;
            }
            size_t end = at + 1;
            if (wordCharacter(c))
            {
                // A word, or a number with its fraction and suffix (0x7FC00000u, 2.0f).
                const bool number = std::isdigit((unsigned char) c);
                while (end < code.size() && (wordCharacter(code[end]) || (number && code[end] == '.')))
                {
                    ++end;
                }
            } else
            {
                for (const char *twoCharacterOperator: kTwoCharacterOperators)
                {
                    if (code.compare(at, kTwoCharacters, twoCharacterOperator) == 0)
                    {
                        end = at + kTwoCharacters;
                        break;
                    }
                }
            }
            tokens.push_back(code.substr(at, end - at));
            at = end;
        }
        std::vector<std::string> normalized;
        for (size_t index = 0; index < tokens.size(); ++index)
        {
            std::string token = tokens[index];
            if (token == "precise")
            {
                continue;
            }
            if (token == "std" && index + 1 < tokens.size() && tokens[index + 1] == "::")
            {
                ++index;
                continue;
            }
            if (token == "fabs")
            {
                token = "abs";
            }
            if (token.compare(0, kTranscribedPrefix.size(), kTranscribedPrefix) == 0 && token.size() > kTranscribedPrefix.size() &&
                std::isupper((unsigned char) token[kTranscribedPrefix.size()]))
            {
                token = std::string(1, (char) std::tolower((unsigned char) token[kTranscribedPrefix.size()])) + token.substr(kTranscribedPrefix.size() + 1);
            }
            if (std::isdigit((unsigned char) token[0]) && token.find('.') != std::string::npos && (token.back() == 'f' || token.back() == 'F'))
            {
                token.pop_back();
            }
            normalized.push_back(token);
        }
        return normalized;
    }

    // Every `const <type> <name> = <integer literal>;` declaration of a shader, name -> value.
    std::map<std::string, int64_t> shaderIntegerConstants(const std::string &code) {
        // const, type, name, '=', literal, ';'
        static constexpr size_t        kDeclarationTokens     = 6;
        static constexpr size_t        kNameToken             = 2;
        static constexpr size_t        kAssignToken           = 3;
        static constexpr size_t        kLiteralToken          = 4;
        static constexpr size_t        kEndToken              = 5;
        static constexpr int           kLiteralBaseFromPrefix = 0; // std::stoll reads 0x as hexadecimal
        std::map<std::string, int64_t> constants;
        std::istringstream             lines(code);
        std::string                    line;
        while (std::getline(lines, line))
        {
            const std::vector<std::string> tokens = normalizedTokens(line);
            if (tokens.size() != kDeclarationTokens || tokens[0] != "const" || tokens[kAssignToken] != "=" || tokens[kEndToken] != ";")
            {
                continue;
            }
            std::string literal = tokens[kLiteralToken];
            if (literal.back() == 'u' || literal.back() == 'U')
            {
                literal.pop_back();
            }
            constants[tokens[kNameToken]] = std::stoll(literal, nullptr, kLiteralBaseFromPrefix);
        }
        return constants;
    }

    // ---- single-node CPU session ------------------------------------------------------------------

    // fmod value that leaves the attribute off the node (the ONNX default, floor remainder, applies).
    constexpr int64_t kFmodAttributeAbsent = -1;
    // An fmod value outside ONNX's {0, 1}.
    constexpr int64_t kUnsupportedFmod = 2;
    // ONNX TensorProto.DataType codes a Cast in front of an operand targets.
    constexpr int64_t kOnnxFloat = 1;
    constexpr int64_t kOnnxInt32 = 6;
    constexpr int64_t kOnnxInt64 = 7;
    // ModOperand::castTo value for an operand read directly (no Cast in front of Mod).
    constexpr int64_t kNoCast = -1;

    Attr intAttr(int64_t value) {
        Attr a;
        a.kind = Attr::Int;
        a.i    = value;
        return a;
    }

    // One Mod operand: a runtime graph input or a constant initializer. Float32 operands carry `floats`;
    // Int64, Int32 and UInt8 operands carry `ints` (Int32 and UInt8 only as graph inputs, which the
    // session binds into fp32 lanes). A `castTo` other than kNoCast reads the operand through a Cast to
    // that ONNX type.
    struct ModOperand {
        Shape                shape;
        DType                dtype    = DType::Float32;
        bool                 constant = false;
        std::vector<float>   floats; // payload when dtype is Float32
        std::vector<int64_t> ints;   // payload when dtype is Int64, Int32 or UInt8
        int64_t              castTo = kNoCast;
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

    // A runtime graph input declared Int32 or UInt8 (the session carries it in fp32 lanes).
    ModOperand narrowIntegerInput(DType dtype, Shape shape, std::vector<int64_t> values) {
        ModOperand operand;
        operand.shape = std::move(shape);
        operand.dtype = dtype;
        operand.ints  = std::move(values);
        return operand;
    }

    // The operand's payload in its declared dtype's bytes.
    std::vector<uint8_t> operandBytes(const ModOperand &operand) {
        std::vector<uint8_t> bytes;
        auto                 append = [&](const void *value, size_t width) {
            const auto *first = static_cast<const uint8_t *>(value);
            bytes.insert(bytes.end(), first, first + width);
        };
        if (operand.dtype == DType::Float32)
        {
            for (float value: operand.floats)
            {
                append(&value, sizeof(value));
            }
            return bytes;
        }
        for (int64_t value: operand.ints)
        {
            if (operand.dtype == DType::Int32)
            {
                const int32_t narrow = (int32_t) value;
                append(&narrow, sizeof(narrow));
            } else if (operand.dtype == DType::UInt8)
            {
                const uint8_t narrow = (uint8_t) value;
                append(&narrow, sizeof(narrow));
            } else
            {
                append(&value, sizeof(value));
            }
        }
        return bytes;
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
        std::vector<int32_t> ints32() const {
            std::vector<int32_t> values(bytes.size() / sizeof(int32_t));
            std::memcpy(values.data(), bytes.data(), values.size() * sizeof(int32_t));
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
            d.name                         = name;
            d.shape                        = operand.shape;
            d.dtype                        = operand.dtype;
            d.isInitializer                = operand.constant;
            d.isInput                      = !operand.constant;
            TensorId                   id  = g.addTensor(d);
            const std::vector<uint8_t> raw = operandBytes(operand);
            if (operand.constant)
            {
                EXPECT_TRUE(operand.dtype == DType::Float32 || operand.dtype == DType::Int64) << name << ": constants are Float32 or Int64";
                HostBuffer hb;
                hb.resizeElems((int64_t) (operand.dtype == DType::Int64 ? operand.ints.size() : operand.floats.size()), operand.dtype);
                if (!raw.empty())
                {
                    std::memcpy(hb.bytes.data(), raw.data(), raw.size());
                }
                g.initializers[id] = hb;
            } else
            {
                g.inputs.push_back(id);
                IOTensor feed;
                feed.name  = name;
                feed.shape = operand.shape;
                feed.dtype = operand.dtype;
                feed.data.resize(raw.size());
                if (!raw.empty())
                {
                    std::memcpy(feed.data.data(), raw.data(), raw.size());
                }
                feeds.push_back(std::move(feed));
            }
            if (operand.castTo == kNoCast)
            {
                return id;
            }
            TensorDesc castDesc;
            castDesc.name    = name + "_cast";
            TensorId castOut = g.addTensor(castDesc);
            Node     cast;
            cast.type           = OpType::Cast;
            cast.name           = name + "_cast";
            cast.inputs         = {id};
            cast.outputs        = {castOut};
            cast.attr.map["to"] = intAttr(operand.castTo);
            g.nodes.push_back(cast);
            return castOut;
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
    // 2^63 in fp32, the first magnitude outside the int64 range.
    const float kTwoToThe63 = std::ldexp(1.0f, std::numeric_limits<int64_t>::digits);

    // Every result is compared by bit pattern: -0 vs +0 and the canonical NaN are part of the contract.
    void expectFloatBits(const std::vector<float> &got, const std::vector<float> &expected, const std::string &what) {
        ASSERT_EQ(got.size(), expected.size()) << what;
        for (size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_EQ(floatBitsToUint(got[i]), floatBitsToUint(expected[i])) << what << " element " << i << ": got " << got[i] << ", expected " << expected[i];
        }
    }

    // Row-major index into an operand of shape `operandShape` for output element `outputIndex` of a
    // NumPy broadcast to `outputShape`, by an independent per-element coordinate unravel.
    int64_t broadcastSourceIndex(const Shape &outputShape, const Shape &operandShape, int64_t outputIndex) {
        int64_t remainingIndex = outputIndex, index = 0, stride = 1;
        for (int axis = (int) outputShape.size() - 1; axis >= 0; --axis)
        {
            const int64_t coordinate = remainingIndex % outputShape[axis];
            remainingIndex /= outputShape[axis];
            const int operandAxis = axis - ((int) outputShape.size() - (int) operandShape.size());
            if (operandAxis >= 0)
            {
                index += (operandShape[operandAxis] == 1 ? 0 : coordinate) * stride;
                stride *= operandShape[operandAxis];
            }
        }
        return index;
    }

    // ---- graph builders for the resolver and pin tests ---------------------------------------------

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

    // A constant whose values do not matter to the resolver or the pin: Float32 (how an INT32, INT16 or
    // UINT16 initializer imports) or Int64.
    TensorId addConstant(Graph &g, const std::string &name, Shape shape, DType dtype = DType::Float32) {
        TensorId id              = addTensor(g, name, shape, dtype);
        g.desc(id).isInitializer = true;
        HostBuffer hb;
        hb.resizeElems(std::max<int64_t>(1, numElements(shape)), dtype);
        g.initializers[id] = hb;
        return id;
    }

    void addNode(Graph &g, OpType type, const std::string &name, std::vector<TensorId> inputs, std::vector<TensorId> outputs, const std::map<std::string, int64_t> &attributes = {}, int32_t subOp = 0) {
        Node n;
        n.type    = type;
        n.name    = name;
        n.inputs  = std::move(inputs);
        n.outputs = std::move(outputs);
        n.subOp   = subOp;
        for (const auto &[key, value]: attributes)
        {
            n.attr.map[key] = intAttr(value);
        }
        g.nodes.push_back(std::move(n));
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

    // Append `result = Mod(dividend, divisor, fmod 1)` and resolve it with both resolver overloads,
    // which must agree.
    bool resolveMod(Graph &g, TensorId dividend, TensorId divisor, DType resultDtype = DType::Float32) {
        TensorId result = addTensor(g, "result", {}, resultDtype);
        addNode(g, OpType::Mod, "mod", {dividend, divisor}, {result}, {{"fmod", cpu::kModTruncRemainder}});
        const Node &mod         = g.nodes.back();
        const bool  withIndex   = modOperandsAreInteger(g, mod, modTensorProducers(g));
        const bool  buildsIndex = modOperandsAreInteger(g, mod);
        EXPECT_EQ(withIndex, buildsIndex);
        return withIndex;
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

// ONNX integer element types the engine carries in fp32 lanes take the int64 rules, exactly like Int64
// storage: INT32 graph inputs, a UINT8 graph input over a constant imported as Float32, and an operand
// produced by a Cast to INT32. A zero divisor yields 0 in both modes (not fmod 1's NaN), the value reads
// back exactly in an Int32 or a Float32 output, and a fractional value cast to INT32 truncates first.
TEST(ModOps, IntegerElementTypesInFloatLanesTakeTheInt64Path) {
    const Shape shape {6};
    // 2^24 - 1 is the largest odd integer an fp32 lane holds exactly.
    const std::vector<int64_t> dividends {5, -5, 7, -7, (1LL << 24) - 1, 0};
    const std::vector<int64_t> divisors {0, 0, 3, 3, -2, 0};
    const std::vector<int32_t> truncExpected {0, 0, 1, -1, 1, 0};
    const std::vector<int32_t> floorExpected {0, 0, 1, 2, -1, 0};
    for (int64_t fmodMode: {cpu::kModFloorRemainder, cpu::kModTruncRemainder})
    {
        const std::vector<int32_t> &expected = fmodMode == cpu::kModTruncRemainder ? truncExpected : floorExpected;
        ModRun int32Output = runMod(fmodMode, narrowIntegerInput(DType::Int32, shape, dividends), narrowIntegerInput(DType::Int32, shape, divisors), DType::Int32);
        ASSERT_EQ(int32Output.status, Status::Ok);
        EXPECT_EQ(int32Output.ints32(), expected) << "INT32 inputs, fmod " << fmodMode;
        ModRun floatOutput = runMod(fmodMode, narrowIntegerInput(DType::Int32, shape, dividends), narrowIntegerInput(DType::Int32, shape, divisors), DType::Float32);
        ASSERT_EQ(floatOutput.status, Status::Ok);
        std::vector<float> expectedFloats(expected.begin(), expected.end());
        expectFloatBits(floatOutput.floats(), expectedFloats, "INT32 inputs into a Float32 output, fmod " + std::to_string(fmodMode));
    }

    const Shape byteShape {4};
    const ModRun uint8Dividend = runMod(cpu::kModTruncRemainder, narrowIntegerInput(DType::UInt8, byteShape, {200, 250, 7, 0}), floatOperand(byteShape, {0, 7, 0, 3}, true), DType::Float32);
    ASSERT_EQ(uint8Dividend.status, Status::Ok);
    expectFloatBits(uint8Dividend.floats(), {0, 5, 0, 0}, "UINT8 input over a Float32-imported constant");

    ModOperand castDividend = floatOperand(byteShape, {5.7f, 7.2f, -5.0f, -7.9f}, false);
    castDividend.castTo     = kOnnxInt32;
    const ModRun castTrunc  = runMod(cpu::kModTruncRemainder, castDividend, floatOperand(byteShape, {0, 3, 0, 3}, true), DType::Int32);
    ASSERT_EQ(castTrunc.status, Status::Ok);
    EXPECT_EQ(castTrunc.ints32(), (std::vector<int32_t> {0, 1, 0, -1}));
    const ModRun castFloor = runMod(cpu::kModFloorRemainder, castDividend, floatOperand(byteShape, {0, 3, 0, 3}, true), DType::Int32);
    ASSERT_EQ(castFloor.status, Status::Ok);
    EXPECT_EQ(castFloor.ints32(), (std::vector<int32_t> {0, 1, 0, 2}));

    // The same values cast to FLOAT stay float operands: fmod 1 over a zero divisor is NaN.
    ModOperand floatCastDividend = castDividend;
    floatCastDividend.castTo     = kOnnxFloat;
    const ModRun floatCast       = runMod(cpu::kModTruncRemainder, floatCastDividend, floatOperand(byteShape, {0, 3, 0, 3}, true), DType::Float32);
    ASSERT_EQ(floatCast.status, Status::Ok);
    EXPECT_EQ(floatBitsToUint(floatCast.floats()[0]), kQuietNaNBits);
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
// and the result is int64 in both operand orders. An fp32 operand outside the int64 range saturates
// (+inf, 1e30 and 2^63 to INT64_MAX; -inf, -2^63 and below to INT64_MIN), never an undefined
// conversion.
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

    EXPECT_EQ(cpu::modOperandToInt64(kInf), kInt64Max);
    EXPECT_EQ(cpu::modOperandToInt64(1e30f), kInt64Max);
    EXPECT_EQ(cpu::modOperandToInt64(kTwoToThe63), kInt64Max);
    EXPECT_EQ(cpu::modOperandToInt64(-kInf), kInt64Min);
    EXPECT_EQ(cpu::modOperandToInt64(-kTwoToThe63), kInt64Min);
    EXPECT_EQ(cpu::modOperandToInt64(std::ldexp(1.0f, 62)), 1LL << 62);
    // The largest fp32 below 2^63 is 2^63 - 2^39, still inside the range.
    EXPECT_EQ(cpu::modOperandToInt64(std::nextafter(kTwoToThe63, 0.0f)), (1LL << 62) + ((1LL << 62) - (1LL << 39)));
    // Saturated dividends over Int64 divisors: INT64_MAX = ...807 and INT64_MIN = -...808, so % 10
    // leaves 7 and -8; -9.3e18 lies below -2^63 and 2^63 = 1 (mod 7), so INT64_MIN % 7 = -1.
    const std::vector<float>   saturatingDividends {kInf, -kInf, 1e30f, -9.3e18f};
    const std::vector<int64_t> saturatedDivisors {10, 10, -10, 7};
    ModRun saturatedTrunc = runMod(cpu::kModTruncRemainder, floatOperand(shape, saturatingDividends, false), int64Operand(shape, saturatedDivisors, true), DType::Int64);
    ASSERT_EQ(saturatedTrunc.status, Status::Ok);
    EXPECT_EQ(saturatedTrunc.ints(), (std::vector<int64_t> {7, -8, 7, -1}));
    ModRun saturatedFloor = runMod(cpu::kModFloorRemainder, floatOperand(shape, saturatingDividends, false), int64Operand(shape, saturatedDivisors, true), DType::Int64);
    ASSERT_EQ(saturatedFloor.status, Status::Ok);
    EXPECT_EQ(saturatedFloor.ints(), (std::vector<int64_t> {7, 2, -3, 6}));
}

// NumPy broadcasting on the float path: a row operand, operands broadcasting against each other on
// different axes, a rank-0 and a 1-element divisor, a rank-0 dividend, and a zero-extent axis. Expected
// values come from an independent per-element coordinate unravel.
TEST(ModOps, BroadcastShapes) {
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
            std::vector<float> expected;
            const int64_t      total = c.outShape.empty() ? 1 : numElements(c.outShape);
            for (int64_t outputIndex = 0; outputIndex < total; ++outputIndex)
            {
                expected.push_back(cpu::modRemainderFloat(c.dividends[(size_t) broadcastSourceIndex(c.outShape, c.dividendShape, outputIndex)], c.divisors[(size_t) broadcastSourceIndex(c.outShape, c.divisorShape, outputIndex)], fmodMode == cpu::kModFloorRemainder));
            }
            expectFloatBits(run.floats(), expected, std::string(c.what) + " fmod " + std::to_string(fmodMode));
        }
    }
    // A zero extent broadcasts against a size-1 axis to an empty result.
    ModRun empty = runMod(cpu::kModFloorRemainder, floatOperand({1, 3}, {1, 2, 3}, false), floatOperand({0, 1}, {}, true), DType::Float32);
    ASSERT_EQ(empty.status, Status::Ok);
    EXPECT_EQ(empty.shape, (Shape {0, 3}));
    EXPECT_TRUE(empty.bytes.empty());
}

// NumPy broadcasting on the int64 path, against the same independent unravel with the int64 rules:
// a runtime Int64 tensor over a rank-0 Int64 constant, two runtime Int64 operands broadcasting against
// each other, an Int64 row against a column, and an Int64 tensor over an fp32 rank-0 divisor. Values lie
// past 2^24, so a float fallback or a swapped operand offset changes the answer.
TEST(ModOps, Int64BroadcastShapes) {
    static constexpr int64_t kLarge = 1LL << 40;
    struct Case {
        const char *what;
        ModOperand  dividend, divisor;
        Shape       outShape;
    };
    const std::vector<Case> cases {
        {"runtime Int64 over a rank-0 Int64 constant", int64Operand({2, 3}, {kLarge + 1, -(kLarge + 2), kLarge + 3, -(kLarge + 4), kLarge + 5, -(kLarge + 6)}, false), int64Operand({}, {(1LL << 33) + 3}, true), {2, 3}},
        {"runtime Int64 operands broadcasting against each other", int64Operand({2, 1, 3}, {kLarge + 11, -(kLarge + 13), kLarge + 17, -(kLarge + 19), kLarge + 23, -(kLarge + 29)}, false), int64Operand({4, 1}, {(1LL << 30) + 1, -((1LL << 29) + 3), (1LL << 31) + 5, -((1LL << 28) + 7)}, false), {2, 4, 3}},
        {"Int64 row against a column", int64Operand({3}, {kLarge + 31, -(kLarge + 37), kLarge + 41}, false), int64Operand({2, 1}, {(1LL << 32) + 9, -((1LL << 27) + 1)}, true), {2, 3}},
        {"Int64 tensor over an fp32 rank-0 divisor", int64Operand({2, 3}, {kLarge + 43, -(kLarge + 47), kLarge + 53, -(kLarge + 59), kLarge + 61, -(kLarge + 67)}, false), floatOperand({}, {-7.9f}, true), {2, 3}},
    };
    auto operandValue = [](const ModOperand &operand, int64_t index) {
        return operand.dtype == DType::Int64 ? operand.ints[(size_t) index] : cpu::modOperandToInt64(operand.floats[(size_t) index]);
    };
    for (const Case &c: cases)
    {
        for (int64_t fmodMode: {cpu::kModFloorRemainder, cpu::kModTruncRemainder})
        {
            ModRun run = runMod(fmodMode, c.dividend, c.divisor, DType::Int64);
            ASSERT_EQ(run.status, Status::Ok) << c.what;
            EXPECT_EQ(run.shape, c.outShape) << c.what;
            std::vector<int64_t> expected;
            for (int64_t outputIndex = 0; outputIndex < numElements(c.outShape); ++outputIndex)
            {
                const int64_t dividend = operandValue(c.dividend, broadcastSourceIndex(c.outShape, c.dividend.shape, outputIndex));
                const int64_t divisor  = operandValue(c.divisor, broadcastSourceIndex(c.outShape, c.divisor.shape, outputIndex));
                expected.push_back(cpu::modRemainderInt(dividend, divisor, fmodMode == cpu::kModFloorRemainder));
            }
            EXPECT_EQ(run.ints(), expected) << c.what << " fmod " << fmodMode;
        }
    }
}

// Invalid nodes fail the run with InvalidArgument rather than computing a value: an fmod outside
// {0, 1}, and operand shapes that do not broadcast.
TEST(ModOps, InvalidNodesFailTheRun) {
    const Shape shape {3};
    ModRun      badFmod = runMod(kUnsupportedFmod, floatOperand(shape, {1, 2, 3}, false), floatOperand(shape, {2, 2, 2}, true), DType::Float32);
    EXPECT_NE(badFmod.status, Status::Ok);
    ModRun badShapes = runMod(cpu::kModTruncRemainder, floatOperand(shape, {1, 2, 3}, false), floatOperand({2}, {2, 2}, true), DType::Float32);
    EXPECT_NE(badShapes.status, Status::Ok);
}

// The Vulkan kernel's operand geometry: an operand right-aligns into the output rank when every extent
// is 1 or the output's, and is rejected with InvalidArgument otherwise, before any stride could index
// past its buffer. inferShapes takes the per-axis maximum of operands that do not broadcast ({3} and
// {2} give {3}), so the check runs against the shapes prepare() actually receives.
TEST(ModOps, VulkanOperandGeometryRejectsOperandsThatDoNotBroadcast) {
    EXPECT_EQ(modAlignedOperandExtents("mod", {3}, {2, 3}), (std::vector<int64_t> {1, 3}));
    EXPECT_EQ(modAlignedOperandExtents("mod", {}, {2, 3}), (std::vector<int64_t> {1, 1}));
    EXPECT_EQ(modAlignedOperandExtents("mod", {4, 1}, {2, 4, 3}), (std::vector<int64_t> {1, 4, 1}));
    EXPECT_EQ(modAlignedOperandExtents("mod", {0, 1}, {0, 3}), (std::vector<int64_t> {0, 1}));
    EXPECT_TRUE(modAlignedOperandExtents("mod", {}, {}).empty());

    auto expectRejected = [](const Shape &operandShape, const Shape &outputShape) {
        try
        {
            modAlignedOperandExtents("bad_mod", operandShape, outputShape);
            ADD_FAILURE() << shapeStr(operandShape) << " against " << shapeStr(outputShape) << " must be rejected";
        } catch (const Error &error)
        {
            EXPECT_EQ(error.status(), Status::InvalidArgument);
            EXPECT_NE(std::string(error.what()).find("bad_mod"), std::string::npos) << error.what();
        }
    };
    expectRejected({4}, {3});
    expectRejected({3, 1}, {3}); // higher rank than the output (an unresolved output shape)
    expectRejected({2}, {});
    expectRejected({5}, {0});

    Graph    g;
    TensorId dividend = addInput(g, "dividend", {3});
    TensorId divisor  = addConstant(g, "divisor", {2});
    TensorId result   = addTensor(g, "result", {});
    addNode(g, OpType::Mod, "mod", {dividend, divisor}, {result});
    inferShapes(g);
    ASSERT_EQ(g.desc(result).shape, (Shape {3}));
    EXPECT_EQ(modAlignedOperandExtents("mod", g.desc(dividend).shape, g.desc(result).shape), (std::vector<int64_t> {3}));
    expectRejected(g.desc(divisor).shape, g.desc(result).shape);
}

// modOperandsAreInteger reads the element type from dtypes and producers, following only the operand
// slots that carry the result's element type: never an index, shape, exponent or condition operand.
TEST(ModOps, IntegerOperandResolution) {
    {
        Graph g;
        EXPECT_FALSE(resolveMod(g, addInput(g, "x", {4}), addConstant(g, "k", {4}))) << "float input over a Float32 constant";
    }
    for (DType integerType: {DType::Int32, DType::Int64, DType::Int8, DType::UInt8})
    {
        Graph g;
        EXPECT_TRUE(resolveMod(g, addInput(g, "x", {4}, integerType), addConstant(g, "k", {4}))) << "integer-typed input, dtype " << (int) integerType;
    }
    {
        Graph g;
        EXPECT_TRUE(resolveMod(g, addInput(g, "x", {4}), addConstant(g, "k", {4}, DType::Int64))) << "Int64 constant divisor";
    }
    {
        Graph g;
        EXPECT_TRUE(resolveMod(g, addInput(g, "x", {4}), addConstant(g, "k", {4}), DType::Int32)) << "integer-declared result";
    }
    for (const auto &[to, integer]: std::vector<std::pair<int64_t, bool>> {{kOnnxInt32, true}, {kOnnxInt64, true}, {kOnnxFloat, false}})
    {
        Graph    g;
        TensorId x        = addInput(g, "x", {2, 2}, DType::Int64);
        TensorId cast     = addTensor(g, "cast", {2, 2});
        TensorId shape    = addConstant(g, "shape", {1}, DType::Int64);
        TensorId reshaped = addTensor(g, "reshaped", {4});
        addNode(g, OpType::Cast, "cast", {x}, {cast}, {{"to", to}});
        addNode(g, OpType::Reshape, "reshape", {cast, shape}, {reshaped});
        EXPECT_EQ(resolveMod(g, reshaped, addConstant(g, "k", {4})), integer) << "Cast to " << to << " then Reshape";
    }
    {
        Graph    g;
        TensorId x       = addInput(g, "x", {2, 3});
        TensorId shape   = addTensor(g, "shape", {2});
        TensorId index   = addConstant(g, "index", {}, DType::Int64);
        TensorId element = addTensor(g, "element", {});
        addNode(g, OpType::Shape, "shape", {x}, {shape});
        addNode(g, OpType::Gather, "gather", {shape, index}, {element});
        EXPECT_TRUE(resolveMod(g, element, addConstant(g, "k", {}))) << "Gather of a Shape";
    }
    {
        Graph    g;
        TensorId table = addConstant(g, "table", {16, 4});
        TensorId ids   = addInput(g, "ids", {2}, DType::Int64);
        TensorId rows  = addTensor(g, "rows", {2, 4});
        addNode(g, OpType::Gather, "gather", {table, ids}, {rows});
        EXPECT_FALSE(resolveMod(g, rows, addConstant(g, "k", {4}))) << "Gather of a float table by int64 ids";
    }
    {
        Graph    g;
        TensorId x        = addInput(g, "x", {1, 4});
        TensorId shape    = addConstant(g, "shape", {2}, DType::Int64);
        TensorId expanded = addTensor(g, "expanded", {3, 4});
        addNode(g, OpType::Expand, "expand", {x, shape}, {expanded});
        EXPECT_FALSE(resolveMod(g, expanded, addConstant(g, "k", {4}))) << "Expand by an int64 shape";
    }
    {
        Graph    g;
        TensorId condition = addInput(g, "condition", {4}, DType::UInt8);
        TensorId selected  = addTensor(g, "selected", {4});
        addNode(g, OpType::Where, "where", {condition, addInput(g, "x", {4}), addConstant(g, "y", {4})}, {selected});
        EXPECT_FALSE(resolveMod(g, selected, addConstant(g, "k", {4}))) << "Where over float values";
    }
    {
        Graph    g;
        TensorId power = addTensor(g, "power", {4});
        addNode(g, OpType::Binary, "pow", {addInput(g, "base", {4}), addConstant(g, "exponent", {4}, DType::Int64)}, {power}, {}, (int32_t) BinaryType::Pow);
        EXPECT_FALSE(resolveMod(g, power, addConstant(g, "k", {4}))) << "Pow of a float base by an int64 exponent";
    }
    {
        Graph    g;
        TensorId cast       = addTensor(g, "cast", {4});
        TensorId difference = addTensor(g, "difference", {4});
        addNode(g, OpType::Cast, "cast", {addInput(g, "x", {4})}, {cast}, {{"to", kOnnxInt64}});
        addNode(g, OpType::Binary, "sub", {addConstant(g, "offset", {4}), cast}, {difference}, {}, (int32_t) BinaryType::Sub);
        EXPECT_TRUE(resolveMod(g, difference, addConstant(g, "k", {4}))) << "Sub whose second operand is integer";
    }
    {
        Graph    g;
        TensorId x       = addInput(g, "x", {2, 8});
        TensorId values  = addTensor(g, "values", {2, 3});
        TensorId indices = addTensor(g, "indices", {2, 3});
        addNode(g, OpType::TopK, "topk", {x, addConstant(g, "count", {1}, DType::Int64)}, {values, indices});
        EXPECT_TRUE(resolveMod(g, indices, addConstant(g, "k", {}))) << "TopK indices";
        Graph    valuesGraph;
        TensorId valuesX     = addInput(valuesGraph, "x", {2, 8});
        TensorId valuesOut   = addTensor(valuesGraph, "values", {2, 3});
        TensorId valuesIndex = addTensor(valuesGraph, "indices", {2, 3});
        addNode(valuesGraph, OpType::TopK, "topk", {valuesX, addConstant(valuesGraph, "count", {1}, DType::Int64)}, {valuesOut, valuesIndex});
        EXPECT_FALSE(resolveMod(valuesGraph, valuesOut, addConstant(valuesGraph, "k", {}))) << "TopK values of float data";
    }
    {
        Graph    g;
        TensorId indices = addTensor(g, "indices", {2});
        TensorId joined  = addTensor(g, "joined", {6});
        addNode(g, OpType::ArgMax, "argmax", {addInput(g, "logits", {2, 5})}, {indices});
        addNode(g, OpType::Concat, "concat", {addInput(g, "x", {4}), indices}, {joined}, {{"axis", 0}});
        EXPECT_TRUE(resolveMod(g, joined, addConstant(g, "k", {}))) << "Concat with ArgMax indices";
    }
    for (const Attr::Kind fillKind: {Attr::Ints, Attr::Floats})
    {
        Graph    g;
        TensorId filled = addTensor(g, "filled", {4});
        addNode(g, OpType::ConstantOfShape, "fill", {addConstant(g, "shape", {1}, DType::Int64)}, {filled});
        Attr value;
        value.kind                       = fillKind;
        value.ints                       = {0};
        value.floats                     = {0.0f};
        g.nodes.back().attr.map["value"] = value;
        EXPECT_EQ(resolveMod(g, filled, addConstant(g, "k", {4})), fillKind == Attr::Ints) << "ConstantOfShape fill kind " << (int) fillKind;
    }
    {
        Graph    g;
        TensorId cast  = addTensor(g, "cast", {4});
        TensorId fused = addTensor(g, "fused", {4});
        addNode(g, OpType::Cast, "cast", {addInput(g, "x", {4})}, {cast}, {{"to", kOnnxInt32}});
        addNode(g, OpType::Add, "fused_add", {cast, addConstant(g, "bias", {4})}, {fused}, {{"pw_steps", 1}});
        EXPECT_FALSE(resolveMod(g, fused, addConstant(g, "k", {4}))) << "a producer hosting a fused pointwise chain";
    }
}

// The fp32 pin follows the same resolver: Mod fmod 1 over a Cast to INT32 and a constant imported as
// Float32 (an INT32 initializer) pins the Cast output, the Mod operand in front of the kernel and the
// result, so the fp16-segment kernel never rounds an integer above 2^11. The same graph with a Cast to
// FLOAT is a float Mod and keeps the segment's precision.
TEST(ModOps, IntegerModPinsFp32ThroughItsCastProducer) {
    for (const auto &[to, integer]: std::vector<std::pair<int64_t, bool>> {{kOnnxInt32, true}, {kOnnxFloat, false}})
    {
        Graph    g;
        TensorId x      = addInput(g, "x", {1, 8});
        TensorId cast   = addTensor(g, "cast", {1, 8});
        TensorId result = addTensor(g, "result", {1, 8});
        addNode(g, OpType::Cast, "cast", {x}, {cast}, {{"to", to}});
        addNode(g, OpType::Mod, "mod", {cast, addConstant(g, "divisor", {1})}, {result}, {{"fmod", cpu::kModTruncRemainder}});
        g.desc(result).isOutput = true;
        g.outputs.push_back(result);

        planFlatLayoutAndStorage(g, "", nullptr);
        const Node *mod = findNode(g, "mod");
        ASSERT_NE(mod, nullptr);
        EXPECT_EQ(g.desc(result).storeFp32, integer) << "Cast to " << to;
        EXPECT_EQ(g.desc(mod->inputs[0]).storeFp32, integer) << "Cast to " << to;
        EXPECT_EQ(g.desc(cast).storeFp32, integer) << "Cast to " << to;
    }
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

    inferShapes(g);
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

// The shader's per-element arithmetic on float operands, transcribed, against std::fmod and the CPU
// oracle over more than a million randomized pairs: uniformly random bit patterns (every class: NaN,
// inf, subnormal, huge), log-uniform magnitudes over the whole fp32 exponent span, huge quotients (1e30 /
// 3e-30 up to FLT_MAX / 2^-149), subnormal pairs, integer-valued operands within 2^24, and a cross
// product of special values, all with random signs. A finite fmod 1 result must match std::fmod bit for
// bit, a NaN must be the canonical NaN, and both modes must match cpu::modRemainderFloat bit for bit
// (the fmod 0 fix-up included).
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
    static constexpr size_t   kMismatchLineBytes          = 256;

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
    const std::vector<float>           specials  = {0.0f,       -0.0f,     kInf,     -kInf,     kNaN,       uintBitsToFloat(kNegativeNaNBits),
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
            char line[kMismatchLineBytes];
            std::snprintf(line, sizeof(line), "%s: fmod(%a, %a) got %a (0x%08x) expected %a (0x%08x)\n", what, dividend, divisor, got, floatBitsToUint(got), expected, floatBitsToUint(expected));
            firstMismatches += line;
        }
    };
    for (const auto &pair: pairs)
    {
        const float dividend = pair.first, divisor = pair.second;
        const float truncated = transcribedModElement(dividend, divisor, (int) cpu::kModTruncRemainder, false);
        const float floored   = transcribedModElement(dividend, divisor, (int) cpu::kModFloorRemainder, false);
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

// The shader's integer mode, transcribed, against the CPU int64 path (modOperandToInt64 then
// modRemainderInt, converted to fp32) bit for bit: random bit patterns, integers within 2^24 over small
// divisors that include zero, integer magnitudes up to 2^62, fractional values, and a cross product of
// special values (signed zeros, NaN, infinities, -2^63, the largest fp32 below 2^63). Zero divisors
// yield +0 in both modes and no remainder is -0. An operand at or above +2^63 is outside the check:
// the GPU reads it as 2^63, the CPU as INT64_MAX (2^63 - 1), which no fp32 value holds.
TEST(ModOps, ShaderIntegerModeTranscriptionMatchesTheInt64Path) {
    static constexpr uint32_t kFuzzSeed              = 20260917u;
    static constexpr int      kPairsPerCategory      = 80000;
    static constexpr int      kMantissaExactExponent = 24;
    static constexpr int      kSmallDivisorMagnitude = 8;
    static constexpr int      kLargeIntegerExponent  = 62;
    static constexpr double   kFractionalMagnitude   = 1000.0;
    static constexpr size_t   kMinimumCheckedPairs   = 350000;
    static constexpr int      kReportedMismatches    = 8;
    static constexpr uint32_t kSignSelectBit         = 1u;
    static constexpr size_t   kMismatchLineBytes     = 256;

    std::mt19937                            rng(kFuzzSeed);
    std::uniform_int_distribution<uint32_t> anyBits;
    std::uniform_real_distribution<double>  unit(0.0, 1.0);
    std::uniform_int_distribution<int>      smallInteger(-(1 << kMantissaExactExponent), 1 << kMantissaExactExponent);
    std::uniform_int_distribution<int>      smallDivisor(-kSmallDivisorMagnitude, kSmallDivisorMagnitude);
    std::uniform_int_distribution<int>      largeExponent(0, kLargeIntegerExponent);
    auto                                    randomSign = [&](float magnitude) {
        return (anyBits(rng) & kSignSelectBit) ? -magnitude : magnitude;
    };
    auto largeInteger = [&]() {
        return randomSign(std::trunc(std::ldexp((float) (1.0 + unit(rng)), largeExponent(rng))));
    };
    auto fractional = [&]() {
        return (float) ((unit(rng) * 2.0 - 1.0) * kFractionalMagnitude);
    };
    const float largestBelowTwoToThe63 = std::nextafter(kTwoToThe63, 0.0f);
    const std::vector<float> specials {0.0f, -0.0f, kNaN, uintBitsToFloat(kNegativeNaNBits), -kInf, -kTwoToThe63, largestBelowTwoToThe63, -largestBelowTwoToThe63, 1.0f, -1.0f, 0.5f, -0.5f, 3.0f, -3.0f, 16777215.0f, 16777216.0f};

    std::vector<std::pair<float, float>> pairs;
    for (int i = 0; i < kPairsPerCategory; ++i)
    {
        pairs.emplace_back(uintBitsToFloat(anyBits(rng)), uintBitsToFloat(anyBits(rng)));
        pairs.emplace_back((float) smallInteger(rng), (float) smallDivisor(rng));
        pairs.emplace_back(largeInteger(), largeInteger());
        pairs.emplace_back(largeInteger(), (float) smallDivisor(rng));
        pairs.emplace_back(fractional(), fractional());
    }
    for (float dividend: specials)
    {
        for (float divisor: specials)
        {
            pairs.emplace_back(dividend, divisor);
        }
    }

    size_t      checked    = 0;
    int64_t     mismatches = 0;
    std::string firstMismatches;
    for (const auto &pair: pairs)
    {
        const float dividend = pair.first, divisor = pair.second;
        if (dividend >= kTwoToThe63 || divisor >= kTwoToThe63)
        {
            continue;
        }
        ++checked;
        for (int64_t fmodMode: {cpu::kModFloorRemainder, cpu::kModTruncRemainder})
        {
            const float shaderResult = transcribedModElement(dividend, divisor, (int) fmodMode, true);
            const float oracleResult = (float) cpu::modRemainderInt(cpu::modOperandToInt64(dividend), cpu::modOperandToInt64(divisor), fmodMode == cpu::kModFloorRemainder);
            if (floatBitsToUint(shaderResult) != floatBitsToUint(oracleResult) && ++mismatches <= kReportedMismatches)
            {
                char line[kMismatchLineBytes];
                std::snprintf(line, sizeof(line), "fmod %d: mod(%a, %a) got %a (0x%08x) expected %a (0x%08x)\n", (int) fmodMode, dividend, divisor, shaderResult, floatBitsToUint(shaderResult), oracleResult, floatBitsToUint(oracleResult));
                firstMismatches += line;
            }
        }
    }
    ASSERT_GE(checked, kMinimumCheckedPairs);
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
    EXPECT_EQ(floatBitsToUint(uintBitsToFloat(kInt64RangeEndBits)), floatBitsToUint(kTwoToThe63));
    struct Extreme {
        float dividend, divisor;
    };
    const std::vector<Extreme> extremes {{floatMax, denormMin}, {-floatMax, denormMin}, {floatMax, -3.0f * denormMin}, {floatMax, std::numeric_limits<float>::min()}, {floatMax, uintBitsToFloat(kFloatMaxHalfBits)}, {floatMax, floatMax}, {1e30f, 3e-30f}, {-1e30f, 3e-30f}, {std::nextafter(floatMax, 0.0f), 7.0f * denormMin}};
    for (const Extreme &e: extremes)
    {
        const float reference = std::fmod(e.dividend, e.divisor);
        EXPECT_EQ(floatBitsToUint(transcribedModRemainder(e.dividend, e.divisor, (int) cpu::kModTruncRemainder, false)), floatBitsToUint(reference)) << e.dividend << " fmod " << e.divisor;
        EXPECT_EQ(floatBitsToUint(transcribedModRemainder(e.dividend, e.divisor, (int) cpu::kModFloorRemainder, false)), floatBitsToUint(cpu::modRemainderFloat(e.dividend, e.divisor, true))) << e.dividend << " mod " << e.divisor;
    }
}

// shaders/mod.comp and its transcription stay token for token the same (after the normalization of
// normalizedTokens), and the shader's constants carry the transcription's names and values, so a shader
// edit that no longer matches the host-proven arithmetic fails here. Skips when the sources are not
// readable next to this file (a device-side test build).
TEST(ModOps, ShaderTranscriptionMatchesCompSource) {
    const std::string testPath      = __FILE__;
    const size_t      separator     = testPath.find_last_of("/\\");
    const std::string testDirectory = separator == std::string::npos ? std::string(".") : testPath.substr(0, separator);
    const std::string shaderSource  = withoutComments(readTextFile(testDirectory + "/../shaders/mod.comp"));
    const std::string testSource    = withoutComments(readTextFile(testPath));
    if (shaderSource.empty() || testSource.empty())
    {
        GTEST_SKIP() << "shaders/mod.comp or this test's source is not readable from " << testDirectory;
    }
    static constexpr const char *kTranscribedFunctions[][2] = {
        {"modIntegerOperand", "transcribedModIntegerOperand"},
        {"modRemainder", "transcribedModRemainder"},
        {"modElement", "transcribedModElement"},
    };
    for (const auto &function: kTranscribedFunctions)
    {
        const std::string shaderFunction = floatFunctionText(shaderSource, function[0]);
        const std::string transcription  = floatFunctionText(testSource, function[1]);
        ASSERT_FALSE(shaderFunction.empty()) << function[0] << " not found in shaders/mod.comp";
        ASSERT_FALSE(transcription.empty()) << function[1] << " not found in " << testPath;
        EXPECT_EQ(normalizedTokens(shaderFunction), normalizedTokens(transcription)) << function[0] << " and " << function[1] << " differ";
    }
    const std::map<std::string, int64_t> transcribedConstants {
        {"kModFloorRemainder", kModFloorRemainder},
        {"kQuietNaNBits", kQuietNaNBits},
        {"kSignBit", kSignBit},
        {"kFloatMaxHalfBits", kFloatMaxHalfBits},
        {"kInt64RangeEndBits", kInt64RangeEndBits},
        {"kMaxDivisorDoublings", kMaxDivisorDoublings},
        {"kDividendStrideBlock", kDividendStrideBlock},
        {"kDivisorStrideBlock", kDivisorStrideBlock},
    };
    EXPECT_EQ(shaderIntegerConstants(shaderSource), transcribedConstants);
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
