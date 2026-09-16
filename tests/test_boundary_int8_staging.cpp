// Boundary dtype contract for 8-bit graph I/O (core/boundary_convert_rule.h).
//
// A whole-GPU run keeps a caller's uint8/int8 input bytes raw and converts them on the device with the
// boundary_convert variant for the declared dtype; a declared-format dma-buf converts the same way in
// both directions. The variant is chosen by the exact (source, destination) dtype pair, so a dtype the
// Session stages raw must have a compiled variant to both device storage precisions, and a pair with no
// variant must be refused rather than read through another dtype's variant (an int8 payload read by
// the fp32 variant decodes as zeros).
//
// - The rule is swept over every underlying DType value: the storage tags, the variant set, the raw
//   staging dtypes, the 8-bit storage requirement and the boundary storage dtype.
// - CMakeLists.txt's bc_variant lines build exactly the rule's variant set, with the element types and
//   variant defines each pair needs, and the shader tests exactly those defines.
// - GLSL does not run on the host, so boundary_convert.comp's int8 source and int8 destination arms are
//   transcribed below; a source check pins the shader's lines and constants to the transcription. The
//   int8 source arm is compared with the CPU Session's bindInput decode over all 256 bytes, the int8
//   destination arm with the Session's readbackOutput narrowing over every finite fp16 value and an
//   fp32 grid. The host lane decode (bindInput and the Vulkan host upload) is checked against the value
//   the test states for each dtype's lanes, and int64 lanes also against a CPU Session run.
#include "core/boundary_convert_rule.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace vknn;

namespace {

    // DType is a uint8_t enum: sweeping every underlying value covers each enumerator and every value no
    // enumerator names.
    constexpr int kDtypeUnderlyingValueCount = 1 << 8;

    std::vector<DType> everyDtypeValue() {
        std::vector<DType> values;
        for (int raw = 0; raw < kDtypeUnderlyingValueCount; ++raw)
        {
            values.push_back(static_cast<DType>(raw));
        }
        return values;
    }

    bool isRecognizedDtype(DType dtype) {
        return dtypeSize(dtype) > 0;
    }

    bool isFloatStorage(DType dtype) {
        return dtype == DType::Float32 || dtype == DType::Float16;
    }

    // Independent statement of the boundary_convert variant set: every fp32/fp16 pair, each 8-bit dtype
    // to and from both float storages, and uint8 to uint8.
    bool expectedHasVariant(DType source, DType destination) {
        const bool sourceEightBit = source == DType::UInt8 || source == DType::Int8;
        const bool destEightBit   = destination == DType::UInt8 || destination == DType::Int8;
        return (isFloatStorage(source) && isFloatStorage(destination)) || (sourceEightBit && isFloatStorage(destination)) || (isFloatStorage(source) && destEightBit) || (source == DType::UInt8 && destination == DType::UInt8);
    }

    // Variant-name tag of each storage dtype; empty for a dtype without a storage lane.
    std::string expectedTag(DType dtype) {
        switch (dtype)
        {
            case DType::Float32:
                return "f32";
            case DType::Float16:
                return "f16";
            case DType::UInt8:
                return "u8";
            case DType::Int8:
                return "i8";
            default:
                return {};
        }
    }

    // GLSL element type the shader declares for each storage dtype; empty for a dtype without a lane.
    std::string glslElementType(DType dtype) {
        switch (dtype)
        {
            case DType::Float32:
                return "float";
            case DType::Float16:
                return "float16_t";
            case DType::UInt8:
                return "uint8_t";
            case DType::Int8:
                return "int8_t";
            default:
                return {};
        }
    }

    // --- source tree access ---------------------------------------------------------------------------

    // `relative` inside the source tree. CMake globs tests/*.cpp as absolute paths, so __FILE__ names this
    // file inside the tree and the root is its parent directory's parent.
    std::string sourceTreePath(const std::string &relative) {
        const std::string thisFile      = __FILE__;
        const size_t      testsDirEnd   = thisFile.find_last_of("/\\");
        const size_t      sourceRootEnd = testsDirEnd == std::string::npos ? std::string::npos : thisFile.find_last_of("/\\", testsDirEnd - 1);
        if (sourceRootEnd == std::string::npos)
        {
            return relative;
        }
        return thisFile.substr(0, sourceRootEnd + 1) + relative;
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

    // The normalized lines of a source-tree file, without blank lines and lines opening with
    // `commentPrefix`; empty when the file cannot be read.
    std::vector<std::string> sourceLines(const std::string &relative, const std::string &commentPrefix) {
        std::ifstream            file(sourceTreePath(relative));
        std::vector<std::string> lines;
        for (std::string raw; std::getline(file, raw);)
        {
            std::string line = normalizedSourceLine(raw);
            if (!line.empty() && line.rfind(commentPrefix, 0) != 0)
            {
                lines.push_back(line);
            }
        }
        return lines;
    }

    const char *const kShaderRelativePath = "shaders/boundary_convert.comp";
    const char *const kCMakeRelativePath  = "CMakeLists.txt";

    // The lines strictly between the preprocessor line `opening` and the next preprocessor line. Empty
    // when `opening` is absent.
    std::vector<std::string> preprocessorArm(const std::vector<std::string> &lines, const std::string &opening) {
        const auto               start = std::find(lines.begin(), lines.end(), opening);
        std::vector<std::string> body;
        if (start == lines.end())
        {
            return body;
        }
        for (auto it = start + 1; it != lines.end() && it->rfind("#", 0) != 0; ++it)
        {
            body.push_back(*it);
        }
        return body;
    }

    // --- transcription of shaders/boundary_convert.comp ------------------------------------------------
    // int8DestinationStoredValue and int8SourceElement transcribe the INT_DST_I8 and INT_SRC_I8 arms of
    // main() line for line, each C++ line trailed by the shader line it mirrors; `source` stands for the
    // source lane s[encode(pc.srcFmt, n, c, h, w)]. The tables below hold the shader's own lines
    // (normalized, comments dropped) and BoundaryConvertShaderSource checks the shader against them, so a
    // shader edit the transcription does not follow fails a test.

    const std::string kInt8DestinationArmOpening = "#elif defined(INT_DST_I8)";
    const std::string kInt8SourceArmOpening      = "#elif defined(INT_SRC_I8)";

    const std::vector<std::string> kInt8DestinationArmLines {
        "int lowByte = int(float(s[encode(pc.srcFmt, n, c, h, w)])) & kInt8LowByteMask;",
        "d[i] = DST_T(lowByte >= kInt8SignBit ? lowByte - kInt8Modulus : lowByte);",
    };

    const std::vector<std::string> kInt8SourceArmLines {
        "d[i] = DST_T(int(s[encode(pc.srcFmt, n, c, h, w)]));",
    };

    constexpr int kInt8LowByteMask = 0xFF;  // const int kInt8LowByteMask = 0xFF;
    constexpr int kInt8SignBit     = 0x80;  // const int kInt8SignBit     = 0x80;
    constexpr int kInt8Modulus     = 0x100; // const int kInt8Modulus     = 0x100;

    // GLSL int(float): truncation toward zero. Like the GLSL conversion it is defined only for values
    // inside the int range; the sweeps feed it nothing else.
    int glslInt(float value) {
        return (int) value;
    }

    // INT_DST_I8 with SRC_T float (an fp16 source widens exactly to the same float first). Returns the int
    // the arm hands to DST_T (int8_t), so a caller can check it is already inside the int8 range and the
    // store never relies on a narrowing conversion.
    int int8DestinationStoredValue(float source) {
        int lowByte = glslInt(source) & kInt8LowByteMask;                  // int lowByte = int(float(s[encode(pc.srcFmt, n, c, h, w)])) & kInt8LowByteMask;
        return lowByte >= kInt8SignBit ? lowByte - kInt8Modulus : lowByte; // d[i] = DST_T(lowByte >= kInt8SignBit ? lowByte - kInt8Modulus : lowByte);
    }

    // INT_SRC_I8 with DST_T float. The fp16 variant stores the same integer value, which fp16 holds
    // exactly.
    float int8SourceElement(int8_t source) {
        return (float) (int) source; // d[i] = DST_T(int(s[encode(pc.srcFmt, n, c, h, w)]));
    }

    // --- CPU Session references -------------------------------------------------------------------------

    constexpr float kInt32RangeLimit = 2147483648.0f; // 2^31: the first float outside the int range

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

    // One graph input declared `inputDtype` copied by Identity to one output declared `outputDtype`, run
    // on the CPU backend with `payload` as the input bytes. The output is what the Session's bindInput
    // decode followed by its readbackOutput narrowing produce. Empty when the session fails (reported
    // through a non-fatal expectation).
    std::vector<IOTensor> runIdentityOnCpu(DType inputDtype, DType outputDtype, int64_t count, const std::vector<uint8_t> &payload) {
        const Shape shape {count};
        Graph       g;
        TensorId    x = addGraphInput(g, "x", shape, inputDtype);
        TensorId    y = addGraphOutput(g, "y", outputDtype);
        Node        n;
        n.type    = OpType::Identity;
        n.name    = "copy";
        n.inputs  = {x};
        n.outputs = {y};
        g.nodes.push_back(n);
        Config cfg;
        cfg.backend    = BackendKind::Cpu;
        cfg.cpuThreads = 1;
        auto session   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(session);
        if (!session)
        {
            return {};
        }
        IOTensor in;
        in.name  = "x";
        in.shape = shape;
        in.dtype = inputDtype;
        in.data  = payload;
        std::vector<IOTensor> outs;
        EXPECT_EQ(session->run({in}, outs), Status::Ok);
        return outs;
    }

    std::vector<float> float32Payload(const IOTensor &t) {
        EXPECT_EQ(t.dtype, DType::Float32) << t.name;
        std::vector<float> values(t.data.size() / sizeof(float));
        std::memcpy(values.data(), t.data.data(), values.size() * sizeof(float));
        return values;
    }

    std::vector<uint8_t> bytesOf(const float *values, size_t count) {
        std::vector<uint8_t> bytes(count * sizeof(float));
        std::memcpy(bytes.data(), values, bytes.size());
        return bytes;
    }

    bool sameBits(float a, float b) {
        return std::memcmp(&a, &b, sizeof(float)) == 0;
    }

} // namespace

// --- the rule ---------------------------------------------------------------------------------------------

TEST(BoundaryConvertRule, StorageTagsNameExactlyTheShaderLanes) {
    for (DType dtype: everyDtypeValue())
    {
        const char       *tag      = boundaryConvertDtypeTag(dtype);
        const std::string expected = expectedTag(dtype);
        if (expected.empty())
        {
            EXPECT_EQ(tag, nullptr) << "dtype value " << (int) dtype << " has no storage lane in boundary_convert.comp";
        } else
        {
            EXPECT_STREQ(tag, expected.c_str()) << "dtype value " << (int) dtype;
        }
    }
}

TEST(BoundaryConvertRule, VariantSetIsEveryFloatPairAndEachEightBitDtypeToAndFromFloat) {
    const std::vector<DType> dtypes = everyDtypeValue();
    for (DType source: dtypes)
    {
        for (DType destination: dtypes)
        {
            const bool expected = expectedHasVariant(source, destination);
            EXPECT_EQ(boundaryConvertHasVariant(source, destination), expected) << (int) source << " -> " << (int) destination;
            const std::string name = boundaryConvertVariantName(source, destination);
            if (expected)
            {
                EXPECT_EQ(name, "boundary_convert_" + expectedTag(source) + "_" + expectedTag(destination));
            } else
            {
                EXPECT_TRUE(name.empty()) << "a pair with no variant must not name one: " << name;
            }
        }
    }
    // The table lists each pair once.
    std::set<std::pair<DType, DType>> listed;
    for (const BoundaryConvertVariant &variant: kBoundaryConvertVariants)
    {
        EXPECT_TRUE(listed.insert({variant.source, variant.destination}).second) << "duplicate " << dtypeStr(variant.source) << " -> " << dtypeStr(variant.destination);
    }
}

TEST(BoundaryConvertRule, SignedBytesHaveBothDirectionsAtBothDevicePrecisions) {
    // An int8 input is staged raw, so it needs its own variants: read through the fp32 variant, its
    // 1-byte lanes decode as zeros.
    EXPECT_EQ(boundaryConvertVariantName(DType::Int8, DType::Float16), "boundary_convert_i8_f16");
    EXPECT_EQ(boundaryConvertVariantName(DType::Int8, DType::Float32), "boundary_convert_i8_f32");
    EXPECT_EQ(boundaryConvertVariantName(DType::Float16, DType::Int8), "boundary_convert_f16_i8");
    EXPECT_EQ(boundaryConvertVariantName(DType::Float32, DType::Int8), "boundary_convert_f32_i8");
    // Integer dtypes wider than a byte have no lane: they are host-decoded, never converted on the GPU.
    for (DType wide: {DType::Int32, DType::Int64})
    {
        for (DType storage: {DType::Float16, DType::Float32})
        {
            EXPECT_FALSE(boundaryConvertHasVariant(wide, storage)) << dtypeStr(wide);
            EXPECT_FALSE(boundaryConvertHasVariant(storage, wide)) << dtypeStr(wide);
        }
    }
}

TEST(BoundaryConvertRule, RawStagedInputDtypesConvertToBothDeviceStorages) {
    for (DType dtype: everyDtypeValue())
    {
        const bool eightBit = dtype == DType::UInt8 || dtype == DType::Int8;
        EXPECT_EQ(boundaryStagesRawInputBytes(dtype), eightBit) << (int) dtype;
        if (boundaryStagesRawInputBytes(dtype))
        {
            for (bool segmentFp16: {false, true})
            {
                for (bool storeFp32: {false, true})
                {
                    const DType device = boundaryDeviceDtype(segmentFp16, storeFp32);
                    EXPECT_TRUE(boundaryConvertHasVariant(dtype, device)) << dtypeStr(dtype) << " staged raw has no variant to " << dtypeStr(device);
                    EXPECT_TRUE(boundaryConvertNeedsEightBitStorage(dtype, device)) << dtypeStr(dtype);
                }
            }
        }
    }
}

TEST(BoundaryConvertRule, EightBitVariantsRunOnlyWithEightBitStorage) {
    const std::vector<DType> dtypes = everyDtypeValue();
    for (DType source: dtypes)
    {
        for (DType destination: dtypes)
        {
            const bool eightBitSide = source == DType::UInt8 || source == DType::Int8 || destination == DType::UInt8 || destination == DType::Int8;
            EXPECT_EQ(boundaryConvertNeedsEightBitStorage(source, destination), eightBitSide);
            for (bool storage8bit: {false, true})
            {
                for (bool shaderInt8: {false, true})
                {
                    const bool expected = expectedHasVariant(source, destination) && (!eightBitSide || (storage8bit && shaderInt8));
                    EXPECT_EQ(boundaryConvertDeviceSupports(source, destination, storage8bit, shaderInt8), expected) << (int) source << " -> " << (int) destination << " storage8bit=" << storage8bit << " shaderInt8=" << shaderInt8;
                }
            }
        }
    }
}

TEST(BoundaryConvertRule, DeviceDtypeIsHalfOnlyForAnUnpinnedHalfPrecisionSegment) {
    EXPECT_EQ(boundaryDeviceDtype(/*segmentFp16=*/true, /*storeFp32=*/false), DType::Float16);
    EXPECT_EQ(boundaryDeviceDtype(/*segmentFp16=*/true, /*storeFp32=*/true), DType::Float32);
    EXPECT_EQ(boundaryDeviceDtype(/*segmentFp16=*/false, /*storeFp32=*/false), DType::Float32);
    EXPECT_EQ(boundaryDeviceDtype(/*segmentFp16=*/false, /*storeFp32=*/true), DType::Float32);
}

// --- the build and the shader agree with the rule ---------------------------------------------------------

TEST(BoundaryConvertVariants, CMakeBuildsExactlyTheRuleVariantSet) {
    const std::vector<std::string> lines = sourceLines(kCMakeRelativePath, "#");
#if defined(__ANDROID__)
    if (lines.empty())
    {
        GTEST_SKIP() << "the source tree is not present on the device";
    }
#endif
    ASSERT_FALSE(lines.empty()) << "cannot read " << sourceTreePath(kCMakeRelativePath);

    struct BuiltVariant {
        std::string           srcType, dstType;
        std::set<std::string> defines;
    };
    const std::string                   call = "bc_variant(";
    std::map<std::string, BuiltVariant> built;
    for (const std::string &line: lines)
    {
        if (line.rfind(call, 0) != 0)
        {
            continue;
        }
        std::string body = line.substr(call.size());
        body.erase(std::find(body.begin(), body.end(), ')'), body.end());
        std::istringstream words(body);
        std::string        suffix;
        BuiltVariant       variant;
        words >> suffix >> variant.srcType >> variant.dstType;
        for (std::string define; words >> define;)
        {
            variant.defines.insert(define);
        }
        EXPECT_TRUE(built.emplace(suffix, variant).second) << "bc_variant " << suffix << " is built twice";
    }

    std::set<std::string> expectedSuffixes;
    for (const BoundaryConvertVariant &variant: kBoundaryConvertVariants)
    {
        const std::string suffix = expectedTag(variant.source) + "_" + expectedTag(variant.destination);
        expectedSuffixes.insert(suffix);
        const auto it = built.find(suffix);
        if (it == built.end())
        {
            ADD_FAILURE() << "CMakeLists.txt does not build boundary_convert_" << suffix;
            continue;
        }
        EXPECT_EQ(it->second.srcType, glslElementType(variant.source)) << suffix;
        EXPECT_EQ(it->second.dstType, glslElementType(variant.destination)) << suffix;
        std::set<std::string> defines;
        if (variant.source == DType::Float16 || variant.destination == DType::Float16)
        {
            defines.insert("-DNEED_FP16=1");
        }
        if (boundaryConvertNeedsEightBitStorage(variant.source, variant.destination))
        {
            defines.insert("-DNEED_INT8=1");
        }
        if (variant.destination == DType::UInt8)
        {
            defines.insert("-DINT_DST_U8=1");
        }
        if (variant.destination == DType::Int8)
        {
            defines.insert("-DINT_DST_I8=1");
        }
        if (variant.source == DType::Int8)
        {
            defines.insert("-DINT_SRC_I8=1");
        }
        EXPECT_EQ(it->second.defines, defines) << suffix;
    }
    for (const auto &entry: built)
    {
        EXPECT_TRUE(expectedSuffixes.count(entry.first)) << "CMakeLists.txt builds boundary_convert_" << entry.first << ", which kBoundaryConvertVariants does not list";
    }
}

TEST(BoundaryConvertShaderSource, TranscribedInt8ArmsAndConstantsMatchBoundaryConvertComp) {
    const std::vector<std::string> lines = sourceLines(kShaderRelativePath, "//");
#if defined(__ANDROID__)
    if (lines.empty())
    {
        GTEST_SKIP() << "the shader sources are not present on the device";
    }
#endif
    ASSERT_FALSE(lines.empty()) << "cannot read " << sourceTreePath(kShaderRelativePath);

    struct ConstantDeclaration {
        const char *name;
        int         value;
    };
    const std::vector<ConstantDeclaration> constants {
        {"kInt8LowByteMask", kInt8LowByteMask},
        {"kInt8SignBit", kInt8SignBit},
        {"kInt8Modulus", kInt8Modulus},
    };
    static constexpr size_t kLineCapacity = 96;
    for (const ConstantDeclaration &constant: constants)
    {
        char declaration[kLineCapacity];
        std::snprintf(declaration, sizeof(declaration), "const int %s = 0x%X;", constant.name, (unsigned) constant.value);
        EXPECT_NE(std::find(lines.begin(), lines.end(), std::string(declaration)), lines.end()) << "boundary_convert.comp lacks `" << declaration << "`";
    }
    EXPECT_EQ(preprocessorArm(lines, kInt8DestinationArmOpening), kInt8DestinationArmLines);
    EXPECT_EQ(preprocessorArm(lines, kInt8SourceArmOpening), kInt8SourceArmLines);

    // The shader tests exactly the variant defines the build passes: a define it never tests would
    // compile a variant through the generic arm.
    const std::vector<std::string> cmakeLines = sourceLines(kCMakeRelativePath, "#");
    ASSERT_FALSE(cmakeLines.empty()) << "cannot read " << sourceTreePath(kCMakeRelativePath);
    const std::string     defineFlag   = "-D";
    const std::string     definePrefix = defineFlag + "INT_";
    std::set<std::string> passed;
    for (const std::string &line: cmakeLines)
    {
        if (line.rfind("bc_variant(", 0) != 0)
        {
            continue;
        }
        for (size_t at = line.find(definePrefix); at != std::string::npos; at = line.find(definePrefix, at + 1))
        {
            const size_t nameStart = at + defineFlag.size();
            passed.insert(line.substr(nameStart, line.find('=', nameStart) - nameStart));
        }
    }
    const std::string     definedCall  = "defined(";
    const std::string     testedPrefix = definedCall + "INT_";
    std::set<std::string> tested;
    for (const std::string &line: lines)
    {
        if (line.rfind("#", 0) != 0)
        {
            continue;
        }
        for (size_t at = line.find(testedPrefix); at != std::string::npos; at = line.find(testedPrefix, at + 1))
        {
            const size_t nameStart = at + definedCall.size();
            tested.insert(line.substr(nameStart, line.find(')', nameStart) - nameStart));
        }
    }
    EXPECT_EQ(tested, passed);
}

// --- the transcription agrees with the Session's host decode and readback --------------------------------

TEST(BoundaryInt8Staging, SignedByteSourceMatchesCpuBindInputForEveryByte) {
    constexpr int        kByteValueCount = 1 << 8;
    std::vector<uint8_t> payload(kByteValueCount);
    for (int value = 0; value < kByteValueCount; ++value)
    {
        payload[(size_t) value] = (uint8_t) value;
    }
    const std::vector<IOTensor> outs = runIdentityOnCpu(DType::Int8, DType::Float32, kByteValueCount, payload);
    ASSERT_EQ(outs.size(), 1u);
    const std::vector<float> bound = float32Payload(outs[0]);
    ASSERT_EQ(bound.size(), payload.size());

    std::vector<float> decoded(payload.size());
    decodeHostLanesToFloat32(DType::Int8, payload.data(), (int64_t) payload.size(), decoded.data());
    for (size_t k = 0; k < payload.size(); ++k)
    {
        int8_t signedByte;
        std::memcpy(&signedByte, &payload[k], sizeof(signedByte));
        const float onDevice = int8SourceElement(signedByte);
        EXPECT_TRUE(sameBits(onDevice, bound[k])) << "byte " << k << ": variant " << onDevice << ", bindInput " << bound[k];
        EXPECT_TRUE(sameBits(decoded[k], bound[k])) << "byte " << k << ": host lane decode " << decoded[k] << ", bindInput " << bound[k];
        // The fp16 variant stores the same value: fp16 holds every int8 value exactly.
        EXPECT_TRUE(sameBits(halfToFloat(floatToHalf(onDevice)), bound[k])) << "byte " << k;
        EXPECT_EQ(onDevice, (float) signedByte) << "byte " << k << " must sign-extend, never normalize";
    }
}

TEST(BoundaryInt8Staging, SignedByteDestinationMatchesReadbackNarrowing) {
    // Every finite fp16 value (the f16_i8 variant's whole input domain) and an fp32 grid inside the int
    // range (the f32_i8 variant): wrap boundaries, half-integers, negative zero and the int extremes.
    std::vector<float> values;
    constexpr uint32_t kFp16PatternCount = 1u << 16;
    for (uint32_t bits = 0; bits < kFp16PatternCount; ++bits)
    {
        const float value = halfToFloat((fp16_t) bits);
        if (std::isfinite(value))
        {
            values.push_back(value);
        }
    }
    constexpr int   kWrapGridHalfWidth = 1024;
    constexpr float kWrapGridStep      = 0.5f;
    for (int step = -kWrapGridHalfWidth; step <= kWrapGridHalfWidth; ++step)
    {
        values.push_back((float) step * kWrapGridStep);
    }
    const float largestBelowInt32Limit = std::nextafter(kInt32RangeLimit, 0.0f);
    for (float edge: {-0.0f, 127.0f, 127.99f, 128.0f, -128.0f, -128.99f, -129.0f, 255.0f, 256.0f, 65535.0f, 65536.0f, 16777216.0f, largestBelowInt32Limit, -kInt32RangeLimit})
    {
        values.push_back(edge);
    }
    const std::vector<IOTensor> outs = runIdentityOnCpu(DType::Float32, DType::Int8, (int64_t) values.size(), bytesOf(values.data(), values.size()));
    ASSERT_EQ(outs.size(), 1u);
    ASSERT_EQ(outs[0].dtype, DType::Int8);
    ASSERT_EQ(outs[0].data.size(), values.size());
    for (size_t k = 0; k < values.size(); ++k)
    {
        int8_t readback;
        std::memcpy(&readback, &outs[0].data[k], sizeof(readback));
        const int stored = int8DestinationStoredValue(values[k]);
        EXPECT_GE(stored, (int) std::numeric_limits<int8_t>::min()) << values[k];
        EXPECT_LE(stored, (int) std::numeric_limits<int8_t>::max()) << values[k];
        EXPECT_EQ(stored, (int) readback) << values[k];
    }
}

TEST(BoundaryInt8Staging, HostLaneDecodeGivesEachDtypeItsValue) {
    // The Vulkan host upload decodes a non-fp32 host tensor (int64/int32 lanes, or raw 8-bit bytes the
    // staging conversion did not take) with decodeHostLanesToFloat32, and bindInput decodes caller bytes
    // with it, so a CPU Session cannot serve as the reference: each lane is checked against the value the
    // test states for it. Int64 lanes are also checked against a CPU Session run, whose bindInput keeps
    // them undecoded and whose readback widens them on its own path.
    struct DecodeCase {
        DType                dtype;
        std::vector<uint8_t> bytes;
        std::vector<float>   expected;
    };
    auto lanes = [](const auto &values) {
        std::vector<uint8_t> bytes(values.size() * sizeof(values[0]));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        return bytes;
    };
    auto integerValues = [](const auto &values) {
        std::vector<float> expected;
        for (const auto value: values)
        {
            expected.push_back((float) value);
        }
        return expected;
    };
    // fp32 lanes copy bit for bit, a NaN payload and a subnormal included.
    const std::vector<uint32_t> float32Patterns {0x00000000u, 0x80000000u, 0x3FC00000u, 0xC0500000u, 0x7F800000u, 0x7FC00001u, 0x00000001u};
    const std::vector<uint8_t>  float32Bytes = lanes(float32Patterns);
    std::vector<float>          float32Values(float32Patterns.size());
    std::memcpy(float32Values.data(), float32Patterns.data(), float32Patterns.size() * sizeof(uint32_t));
    // fp16 lanes widen to the value each bit pattern encodes.
    struct HalfLane {
        fp16_t pattern;
        float  value;
    };
    const std::vector<HalfLane> halfLanes {
        {0x0000, 0.0f},                   // +0
        {0x8000, -0.0f},                  // -0
        {0x3C00, 1.0f},                   // exponent bias, mantissa 0
        {0xBC00, -1.0f},                  // sign bit set
        {0x3555, 0.333251953125f},        // (1 + 341/1024) * 2^-2
        {0x7BFF, 65504.0f},               // largest finite
        {0x0001, std::ldexp(1.0f, -24)},  // smallest subnormal: 2^-14 / 2^10
        {0x8400, -std::ldexp(1.0f, -14)}, // smallest normal, negative
        {0x7C00, std::numeric_limits<float>::infinity()},
        {0xFC00, -std::numeric_limits<float>::infinity()},
    };
    std::vector<fp16_t> halfPatterns;
    std::vector<float>  halfValues;
    for (const HalfLane &lane: halfLanes)
    {
        halfPatterns.push_back(lane.pattern);
        halfValues.push_back(lane.value);
    }
    const std::vector<uint8_t> unsignedBytes {0, 1, 127, 128, 200, 254, 255};
    const std::vector<int8_t>  signedBytes {0, 1, -1, 127, -128, 100, -5};
    const std::vector<int32_t> ints32 {0, -1, 16777216, -16777217, std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min(), 123456};
    const std::vector<int64_t> ints64 {0, -1, int64_t(1) << 40, -(int64_t(1) << 24), std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min(),
                                       99};
    const std::vector<uint8_t> int64Bytes = lanes(ints64);
    const std::vector<DecodeCase> cases {
        {DType::Float32, float32Bytes, float32Values},
        {DType::Float16, lanes(halfPatterns), halfValues},
        {DType::UInt8, lanes(unsignedBytes), integerValues(unsignedBytes)},
        {DType::Int8, lanes(signedBytes), integerValues(signedBytes)},
        {DType::Int32, lanes(ints32), integerValues(ints32)},
        {DType::Int64, int64Bytes, integerValues(ints64)},
    };
    // A recognized dtype missing from the table would fall through to the fp32 copy unchecked; an
    // unrecognized value decodes as fp32 lanes.
    for (DType dtype: everyDtypeValue())
    {
        if (!isRecognizedDtype(dtype))
        {
            EXPECT_EQ(boundaryHostLaneBytes(dtype), sizeof(float)) << "dtype value " << (int) dtype;
            std::vector<float> decoded(float32Values.size());
            decodeHostLanesToFloat32(dtype, float32Bytes.data(), (int64_t) decoded.size(), decoded.data());
            for (size_t k = 0; k < decoded.size(); ++k)
            {
                EXPECT_TRUE(sameBits(decoded[k], float32Values[k])) << "dtype value " << (int) dtype << " element " << k;
            }
            continue;
        }
        const bool covered = std::any_of(cases.begin(), cases.end(), [&](const DecodeCase &c) {
            return c.dtype == dtype;
        });
        EXPECT_TRUE(covered) << dtypeStr(dtype) << " has no decode check";
        EXPECT_EQ(boundaryHostLaneBytes(dtype), dtypeSize(dtype)) << dtypeStr(dtype);
    }
    for (const DecodeCase &c: cases)
    {
        const int64_t count = (int64_t) (c.bytes.size() / dtypeSize(c.dtype));
        ASSERT_EQ((size_t) count, c.expected.size()) << dtypeStr(c.dtype);
        std::vector<float> decoded((size_t) count);
        decodeHostLanesToFloat32(c.dtype, c.bytes.data(), count, decoded.data());
        for (size_t k = 0; k < decoded.size(); ++k)
        {
            EXPECT_TRUE(sameBits(decoded[k], c.expected[k])) << dtypeStr(c.dtype) << " element " << k << ": decode " << decoded[k] << ", expected " << c.expected[k];
        }
    }
    const std::vector<IOTensor> outs = runIdentityOnCpu(DType::Int64, DType::Float32, (int64_t) ints64.size(), int64Bytes);
    ASSERT_EQ(outs.size(), 1u);
    const std::vector<float> bound = float32Payload(outs[0]);
    ASSERT_EQ(bound.size(), ints64.size());
    std::vector<float> decoded(ints64.size());
    decodeHostLanesToFloat32(DType::Int64, int64Bytes.data(), (int64_t) decoded.size(), decoded.data());
    for (size_t k = 0; k < decoded.size(); ++k)
    {
        EXPECT_TRUE(sameBits(decoded[k], bound[k])) << "i64 element " << k << ": decode " << decoded[k] << ", bindInput + readback " << bound[k];
    }
}
