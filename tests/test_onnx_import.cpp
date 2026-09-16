// ONNX importer wire-level tests: TensorProto / AttributeProto / NodeProto / ModelProto messages are
// encoded by hand (protobuf varint/length-delimited framing) and decoded through TensorProtoParser /
// NodeParser / importOnnx, asserting integer payloads materialize exactly for every dtype x payload-field
// combination (raw_data bytes vs the typed int32_data / uint64_data varint arrays) and that BitShift /
// BitwiseNot nodes carry the integer width their operand's ONNX element type resolves to.
#include "core/bitwise_attrs.h"
#include "import/onnx/onnx_node_parser.h"
#include "import/onnx/onnx_tensor_parser.h"
#include "vknn/dtype.h"
#include "vknn/graph.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

using namespace vknn;
using namespace vknn::onnx;

namespace {

    // --- protobuf wire encoding helpers -------------------------------------------------------
    void putVarint(std::vector<uint8_t> &b, uint64_t v) {
        while (v >= 0x80)
        {
            b.push_back((uint8_t) ((v & 0x7F) | 0x80));
            v >>= 7;
        }
        b.push_back((uint8_t) v);
    }
    void putTag(std::vector<uint8_t> &b, uint32_t field, uint32_t wire) {
        putVarint(b, ((uint64_t) field << 3) | wire);
    }
    void putLenField(std::vector<uint8_t> &b, uint32_t field, const std::vector<uint8_t> &payload) {
        putTag(b, field, 2);
        putVarint(b, payload.size());
        b.insert(b.end(), payload.begin(), payload.end());
    }

    // TensorProto header: dims (field 1, packed) + data_type (field 2).
    std::vector<uint8_t> tensorHeader(OnnxType dt, const std::vector<int64_t> &dims) {
        std::vector<uint8_t> b, d;
        for (int64_t x: dims)
        {
            putVarint(d, (uint64_t) x);
        }
        if (!d.empty())
        {
            putLenField(b, 1, d);
        }
        putTag(b, 2, 0);
        putVarint(b, (uint64_t) (int32_t) dt);
        return b;
    }

    // TensorProto carrying its payload in raw_data (field 9).
    std::vector<uint8_t> protoWithRaw(OnnxType dt, const std::vector<int64_t> &dims, const std::vector<uint8_t> &raw) {
        std::vector<uint8_t> b = tensorHeader(dt, dims);
        putLenField(b, 9, raw);
        return b;
    }

    // TensorProto carrying its payload in int32_data (field 5). Negative values are encoded the
    // way protobuf encodes negative int32: sign-extended to 64 bits (a 10-byte varint). `packed`
    // selects the length-delimited blob form vs one varint per tag; exporters emit both.
    std::vector<uint8_t> protoWithInt32Data(OnnxType dt, const std::vector<int64_t> &dims, const std::vector<int64_t> &vals, bool packed) {
        std::vector<uint8_t> b = tensorHeader(dt, dims);
        if (packed)
        {
            std::vector<uint8_t> p;
            for (int64_t v: vals)
            {
                putVarint(p, (uint64_t) v);
            }
            putLenField(b, 5, p);
        } else
        {
            for (int64_t v: vals)
            {
                putTag(b, 5, 0);
                putVarint(b, (uint64_t) v);
            }
        }
        return b;
    }

    // parse + fillHostFloat, returning the materialized fp32 values.
    std::vector<float> decodeF32(const std::vector<uint8_t> &msg, int64_t elems) {
        TensorProto t = TensorProtoParser::parse(Reader(msg.data(), msg.size()));
        HostBuffer  hb;
        TensorProtoParser::fillHostFloat(t, hb, elems);
        return std::vector<float>(hb.f32(), hb.f32() + elems);
    }

} // namespace

// --- raw_data payloads ------------------------------------------------------------------------

TEST(OnnxTensorProto, Int8RawWidensExact) {
    const std::vector<int8_t> v {-128, -1, 0, 1, 127};
    std::vector<uint8_t>      raw(v.size());
    std::memcpy(raw.data(), v.data(), v.size());
    auto f = decodeF32(protoWithRaw(OnnxType::Int8, {(int64_t) v.size()}, raw), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], (float) v[i]) << "i=" << i;
    }
}

TEST(OnnxTensorProto, Uint8RawWidensExact) {
    const std::vector<uint8_t> v {0, 1, 128, 255};
    auto                       f = decodeF32(protoWithRaw(OnnxType::Uint8, {(int64_t) v.size()}, v), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], (float) v[i]) << "i=" << i;
    }
}

TEST(OnnxTensorProto, Int32RawWidensExact) {
    const std::vector<int32_t> v {std::numeric_limits<int32_t>::min(), -7, 0, 1, 1 << 24, std::numeric_limits<int32_t>::max()};
    std::vector<uint8_t>       raw(v.size() * 4);
    std::memcpy(raw.data(), v.data(), raw.size());
    auto f = decodeF32(protoWithRaw(OnnxType::Int32, {(int64_t) v.size()}, raw), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], (float) v[i]) << "i=" << i;
    }
}

TEST(OnnxTensorProto, BoolRawWidensToZeroOne) {
    const std::vector<uint8_t> v {0, 1, 1, 0};
    auto                       f = decodeF32(protoWithRaw(OnnxType::Bool, {(int64_t) v.size()}, v), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], v[i] ? 1.0f : 0.0f) << "i=" << i;
    }
}

TEST(OnnxTensorProto, TruncatedRawLeavesTailZero) {
    // 3 payload bytes for a 5-element tensor: decoded head, zero tail (clamped copy contract).
    const std::vector<int8_t> v {-3, 4, -5};
    std::vector<uint8_t>      raw(v.size());
    std::memcpy(raw.data(), v.data(), v.size());
    auto f = decodeF32(protoWithRaw(OnnxType::Int8, {5}, raw), 5);
    EXPECT_EQ(f[0], -3.0f);
    EXPECT_EQ(f[1], 4.0f);
    EXPECT_EQ(f[2], -5.0f);
    EXPECT_EQ(f[3], 0.0f);
    EXPECT_EQ(f[4], 0.0f);
}

// --- int32_data payloads ----------------------------------------------------------------------

TEST(OnnxTensorProto, Int8Int32DataPacked) {
    const std::vector<int64_t> v {-128, 127, -1, 0, 42};
    auto                       f = decodeF32(protoWithInt32Data(OnnxType::Int8, {(int64_t) v.size()}, v, true), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], (float) v[i]) << "i=" << i;
    }
}

TEST(OnnxTensorProto, Uint8Int32DataPacked) {
    const std::vector<int64_t> v {0, 255, 7, 128};
    auto                       f = decodeF32(protoWithInt32Data(OnnxType::Uint8, {(int64_t) v.size()}, v, true), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], (float) v[i]) << "i=" << i;
    }
}

TEST(OnnxTensorProto, Int32Int32DataPacked) {
    const std::vector<int64_t> v {std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(), -123456, 1 << 24};
    auto                       f = decodeF32(protoWithInt32Data(OnnxType::Int32, {(int64_t) v.size()}, v, true), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], (float) (int32_t) v[i]) << "i=" << i;
    }
}

TEST(OnnxTensorProto, BoolInt32DataPacked) {
    const std::vector<int64_t> v {0, 1, 1, 0};
    auto                       f = decodeF32(protoWithInt32Data(OnnxType::Bool, {(int64_t) v.size()}, v, true), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], v[i] ? 1.0f : 0.0f) << "i=" << i;
    }
}

TEST(OnnxTensorProto, Float16Int32DataDecodesBits) {
    // FLOAT16 rides int32_data as raw bit patterns, not numeric values.
    const std::vector<float> want {1.0f, -2.0f, 0.5f, 65504.0f};
    std::vector<int64_t>     bits;
    for (float w: want)
    {
        bits.push_back((int64_t) floatToHalf(w));
    }
    auto f = decodeF32(protoWithInt32Data(OnnxType::Float16, {(int64_t) want.size()}, bits, true), (int64_t) want.size());
    for (size_t i = 0; i < want.size(); ++i)
    {
        EXPECT_EQ(f[i], want[i]) << "i=" << i;
    }
}

TEST(OnnxTensorProto, Int8Int32DataUnpacked) {
    // One varint per tag instead of the packed blob; both encodings appear in the wild.
    const std::vector<int64_t> v {-5, 6};
    auto                       f = decodeF32(protoWithInt32Data(OnnxType::Int8, {(int64_t) v.size()}, v, false), (int64_t) v.size());
    EXPECT_EQ(f[0], -5.0f);
    EXPECT_EQ(f[1], 6.0f);
}

// --- Constant-node tensor attribute (AttributeProto field 5) -----------------------------------

namespace {
    // AttributeProto: field 1 = name, field 5 = t (TensorProto).
    Attr parseTensorAttr(const std::vector<uint8_t> &tpMsg) {
        std::vector<uint8_t> msg;
        putLenField(msg, 1, {'v', 'a', 'l', 'u', 'e'});
        putLenField(msg, 5, tpMsg);
        Node node;
        NodeParser::parseAttr(Reader(msg.data(), msg.size()), node);
        return node.attr.map.at("value");
    }
} // namespace

TEST(OnnxAttr, ConstantInt8RawPayload) {
    const std::vector<int8_t> v {-128, 0, 127};
    std::vector<uint8_t>      raw(v.size());
    std::memcpy(raw.data(), v.data(), v.size());
    Attr a = parseTensorAttr(protoWithRaw(OnnxType::Int8, {(int64_t) v.size()}, raw));
    EXPECT_EQ(a.kind, Attr::Floats);
    ASSERT_EQ(a.floats.size(), v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(a.floats[i], (float) v[i]) << "i=" << i;
    }
}

TEST(OnnxAttr, ConstantUint8Int32DataPayload) {
    const std::vector<int64_t> v {0, 200, 255};
    Attr                       a = parseTensorAttr(protoWithInt32Data(OnnxType::Uint8, {(int64_t) v.size()}, v, true));
    EXPECT_EQ(a.kind, Attr::Floats);
    ASSERT_EQ(a.floats.size(), v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(a.floats[i], (float) v[i]) << "i=" << i;
    }
}

// --- NodeProto (parseNode) ----------------------------------------------------------------------

namespace {
    // NodeProto: field 1 = input (repeated string), field 4 = op_type, field 7 = domain.
    std::vector<uint8_t> nodeProto(const char *opType, const std::vector<std::string> &inputs, const char *domain = nullptr) {
        std::vector<uint8_t> msg;
        for (const std::string &in: inputs)
        {
            putLenField(msg, 1, std::vector<uint8_t>(in.begin(), in.end()));
        }
        std::string t(opType);
        putLenField(msg, 4, std::vector<uint8_t>(t.begin(), t.end()));
        if (domain)
        {
            std::string d(domain);
            putLenField(msg, 7, std::vector<uint8_t>(d.begin(), d.end()));
        }
        return msg;
    }
} // namespace

TEST(OnnxNode, UpsampleScalesNormalizedToResizeSlot) {
    // Opset-9 Upsample(X, scales) imports as Resize with the absent-roi slot inserted, so scales
    // sits at input 2 where the Resize shape rule and kernels read it.
    std::vector<uint8_t>     msg = nodeProto("Upsample", {"x", "s"});
    Node                     node;
    std::vector<std::string> ins, outs;
    NodeParser::parseNode(Reader(msg.data(), msg.size()), node, ins, outs);
    EXPECT_EQ(node.type, OpType::Resize);
    ASSERT_EQ(ins.size(), 3u);
    EXPECT_EQ(ins[0], "x");
    EXPECT_EQ(ins[1], "") << "roi placeholder (resolves to kNoTensor)";
    EXPECT_EQ(ins[2], "s");
}

TEST(OnnxNode, UpsampleAttrFormKeepsInputs) {
    // Opset-7 Upsample carries scales as an attribute and has one input: nothing is inserted.
    std::vector<uint8_t>     msg = nodeProto("Upsample", {"x"});
    Node                     node;
    std::vector<std::string> ins, outs;
    NodeParser::parseNode(Reader(msg.data(), msg.size()), node, ins, outs);
    EXPECT_EQ(node.type, OpType::Resize);
    ASSERT_EQ(ins.size(), 1u);
    EXPECT_EQ(ins[0], "x");
}

// --- ONNX quantized operator family --------------------------------------------------------------

TEST(OnnxOpMap, QuantizedFamilyRoundTrips) {
    // Each quantized op maps to its own OpType and spells back to the ONNX name.
    const std::pair<const char *, OpType> want[] = {
        {"QuantizeLinear", OpType::QuantizeLinear},
        {"DequantizeLinear", OpType::DequantizeLinear},
        {"DynamicQuantizeLinear", OpType::DynamicQuantizeLinear},
        {"QLinearConv", OpType::QLinearConv},
        {"QLinearMatMul", OpType::QLinearMatMul},
        {"QLinearAdd", OpType::QLinearAdd},
        {"QLinearGlobalAveragePool", OpType::QLinearGlobalAveragePool},
        {"MatMulInteger", OpType::MatMulInteger},
        {"ConvInteger", OpType::ConvInteger},
        {"QGemm", OpType::QGemm},
    };
    for (const auto &[name, type]: want)
    {
        EXPECT_EQ(opTypeFromOnnx(name), type) << name;
        EXPECT_STREQ(opTypeName(type), name);
        EXPECT_TRUE(opTypeIsQuantized(type)) << name;
    }
    EXPECT_FALSE(opTypeIsQuantized(OpType::Conv));
    EXPECT_FALSE(opTypeIsQuantized(OpType::Unknown));
}

TEST(OnnxNode, QLinearConvImportsAsItsOpType) {
    // The full 9-input QLinearConv form parses to its own OpType with the input list intact
    // (no Unknown, so the unrecognized-op WARN does not fire for quantized checkpoints).
    std::vector<uint8_t>     msg = nodeProto("QLinearConv", {"x", "x_s", "x_zp", "w", "w_s", "w_zp", "y_s", "y_zp", "b"});
    Node                     node;
    std::vector<std::string> ins, outs;
    NodeParser::parseNode(Reader(msg.data(), msg.size()), node, ins, outs);
    EXPECT_EQ(node.type, OpType::QLinearConv);
    ASSERT_EQ(ins.size(), 9u);
    EXPECT_EQ(ins[0], "x");
    EXPECT_EQ(ins[8], "b");
}

TEST(OnnxNode, MicrosoftDomainQGemmMatchesByName) {
    // The wire parser drops NodeProto.domain (field 7), so the com.microsoft members of the
    // family (QGemm, QLinearAdd, QLinearGlobalAveragePool) resolve through name matching alone.
    std::vector<uint8_t>     msg = nodeProto("QGemm", {"a", "a_s", "a_zp", "b", "b_s", "b_zp"}, "com.microsoft");
    Node                     node;
    std::vector<std::string> ins, outs;
    NodeParser::parseNode(Reader(msg.data(), msg.size()), node, ins, outs);
    EXPECT_EQ(node.type, OpType::QGemm);
    ASSERT_EQ(ins.size(), 6u);
}

// --- INT16 / UINT16 / UINT32 / UINT64 payloads ------------------------------------------------------

namespace {
    constexpr uint32_t kUint64DataField = 11; // TensorProto.uint64_data (packed varints)

    // Little-endian bytes of `values` as raw_data.
    template <class T> std::vector<uint8_t> rawBytes(const std::vector<T> &values) {
        std::vector<uint8_t> raw(values.size() * sizeof(T));
        if (!raw.empty())
        {
            std::memcpy(raw.data(), values.data(), raw.size());
        }
        return raw;
    }

    // TensorProto carrying its payload in uint64_data (packed blob, or one varint per tag).
    std::vector<uint8_t> protoWithUint64Data(OnnxType dt, const std::vector<int64_t> &dims, const std::vector<uint64_t> &vals, bool packed) {
        std::vector<uint8_t> b = tensorHeader(dt, dims);
        if (packed)
        {
            std::vector<uint8_t> p;
            for (uint64_t v: vals)
            {
                putVarint(p, v);
            }
            putLenField(b, kUint64DataField, p);
        } else
        {
            for (uint64_t v: vals)
            {
                putTag(b, kUint64DataField, 0);
                putVarint(b, v);
            }
        }
        return b;
    }

    // parse + fillHostI64, returning the materialized int64 lanes.
    std::vector<int64_t> decodeI64(const std::vector<uint8_t> &msg, int64_t elems) {
        TensorProto t = TensorProtoParser::parse(Reader(msg.data(), msg.size()));
        HostBuffer  hb;
        TensorProtoParser::fillHostI64(t, hb, elems);
        return std::vector<int64_t>(hb.i64(), hb.i64() + elems);
    }

    // The int64 whose two's-complement bits are `bits`.
    int64_t int64FromBits(uint64_t bits) {
        int64_t v;
        std::memcpy(&v, &bits, sizeof v);
        return v;
    }
} // namespace

TEST(OnnxTensorProto, Int16RawWidensExact) {
    const std::vector<int16_t> v {std::numeric_limits<int16_t>::min(), -1, 0, 1, std::numeric_limits<int16_t>::max()};
    auto                       f = decodeF32(protoWithRaw(OnnxType::Int16, {(int64_t) v.size()}, rawBytes(v)), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], (float) v[i]) << "i=" << i;
    }
}

TEST(OnnxTensorProto, Uint16RawWidensExact) {
    const std::vector<uint16_t> v {0, 1, 32768, 40000, std::numeric_limits<uint16_t>::max()};
    auto                        f = decodeF32(protoWithRaw(OnnxType::Uint16, {(int64_t) v.size()}, rawBytes(v)), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], (float) v[i]) << "i=" << i;
    }
}

TEST(OnnxTensorProto, Uint32RawZeroExtendsToInt64) {
    const std::vector<uint32_t> v {0, 1, 2147483648u, 3000000000u, std::numeric_limits<uint32_t>::max()};
    auto                        got = decodeI64(protoWithRaw(OnnxType::Uint32, {(int64_t) v.size()}, rawBytes(v)), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(got[i], (int64_t) v[i]) << "i=" << i;
    }
}

TEST(OnnxTensorProto, Uint64RawKeepsBitPattern) {
    const std::vector<uint64_t> v {0, 5, (uint64_t) 1 << 63, ((uint64_t) 1 << 63) + 5, std::numeric_limits<uint64_t>::max()};
    auto                        got = decodeI64(protoWithRaw(OnnxType::Uint64, {(int64_t) v.size()}, rawBytes(v)), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(got[i], int64FromBits(v[i])) << "i=" << i;
    }
}

TEST(OnnxTensorProto, Uint64DataFieldKeepsBitPattern) {
    // uint64_data carries UINT64 and UINT32 values, packed or one varint per tag; a value at or above
    // 2^63 keeps its bits.
    const std::vector<uint64_t> wide {((uint64_t) 1 << 63) + 5, 7, std::numeric_limits<uint64_t>::max()};
    for (bool packed: {true, false})
    {
        auto got = decodeI64(protoWithUint64Data(OnnxType::Uint64, {(int64_t) wide.size()}, wide, packed), (int64_t) wide.size());
        for (size_t i = 0; i < wide.size(); ++i)
        {
            EXPECT_EQ(got[i], int64FromBits(wide[i])) << "packed=" << packed << " i=" << i;
        }
    }
    const std::vector<uint64_t> narrow {std::numeric_limits<uint32_t>::max(), 1};
    auto                        got = decodeI64(protoWithUint64Data(OnnxType::Uint32, {2}, narrow, true), 2);
    EXPECT_EQ(got[0], (int64_t) std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(got[1], 1);
}

TEST(OnnxAttr, ConstantUint64RawPayloadIsInts) {
    // A UINT64 Constant tensor materializes as int64 `ints` (bit pattern kept), not as fp32 floats.
    const std::vector<uint64_t> v {((uint64_t) 1 << 63) + 1, 42};
    Attr                        a = parseTensorAttr(protoWithRaw(OnnxType::Uint64, {(int64_t) v.size()}, rawBytes(v)));
    EXPECT_EQ(a.kind, Attr::Ints);
    ASSERT_EQ(a.ints.size(), v.size());
    EXPECT_EQ(a.ints[0], int64FromBits(v[0]));
    EXPECT_EQ(a.ints[1], 42);
}

// --- Whole-model import: initializer storage and BitShift / BitwiseNot integer widths --------------

namespace {
    // Field numbers of the ModelProto / GraphProto / NodeProto / AttributeProto / ValueInfoProto parts the
    // model encoder writes (onnx.proto).
    constexpr uint32_t kModelGraphField         = 7;
    constexpr uint32_t kGraphNodeField          = 1;
    constexpr uint32_t kGraphInitializerField   = 5;
    constexpr uint32_t kGraphInputField         = 11;
    constexpr uint32_t kGraphOutputField        = 12;
    constexpr uint32_t kGraphValueInfoField     = 13;
    constexpr uint32_t kNodeInputField          = 1;
    constexpr uint32_t kNodeOutputField         = 2;
    constexpr uint32_t kNodeNameField           = 3;
    constexpr uint32_t kNodeOpTypeField         = 4;
    constexpr uint32_t kNodeAttributeField      = 5;
    constexpr uint32_t kAttributeNameField      = 1;
    constexpr uint32_t kAttributeIntField       = 3;
    constexpr uint32_t kAttributeStringField    = 4;
    constexpr uint32_t kAttributeTensorField    = 5;
    constexpr uint32_t kAttributeTypeField      = 20;
    constexpr uint64_t kAttributeTypeInt        = 2;
    constexpr uint64_t kAttributeTypeString     = 3;
    constexpr uint64_t kAttributeTypeTensor     = 4;
    constexpr uint32_t kValueInfoNameField      = 1;
    constexpr uint32_t kValueInfoTypeField      = 2;
    constexpr uint32_t kTypeTensorTypeField     = 1;
    constexpr uint32_t kTensorTypeElemTypeField = 1;
    constexpr uint32_t kTensorTypeShapeField    = 2;
    constexpr uint32_t kShapeDimField           = 1;
    constexpr uint32_t kDimValueField           = 1;
    constexpr uint32_t kTensorNameField         = 8;

    std::vector<uint8_t> bytesOf(const std::string &s) {
        return std::vector<uint8_t>(s.begin(), s.end());
    }

    // ValueInfoProto `name` with a tensor type of `dims`; `elemType` Undefined leaves elem_type unset.
    std::vector<uint8_t> valueInfo(const std::string &name, OnnxType elemType, const std::vector<int64_t> &dims) {
        std::vector<uint8_t> shape, tensorType, type, msg;
        for (int64_t d: dims)
        {
            std::vector<uint8_t> dim;
            putTag(dim, kDimValueField, kWireVarint);
            putVarint(dim, (uint64_t) d);
            putLenField(shape, kShapeDimField, dim);
        }
        if (elemType != OnnxType::Undefined)
        {
            putTag(tensorType, kTensorTypeElemTypeField, kWireVarint);
            putVarint(tensorType, (uint64_t) (int32_t) elemType);
        }
        putLenField(tensorType, kTensorTypeShapeField, shape);
        putLenField(type, kTypeTensorTypeField, tensorType);
        putLenField(msg, kValueInfoNameField, bytesOf(name));
        putLenField(msg, kValueInfoTypeField, type);
        return msg;
    }

    std::vector<uint8_t> intAttribute(const std::string &name, int64_t value) {
        std::vector<uint8_t> msg;
        putLenField(msg, kAttributeNameField, bytesOf(name));
        putTag(msg, kAttributeIntField, kWireVarint);
        putVarint(msg, (uint64_t) value);
        putTag(msg, kAttributeTypeField, kWireVarint);
        putVarint(msg, kAttributeTypeInt);
        return msg;
    }

    std::vector<uint8_t> stringAttribute(const std::string &name, const std::string &value) {
        std::vector<uint8_t> msg;
        putLenField(msg, kAttributeNameField, bytesOf(name));
        putLenField(msg, kAttributeStringField, bytesOf(value));
        putTag(msg, kAttributeTypeField, kWireVarint);
        putVarint(msg, kAttributeTypeString);
        return msg;
    }

    std::vector<uint8_t> tensorAttribute(const std::string &name, const std::vector<uint8_t> &tensor) {
        std::vector<uint8_t> msg;
        putLenField(msg, kAttributeNameField, bytesOf(name));
        putLenField(msg, kAttributeTensorField, tensor);
        putTag(msg, kAttributeTypeField, kWireVarint);
        putVarint(msg, kAttributeTypeTensor);
        return msg;
    }

    std::vector<uint8_t> graphNode(const std::string &opType, const std::string &name, const std::vector<std::string> &inputs, const std::vector<std::string> &outputs, const std::vector<std::vector<uint8_t>> &attributes = {}) {
        std::vector<uint8_t> msg;
        for (const std::string &in: inputs)
        {
            putLenField(msg, kNodeInputField, bytesOf(in));
        }
        for (const std::string &out: outputs)
        {
            putLenField(msg, kNodeOutputField, bytesOf(out));
        }
        putLenField(msg, kNodeNameField, bytesOf(name));
        putLenField(msg, kNodeOpTypeField, bytesOf(opType));
        for (const auto &attribute: attributes)
        {
            putLenField(msg, kNodeAttributeField, attribute);
        }
        return msg;
    }

    // TensorProto named `name` with raw_data.
    std::vector<uint8_t> namedRawTensor(const std::string &name, OnnxType dt, const std::vector<int64_t> &dims, const std::vector<uint8_t> &raw) {
        std::vector<uint8_t> msg = protoWithRaw(dt, dims, raw);
        putLenField(msg, kTensorNameField, bytesOf(name));
        return msg;
    }

    struct GraphParts {
        std::vector<std::vector<uint8_t>> nodes, initializers, inputs, outputs, valueInfos;
    };

    // Encodes the ModelProto, writes it under the test temp dir and imports it.
    Graph importModel(const std::string &fileStem, const GraphParts &parts) {
        std::vector<uint8_t> graph, model;
        for (const auto &n: parts.nodes)
        {
            putLenField(graph, kGraphNodeField, n);
        }
        for (const auto &t: parts.initializers)
        {
            putLenField(graph, kGraphInitializerField, t);
        }
        for (const auto &v: parts.inputs)
        {
            putLenField(graph, kGraphInputField, v);
        }
        for (const auto &v: parts.outputs)
        {
            putLenField(graph, kGraphOutputField, v);
        }
        for (const auto &v: parts.valueInfos)
        {
            putLenField(graph, kGraphValueInfoField, v);
        }
        putLenField(model, kModelGraphField, graph);
        const std::string path = testing::TempDir() + fileStem + ".onnx";
        {
            std::ofstream file(path, std::ios::binary);
            file.write((const char *) model.data(), (std::streamsize) model.size());
        }
        return importOnnx(path);
    }

    const Node *findNode(const Graph &g, const std::string &name) {
        for (const Node &n: g.nodes)
        {
            if (n.name == name)
            {
                return &n;
            }
        }
        ADD_FAILURE() << "no node named " << name;
        return nullptr;
    }

    // Expects `name` to carry the stamped width attributes.
    void expectWidth(const Graph &g, const std::string &name, int64_t bits, int64_t isSigned) {
        const Node *node = findNode(g, name);
        ASSERT_NE(node, nullptr);
        ASSERT_TRUE(node->attr.has(bitwise::kIntBitsAttr)) << name;
        ASSERT_TRUE(node->attr.has(bitwise::kIntSignedAttr)) << name;
        EXPECT_EQ(node->attr.map.at(bitwise::kIntBitsAttr).kind, Attr::Int) << name;
        EXPECT_EQ(node->attr.geti(bitwise::kIntBitsAttr), bits) << name;
        EXPECT_EQ(node->attr.geti(bitwise::kIntSignedAttr), isSigned) << name;
    }
} // namespace

TEST(OnnxImport, UnsignedAndSixteenBitInitializersMaterializeExactly) {
    // INT16 / UINT16 widen to exact fp32 lanes; UINT32 / UINT64 take int64 storage (bit patterns kept).
    const std::vector<int16_t>  i16 {-32768, -2, 32767};
    const std::vector<uint16_t> u16 {0, 40000, 65535};
    const std::vector<uint32_t> u32 {1, 3000000000u, 4294967295u};
    const std::vector<uint64_t> u64 {3, ((uint64_t) 1 << 63) + 9, std::numeric_limits<uint64_t>::max()};
    GraphParts                  parts;
    parts.initializers = {
        namedRawTensor("i16", OnnxType::Int16, {3}, rawBytes(i16)),
        namedRawTensor("u16", OnnxType::Uint16, {3}, rawBytes(u16)),
        namedRawTensor("u32", OnnxType::Uint32, {3}, rawBytes(u32)),
        namedRawTensor("u64", OnnxType::Uint64, {3}, rawBytes(u64)),
    };
    Graph g = importModel("vknn_import_unsigned_initializers", parts);
    for (const char *name: {"i16", "u16"})
    {
        TensorId id = g.find(name);
        ASSERT_TRUE(g.isInitializer(id)) << name;
        EXPECT_EQ(g.desc(id).dtype, DType::Float32) << name;
        const float *lanes = g.initializers.at(id).f32();
        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_EQ(lanes[i], std::string(name) == "i16" ? (float) i16[i] : (float) u16[i]) << name << " i=" << i;
        }
    }
    for (const char *name: {"u32", "u64"})
    {
        TensorId id = g.find(name);
        ASSERT_TRUE(g.isInitializer(id)) << name;
        EXPECT_EQ(g.desc(id).dtype, DType::Int64) << name;
        const int64_t *lanes = g.initializers.at(id).i64();
        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_EQ(lanes[i], std::string(name) == "u32" ? (int64_t) u32[i] : int64FromBits(u64[i])) << name << " i=" << i;
        }
    }
}

TEST(OnnxImport, IntegerWidthFromGraphInputAndInitializer) {
    // A declared UINT16 graph input types BitShift and BitwiseNot; an untyped shifted operand falls back to
    // the UINT32 shift-count initializer; a BitwiseNot of an INT8 initializer is signed 8-bit.
    const std::vector<uint32_t> counts {1, 2};
    const std::vector<int8_t>   bytes {-3, 5};
    GraphParts                  parts;
    parts.inputs       = {valueInfo("x", OnnxType::Uint16, {2}), valueInfo("untyped", OnnxType::Undefined, {2})};
    parts.initializers = {namedRawTensor("counts", OnnxType::Uint32, {2}, rawBytes(counts)), namedRawTensor("bytes", OnnxType::Int8, {2}, rawBytes(bytes))};
    parts.nodes        = {
        graphNode("BitShift", "shift_input", {"x", "x"}, {"shifted"}, {stringAttribute("direction", "LEFT")}),
        graphNode("BitwiseNot", "not_input", {"x"}, {"flipped"}),
        graphNode("BitShift", "shift_count_fallback", {"untyped", "counts"}, {"shifted_untyped"}, {stringAttribute("direction", "RIGHT")}),
        graphNode("BitwiseNot", "not_initializer", {"bytes"}, {"flipped_bytes"}),
    };
    parts.outputs = {valueInfo("shifted", OnnxType::Uint16, {2}), valueInfo("flipped", OnnxType::Uint16, {2}), valueInfo("shifted_untyped", OnnxType::Undefined, {2}), valueInfo("flipped_bytes", OnnxType::Int8, {2})};
    Graph g = importModel("vknn_import_width_input_initializer", parts);
    expectWidth(g, "shift_input", 16, 0);
    expectWidth(g, "not_input", 16, 0);
    expectWidth(g, "shift_count_fallback", 32, 0);
    expectWidth(g, "not_initializer", 8, 1);
    const Node *shift = findNode(g, "shift_input");
    ASSERT_NE(shift, nullptr);
    EXPECT_EQ(shift->attr.gets(bitwise::kDirectionAttr), "LEFT") << "the direction string survives import";
}

TEST(OnnxImport, IntegerWidthFromValueInfoAndProducers) {
    // value_info on an intermediate wins over its producer chain; a Cast chain through pass-through ops
    // resolves to the Cast target; Shape is INT64; a comparison is BOOL (unsigned 8-bit); a Constant's
    // value tensor carries its own type.
    const std::vector<uint64_t> constant {7, 9};
    GraphParts                  parts;
    parts.inputs     = {valueInfo("x", OnnxType::Int64, {2}), valueInfo("f", OnnxType::Float, {2})};
    parts.valueInfos = {valueInfo("relabeled", OnnxType::Uint8, {2})};
    parts.nodes      = {
        graphNode("Identity", "relabel", {"x"}, {"relabeled"}),
        graphNode("BitwiseNot", "not_value_info", {"relabeled"}, {"flipped_value_info"}),
        graphNode("Cast", "cast_int16", {"f"}, {"as_int16"}, {intAttribute("to", (int64_t) OnnxType::Int16)}),
        graphNode("Unsqueeze", "unsqueeze", {"as_int16"}, {"expanded"}, {}),
        graphNode("Identity", "pass", {"expanded"}, {"passed"}),
        graphNode("BitwiseNot", "not_cast_chain", {"passed"}, {"flipped_cast"}),
        graphNode("Shape", "shape", {"f"}, {"dims"}),
        graphNode("BitwiseNot", "not_shape", {"dims"}, {"flipped_dims"}),
        graphNode("Less", "less", {"f", "f"}, {"mask"}),
        graphNode("BitwiseNot", "not_bool", {"mask"}, {"flipped_mask"}),
        graphNode("Constant", "constant", {}, {"constant_u64"}, {tensorAttribute("value", protoWithRaw(OnnxType::Uint64, {2}, rawBytes(constant)))}),
        graphNode("BitShift", "shift_constant", {"constant_u64", "constant_u64"}, {"shifted_constant"}, {stringAttribute("direction", "RIGHT")}),
    };
    parts.outputs = {valueInfo("flipped_value_info", OnnxType::Undefined, {2}), valueInfo("flipped_cast", OnnxType::Undefined, {}), valueInfo("flipped_dims", OnnxType::Undefined, {1}), valueInfo("flipped_mask", OnnxType::Undefined, {2}), valueInfo("shifted_constant", OnnxType::Undefined, {2})};
    Graph g = importModel("vknn_import_width_value_info_producers", parts);
    expectWidth(g, "not_value_info", 8, 0);
    expectWidth(g, "not_cast_chain", 16, 1);
    expectWidth(g, "not_shape", 64, 1);
    expectWidth(g, "not_bool", 8, 0);
    expectWidth(g, "shift_constant", 64, 0);
}

TEST(OnnxImport, UnresolvedIntegerWidthLeavesDefaults) {
    // An operand with no declared type and no typing producer (an untyped input through Relu), or a float
    // operand, stamps nothing: the kernels read the 64-bit signed defaults.
    GraphParts parts;
    parts.inputs = {valueInfo("untyped", OnnxType::Undefined, {2}), valueInfo("f", OnnxType::Float, {2})};
    parts.nodes  = {
        graphNode("Relu", "relu", {"untyped"}, {"activated"}),
        graphNode("BitwiseNot", "not_unresolved", {"activated"}, {"flipped"}),
        graphNode("BitShift", "shift_float", {"f", "f"}, {"shifted"}, {stringAttribute("direction", "LEFT")}),
    };
    parts.outputs = {valueInfo("flipped", OnnxType::Undefined, {2}), valueInfo("shifted", OnnxType::Undefined, {2})};
    Graph g       = importModel("vknn_import_width_unresolved", parts);
    for (const char *name: {"not_unresolved", "shift_float"})
    {
        const Node *node = findNode(g, name);
        ASSERT_NE(node, nullptr);
        EXPECT_FALSE(node->attr.has(bitwise::kIntBitsAttr)) << name;
        EXPECT_FALSE(node->attr.has(bitwise::kIntSignedAttr)) << name;
        const bitwise::IntegerWidth width = bitwise::readIntegerWidth(*node);
        EXPECT_EQ(width.bits, bitwise::kInt64Bits) << name;
        EXPECT_TRUE(width.isSigned) << name;
    }
}
