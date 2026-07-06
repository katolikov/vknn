// ONNX importer wire-level tests: TensorProto / AttributeProto messages are encoded by hand
// (protobuf varint/length-delimited framing) and decoded through TensorProtoParser / NodeParser,
// asserting integer payloads widen to exact fp32 values for every dtype x payload-field
// combination (raw_data bytes vs the typed int32_data varint array).
#include "import/onnx/onnx_node_parser.h"
#include "import/onnx/onnx_tensor_parser.h"
#include "vknn/dtype.h"
#include "vknn/graph.h"
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
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
    const std::vector<int8_t> v{-128, -1, 0, 1, 127};
    std::vector<uint8_t>      raw(v.size());
    std::memcpy(raw.data(), v.data(), v.size());
    auto f = decodeF32(protoWithRaw(OnnxType::Int8, {(int64_t) v.size()}, raw), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], (float) v[i]) << "i=" << i;
    }
}

TEST(OnnxTensorProto, Uint8RawWidensExact) {
    const std::vector<uint8_t> v{0, 1, 128, 255};
    auto                       f = decodeF32(protoWithRaw(OnnxType::Uint8, {(int64_t) v.size()}, v), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], (float) v[i]) << "i=" << i;
    }
}

TEST(OnnxTensorProto, Int32RawWidensExact) {
    const std::vector<int32_t> v{std::numeric_limits<int32_t>::min(), -7, 0, 1, 1 << 24, std::numeric_limits<int32_t>::max()};
    std::vector<uint8_t>       raw(v.size() * 4);
    std::memcpy(raw.data(), v.data(), raw.size());
    auto f = decodeF32(protoWithRaw(OnnxType::Int32, {(int64_t) v.size()}, raw), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], (float) v[i]) << "i=" << i;
    }
}

TEST(OnnxTensorProto, BoolRawWidensToZeroOne) {
    const std::vector<uint8_t> v{0, 1, 1, 0};
    auto                       f = decodeF32(protoWithRaw(OnnxType::Bool, {(int64_t) v.size()}, v), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], v[i] ? 1.0f : 0.0f) << "i=" << i;
    }
}

TEST(OnnxTensorProto, TruncatedRawLeavesTailZero) {
    // 3 payload bytes for a 5-element tensor: decoded head, zero tail (clamped copy contract).
    const std::vector<int8_t> v{-3, 4, -5};
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
    const std::vector<int64_t> v{-128, 127, -1, 0, 42};
    auto                       f = decodeF32(protoWithInt32Data(OnnxType::Int8, {(int64_t) v.size()}, v, true), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], (float) v[i]) << "i=" << i;
    }
}

TEST(OnnxTensorProto, Uint8Int32DataPacked) {
    const std::vector<int64_t> v{0, 255, 7, 128};
    auto                       f = decodeF32(protoWithInt32Data(OnnxType::Uint8, {(int64_t) v.size()}, v, true), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], (float) v[i]) << "i=" << i;
    }
}

TEST(OnnxTensorProto, Int32Int32DataPacked) {
    const std::vector<int64_t> v{std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(), -123456, 1 << 24};
    auto                       f = decodeF32(protoWithInt32Data(OnnxType::Int32, {(int64_t) v.size()}, v, true), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], (float) (int32_t) v[i]) << "i=" << i;
    }
}

TEST(OnnxTensorProto, BoolInt32DataPacked) {
    const std::vector<int64_t> v{0, 1, 1, 0};
    auto                       f = decodeF32(protoWithInt32Data(OnnxType::Bool, {(int64_t) v.size()}, v, true), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        EXPECT_EQ(f[i], v[i] ? 1.0f : 0.0f) << "i=" << i;
    }
}

TEST(OnnxTensorProto, Float16Int32DataDecodesBits) {
    // FLOAT16 rides int32_data as raw bit patterns, not numeric values.
    const std::vector<float> want{1.0f, -2.0f, 0.5f, 65504.0f};
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
    const std::vector<int64_t> v{-5, 6};
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
    const std::vector<int8_t> v{-128, 0, 127};
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
    const std::vector<int64_t> v{0, 200, 255};
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
    // NodeProto: field 1 = input (repeated string), field 4 = op_type.
    std::vector<uint8_t> nodeProto(const char *opType, const std::vector<std::string> &inputs) {
        std::vector<uint8_t> msg;
        for (const std::string &in: inputs)
        {
            putLenField(msg, 1, std::vector<uint8_t>(in.begin(), in.end()));
        }
        std::string t(opType);
        putLenField(msg, 4, std::vector<uint8_t>(t.begin(), t.end()));
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
