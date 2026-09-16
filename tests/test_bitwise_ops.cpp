// BitwiseAnd / BitwiseOr / BitwiseXor / BitwiseNot / BitShift on the CPU oracle, and host checks of the
// GLSL kernels' integer math.
//
// The CPU tests run each op through a Session (runtime int64 and fp32-carried operands), through
// constFold (all-initializer int64 graphs), and directly through the registered kernel (zero-extent and
// rank-0 broadcasts), and check the thread partition leaves the output bytes unchanged.
//
// GLSL never runs on the host, so the second half carries a C++ transcription of every function of
// shaders/bitwise_int.glsl and of the per-element value functions of shaders/bitwise.comp,
// shaders/bitwise_not.comp and shaders/bitshift.comp. Each transcription keeps the GLSL statements in the
// same order with the same names (GLSL builtins spelled through the glsl:: helpers below, control flow
// braced by the C++ format) so a reviewer can diff the two, and is swept bit for bit against the CPU
// oracle's fp32 lane over integer operands within +-2^24, fractional and NaN operands, every width, and
// shift counts 0..70.
#include "backend/cpu/bitwise_int.h"
#include "backend/cpu/parallel.h"
#include "core/bitwise_attrs.h"
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

// The transcriptions round every floating-point operation separately, as the GLSL `precise` qualifier makes
// the shaders do, so this file forbids contraction into fused multiply-add instructions.
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

using namespace vknn;

namespace {

    constexpr int64_t kTwoPow24 = int64_t {1} << 24;
    constexpr int64_t kTwoPow40 = int64_t {1} << 40;

    Attr intAttr(int64_t value) {
        Attr a;
        a.kind = Attr::Int;
        a.i    = value;
        return a;
    }
    Attr stringAttr(const std::string &value) {
        Attr a;
        a.kind = Attr::String;
        a.str  = value;
        return a;
    }

    TensorId addInput(Graph &g, const std::string &name, const Shape &shape, DType dtype) {
        TensorDesc d;
        d.name      = name;
        d.shape     = shape;
        d.dtype     = dtype;
        d.isInput   = true;
        TensorId id = g.addTensor(d);
        g.inputs.push_back(id);
        return id;
    }

    TensorId addInt64Initializer(Graph &g, const std::string &name, const Shape &shape, const std::vector<int64_t> &values) {
        TensorDesc d;
        d.name          = name;
        d.shape         = shape;
        d.dtype         = DType::Int64;
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) values.size(), DType::Int64);
        for (size_t k = 0; k < values.size(); ++k)
        {
            hb.i64()[k] = values[k];
        }
        g.initializers[id] = hb;
        return id;
    }

    TensorId addFloatInitializer(Graph &g, const std::string &name, const Shape &shape, const std::vector<float> &values) {
        TensorDesc d;
        d.name          = name;
        d.shape         = shape;
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) values.size(), DType::Float32);
        for (size_t k = 0; k < values.size(); ++k)
        {
            hb.f32()[k] = values[k];
        }
        g.initializers[id] = hb;
        return id;
    }

    TensorId addOutput(Graph &g, const std::string &name, DType dtype) {
        TensorDesc d;
        d.name      = name;
        d.dtype     = dtype;
        d.isOutput  = true;
        TensorId id = g.addTensor(d);
        g.outputs.push_back(id);
        return id;
    }

    Node &addNode(Graph &g, OpType type, const std::string &name, std::vector<TensorId> inputs, std::vector<TensorId> outputs) {
        Node n;
        n.type    = type;
        n.name    = name;
        n.inputs  = std::move(inputs);
        n.outputs = std::move(outputs);
        g.nodes.push_back(std::move(n));
        return g.nodes.back();
    }

    void setWidth(Node &node, int bits, bool isSigned) {
        node.attr.map[bitwise::kIntBitsAttr]   = intAttr(bits);
        node.attr.map[bitwise::kIntSignedAttr] = intAttr(isSigned ? 1 : 0);
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

    // Run `g` on the CPU backend; returns the run status and fills `outs`.
    Status runCpu(Graph g, const std::vector<IOTensor> &inputs, std::vector<IOTensor> &outs, int threads = 1) {
        Config cfg;
        cfg.backend    = BackendKind::Cpu;
        cfg.cpuThreads = threads;
        auto sess      = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return Status::RuntimeError;
        }
        return sess->run(inputs, outs);
    }

    std::vector<int64_t> int64Values(const IOTensor &t) {
        EXPECT_EQ(t.dtype, DType::Int64);
        std::vector<int64_t> v(t.data.size() / sizeof(int64_t));
        std::memcpy(v.data(), t.data.data(), v.size() * sizeof(int64_t));
        return v;
    }

    std::vector<float> floatValues(const IOTensor &t) {
        EXPECT_EQ(t.dtype, DType::Float32);
        std::vector<float> v(t.data.size() / sizeof(float));
        std::memcpy(v.data(), t.data.data(), v.size() * sizeof(float));
        return v;
    }

    // Run the registered CPU kernel for `node` on a hand-bound pool (no Session, so zero-extent and
    // rank-0 operands reach the kernel as-is).
    void runKernel(const Node &node, std::vector<RtTensor> &pool) {
        Graph       g;
        Config      cfg;
        ExecContext ctx;
        ctx.pool   = &pool;
        ctx.graph  = &g;
        ctx.config = &cfg;
        auto op    = CpuOpRegistry::instance().create(node.type);
        ASSERT_TRUE(op) << opTypeName(node.type);
        op->run(node, ctx);
    }

    RtTensor int64Rt(const Shape &shape, const std::vector<int64_t> &values) {
        RtTensor t;
        t.shape = shape;
        t.dtype = DType::Int64;
        t.host.resizeElems((int64_t) values.size(), DType::Int64);
        if (!values.empty())
        {
            std::memcpy(t.host.i64(), values.data(), values.size() * sizeof(int64_t));
        }
        t.hostValid = true;
        return t;
    }

    // Two-operand broadcast of int64 operands a {2,3} and b {3} through a Session: a runtime Int64 input
    // and an Int64 initializer, output declared Int64.
    std::vector<int64_t> runInt64Binary(OpType type, const std::vector<int64_t> &a, const std::vector<int64_t> &b, const std::function<void(Node &)> &configure = nullptr) {
        Graph    g;
        TensorId x = addInput(g, "x", {2, 3}, DType::Int64);
        TensorId c = addInt64Initializer(g, "c", {3}, b);
        TensorId y = addOutput(g, "y", DType::Int64);
        Node    &n = addNode(g, type, "op", {x, c}, {y});
        if (configure)
        {
            configure(n);
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(runCpu(std::move(g), {int64Tensor("x", {2, 3}, a)}, outs), Status::Ok);
        if (outs.empty())
        {
            return {};
        }
        EXPECT_EQ(outs[0].shape, (Shape {2, 3}));
        return int64Values(outs[0]);
    }

} // namespace

// ---------------------------------------------------------------------------------------------------
// CPU oracle
// ---------------------------------------------------------------------------------------------------

TEST(BitwiseOps, AndOrXorInt64RuntimeInputBroadcastsInitializer) {
    // Negative operands and values past 2^24 stay exact on the int64 path; b broadcasts along the rows.
    const std::vector<int64_t> a {-1, -8, (kTwoPow40 | 0x0F0F), 0, 12345, std::numeric_limits<int64_t>::min()};
    const std::vector<int64_t> b {0x00FF, -256, kTwoPow24 + 3};
    auto                       reference = [&](OpType type) {
        std::vector<int64_t> r(a.size());
        for (size_t i = 0; i < a.size(); ++i)
        {
            const int64_t rhs = b[i % b.size()];
            r[i]              = type == OpType::BitwiseAnd ? (a[i] & rhs) : (type == OpType::BitwiseOr ? (a[i] | rhs) : (a[i] ^ rhs));
        }
        return r;
    };
    for (OpType type: {OpType::BitwiseAnd, OpType::BitwiseOr, OpType::BitwiseXor})
    {
        EXPECT_EQ(runInt64Binary(type, a, b), reference(type)) << opTypeName(type);
    }
}

TEST(BitwiseOps, AndOrXorInt64InitializersConstFold) {
    // All-initializer int64 graph: constFold evaluates the kernel and keeps the result Int64. A rank-0
    // operand broadcasts over the vector and a rank-0 pair folds to a rank-0 result.
    Graph    g;
    TensorId vec    = addInt64Initializer(g, "vec", {4}, {-5, 6, (int64_t {1} << 50) + 1, -(int64_t {1} << 33)});
    TensorId scalar = addInt64Initializer(g, "scalar", {}, {-3});
    TensorId other  = addInt64Initializer(g, "other", {}, {10});
    TensorId yAnd   = g.addTensor({"and_out"});
    TensorId yOr    = g.addTensor({"or_out"});
    TensorId yXor   = g.addTensor({"xor_out"});
    TensorId yRank0 = g.addTensor({"rank0_out"});
    addNode(g, OpType::BitwiseAnd, "and", {vec, scalar}, {yAnd});
    addNode(g, OpType::BitwiseOr, "or", {scalar, vec}, {yOr});
    addNode(g, OpType::BitwiseXor, "xor", {vec, scalar}, {yXor});
    addNode(g, OpType::BitwiseXor, "xor_rank0", {scalar, other}, {yRank0});
    g.outputs = {yAnd, yOr, yXor, yRank0};
    inferShapes(g, 1);
    constFold(g);
    EXPECT_TRUE(g.nodes.empty()) << "all-constant bitwise ops fold away";
    const int64_t v[4] = {-5, 6, (int64_t {1} << 50) + 1, -(int64_t {1} << 33)};
    for (TensorId out: {yAnd, yOr, yXor})
    {
        ASSERT_TRUE(g.isInitializer(out));
        EXPECT_EQ(g.desc(out).dtype, DType::Int64);
        EXPECT_EQ(g.desc(out).shape, (Shape {4}));
    }
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(g.initializers[yAnd].i64()[i], v[i] & -3) << i;
        EXPECT_EQ(g.initializers[yOr].i64()[i], -3 | v[i]) << i;
        EXPECT_EQ(g.initializers[yXor].i64()[i], v[i] ^ -3) << i;
    }
    ASSERT_TRUE(g.isInitializer(yRank0));
    EXPECT_TRUE(g.desc(yRank0).shape.empty());
    EXPECT_EQ(g.initializers[yRank0].i64()[0], -3 ^ 10);
}

TEST(BitwiseOps, AndOrXorFloatCarriedTruncatesAndBroadcasts) {
    // fp32-carried operands (the IR's int32/uint8 storage): truncation toward zero, NaN reads 0, and the
    // result stores as fp32. The initializer {2,1} broadcasts against x {1,4}.
    const float                nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float>   x {-6.0f, 5.9f, -3.7f, nan};
    const std::vector<float>   c {12.0f, -1.5f};
    const std::vector<int64_t> xi {-6, 5, -3, 0};
    const std::vector<int64_t> ci {12, -1};
    for (OpType type: {OpType::BitwiseAnd, OpType::BitwiseOr, OpType::BitwiseXor})
    {
        Graph    g;
        TensorId in = addInput(g, "x", {1, 4}, DType::Float32);
        TensorId k  = addFloatInitializer(g, "c", {2, 1}, c);
        TensorId y  = addOutput(g, "y", DType::Float32);
        addNode(g, type, "op", {in, k}, {y});
        std::vector<IOTensor> outs;
        ASSERT_EQ(runCpu(std::move(g), {floatTensor("x", {1, 4}, x)}, outs), Status::Ok);
        ASSERT_EQ(outs.size(), 1u);
        EXPECT_EQ(outs[0].shape, (Shape {2, 4}));
        const std::vector<float> got = floatValues(outs[0]);
        ASSERT_EQ(got.size(), 8u);
        for (size_t row = 0; row < 2; ++row)
        {
            for (size_t col = 0; col < 4; ++col)
            {
                const int64_t lhs = xi[col], rhs = ci[row];
                const int64_t want = type == OpType::BitwiseAnd ? (lhs & rhs) : (type == OpType::BitwiseOr ? (lhs | rhs) : (lhs ^ rhs));
                EXPECT_EQ(got[row * 4 + col], (float) want) << opTypeName(type) << " row " << row << " col " << col;
            }
        }
    }
}

TEST(BitwiseOps, MixedInt64AndFloatOperandsStoreInt64) {
    // Either operand Int64 -> int64 result; the fp32 operand truncates toward zero.
    Graph    g;
    TensorId x = addInput(g, "x", {3}, DType::Int64);
    TensorId c = addFloatInitializer(g, "c", {3}, {7.9f, -2.5f, 1e10f});
    TensorId y = addOutput(g, "y", DType::Int64);
    addNode(g, OpType::BitwiseXor, "xor", {x, c}, {y});
    std::vector<IOTensor> outs;
    ASSERT_EQ(runCpu(std::move(g), {int64Tensor("x", {3}, {1, 1, kTwoPow40})}, outs), Status::Ok);
    ASSERT_EQ(outs.size(), 1u);
    EXPECT_EQ(int64Values(outs[0]), (std::vector<int64_t> {1 ^ 7, 1 ^ -2, kTwoPow40 ^ int64_t {10000000000}}));
}

TEST(BitwiseOps, ZeroExtentAndRankZeroBroadcast) {
    // A 0 extent broadcasts to 0 (never to 1); a rank-0 operand broadcasts over any shape.
    Node node;
    node.type    = OpType::BitwiseOr;
    node.name    = "or";
    node.inputs  = {0, 1};
    node.outputs = {2};
    std::vector<RtTensor> pool(3);
    pool[0] = int64Rt({0, 3}, {});
    pool[1] = int64Rt({3}, {1, 2, 3});
    runKernel(node, pool);
    EXPECT_EQ(pool[2].shape, (Shape {0, 3}));
    EXPECT_EQ(pool[2].dtype, DType::Int64);
    EXPECT_EQ(pool[2].host.bytes.size(), 0u);

    pool[0] = int64Rt({}, {-16});
    pool[1] = int64Rt({2, 2}, {1, 2, 3, 4});
    runKernel(node, pool);
    EXPECT_EQ(pool[2].shape, (Shape {2, 2}));
    ASSERT_EQ(pool[2].host.bytes.size(), 4 * sizeof(int64_t));
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(pool[2].host.i64()[i], -16 | (i + 1)) << i;
    }
}

TEST(BitwiseOps, BitwiseNotSignedAndUnsignedWidths) {
    // int64 operands so every result is exact: unsigned narrower than 64 bits keeps its low bits
    // (mask - v); signed widths and the 64-bit width complement all bits.
    const std::vector<int64_t> operand {0, 5, 200, 255, 65535, 4294967295LL};
    struct Case {
        int                  bits;
        bool                 isSigned;
        std::vector<int64_t> want;
    };
    const std::vector<Case> cases {
        {8, false, {255, 250, 55, 0, 0, 0}},
        {8, true, {-1, -6, -201, -256, -65536, -4294967296LL}},
        {16, false, {65535, 65530, 65335, 65280, 0, 0}},
        {16, true, {-1, -6, -201, -256, -65536, -4294967296LL}},
        {32, false, {4294967295LL, 4294967290LL, 4294967095LL, 4294967040LL, 4294901760LL, 0}},
        {32, true, {-1, -6, -201, -256, -65536, -4294967296LL}},
        {64, false, {-1, -6, -201, -256, -65536, -4294967296LL}},
        {64, true, {-1, -6, -201, -256, -65536, -4294967296LL}},
    };
    for (const Case &c: cases)
    {
        Graph    g;
        TensorId x = addInput(g, "x", {6}, DType::Int64);
        TensorId y = addOutput(g, "y", DType::Int64);
        Node    &n = addNode(g, OpType::BitwiseNot, "not", {x}, {y});
        setWidth(n, c.bits, c.isSigned);
        std::vector<IOTensor> outs;
        ASSERT_EQ(runCpu(std::move(g), {int64Tensor("x", {6}, operand)}, outs), Status::Ok);
        ASSERT_EQ(outs.size(), 1u);
        EXPECT_EQ(int64Values(outs[0]), c.want) << "bits " << c.bits << " signed " << c.isSigned;
    }
}

TEST(BitwiseOps, BitwiseNotDefaultsAndFloatCarried) {
    // Absent attributes are 64-bit signed; an fp32-carried uint8 operand stores its fp32 complement.
    {
        Graph    g;
        TensorId x = addInput(g, "x", {3}, DType::Int64);
        TensorId y = addOutput(g, "y", DType::Int64);
        addNode(g, OpType::BitwiseNot, "not", {x}, {y});
        std::vector<IOTensor> outs;
        ASSERT_EQ(runCpu(std::move(g), {int64Tensor("x", {3}, {0, -1, kTwoPow40})}, outs), Status::Ok);
        EXPECT_EQ(int64Values(outs[0]), (std::vector<int64_t> {-1, 0, ~kTwoPow40}));
    }
    {
        Graph    g;
        TensorId x = addInput(g, "x", {4}, DType::Float32);
        TensorId y = addOutput(g, "y", DType::Float32);
        Node    &n = addNode(g, OpType::BitwiseNot, "not", {x}, {y});
        setWidth(n, 8, false);
        std::vector<IOTensor> outs;
        ASSERT_EQ(runCpu(std::move(g), {floatTensor("x", {4}, {0.0f, 15.0f, 255.0f, 7.5f})}, outs), Status::Ok);
        EXPECT_EQ(floatValues(outs[0]), (std::vector<float> {255.0f, 240.0f, 0.0f, 248.0f}));
    }
}

TEST(BitwiseOps, BitwiseNotInvalidWidthIsInvalidArgument) {
    Graph    g;
    TensorId x                        = addInput(g, "x", {2}, DType::Int64);
    TensorId y                        = addOutput(g, "y", DType::Int64);
    Node    &n                        = addNode(g, OpType::BitwiseNot, "not", {x}, {y});
    n.attr.map[bitwise::kIntBitsAttr] = intAttr(12);
    std::vector<IOTensor> outs;
    EXPECT_NE(runCpu(std::move(g), {int64Tensor("x", {2}, {1, 2})}, outs), Status::Ok);
    Node probe;
    probe.type                            = OpType::BitwiseNot;
    probe.name                            = "probe";
    probe.attr.map[bitwise::kIntBitsAttr] = intAttr(12);
    try
    {
        bitwise::readIntegerWidth(probe);
        ADD_FAILURE() << "an int_bits of 12 must be rejected";
    } catch (const Error &e)
    {
        EXPECT_EQ(e.status(), Status::InvalidArgument);
        EXPECT_NE(std::string(e.what()).find("BitwiseNot 'probe'"), std::string::npos) << e.what();
    }
}

TEST(BitwiseOps, BitShiftLeftUint8Wraps) {
    // fp32-carried uint8: LEFT keeps the low 8 bits (200 << 1 = 144); a count of 8 or more, or a
    // negative count, yields 0.
    Graph    g;
    TensorId x                          = addInput(g, "x", {6}, DType::Float32);
    TensorId s                          = addFloatInitializer(g, "s", {6}, {1, 1, 7, 8, 9, -1});
    TensorId y                          = addOutput(g, "y", DType::Float32);
    Node    &n                          = addNode(g, OpType::BitShift, "shl", {x, s}, {y});
    n.attr.map[bitwise::kDirectionAttr] = stringAttr("LEFT");
    setWidth(n, 8, false);
    std::vector<IOTensor> outs;
    ASSERT_EQ(runCpu(std::move(g), {floatTensor("x", {6}, {200, 255, 1, 1, 1, 1})}, outs), Status::Ok);
    ASSERT_EQ(outs.size(), 1u);
    EXPECT_EQ(floatValues(outs[0]), (std::vector<float> {144, 254, 128, 0, 0, 0}));
}

TEST(BitwiseOps, BitShiftCountAtOrAboveWidthIsZero) {
    for (int bits: {8, 16, 32, 64})
    {
        for (const char *direction: {"LEFT", "RIGHT"})
        {
            const std::vector<int64_t> counts {bits - 1, bits, bits + 1, 70};
            std::vector<int64_t>       want(counts.size());
            for (size_t k = 0; k < counts.size(); ++k)
            {
                want[k] = cpu::bitwise::bitShift(1, counts[k], std::string(direction) == "LEFT" ? bitwise::ShiftDirection::Left : bitwise::ShiftDirection::Right, bits);
            }
            Graph    g;
            TensorId x                          = addInput(g, "x", {4}, DType::Int64);
            TensorId s                          = addInt64Initializer(g, "s", {4}, counts);
            TensorId y                          = addOutput(g, "y", DType::Int64);
            Node    &n                          = addNode(g, OpType::BitShift, "shift", {x, s}, {y});
            n.attr.map[bitwise::kDirectionAttr] = stringAttr(direction);
            setWidth(n, bits, false);
            std::vector<IOTensor> outs;
            ASSERT_EQ(runCpu(std::move(g), {int64Tensor("x", {4}, {1, 1, 1, 1})}, outs), Status::Ok);
            const std::vector<int64_t> got = int64Values(outs[0]);
            ASSERT_EQ(got.size(), 4u);
            EXPECT_EQ(got[1], 0) << direction << " by the width " << bits;
            EXPECT_EQ(got[2], 0) << direction << " past the width " << bits;
            EXPECT_EQ(got[3], 0) << direction << " by 70 at width " << bits;
            EXPECT_EQ(got, want);
        }
        EXPECT_EQ(cpu::bitwise::bitShift(1, bits - 1, bitwise::ShiftDirection::Left, bits), bits == 64 ? std::numeric_limits<int64_t>::min() : (int64_t {1} << (bits - 1)));
    }
}

TEST(BitwiseOps, BitShiftRightOfUint64BitPatternIsLogical) {
    // A UINT64 value at or above 2^63 rides int64 storage as its two's-complement bits; RIGHT at 64 bits
    // is a logical shift of those bits, never an arithmetic one.
    const int64_t highBits = cpu::bitwise::signedFromBits(0xF000000000000010ULL);
    auto          shifted  = runInt64Binary(OpType::BitShift, {highBits, highBits, highBits, highBits, highBits, highBits}, {0, 4, 63}, [](Node &n) {
        n.attr.map[bitwise::kDirectionAttr] = stringAttr("RIGHT");
        setWidth(n, 64, false);
    });
    ASSERT_EQ(shifted.size(), 6u);
    EXPECT_EQ(shifted[0], highBits);
    EXPECT_EQ(shifted[1], cpu::bitwise::signedFromBits(0x0F00000000000001ULL));
    EXPECT_EQ(shifted[2], 1);
    // LEFT at 64 bits wraps into the sign bit.
    auto          left        = runInt64Binary(OpType::BitShift, {3, 3, 3, -1, -1, -1}, {1, 62, 63}, [](Node &n) {
        n.attr.map[bitwise::kDirectionAttr] = stringAttr("LEFT");
        setWidth(n, 64, false);
    });
    const int64_t twoHighBits = std::numeric_limits<int64_t>::min() + (int64_t {1} << 62); // 0xC000000000000000
    EXPECT_EQ(left, (std::vector<int64_t> {6, twoHighBits, std::numeric_limits<int64_t>::min(), -2, twoHighBits, std::numeric_limits<int64_t>::min()}));
}

TEST(BitwiseOps, BitShiftUint32RightReducesOperandToWidth) {
    // RIGHT shifts the operand's low int_bits bits: at 32 bits a negative int64 reads as its low 32 bits.
    auto got = runInt64Binary(OpType::BitShift, {-1, -1, -1, int64_t {1} << 33, 256, 256}, {0, 4, 31}, [](Node &n) {
        n.attr.map[bitwise::kDirectionAttr] = stringAttr("RIGHT");
        setWidth(n, 32, false);
    });
    EXPECT_EQ(got, (std::vector<int64_t> {4294967295LL, 268435455LL, 1, 0, 16, 0}));
}

TEST(BitwiseOps, BitShiftInt64ConstFold) {
    Graph    g;
    TensorId value                      = addInt64Initializer(g, "value", {3}, {1, 5, int64_t {1} << 62});
    TensorId count                      = addInt64Initializer(g, "count", {3}, {3, 64, 1});
    TensorId y                          = g.addTensor({"shifted"});
    Node    &n                          = addNode(g, OpType::BitShift, "shl", {value, count}, {y});
    n.attr.map[bitwise::kDirectionAttr] = stringAttr("LEFT");
    g.outputs                           = {y};
    inferShapes(g, 1);
    constFold(g);
    ASSERT_TRUE(g.isInitializer(y));
    EXPECT_EQ(g.desc(y).dtype, DType::Int64);
    EXPECT_EQ(g.initializers[y].i64()[0], 8);
    EXPECT_EQ(g.initializers[y].i64()[1], 0) << "count 64 at the default 64-bit width";
    EXPECT_EQ(g.initializers[y].i64()[2], std::numeric_limits<int64_t>::min());
}

TEST(BitwiseOps, BitShiftInvalidDirectionIsInvalidArgument) {
    // ONNX spells the direction exactly "LEFT" or "RIGHT": another spelling, a missing attribute, or a
    // non-string attribute (read through gets as the empty string) fails the run, and the kernel names
    // the node in an InvalidArgument error.
    struct Case {
        std::string label;
        bool        present;
        Attr        direction;
    };
    const std::vector<Case> cases {
        {"lowercase", true, stringAttr("left")}, {"unknown", true, stringAttr("UP")}, {"empty", true, stringAttr("")}, {"missing", false, Attr {}},
        {"integer", true, intAttr(1)},
    };
    for (const Case &c: cases)
    {
        Graph    g;
        TensorId x = addInput(g, "x", {2}, DType::Int64);
        TensorId s = addInt64Initializer(g, "s", {2}, {1, 2});
        TensorId y = addOutput(g, "y", DType::Int64);
        Node    &n = addNode(g, OpType::BitShift, "shift_" + c.label, {x, s}, {y});
        if (c.present)
        {
            n.attr.map[bitwise::kDirectionAttr] = c.direction;
        }
        Node node    = n;
        node.inputs  = {0, 1};
        node.outputs = {2};
        std::vector<IOTensor> outs;
        EXPECT_NE(runCpu(std::move(g), {int64Tensor("x", {2}, {1, 2})}, outs), Status::Ok) << c.label;

        std::vector<RtTensor> pool(3);
        pool[0] = int64Rt({2}, {1, 2});
        pool[1] = int64Rt({2}, {1, 2});
        try
        {
            runKernel(node, pool);
            ADD_FAILURE() << c.label << ": the kernel must reject the direction";
        } catch (const Error &e)
        {
            EXPECT_EQ(e.status(), Status::InvalidArgument) << c.label;
            EXPECT_NE(std::string(e.what()).find("BitShift '" + node.name + "'"), std::string::npos) << e.what();
        }
    }
}

TEST(BitwiseOps, BitwiseNotInt64InitializerConstFold) {
    // An all-initializer BitwiseNot folds through the kernel at the stamped width, int64 in, int64 out;
    // a rank-0 operand folds to a rank-0 result.
    Graph    g;
    TensorId vec    = addInt64Initializer(g, "vec", {4}, {-1, 0, 200, kTwoPow40 + 7});
    TensorId scalar = addInt64Initializer(g, "scalar", {}, {kTwoPow40});
    TensorId yVec   = g.addTensor({"vec_not"});
    TensorId yRank0 = g.addTensor({"scalar_not"});
    setWidth(addNode(g, OpType::BitwiseNot, "not_uint8", {vec}, {yVec}), 8, false);
    addNode(g, OpType::BitwiseNot, "not_default", {scalar}, {yRank0});
    g.outputs = {yVec, yRank0};
    inferShapes(g, 1);
    constFold(g);
    EXPECT_TRUE(g.nodes.empty()) << "all-constant BitwiseNot folds away";
    ASSERT_TRUE(g.isInitializer(yVec));
    EXPECT_EQ(g.desc(yVec).dtype, DType::Int64);
    const std::vector<int64_t> wantVec {0, 255, 55, 248};
    for (size_t i = 0; i < wantVec.size(); ++i)
    {
        EXPECT_EQ(g.initializers[yVec].i64()[i], wantVec[i]) << i;
    }
    ASSERT_TRUE(g.isInitializer(yRank0));
    EXPECT_TRUE(g.desc(yRank0).shape.empty());
    EXPECT_EQ(g.initializers[yRank0].i64()[0], ~kTwoPow40);
}

TEST(BitwiseOps, BitShiftAttributesRoundTripVxm) {
    // The string `direction` and the stamped `int_bits` / `int_signed` survive a .vxm save and load, and
    // the loaded graph computes the same RIGHT shift at the stamped width.
    auto build = [] {
        Graph    g;
        TensorId x                          = addInput(g, "x", {4}, DType::Float32);
        TensorId s                          = addFloatInitializer(g, "s", {4}, {0, 1, 3, 8});
        TensorId y                          = addOutput(g, "y", DType::Float32);
        Node    &n                          = addNode(g, OpType::BitShift, "shift_right_u8", {x, s}, {y});
        n.attr.map[bitwise::kDirectionAttr] = stringAttr("RIGHT");
        setWidth(n, 8, false);
        return g;
    };
    const std::string path = testing::TempDir() + "vknn_bitshift_attributes.vxm";
    ASSERT_TRUE(saveGraphBin(build(), path));
    Graph loaded;
    ASSERT_TRUE(loadGraphBin(loaded, path));
    ASSERT_EQ(loaded.nodes.size(), 1u);
    const Node &node = loaded.nodes[0];
    EXPECT_EQ(node.type, OpType::BitShift);
    ASSERT_TRUE(node.attr.has(bitwise::kDirectionAttr));
    EXPECT_EQ(node.attr.map.at(bitwise::kDirectionAttr).kind, Attr::String);
    EXPECT_EQ(node.attr.gets(bitwise::kDirectionAttr), "RIGHT");
    EXPECT_EQ(node.attr.geti(bitwise::kIntBitsAttr), 8);
    EXPECT_EQ(node.attr.geti(bitwise::kIntSignedAttr), 0);
    std::vector<IOTensor> outs;
    ASSERT_EQ(runCpu(std::move(loaded), {floatTensor("x", {4}, {-1.0f, 200.0f, 255.0f, 255.0f})}, outs), Status::Ok);
    ASSERT_EQ(outs.size(), 1u);
    EXPECT_EQ(floatValues(outs[0]), (std::vector<float> {255.0f, 100.0f, 31.0f, 0.0f})) << "-1 reads as its low 8 bits (255)";
}

TEST(BitwiseOps, ThreadPartitionKeepsBytesIdentical) {
    // Shapes past cpu::kMinChunkOps so the sweeps partition; the extents divide evenly by none of the
    // probed thread counts.
    const Shape        shape {7, 13, 1447};
    const int64_t      count = 7 * 13 * 1447;
    std::vector<float> xs((size_t) count);
    uint32_t           state = 12345u;
    for (float &v: xs)
    {
        state = state * 1664525u + 1013904223u;
        v     = (float) ((int32_t) (state >> 8) % 70001 - 35000);
    }
    const std::vector<std::function<Graph()>> builds {
        [&] {
            Graph    g;
            TensorId x = addInput(g, "x", shape, DType::Float32);
            TensorId c = addFloatInitializer(g, "c", {1447}, std::vector<float>(xs.begin(), xs.begin() + 1447));
            TensorId y = addOutput(g, "y", DType::Float32);
            addNode(g, OpType::BitwiseXor, "xor", {x, c}, {y});
            return g;
        },
        [&] {
            Graph    g;
            TensorId x                          = addInput(g, "x", shape, DType::Float32);
            TensorId c                          = addFloatInitializer(g, "c", {13, 1}, {0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 23});
            TensorId y                          = addOutput(g, "y", DType::Float32);
            Node    &n                          = addNode(g, OpType::BitShift, "shift", {x, c}, {y});
            n.attr.map[bitwise::kDirectionAttr] = stringAttr("LEFT");
            setWidth(n, 16, false);
            return g;
        },
        [&] {
            Graph    g;
            TensorId x = addInput(g, "x", shape, DType::Float32);
            TensorId y = addOutput(g, "y", DType::Float32);
            Node    &n = addNode(g, OpType::BitwiseNot, "not", {x}, {y});
            setWidth(n, 32, false);
            return g;
        },
    };
    const std::vector<int> threadCounts {2, 3, 5, 8};
    for (size_t b = 0; b < builds.size(); ++b)
    {
        const int64_t         dispatchesBefore = cpu::detail::poolDispatches();
        std::vector<IOTensor> reference;
        ASSERT_EQ(runCpu(builds[b](), {floatTensor("x", shape, xs)}, reference, 1), Status::Ok);
        ASSERT_EQ(reference.size(), 1u);
        for (int threads: threadCounts)
        {
            std::vector<IOTensor> outs;
            ASSERT_EQ(runCpu(builds[b](), {floatTensor("x", shape, xs)}, outs, threads), Status::Ok);
            ASSERT_EQ(outs.size(), 1u);
            ASSERT_EQ(outs[0].data.size(), reference[0].data.size());
            EXPECT_EQ(0, std::memcmp(outs[0].data.data(), reference[0].data.data(), reference[0].data.size())) << "graph " << b << " threads " << threads;
        }
        EXPECT_GT(cpu::detail::poolDispatches(), dispatchesBefore) << "graph " << b << ": the sweep did not partition";
    }
}

// ---------------------------------------------------------------------------------------------------
// Host transcriptions of the GLSL kernels
// ---------------------------------------------------------------------------------------------------

namespace glsl {

    // GLSL builtins used by the transcriptions, with GLSL semantics on the values they receive (NaN never
    // reaches clamp: every caller tests isnan first).
    inline float trunc(float x) {
        return std::trunc(x);
    }
    inline float floor(float x) {
        return std::floor(x);
    }
    inline float clamp(float x, float minVal, float maxVal) {
        return std::min(std::max(x, minVal), maxVal);
    }
    inline bool isnan(float x) {
        return std::isnan(x);
    }

    // ---- transcription of shaders/bitwise_int.glsl ----

    const int   kInt64Bits             = 64;
    const int   kShiftWordBits         = 32;
    const int   kFloatExactIntegerBits = 24;
    const float kTwoPow32              = 4294967296.0f;
    const float kTwoPow63              = 9223372036854775808.0f;
    const float kTwoPow64              = 18446744073709551616.0f;
    const float kInt32MinFloat         = -2147483648.0f;
    const float kInt32MaxFloat         = 2147483520.0f;

    float integerOperand(float value) {
        if (isnan(value))
        {
            return 0.0f;
        }
        float integer = trunc(clamp(value, -kTwoPow63, kTwoPow63));
        return integer == 0.0f ? 0.0f : integer;
    }

    int int32Operand(float value) {
        if (isnan(value))
        {
            return 0;
        }
        return int(trunc(clamp(value, kInt32MinFloat, kInt32MaxFloat)));
    }

    float powerOfTwo(int exponent) {
        float value     = 1.0f;
        int   remaining = exponent;
        if (remaining >= kShiftWordBits)
        {
            value *= kTwoPow32;
            remaining -= kShiftWordBits;
        }
        if (remaining >= kShiftWordBits)
        {
            value *= kTwoPow32;
            remaining -= kShiftWordBits;
        }
        float result = value * float(1u << unsigned(remaining));
        return result;
    }

    float residueModPowerOfTwo(float x, int bits) {
        float modulus = powerOfTwo(bits);
        float residue = x - floor(x / modulus) * modulus;
        return residue;
    }

    float wrapToWidth(float x, int bits) {
        if (bits < kInt64Bits)
        {
            return residueModPowerOfTwo(x, bits);
        }
        if (x >= -kTwoPow63 && x < kTwoPow63)
        {
            return x;
        }
        float unsignedResidue = x - floor(x / kTwoPow64) * kTwoPow64;
        float signedResidue   = unsignedResidue >= kTwoPow63 ? unsignedResidue - kTwoPow64 : unsignedResidue;
        return signedResidue;
    }

    // ---- transcription of shaders/bitwise.comp (kOperator is a parameter instead of a spec constant) ----

    const int kOperatorAnd = 0;
    const int kOperatorOr  = 1;
    const int kOperatorXor = 2;

    float bitwiseValue(float left, float right, int kOperator) {
        int lhs    = int32Operand(left);
        int rhs    = int32Operand(right);
        int result = kOperator == kOperatorAnd ? (lhs & rhs) : (kOperator == kOperatorOr ? (lhs | rhs) : (lhs ^ rhs));
        return float(result);
    }

    // ---- transcription of shaders/bitwise_not.comp ----

    float bitwiseNotValue(float value, int intBits, int intSigned) {
        float integer = integerOperand(value);
        if (intSigned != 0 || intBits >= kInt64Bits)
        {
            float complement = -integer - 1.0f;
            return complement;
        }
        if (intBits <= kFloatExactIntegerBits)
        {
            float mask    = powerOfTwo(intBits) - 1.0f;
            float flipped = mask - residueModPowerOfTwo(integer, intBits);
            return flipped;
        }
        float wideComplement = -integer - 1.0f;
        return residueModPowerOfTwo(wideComplement, intBits);
    }

    // ---- transcription of shaders/bitshift.comp ----

    float bitShiftValue(float value, float shift, int directionLeft, int intBits) {
        float shiftCount = integerOperand(shift);
        if (shiftCount < 0.0f || shiftCount >= float(intBits))
        {
            return 0.0f;
        }
        int   s       = int(shiftCount);
        float operand = integerOperand(value);
        if (directionLeft != 0)
        {
            float shifted = operand * powerOfTwo(s);
            return wrapToWidth(shifted, intBits);
        }
        if (s == 0)
        {
            return wrapToWidth(operand, intBits);
        }
        float quotient = floor(operand / powerOfTwo(s));
        return residueModPowerOfTwo(quotient, intBits - s);
    }

} // namespace glsl

namespace {

    // Operand values for the transcription sweeps: every integer in a dense window around 0, +-2^k and
    // +-2^k +- 1 up to +-2^24, pseudo-random integers within +-2^24, fractional values and NaN.
    std::vector<float> sweepOperands(int denseRadius, int randomCount) {
        std::vector<float> values;
        for (int v = -denseRadius; v <= denseRadius; ++v)
        {
            values.push_back((float) v);
        }
        for (int k = 0; k <= 24; ++k)
        {
            const float p = (float) (int64_t {1} << k);
            for (float v: {p, p - 1.0f, p + 1.0f})
            {
                if (std::fabs(v) <= (float) kTwoPow24)
                {
                    values.push_back(v);
                    values.push_back(-v);
                }
            }
        }
        uint32_t state = 2463534242u;
        for (int i = 0; i < randomCount; ++i)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            values.push_back((float) ((int64_t) (state % (2 * kTwoPow24 + 1)) - kTwoPow24));
        }
        for (float v: {0.5f, -0.5f, 2.75f, -2.75f, 255.9f, -255.9f, 16777215.5f, -0.0f})
        {
            values.push_back(v);
        }
        values.push_back(std::numeric_limits<float>::quiet_NaN());
        return values;
    }

    bool sameBits(float a, float b) {
        return std::memcmp(&a, &b, sizeof(float)) == 0;
    }

    // The fp32 lane the CPU oracle stores for an int64 result (Binary rule: fp32 operands -> fp32 result),
    // compared bit for bit (a -0.0 from the shader would differ from the oracle's +0.0).
    float oracleLane(int64_t integerResult) {
        return (float) integerResult;
    }

    constexpr int kMaxReportedMismatches = 8;

} // namespace

TEST(BitwiseShaderMath, PowerOfTwoAndWrapHelpersAreExact) {
    for (int e = 0; e <= 64; ++e)
    {
        EXPECT_EQ(glsl::powerOfTwo(e), std::ldexp(1.0f, e)) << e;
    }
    EXPECT_EQ(glsl::wrapToWidth(-6.0f, 64), -6.0f);
    EXPECT_EQ(glsl::wrapToWidth(-3.0f * 9223372036854775808.0f, 64), -9223372036854775808.0f); // -3 << 63 at 64 bits
    EXPECT_EQ(glsl::wrapToWidth(300.0f, 8), 44.0f);
    EXPECT_EQ(glsl::wrapToWidth(-8.0f, 8), 248.0f);
    EXPECT_EQ(glsl::int32Operand(3.0e9f), 2147483520);
    EXPECT_EQ(glsl::int32Operand(-3.0e9f), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(glsl::int32Operand(std::numeric_limits<float>::quiet_NaN()), 0);
    EXPECT_EQ(glsl::integerOperand(std::numeric_limits<float>::infinity()), 9223372036854775808.0f);
}

TEST(BitwiseShaderMath, BitwiseAndOrXorMatchesOracle) {
    const std::vector<float> operands = sweepOperands(128, 300);
    int64_t                  checked = 0, mismatches = 0;
    for (int op: {glsl::kOperatorAnd, glsl::kOperatorOr, glsl::kOperatorXor})
    {
        for (float left: operands)
        {
            for (float right: operands)
            {
                const int64_t lhs = cpu::bitwise::integerFromFloat(left), rhs = cpu::bitwise::integerFromFloat(right);
                const int64_t want = op == glsl::kOperatorAnd ? cpu::bitwise::bitwiseAnd(lhs, rhs) : (op == glsl::kOperatorOr ? cpu::bitwise::bitwiseOr(lhs, rhs) : cpu::bitwise::bitwiseXor(lhs, rhs));
                const float got = glsl::bitwiseValue(left, right, op);
                ++checked;
                if (!sameBits(got, oracleLane(want)) && ++mismatches <= kMaxReportedMismatches)
                {
                    ADD_FAILURE() << "op " << op << " (" << left << ", " << right << "): shader " << got << " oracle " << want;
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << "of " << checked;
}

TEST(BitwiseShaderMath, BitwiseNotMatchesOracle) {
    const std::vector<float> operands = sweepOperands(70000, 20000);
    int64_t                  checked = 0, mismatches = 0;
    for (int bits: {8, 16, 32, 64})
    {
        for (int isSigned: {0, 1})
        {
            bitwise::IntegerWidth width;
            width.bits     = bits;
            width.isSigned = isSigned != 0;
            for (float value: operands)
            {
                const int64_t want = cpu::bitwise::bitwiseNot(cpu::bitwise::integerFromFloat(value), width);
                const float   got  = glsl::bitwiseNotValue(value, bits, isSigned);
                ++checked;
                if (!sameBits(got, oracleLane(want)) && ++mismatches <= kMaxReportedMismatches)
                {
                    ADD_FAILURE() << "bits " << bits << " signed " << isSigned << " value " << value << ": shader " << got << " oracle " << want;
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << "of " << checked;
}

TEST(BitwiseShaderMath, BitShiftMatchesOracle) {
    const std::vector<float> operands = sweepOperands(300, 1500);
    std::vector<float>       shifts;
    for (int s = 0; s <= 70; ++s)
    {
        shifts.push_back((float) s);
    }
    for (float s: {-1.0f, -64.0f, 0.5f, 7.9f, -0.5f, 1.0e10f, std::numeric_limits<float>::quiet_NaN()})
    {
        shifts.push_back(s);
    }
    int64_t checked = 0, mismatches = 0;
    for (int bits: {8, 16, 32, 64})
    {
        for (int left: {0, 1})
        {
            const bitwise::ShiftDirection direction = left ? bitwise::ShiftDirection::Left : bitwise::ShiftDirection::Right;
            for (float value: operands)
            {
                for (float shift: shifts)
                {
                    const int64_t want = cpu::bitwise::bitShift(cpu::bitwise::integerFromFloat(value), cpu::bitwise::integerFromFloat(shift), direction, bits);
                    const float   got  = glsl::bitShiftValue(value, shift, left, bits);
                    ++checked;
                    if (!sameBits(got, oracleLane(want)) && ++mismatches <= kMaxReportedMismatches)
                    {
                        ADD_FAILURE() << "bits " << bits << (left ? " LEFT " : " RIGHT ") << value << " by " << shift << ": shader " << got << " oracle " << want;
                    }
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << "of " << checked;
}
