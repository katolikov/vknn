// BroadcastWalk (src/backend/cpu/broadcast.h) must reproduce, offset for offset, the divide-based
// index unravel the CPU elementwise ops are specified against: for output axis d with row-major
// stride `prod(out[d+1..])`, the operand offset is sum_d ((lin / stride_d) % out[d]) * ostr[d].
// Any drift here silently changes which element an op reads, so the equivalence is pinned here on
// broadcast patterns of every rank the ops see. broadcastOutputShape, the output-shape rule of the same
// ops, is pinned too, and every registered two-operand elementwise kernel (and Where) refuses operand shapes
// that do not broadcast instead of walking past the smaller operand.
#include "backend/cpu/broadcast.h"
#include "backend/cpu/cpu_backend.h"
#include "core/bitwise_attrs.h"
#include "vknn/binary_type.h"
#include "vknn/graph.h"
#include "vknn/shape.h"
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    // Zero-collapsing row-major strides of an operand right-aligned against `out`, the arrays the
    // elementwise ops build back-to-front and hand to BroadcastWalk.
    std::vector<int64_t> bcastStrides(const Shape &operand, const Shape &out) {
        size_t               rank = out.size();
        size_t               off  = rank - operand.size();
        std::vector<int64_t> s(rank, 0);
        int64_t              run = 1;
        for (int i = (int) rank - 1; i >= 0; --i)
        {
            int64_t d = (size_t) i < off ? 1 : operand[i - off];
            s[i]      = (d == 1) ? 0 : run;
            run *= d;
        }
        return s;
    }

    // The per-element unravel BroadcastWalk replaces, kept verbatim as the reference.
    int64_t refOffset(const Shape &out, const std::vector<int64_t> &ostr, int64_t lin) {
        size_t  rank = out.size();
        int64_t io   = 0;
        for (size_t d = 0; d < rank; ++d)
        {
            int64_t stride = 1;
            for (size_t e = d + 1; e < rank; ++e)
            {
                stride *= out[e];
            }
            io += ((lin / stride) % out[d]) * ostr[d];
        }
        return io;
    }

    int64_t elems(const Shape &s) {
        int64_t n = 1;
        for (auto d: s)
        {
            n *= d;
        }
        return n;
    }

    void checkPair(const Shape &out, const Shape &a, const Shape &b) {
        std::vector<int64_t> sa = bcastStrides(a, out), sb = bcastStrides(b, out);
        cpu::BroadcastWalk   w(out, {sa.data(), sb.data()});
        w.seek(0);
        int64_t n = elems(out);
        for (int64_t lin = 0; lin < n; ++lin, w.next())
        {
            ASSERT_EQ(w.offset(0), refOffset(out, sa, lin)) << "operand A, lin=" << lin;
            ASSERT_EQ(w.offset(1), refOffset(out, sb, lin)) << "operand B, lin=" << lin;
        }
    }

} // namespace

TEST(BroadcastWalk, MatchesDivideUnravel) {
    checkPair({5}, {5}, {1});
    checkPair({3, 4}, {3, 4}, {4});
    checkPair({2, 3, 4}, {2, 1, 4}, {3, 1});
    checkPair({2, 3, 4, 5}, {2, 3, 4, 5}, {1, 3, 1, 5});
    checkPair({7, 1, 3, 2, 5}, {7, 1, 1, 2, 1}, {3, 1, 5});
    checkPair({1, 1, 1}, {1}, {1, 1, 1});
}

TEST(BroadcastWalk, RankZeroScalar) {
    Shape                out {};
    std::vector<int64_t> sa, sb;
    cpu::BroadcastWalk   w(out, {sa.data(), sb.data()});
    w.seek(0);
    EXPECT_EQ(w.offset(0), 0);
    EXPECT_EQ(w.offset(1), 0);
    w.next(); // advancing past the single element is a no-op, not a read out of bounds
}

TEST(BroadcastWalk, SeekMatchesSequentialAdvance) {
    // seek() lands on the same offsets a next()-sweep reaches, which is what lets a partitioned
    // sweep start at an arbitrary index. The shape deliberately does not divide evenly by 3.
    Shape                out {5, 2, 7};
    Shape                a {5, 1, 7}, b {2, 1};
    std::vector<int64_t> sa = bcastStrides(a, out), sb = bcastStrides(b, out);
    int64_t              n = elems(out);
    for (int64_t start = 0; start < n; ++start)
    {
        cpu::BroadcastWalk seeked(out, {sa.data(), sb.data()});
        seeked.seek(start);
        cpu::BroadcastWalk swept(out, {sa.data(), sb.data()});
        swept.seek(0);
        for (int64_t i = 0; i < start; ++i)
        {
            swept.next();
        }
        ASSERT_EQ(seeked.offset(0), swept.offset(0)) << "start=" << start;
        ASSERT_EQ(seeked.offset(1), swept.offset(1)) << "start=" << start;
    }
}

namespace {

    Node broadcastNode(OpType type, const std::string &name, size_t operandCount) {
        Node node;
        node.type = type;
        node.name = name;
        for (size_t operand = 0; operand < operandCount; ++operand)
        {
            node.inputs.push_back((TensorId) operand);
        }
        node.outputs = {(TensorId) operandCount};
        return node;
    }

    RtTensor floatOperand(const Shape &shape) {
        RtTensor tensor;
        tensor.shape = shape;
        tensor.dtype = DType::Float32;
        tensor.host.resizeElems(numElements(shape), DType::Float32);
        return tensor;
    }

    RtTensor int64Operand(const Shape &shape) {
        RtTensor tensor;
        tensor.shape = shape;
        tensor.dtype = DType::Int64;
        tensor.host.resizeElems(numElements(shape), DType::Int64);
        return tensor;
    }

    // Run the registered CPU kernel of `node` over `operands` (bound to tensors 0..n-1, the output after
    // them) and return the InvalidArgument message it throws, or an empty string when it does not throw
    // one.
    std::string invalidArgumentMessage(const Node &node, std::vector<RtTensor> operands) {
        operands.emplace_back();
        Graph       g;
        Config      cfg;
        ExecContext ctx;
        ctx.pool   = &operands;
        ctx.graph  = &g;
        ctx.config = &cfg;
        auto op    = CpuOpRegistry::instance().create(node.type);
        if (!op)
        {
            ADD_FAILURE() << "no CPU kernel for " << opTypeName(node.type);
            return {};
        }
        try
        { op->run(node, ctx); } catch (const Error &error)
        {
            if (error.status() == Status::InvalidArgument)
            {
                return error.what();
            }
            ADD_FAILURE() << opTypeName(node.type) << " threw " << error.what();
        }
        return {};
    }

} // namespace

TEST(BroadcastOutputShape, RightAlignsStretchesUnitAxesAndKeepsZeroExtents) {
    const Node node = broadcastNode(OpType::Add, "add", 2);
    EXPECT_EQ(cpu::broadcastOutputShape(node, {2, 3}, {3}), (Shape {2, 3}));
    EXPECT_EQ(cpu::broadcastOutputShape(node, {2, 1, 4}, {3, 1}), (Shape {2, 3, 4}));
    EXPECT_EQ(cpu::broadcastOutputShape(node, {}, {5}), (Shape {5}));
    EXPECT_EQ(cpu::broadcastOutputShape(node, {}, {}), (Shape {}));
    EXPECT_EQ(cpu::broadcastOutputShape(node, {0, 3}, {1, 3}), (Shape {0, 3}));
    EXPECT_EQ(cpu::broadcastOutputShape(node, {1}, {0}), (Shape {0}));
}

TEST(BroadcastOutputShape, ShapesThatDoNotBroadcastThrowNamingTheNodeAndAxis) {
    const Node node = broadcastNode(OpType::BitwiseAnd, "mask", 2);
    for (const auto &[shapeA, shapeB]: std::vector<std::pair<Shape, Shape>> {{{3}, {4}}, {{2, 3}, {2}}, {{0}, {3}}})
    {
        try
        {
            cpu::broadcastOutputShape(node, shapeA, shapeB);
            ADD_FAILURE() << shapeStr(shapeA) << " and " << shapeStr(shapeB) << " must not broadcast";
        } catch (const Error &error)
        {
            EXPECT_EQ(error.status(), Status::InvalidArgument);
            EXPECT_NE(std::string(error.what()).find("BitwiseAnd 'mask'"), std::string::npos) << error.what();
            EXPECT_NE(std::string(error.what()).find("do not broadcast (axis"), std::string::npos) << error.what();
        }
    }
}

TEST(BroadcastOutputShape, EveryBroadcastingKernelRefusesShapesThatDoNotBroadcast) {
    // [3] against [4]: a stride walk over the output [4] would read x[3], past the 3-element operand.
    struct KernelCase {
        OpType type;
        int    subOp;
        bool   int64Operands;
    };
    const std::vector<KernelCase> cases {
        {OpType::Add, 0, false},
        {OpType::Add, 0, true},
        {OpType::Binary, (int) BinaryType::Mul, false},
        {OpType::Binary, (int) BinaryType::Div, true},
        {OpType::Equal, 0, false},
        {OpType::Greater, 0, false},
        {OpType::GreaterEqual, 0, false},
        {OpType::Less, 0, false},
        {OpType::LessEqual, 0, false},
        {OpType::And, 0, false},
        {OpType::Or, 0, false},
        {OpType::Xor, 0, false},
        {OpType::Mod, 0, true},
        {OpType::BitwiseAnd, 0, true},
        {OpType::BitwiseOr, 0, true},
        {OpType::BitwiseXor, 0, true},
        {OpType::BitShift, 0, true},
        {OpType::BitShift, 0, false},
    };
    const Shape smaller {3}, larger {4};
    for (const KernelCase &kernelCase: cases)
    {
        Node node  = broadcastNode(kernelCase.type, "op", 2);
        node.subOp = kernelCase.subOp;
        if (kernelCase.type == OpType::BitShift)
        {
            Attr direction;
            direction.kind                         = Attr::String;
            direction.str                          = bitwise::kDirectionLeft;
            node.attr.map[bitwise::kDirectionAttr] = direction;
        }
        auto operand = [&](const Shape &shape) {
            return kernelCase.int64Operands ? int64Operand(shape) : floatOperand(shape);
        };
        const std::string message = invalidArgumentMessage(node, {operand(smaller), operand(larger)});
        EXPECT_NE(message.find("do not broadcast"), std::string::npos) << opTypeName(kernelCase.type) << " int64=" << kernelCase.int64Operands << ": " << message;
    }
    for (bool int64Values: {false, true})
    {
        const Node where = broadcastNode(OpType::Where, "select", 3);
        auto       value = [&](const Shape &shape) {
            return int64Values ? int64Operand(shape) : floatOperand(shape);
        };
        const std::string message = invalidArgumentMessage(where, {floatOperand({1}), value(smaller), value(larger)});
        EXPECT_NE(message.find("do not broadcast"), std::string::npos) << "Where int64=" << int64Values << ": " << message;
    }
}
