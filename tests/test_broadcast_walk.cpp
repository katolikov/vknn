// BroadcastWalk (src/backend/cpu/broadcast.h) must reproduce, offset for offset, the divide-based
// index unravel the CPU elementwise ops are specified against: for output axis d with row-major
// stride `prod(out[d+1..])`, the operand offset is sum_d ((lin / stride_d) % out[d]) * ostr[d].
// Any drift here silently changes which element an op reads, so the equivalence is pinned here on
// broadcast patterns of every rank the ops see.
#include "backend/cpu/broadcast.h"
#include "vknn/shape.h"
#include <cstdint>
#include <gtest/gtest.h>
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
