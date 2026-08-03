// A channel-last Transpose reads NC4HW4 directly instead of taking a ConvertLayout in front of it.
//
// Transpose is a flat-only op, so the layout pass used to convert its whole input to flat row-major
// first: two full-size passes over the tensor, plus the intermediate buffer, for one reindex. An
// NC4HW4 vec4 holds four consecutive channels of one pixel -- exactly four consecutive elements of
// the NHWC output -- so the packed layout is the BETTER input for this permutation, not the worse
// one. These tests pin which permutations take that path and that the convert really disappears.
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cstring>
#include <gtest/gtest.h>

namespace {
    // MSVC at C++17 rejects designated initializers; a tiny helper names the tensor.
    vknn::TensorDesc namedDesc(const char *name) {
        vknn::TensorDesc d;
        d.name = name;
        return d;
    }
} // namespace

using namespace vknn;

namespace {
    constexpr int64_t kC = 8, kH = 6, kW = 5;

    Attr intsAttr(std::vector<int64_t> v) {
        Attr a;
        a.kind = Attr::Ints;
        a.ints = std::move(v);
        return a;
    }

    // Conv (an NC4HW4 producer) -> Transpose(perm). The conv is what makes the Transpose's input
    // NC4HW4 in the first place; a graph input would be adopted flat and prove nothing.
    Graph buildConvThenTranspose(const std::vector<int64_t> &perm, bool withPerm = true) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {1, kC, kH, kW};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};

        TensorDesc wi;
        wi.name          = "w";
        wi.shape         = {kC, kC, 1, 1};
        wi.isInitializer = true;
        TensorId   w     = g.addTensor(wi);
        HostBuffer hb;
        hb.resizeElems((size_t) (kC * kC), DType::Float32);
        for (int64_t i = 0; i < kC * kC; ++i)
        {
            hb.f32()[i] = (i % (kC + 1) == 0) ? 1.0f : 0.0f; // identity-ish, values are not the point
        }
        g.initializers[w] = hb;

        TensorId t = g.addTensor(namedDesc("t"));
        Node     cv;
        cv.type    = OpType::Conv;
        cv.name    = "conv";
        cv.inputs  = {x, w};
        cv.outputs = {t};
        g.nodes.push_back(cv);

        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     tr;
        tr.type    = OpType::Transpose;
        tr.name    = "tr";
        tr.inputs  = {t};
        tr.outputs = {y};
        if (withPerm)
        {
            tr.attr.map["perm"] = intsAttr(perm);
        }
        g.nodes.push_back(tr);
        g.outputs = {y};
        return g;
    }

    size_t countConvertLayout(const Graph &g) {
        size_t n = 0;
        for (const Node &nd: g.nodes)
        {
            n += nd.type == OpType::ConvertLayout ? 1 : 0;
        }
        return n;
    }
} // namespace

// Only the channel-last permutation takes the NC4HW4 path. The others are genuine flat gathers and
// must keep their convert, or they would read a packed buffer as if it were row-major.
TEST(TransposeNhwc, OnlyChannelLastReadsThePackedLayout) {
    struct Case {
        std::vector<int64_t> perm;
        bool                 nc4;
    };
    const Case cases[] = {
        {{0, 2, 3, 1}, true},  // NCHW -> NHWC
        {{0, 3, 1, 2}, false}, // NCHW -> WNCH
        {{0, 1, 3, 2}, false}, // swap H,W
        {{0, 2, 1, 3}, false}, // swap C,H
        {{0, 1, 2, 3}, false}, // identity: aliased away, never worth a special kernel
    };
    for (const Case &c: cases)
    {
        Graph g = buildConvThenTranspose(c.perm);
        inferShapes(g, 1);
        const Node &tr = g.nodes.back();
        EXPECT_EQ(transposeReadsNc4(g, tr), c.nc4) << "perm {" << c.perm[0] << "," << c.perm[1] << "," << c.perm[2] << "," << c.perm[3] << "}";
    }
}

// An ABSENT perm attribute means the full axis reverse, not channel-last.
TEST(TransposeNhwc, AbsentPermIsNotChannelLast) {
    Graph g = buildConvThenTranspose({}, /*withPerm=*/false);
    inferShapes(g, 1);
    EXPECT_FALSE(transposeReadsNc4(g, g.nodes.back()));
}

// The point of the whole exercise: no ConvertLayout is spliced in front of a channel-last Transpose,
// while a non-channel-last one still gets its convert.
TEST(TransposeNhwc, ChannelLastNeedsNoLayoutConvert) {
    Graph channelLast = buildConvThenTranspose({0, 2, 3, 1});
    inferShapes(channelLast, 1);
    insertLayoutConverts(channelLast);
    EXPECT_EQ(countConvertLayout(channelLast), 0u) << "the packed input feeds transpose_nhwc directly";

    Graph other = buildConvThenTranspose({0, 1, 3, 2});
    inferShapes(other, 1);
    insertLayoutConverts(other);
    EXPECT_GT(countConvertLayout(other), 0u) << "a flat gather still needs its input converted";
}

// A Transpose carrying a fused pointwise epilogue stays on the flat gather: the epilogue is planned
// against the flat output world, and only the gather applies it.
TEST(TransposeNhwc, AFusedEpilogueKeepsTheFlatGather) {
    Graph g = buildConvThenTranspose({0, 2, 3, 1});
    inferShapes(g, 1);
    Attr steps;
    steps.kind                          = Attr::Ints;
    steps.ints                          = {kPwKindUnary, 0, kPwRefEntry, kPwRefNone, kPwRefNone, kPwRefNone, kPwBcastSame, 0};
    g.nodes.back().attr.map["pw_steps"] = steps;
    EXPECT_FALSE(transposeReadsNc4(g, g.nodes.back()));
}
