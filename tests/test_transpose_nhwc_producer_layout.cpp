// Which layout a channel-last Transpose reads its input in is a property of the INPUT, not of the
// permutation alone.
//
// Reading NC4HW4 directly pays off only when the input is already packed: the vec4 is four
// consecutive output channels, so the reindex is one coalesced pass and the ConvertLayout that
// would otherwise precede the flat gather disappears. Behind a FLAT producer the same choice costs
// a full-tensor flat->NC4HW4 convert that the flat gather never needed, so the packed read is worth
// nothing there and the Transpose stays on the gather. Both routes store identical bytes, so this
// is routing only -- and the layout pass and the Vulkan op ask the same predicate, so they cannot
// disagree about which buffer is bound.
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    constexpr int64_t kBatch = 1, kChannels = 16, kHeight = 8, kWidth = 5;
    // NCHW -> NHWC, the only permutation the packed read covers.
    const std::vector<int64_t> kChannelLastPerm {0, 2, 3, 1};

    // MSVC at C++17 rejects designated initializers; a tiny helper names the tensor.
    TensorDesc namedDesc(const char *name) {
        TensorDesc d;
        d.name = name;
        return d;
    }

    Attr intsAttr(std::vector<int64_t> v) {
        Attr a;
        a.kind = Attr::Ints;
        a.ints = std::move(v);
        return a;
    }

    Attr floatAttr(float v) {
        Attr a;
        a.kind = Attr::Float;
        a.f    = v;
        return a;
    }

    void appendChannelLastTranspose(Graph &g, TensorId in) {
        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     tr;
        tr.type             = OpType::Transpose;
        tr.name             = "tr";
        tr.inputs           = {in};
        tr.outputs          = {y};
        tr.attr.map["perm"] = intsAttr(kChannelLastPerm);
        g.nodes.push_back(tr);
        g.outputs = {y};
    }

    // x -> Clip -> Transpose(NHWC). Clip is an unconditionally FLAT op, so the Transpose's input is
    // a flat row-major tensor: exactly the case a packed read would have to pay a convert for.
    Graph buildFlatProducerThenTranspose() {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {kBatch, kChannels, kHeight, kWidth};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};

        TensorId t = g.addTensor(namedDesc("t"));
        Node     clip;
        clip.type            = OpType::Clip;
        clip.name            = "saturate";
        clip.inputs          = {x};
        clip.outputs         = {t};
        clip.attr.map["min"] = floatAttr(0.f);
        clip.attr.map["max"] = floatAttr(6.f);
        g.nodes.push_back(clip);

        appendChannelLastTranspose(g, t);
        return g;
    }

    // x -> Conv -> Transpose(NHWC). Conv is an NC4HW4 producer, so the packed read is free.
    Graph buildPackedProducerThenTranspose() {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {kBatch, kChannels, kHeight, kWidth};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};

        TensorDesc wi;
        wi.name          = "w";
        wi.shape         = {kChannels, kChannels, 1, 1};
        wi.isInitializer = true;
        TensorId   w     = g.addTensor(wi);
        HostBuffer hb;
        hb.resizeElems(kChannels * kChannels, DType::Float32);
        for (int64_t i = 0; i < kChannels * kChannels; ++i)
        {
            hb.f32()[i] = i % (kChannels + 1) == 0 ? 1.0f : 0.0f; // values are not the point
        }
        g.initializers[w] = std::move(hb);

        TensorId t = g.addTensor(namedDesc("t"));
        Node     cv;
        cv.type    = OpType::Conv;
        cv.name    = "conv";
        cv.inputs  = {x, w};
        cv.outputs = {t};
        g.nodes.push_back(cv);

        appendChannelLastTranspose(g, t);
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

    const Node &transposeNode(const Graph &g) {
        for (const Node &nd: g.nodes)
        {
            if (nd.type == OpType::Transpose)
            {
                return nd;
            }
        }
        ADD_FAILURE() << "the graph must still carry its Transpose";
        return g.nodes.front();
    }
} // namespace

// A flat producer must not be converted into the packed layout just to feed the packed read: the
// flat gather reads it as it stands, for zero converts.
TEST(TransposeNhwcProducerLayout, FlatProducerKeepsTheFlatGatherAndAddsNoConvert) {
    Graph g = buildFlatProducerThenTranspose();
    inferShapes(g, 1);
    insertLayoutConverts(g);
    EXPECT_EQ(countConvertLayout(g), 0u) << "a flat-produced input feeds the flat gather directly";
    EXPECT_FALSE(transposeReadsNc4(g, transposeNode(g))) << "the op must read the buffer the layout pass left in place";
}

// The packed producer keeps the win the kernel exists for: no convert, and the packed read engages.
TEST(TransposeNhwcProducerLayout, PackedProducerKeepsThePackedRead) {
    Graph g = buildPackedProducerThenTranspose();
    inferShapes(g, 1);
    insertLayoutConverts(g);
    EXPECT_EQ(countConvertLayout(g), 0u) << "the packed input feeds transpose_nhwc directly";
    EXPECT_TRUE(transposeReadsNc4(g, transposeNode(g))) << "an NC4HW4-produced input is what the packed read exists for";
}
