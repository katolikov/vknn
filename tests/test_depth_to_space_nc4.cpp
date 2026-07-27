// DepthToSpace runs packed when every NC4HW4 block is fully occupied on both sides.
//
// The flat kernel makes the channel-count change a row-major reindex, so an NC4HW4 producer and an
// NC4HW4 consumer each pay a full-size ConvertLayout around it -- three passes over the tensor to
// do one remap, and the two converts cost more than the remap itself. These tests pin when the
// packed path is taken, that the converts really disappear, and that the remap is the ONNX one.
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace vknn {
    bool gpuFlatNode(const Graph &g, const Node &n);
}

namespace {
    Attr intAttr(int64_t v) {
        Attr a;
        a.kind = Attr::Int;
        a.i    = v;
        return a;
    }
    Attr strAttr(std::string v) {
        Attr a;
        a.kind = Attr::String;
        a.str  = std::move(v);
        return a;
    }

    // Conv (an NC4HW4 producer) -> DepthToSpace -> Conv (an NC4HW4 consumer). Both neighbours are
    // packed, so a flat DepthToSpace between them is exactly the sandwich that costs two converts.
    Graph buildSandwich(int64_t cIn, int64_t h, int64_t w, int64_t blockSize, const char *mode) {
        const int64_t cMid = cIn / (blockSize * blockSize);
        Graph         g;
        TensorDesc    xi;
        xi.name    = "x";
        xi.shape   = {1, cIn, h, w};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};

        auto weight = [&](const char *name, int64_t co, int64_t ci) {
            TensorDesc wi;
            wi.name          = name;
            wi.shape         = {co, ci, 1, 1};
            wi.isInitializer = true;
            TensorId   t     = g.addTensor(wi);
            HostBuffer hb;
            hb.resizeElems((size_t) (co * ci), DType::Float32);
            for (int64_t i = 0; i < co * ci; ++i)
            {
                hb.f32()[i] = (i % (ci + 1) == 0) ? 1.0f : 0.0f;
            }
            g.initializers[t] = hb;
            return t;
        };

        TensorId a = g.addTensor({.name = "a"});
        Node     c1;
        c1.type    = OpType::Conv;
        c1.name    = "producer";
        c1.inputs  = {x, weight("w1", cIn, cIn)};
        c1.outputs = {a};
        g.nodes.push_back(c1);

        TensorId b = g.addTensor({.name = "b"});
        Node     d2s;
        d2s.type                  = OpType::DepthToSpace;
        d2s.name                  = "d2s";
        d2s.inputs                = {a};
        d2s.outputs               = {b};
        d2s.attr.map["blocksize"] = intAttr(blockSize);
        d2s.attr.map["mode"]      = strAttr(mode);
        g.nodes.push_back(d2s);

        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     c2;
        c2.type    = OpType::Conv;
        c2.name    = "consumer";
        c2.inputs  = {b, weight("w2", cMid, cMid)};
        c2.outputs = {y};
        g.nodes.push_back(c2);
        g.outputs = {y};
        return g;
    }

    // Converts spliced around the DepthToSpace itself. A whole-graph count would also pick up the
    // trailing convert every NC4HW4 graph output gets so the host readback is a bulk copy, which
    // has nothing to do with this node.
    size_t countConvertsAroundD2s(const Graph &g) {
        TensorId in = kNoTensor, out = kNoTensor;
        for (const Node &nd: g.nodes)
        {
            if (nd.type == OpType::DepthToSpace)
            {
                in  = nd.inputs[0];
                out = nd.outputs[0];
            }
        }
        size_t n = 0;
        for (const Node &nd: g.nodes)
        {
            if (nd.type != OpType::ConvertLayout)
            {
                continue;
            }
            n += (nd.outputs[0] == in || nd.inputs[0] == out) ? 1 : 0;
        }
        return n;
    }
} // namespace

// Both channel counts 4-aligned -> packed. Either side unaligned -> flat, because a partly-filled
// block would draw its four lanes from different source blocks.
TEST(DepthToSpaceNc4, PackedOnlyWhenBothChannelCountsAreBlockAligned) {
    struct Case {
        int64_t cIn, block;
        bool    packed;
    };
    const Case cases[] = {
        {128, 2, true}, // in 128, out 32: both aligned
        {16, 2, true},  // in 16, out 4
        {36, 3, true},  // in 36, out 4: both aligned, block 3
        {8, 2, false},  // in 8 aligned, out 2 NOT aligned
        {12, 2, false}, // in 12 aligned, out 3 NOT aligned
        {18, 3, false}, // in 18 NOT aligned
    };
    for (const Case &c: cases)
    {
        Graph g = buildSandwich(c.cIn, 4, 4, c.block, "DCR");
        inferShapes(g, 1);
        const Node &d2s     = g.nodes[1];
        const bool  aligned = c.cIn % 4 == 0 && (c.cIn / (c.block * c.block)) % 4 == 0;
        EXPECT_EQ(depthToSpaceIsNc4(g, d2s), aligned) << "cIn=" << c.cIn << " block=" << c.block;
        EXPECT_EQ(gpuFlatNode(g, d2s), !aligned) << "gpuFlatNode must mirror the packed decision";
    }
}

// The point of the change: a packed DepthToSpace between two packed neighbours needs no converts,
// while an unaligned one still gets the pair that brackets the flat kernel.
TEST(DepthToSpaceNc4, PackedSandwichNeedsNoLayoutConverts) {
    Graph packed = buildSandwich(128, 8, 8, 2, "DCR");
    inferShapes(packed, 1);
    insertLayoutConverts(packed);
    EXPECT_EQ(countConvertsAroundD2s(packed), 0u) << "both neighbours are packed and so is the remap";

    Graph unaligned = buildSandwich(8, 8, 8, 2, "DCR"); // out channels = 2, not block-aligned
    inferShapes(unaligned, 1);
    insertLayoutConverts(unaligned);
    EXPECT_EQ(countConvertsAroundD2s(unaligned), 2u) << "the flat kernel is still bracketed on both sides";
}

// The remap itself, against the ONNX definition, on the CPU oracle -- the packed kernel has to
// reproduce exactly this. DCR reads the block index as the SLOWEST-varying part of the channel.
TEST(DepthToSpaceNc4, DcrRemapMatchesTheOnnxDefinition) {
    constexpr int64_t kC = 8, kH = 2, kW = 2, kBlock = 2, kC2 = kC / (kBlock * kBlock);
    Graph             g;
    TensorDesc        xi;
    xi.name    = "x";
    xi.shape   = {1, kC, kH, kW};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs   = {x};
    TensorDesc yo;
    yo.name     = "y";
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     nd;
    nd.type                  = OpType::DepthToSpace;
    nd.name                  = "d2s";
    nd.inputs                = {x};
    nd.outputs               = {y};
    nd.attr.map["blocksize"] = intAttr(kBlock);
    nd.attr.map["mode"]      = strAttr("DCR");
    g.nodes.push_back(nd);
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);
    IOTensor in;
    in.name        = "x";
    in.shape       = {1, kC, kH, kW};
    in.dtype       = DType::Float32;
    const size_t n = (size_t) (kC * kH * kW);
    in.data.resize(n * sizeof(float));
    for (size_t i = 0; i < n; ++i)
    {
        reinterpret_cast<float *>(in.data.data())[i] = (float) i;
    }
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_EQ(outs[0].shape, (std::vector<int64_t> {1, kC2, kH * kBlock, kW * kBlock}));

    const float *got = outs[0].f32();
    for (int64_t c = 0; c < kC2; ++c)
    {
        for (int64_t oh = 0; oh < kH * kBlock; ++oh)
        {
            for (int64_t ow = 0; ow < kW * kBlock; ++ow)
            {
                const int64_t blk  = (oh % kBlock) * kBlock + (ow % kBlock);
                const int64_t cIn  = blk * kC2 + c; // DCR
                const float   want = (float) (((cIn) *kH + oh / kBlock) * kW + ow / kBlock);
                EXPECT_EQ(got[(c * kH * kBlock + oh) * kW * kBlock + ow], want) << "c=" << c << " oh=" << oh << " ow=" << ow;
            }
        }
    }
}
