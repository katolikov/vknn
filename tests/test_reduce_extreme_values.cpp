// Max/Min reduction identities at the float extremes.
//
// The GPU kernels used to seed Max/Min folds with finite sentinels (-3.4e38 for reduce, maxpool
// and the flat softmax row max; -1e30 for the NC4 softmax). 3.4e38 is BELOW the largest finite
// float, so an input entirely beyond the sentinel returned the sentinel itself, and a softmax row
// below the seed collapsed to zeros and normalized to NaN. The kernels now seed with exact
// infinities (VX_POS_INF / VX_NEG_INF in shaders/common.glsl), the same identities the CPU
// backend uses. These tests pin the CPU oracle's semantics at the extremes — the GPU side is
// gated by the on-device reproducer suite, since the host build carries no Vulkan backend.
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

using namespace vknn;

namespace {
    constexpr int64_t kC = 4;
    constexpr int64_t kH = 6;
    constexpr int64_t kW = 8;

    TensorDesc namedDesc(const char *name) {
        TensorDesc d;
        d.name = name;
        return d;
    }

    void setAttrInt(Node &n, const char *key, int64_t v) {
        Attr a;
        a.kind          = Attr::Int;
        a.i             = v;
        n.attr.map[key] = a;
    }

    void setAttrInts(Node &n, const char *key, std::vector<int64_t> v) {
        Attr a;
        a.kind          = Attr::Ints;
        a.ints          = std::move(v);
        n.attr.map[key] = a;
    }

    // x[1,kC,kH,kW] -> Reduce(subOp) over W.
    Graph buildReduceGraph(ReduceType op) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {1, kC, kH, kW};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};
        TensorId y = g.addTensor(namedDesc("y"));
        Node     red;
        red.type    = OpType::Reduce;
        red.subOp   = (int32_t) op;
        red.name    = "reduce";
        red.inputs  = {x};
        red.outputs = {y};
        setAttrInts(red, "axes", {3});
        setAttrInt(red, "keepdims", 1);
        g.nodes   = {red};
        g.outputs = {y};
        return g;
    }

    std::vector<float> runCpu(Graph g, const std::vector<float> &src, const Shape &inShape) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_NE(sess, nullptr);
        std::vector<IOTensor> in(1), out;
        in[0].name  = "x";
        in[0].shape = inShape;
        in[0].dtype = DType::Float32;
        in[0].data.resize(src.size() * sizeof(float));
        std::memcpy(in[0].data.data(), src.data(), src.size() * sizeof(float));
        EXPECT_EQ(sess->run(in, out), Status::Ok);
        EXPECT_EQ(out.size(), 1u);
        const float *f = out[0].f32();
        return std::vector<float>(f, f + (size_t) numElements(out[0].shape));
    }
} // namespace

// Every element sits ABOVE the old 3.4e38 sentinel: the reduction must return the true row
// minimum, not any identity value.
TEST(ReduceExtremeValues, MinAboveTheOldSentinel) {
    const size_t       n = (size_t) (kC * kH * kW);
    std::vector<float> src(n);
    const float        base = 3.401e38f;
    for (size_t i = 0; i < n; ++i)
    {
        src[i] = base + (float) (i % 7) * 1.0e34f;
    }
    auto got = runCpu(buildReduceGraph(ReduceType::Min), src, {1, kC, kH, kW});
    ASSERT_EQ(got.size(), (size_t) (kC * kH));
    for (size_t r = 0; r < got.size(); ++r)
    {
        float expect = std::numeric_limits<float>::infinity();
        for (int64_t w = 0; w < kW; ++w)
        {
            expect = std::min(expect, src[r * (size_t) kW + (size_t) w]);
        }
        EXPECT_EQ(got[r], expect) << "row " << r;
        EXPECT_GT(got[r], 3.4e38f) << "row " << r << ": the old sentinel leaked into the result";
    }
}

// Mirror case for Max: every element BELOW -3.4e38.
TEST(ReduceExtremeValues, MaxBelowTheOldSentinel) {
    const size_t       n = (size_t) (kC * kH * kW);
    std::vector<float> src(n);
    for (size_t i = 0; i < n; ++i)
    {
        src[i] = -3.401e38f - (float) (i % 7) * 1.0e34f;
    }
    auto got = runCpu(buildReduceGraph(ReduceType::Max), src, {1, kC, kH, kW});
    ASSERT_EQ(got.size(), (size_t) (kC * kH));
    for (size_t r = 0; r < got.size(); ++r)
    {
        float expect = -std::numeric_limits<float>::infinity();
        for (int64_t w = 0; w < kW; ++w)
        {
            expect = std::max(expect, src[r * (size_t) kW + (size_t) w]);
        }
        EXPECT_EQ(got[r], expect) << "row " << r;
        EXPECT_LT(got[r], -3.4e38f) << "row " << r << ": the old sentinel leaked into the result";
    }
}

// A softmax row whose every element lies below the old -1e30 NC4 seed must still normalize to a
// valid distribution (the wrong seed made exp(x - seed) flush to zero and the sum divide to NaN).
TEST(ReduceExtremeValues, SoftmaxRowFarBelowTheOldSeed) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, kC, kH, kW};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs   = {x};
    TensorId y = g.addTensor(namedDesc("y"));
    Node     sm;
    sm.type    = OpType::Softmax;
    sm.name    = "softmax";
    sm.inputs  = {x};
    sm.outputs = {y};
    setAttrInt(sm, "axis", 3);
    g.nodes   = {sm};
    g.outputs = {y};

    const size_t       n = (size_t) (kC * kH * kW);
    std::vector<float> src(n);
    for (size_t i = 0; i < n; ++i)
    {
        src[i] = -1.1e30f - (float) (i % 5) * 1.0e28f;
    }
    auto got = runCpu(std::move(g), src, {1, kC, kH, kW});
    ASSERT_EQ(got.size(), n);
    for (size_t r = 0; r < (size_t) (kC * kH); ++r)
    {
        float sum = 0.f;
        for (int64_t w = 0; w < kW; ++w)
        {
            const float v = got[r * (size_t) kW + (size_t) w];
            EXPECT_FALSE(std::isnan(v)) << "row " << r << " col " << w;
            sum += v;
        }
        EXPECT_NEAR(sum, 1.f, 1e-4f) << "row " << r;
    }
}
