// A per-pixel [1,1,H,W] operand in a fused pointwise unit.
//
// Such an operand used to classify as kPwBcastGeneral, which is addressable only by the flat
// kernel's per-axis div/mod walk. One of them therefore forced its whole unit off NC4HW4 (see
// `nc4Ok` in fuse_pointwise_chains.cpp) and excluded it from the layout re-vote, so the unit ran
// scalar and picked up a ConvertLayout on each side. It now classifies as kPwBcastSpatial, which
// both kernels index in closed form.
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {
    constexpr int kPwStepInts       = 8;
    constexpr int kPwStepBcastField = 6;
    constexpr int64_t kC            = 8;
    constexpr int64_t kH            = 4;
    constexpr int64_t kW            = 5;

    // x[1,kC,kH,kW] * mask[1,1,kH,kW] -> y, the per-pixel weighting an image pipeline is built from.
    Graph buildSpatialMulGraph(std::vector<float> &maskValues) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {1, kC, kH, kW};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};

        TensorDesc mi;
        mi.name          = "mask";
        mi.shape         = {1, 1, kH, kW};
        mi.isInitializer = true;
        TensorId   m     = g.addTensor(mi);
        HostBuffer hb;
        hb.resizeElems((size_t) (kH * kW), DType::Float32);
        maskValues.resize((size_t) (kH * kW));
        for (int64_t i = 0; i < kH * kW; ++i)
        {
            maskValues[(size_t) i] = 0.25f + 0.5f * (float) (i % 3); // distinct per pixel
            hb.f32()[i]            = maskValues[(size_t) i];
        }
        g.initializers[m] = hb;

        // A producer for the chain to fold into: the fusion pass grows a pointwise region behind a
        // producer's epilogue, so a lone Binary forms no unit.
        TensorId t = g.addTensor({.name = "t"});
        Node     act;
        act.type    = OpType::Unary;
        act.name    = "producer";
        act.subOp   = (int) UnaryType::Abs;
        act.inputs  = {x};
        act.outputs = {t};

        TensorId y = g.addTensor({.name = "y"});
        Node     mul;
        mul.type    = OpType::Binary;
        mul.name    = "spatial_weight";
        mul.subOp   = (int) BinaryType::Mul;
        mul.inputs  = {t, m};
        mul.outputs = {y};
        g.nodes     = {act, mul};
        g.outputs   = {y};
        return g;
    }
} // namespace

TEST(PwSpatialBcast, PerPixelOperandKeepsTheUnitOnTheBlockedPath) {
    std::vector<float> mask;
    Graph              g = buildSpatialMulGraph(mask);
    inferShapes(g, 1);
    fusePointwiseChains(g, /*strictFuse*/ false);

    bool sawSpatial = false, sawGeneral = false;
    for (const Node &nd: g.nodes)
    {
        if (!nd.attr.has("pw_steps"))
        {
            continue;
        }
        const auto &steps = nd.attr.getints("pw_steps");
        for (size_t s = 0; s + kPwStepBcastField < steps.size(); s += kPwStepInts)
        {
            sawSpatial = sawSpatial || steps[s + kPwStepBcastField] == kPwBcastSpatial;
            sawGeneral = sawGeneral || steps[s + kPwStepBcastField] == kPwBcastGeneral;
        }
    }
    EXPECT_TRUE(sawSpatial) << "a [1,1,H,W] operand must classify as the per-pixel class";
    EXPECT_FALSE(sawGeneral) << "no step may stay general: one general step forces the whole unit off NC4HW4";
}

// The closed-form index must select the SAME element the general strided walk would have. The CPU
// backend evaluates the chain unfused, so it is the reference for the value at every position.
TEST(PwSpatialBcast, PerPixelOperandComputesThePerPixelProduct) {
    std::vector<float> mask;
    Graph              g = buildSpatialMulGraph(mask);

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_NE(sess, nullptr);

    std::vector<IOTensor> inputs(1), outputs;
    inputs[0].name  = "x";
    inputs[0].shape = {1, kC, kH, kW};
    inputs[0].dtype = DType::Float32;
    const size_t n  = (size_t) (kC * kH * kW);
    inputs[0].data.resize(n * sizeof(float));
    std::vector<float> src(n);
    for (size_t i = 0; i < n; ++i)
    {
        src[i] = 1.0f + (float) (i % 7);
    }
    std::memcpy(inputs[0].data.data(), src.data(), n * sizeof(float));
    ASSERT_EQ(sess->run(inputs, outputs), Status::Ok);
    ASSERT_EQ(outputs.size(), 1u);

    const float *got = outputs[0].f32();
    for (int64_t c = 0; c < kC; ++c)
    {
        for (int64_t p = 0; p < kH * kW; ++p)
        {
            const size_t i = (size_t) (c * kH * kW + p);
            EXPECT_FLOAT_EQ(got[i], std::abs(src[i]) * mask[(size_t) p]) << "channel " << c << " pixel " << p
                                                               << ": every channel must read the same pixel's mask value";
        }
    }
}

// The GPU must agree with the CPU reference element for element. This is the gate on the closed-form
// index in BOTH kernels: a wrong pixel (or a vec4 load where the four channel lanes need the SAME
// value splatted) shows up as a gross mismatch, not a rounding difference. Vulkan-only, so it skips
// on a host build with no device.
TEST(PwSpatialBcast, GpuMatchesTheCpuReference) {
    std::vector<float> mask;
    Config             gpu;
    gpu.backend      = BackendKind::Vulkan;
    gpu.allowCpuFallback = false;
    Graph gg         = buildSpatialMulGraph(mask);
    std::unique_ptr<Session> gsess;
    try
    {
        gsess = Session::create(std::move(gg), gpu);
    } catch (const std::exception &)
    {
        gsess.reset();
    }
    if (!gsess)
    {
        GTEST_SKIP() << "no Vulkan device";
    }

    const size_t       n = (size_t) (kC * kH * kW);
    std::vector<float> src(n);
    for (size_t i = 0; i < n; ++i)
    {
        src[i] = 1.0f + (float) (i % 7);
    }
    auto bind = [&](std::vector<IOTensor> &in) {
        in.resize(1);
        in[0].name  = "x";
        in[0].shape = {1, kC, kH, kW};
        in[0].dtype = DType::Float32;
        in[0].data.resize(n * sizeof(float));
        std::memcpy(in[0].data.data(), src.data(), n * sizeof(float));
    };

    std::vector<IOTensor> gin, gout;
    bind(gin);
    ASSERT_EQ(gsess->run(gin, gout), Status::Ok);
    ASSERT_EQ(gout.size(), 1u);

    Config cpu;
    cpu.backend = BackendKind::Cpu;
    Graph cg    = buildSpatialMulGraph(mask);
    auto  csess = Session::create(std::move(cg), cpu);
    ASSERT_NE(csess, nullptr);
    std::vector<IOTensor> cin, cout;
    bind(cin);
    ASSERT_EQ(csess->run(cin, cout), Status::Ok);

    const float *gpuOut = gout[0].f32();
    const float *cpuOut = cout[0].f32();
    for (size_t i = 0; i < n; ++i)
    {
        EXPECT_NEAR(gpuOut[i], cpuOut[i], 1e-2f) << "element " << i;
    }
}
