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
    constexpr int64_t kC = 8;
    constexpr int64_t kH = 4;
    constexpr int64_t kW = 5;

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
        TensorId t = g.addTensor(namedDesc("t"));
        Node     act;
        act.type    = OpType::Unary;
        act.name    = "producer";
        act.subOp   = (int) UnaryType::Abs;
        act.inputs  = {x};
        act.outputs = {t};

        TensorId y = g.addTensor(namedDesc("y"));
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
            EXPECT_FLOAT_EQ(got[i], std::abs(src[i]) * mask[(size_t) p]) << "channel " << c << " pixel " << p << ": every channel must read the same pixel's mask value";
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
    gpu.backend                 = BackendKind::Vulkan;
    gpu.allowCpuFallback        = false;
    Graph                    gg = buildSpatialMulGraph(mask);
    std::unique_ptr<Session> gsess;
    try
    { gsess = Session::create(std::move(gg), gpu); } catch (const std::exception &)
    { gsess.reset(); }
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

// The mask an image pipeline actually has is COMPUTED, not baked in: it arrives as a runtime
// activation from a comparison/expand/resize, so it never passes through the initializer packer that
// places a C=1 tensor's value at channel lane 0. This is the case the constant-mask tests above do
// not cover.
namespace {
    Graph buildRuntimeMaskGraph() {
        Graph      g;
        TensorDesc xi;
        xi.name      = "x";
        xi.shape     = {1, kC, kH, kW};
        xi.isInput   = true;
        TensorId   x = g.addTensor(xi);
        TensorDesc mi;
        mi.name    = "m";
        mi.shape   = {1, 1, kH, kW};
        mi.isInput = true;
        TensorId m = g.addTensor(mi);
        g.inputs   = {x, m};

        TensorId t = g.addTensor(namedDesc("t"));
        Node     act;
        act.type    = OpType::Unary;
        act.name    = "producer";
        act.subOp   = (int) UnaryType::Abs;
        act.inputs  = {x};
        act.outputs = {t};

        // A node between the graph input and the use, so the operand is a produced activation.
        TensorId mm = g.addTensor(namedDesc("mm"));
        Node     mact;
        mact.type    = OpType::Unary;
        mact.name    = "mask_producer";
        mact.subOp   = (int) UnaryType::Abs;
        mact.inputs  = {m};
        mact.outputs = {mm};

        TensorId y = g.addTensor(namedDesc("y"));
        Node     mul;
        mul.type    = OpType::Binary;
        mul.name    = "spatial_weight";
        mul.subOp   = (int) BinaryType::Mul;
        mul.inputs  = {t, mm};
        mul.outputs = {y};
        g.nodes     = {act, mact, mul};
        g.outputs   = {y};
        return g;
    }
} // namespace

TEST(PwSpatialBcast, RuntimeMaskGpuMatchesCpu) {
    const size_t       n = (size_t) (kC * kH * kW);
    std::vector<float> src(n), msk((size_t) (kH * kW));
    for (size_t i = 0; i < n; ++i)
    {
        src[i] = 1.0f + (float) (i % 7);
    }
    for (size_t i = 0; i < msk.size(); ++i)
    {
        msk[i] = 0.25f + 0.5f * (float) (i % 3);
    }
    auto bind = [&](std::vector<IOTensor> &in) {
        in.resize(2);
        in[0].name  = "x";
        in[0].shape = {1, kC, kH, kW};
        in[0].dtype = DType::Float32;
        in[0].data.resize(n * sizeof(float));
        std::memcpy(in[0].data.data(), src.data(), n * sizeof(float));
        in[1].name  = "m";
        in[1].shape = {1, 1, kH, kW};
        in[1].dtype = DType::Float32;
        in[1].data.resize(msk.size() * sizeof(float));
        std::memcpy(in[1].data.data(), msk.data(), msk.size() * sizeof(float));
    };

    Config gpu;
    gpu.backend                 = BackendKind::Vulkan;
    gpu.allowCpuFallback        = false;
    Graph                    gg = buildRuntimeMaskGraph();
    std::unique_ptr<Session> gsess;
    try
    { gsess = Session::create(std::move(gg), gpu); } catch (const std::exception &)
    { gsess.reset(); }
    if (!gsess)
    {
        GTEST_SKIP() << "no Vulkan device";
    }
    std::vector<IOTensor> gin, gout;
    bind(gin);
    ASSERT_EQ(gsess->run(gin, gout), Status::Ok);

    Config cpu;
    cpu.backend = BackendKind::Cpu;
    Graph cg    = buildRuntimeMaskGraph();
    auto  csess = Session::create(std::move(cg), cpu);
    ASSERT_NE(csess, nullptr);
    std::vector<IOTensor> cin, cout;
    bind(cin);
    ASSERT_EQ(csess->run(cin, cout), Status::Ok);

    const float *gpuOut = gout[0].f32();
    const float *cpuOut = cout[0].f32();
    for (size_t i = 0; i < n; ++i)
    {
        EXPECT_NEAR(gpuOut[i], cpuOut[i], 1e-2f) << "element " << i << " (channel " << i / (size_t) (kH * kW) << ")";
    }
}

// A channel count that is NOT a multiple of 4 (so the last NC4HW4 block carries padded lanes) and a
// spatial extent large enough that the pixel index is not a tiny number.
TEST(PwSpatialBcast, UnalignedChannelsRuntimeMaskGpuMatchesCpu) {
    constexpr int64_t  C = 3, H = 72, W = 96;
    const size_t       n = (size_t) (C * H * W), np = (size_t) (H * W);
    std::vector<float> src(n), msk(np);
    for (size_t i = 0; i < n; ++i)
    {
        src[i] = 0.5f + (float) (i % 11) * 0.125f;
    }
    for (size_t i = 0; i < np; ++i)
    {
        msk[i] = 0.25f + 0.5f * (float) (i % 3);
    }
    auto build = [&]() {
        Graph      g;
        TensorDesc xi;
        xi.name      = "x";
        xi.shape     = {1, C, H, W};
        xi.isInput   = true;
        TensorId   x = g.addTensor(xi);
        TensorDesc mi;
        mi.name    = "m";
        mi.shape   = {1, 1, H, W};
        mi.isInput = true;
        TensorId m = g.addTensor(mi);
        g.inputs   = {x, m};
        TensorId t = g.addTensor(namedDesc("t"));
        Node     act;
        act.type    = OpType::Unary;
        act.name    = "producer";
        act.subOp   = (int) UnaryType::Abs;
        act.inputs  = {x};
        act.outputs = {t};
        TensorId mm = g.addTensor(namedDesc("mm"));
        Node     mact;
        mact.type    = OpType::Unary;
        mact.name    = "mask_producer";
        mact.subOp   = (int) UnaryType::Abs;
        mact.inputs  = {m};
        mact.outputs = {mm};
        TensorId y   = g.addTensor(namedDesc("y"));
        Node     mul;
        mul.type    = OpType::Binary;
        mul.name    = "spatial_weight";
        mul.subOp   = (int) BinaryType::Mul;
        mul.inputs  = {t, mm};
        mul.outputs = {y};
        g.nodes     = {act, mact, mul};
        g.outputs   = {y};
        return g;
    };
    auto bind = [&](std::vector<IOTensor> &in) {
        in.resize(2);
        in[0].name  = "x";
        in[0].shape = {1, C, H, W};
        in[0].dtype = DType::Float32;
        in[0].data.resize(n * sizeof(float));
        std::memcpy(in[0].data.data(), src.data(), n * sizeof(float));
        in[1].name  = "m";
        in[1].shape = {1, 1, H, W};
        in[1].dtype = DType::Float32;
        in[1].data.resize(np * sizeof(float));
        std::memcpy(in[1].data.data(), msk.data(), np * sizeof(float));
    };

    Config gpu;
    gpu.backend                 = BackendKind::Vulkan;
    gpu.allowCpuFallback        = false;
    Graph                    gg = build();
    std::unique_ptr<Session> gsess;
    try
    { gsess = Session::create(std::move(gg), gpu); } catch (const std::exception &)
    { gsess.reset(); }
    if (!gsess)
    {
        GTEST_SKIP() << "no Vulkan device";
    }
    std::vector<IOTensor> gin, gout;
    bind(gin);
    ASSERT_EQ(gsess->run(gin, gout), Status::Ok);

    Config cpu;
    cpu.backend = BackendKind::Cpu;
    Graph cg    = build();
    auto  csess = Session::create(std::move(cg), cpu);
    ASSERT_NE(csess, nullptr);
    std::vector<IOTensor> cin, cout;
    bind(cin);
    ASSERT_EQ(csess->run(cin, cout), Status::Ok);

    const float *gpuOut = gout[0].f32();
    const float *cpuOut = cout[0].f32();
    size_t       bad    = 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (std::abs(gpuOut[i] - cpuOut[i]) > 1e-2f && bad < 8)
        {
            ++bad;
            ADD_FAILURE() << "element " << i << " channel " << i / np << " pixel " << i % np << ": gpu=" << gpuOut[i] << " cpu=" << cpuOut[i];
        }
    }
}

// The path the constant/runtime tests above do NOT reach: a standalone FusedPointwise emitted into
// the NC4HW4 world, whose 1..8-step form runs the monomorphized applier in fused_pw_nc4.comp rather
// than the shared interpreter. Config::flatLayout=false pins the blocked layout so the test cannot
// drift onto the flat kernel (where the shared pwFlatIdx already handled the class correctly and
// hid this defect). A per-pixel operand read as four consecutive channel lanes shows up here as a
// wrong value on every channel except the first, and as reads past the operand's whole footprint on
// every channel block above the first.
TEST(PwSpatialBcast, BlockedLayoutSpatialOperandMatchesCpu) {
    constexpr int64_t  C = 8, H = 16, W = 24;
    const size_t       n = (size_t) (C * H * W), np = (size_t) (H * W);
    std::vector<float> src(n), msk(np);
    for (size_t i = 0; i < n; ++i)
    {
        src[i] = 0.5f + (float) (i % 11) * 0.125f;
    }
    for (size_t i = 0; i < np; ++i)
    {
        msk[i] = 0.25f + 0.5f * (float) (i % 3);
    }
    auto build = [&]() {
        Graph      g;
        TensorDesc xi;
        xi.name      = "x";
        xi.shape     = {1, C, H, W};
        xi.isInput   = true;
        TensorId   x = g.addTensor(xi);
        TensorDesc mi;
        mi.name    = "m";
        mi.shape   = {1, 1, H, W};
        mi.isInput = true;
        TensorId m = g.addTensor(mi);
        g.inputs   = {x, m};
        TensorId t = g.addTensor(namedDesc("t"));
        Node     act;
        act.type    = OpType::Unary;
        act.name    = "producer";
        act.subOp   = (int) UnaryType::Abs;
        act.inputs  = {x};
        act.outputs = {t};
        TensorId mm = g.addTensor(namedDesc("mm"));
        Node     mact;
        mact.type    = OpType::Unary;
        mact.name    = "mask_producer";
        mact.subOp   = (int) UnaryType::Abs;
        mact.inputs  = {m};
        mact.outputs = {mm};
        TensorId y   = g.addTensor(namedDesc("y"));
        Node     mul;
        mul.type    = OpType::Binary;
        mul.name    = "spatial_weight";
        mul.subOp   = (int) BinaryType::Mul;
        mul.inputs  = {t, mm};
        mul.outputs = {y};
        g.nodes     = {act, mact, mul};
        g.outputs   = {y};
        return g;
    };
    auto bind = [&](std::vector<IOTensor> &in) {
        in.resize(2);
        in[0].name  = "x";
        in[0].shape = {1, C, H, W};
        in[0].dtype = DType::Float32;
        in[0].data.resize(n * sizeof(float));
        std::memcpy(in[0].data.data(), src.data(), n * sizeof(float));
        in[1].name  = "m";
        in[1].shape = {1, 1, H, W};
        in[1].dtype = DType::Float32;
        in[1].data.resize(np * sizeof(float));
        std::memcpy(in[1].data.data(), msk.data(), np * sizeof(float));
    };

    Config gpu;
    gpu.backend          = BackendKind::Vulkan;
    gpu.allowCpuFallback = false;
    gpu.setHint(Hint::FlatLayout, (int) Mode::Off); // pin the blocked world: the NC4HW4 applier is what is under test
    Graph                    gg = build();
    std::unique_ptr<Session> gsess;
    try
    { gsess = Session::create(std::move(gg), gpu); } catch (const std::exception &)
    { gsess.reset(); }
    if (!gsess)
    {
        GTEST_SKIP() << "no Vulkan device";
    }
    std::vector<IOTensor> gin, gout;
    bind(gin);
    ASSERT_EQ(gsess->run(gin, gout), Status::Ok);

    Config cpu;
    cpu.backend = BackendKind::Cpu;
    Graph cg    = build();
    auto  csess = Session::create(std::move(cg), cpu);
    ASSERT_NE(csess, nullptr);
    std::vector<IOTensor> cin, cout;
    bind(cin);
    ASSERT_EQ(csess->run(cin, cout), Status::Ok);

    const float *gpuOut   = gout[0].f32();
    const float *cpuOut   = cout[0].f32();
    size_t       reported = 0;
    for (size_t i = 0; i < n && reported < 8; ++i)
    {
        if (std::abs(gpuOut[i] - cpuOut[i]) > 1e-2f)
        {
            ++reported;
            ADD_FAILURE() << "element " << i << " channel " << i / np << " pixel " << i % np << ": gpu=" << gpuOut[i] << " cpu=" << cpuOut[i];
        }
    }
}
