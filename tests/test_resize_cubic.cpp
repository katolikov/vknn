// Cubic Resize.
//
// `mode="cubic"` used to fall through vxResizeMode's default and run NEAREST. That is not an
// approximation of cubic, it is a different picture: cosine ~0.49 against onnxruntime, on a model
// that reported no CPU fallback and no unsupported op, so nothing anywhere said the result was
// wrong. These tests pin the kernel to the ONNX definition and pin the refusal that replaced the
// silent default.
#include "core/resize_rule.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {
    Attr strAttr(std::string s) {
        Attr a;
        a.kind = Attr::String;
        a.str  = std::move(s);
        return a;
    }
    Attr floatAttr(float v) {
        Attr a;
        a.kind = Attr::Float;
        a.f    = v;
        return a;
    }

    // Resize `in` ([1,1,inH,inW]) to [1,1,outH,outW] on the CPU backend, through the graph path a
    // model takes.
    std::vector<float> runResize(const std::vector<float> &in, int64_t inH, int64_t inW, int64_t outH, int64_t outW, const std::string &mode, const Attributes &extra = {}) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {1, 1, inH, inW};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs.push_back(x);

        TensorDesc sd;
        sd.name          = "sizes";
        sd.shape         = {4};
        sd.dtype         = DType::Int64;
        sd.isInitializer = true;
        TensorId   s     = g.addTensor(sd);
        HostBuffer hb;
        hb.resizeElems(4, DType::Int64);
        hb.i64()[0]       = 1;
        hb.i64()[1]       = 1;
        hb.i64()[2]       = outH;
        hb.i64()[3]       = outW;
        g.initializers[s] = hb;

        TensorDesc yo;
        yo.name     = "y";
        yo.shape    = {1, 1, outH, outW};
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     n;
        n.type             = OpType::Resize;
        n.name             = "rs";
        n.inputs           = {x, kNoTensor, kNoTensor, s}; // X, roi, scales, sizes
        n.outputs          = {y};
        n.attr             = extra;
        n.attr.map["mode"] = strAttr(mode);
        g.nodes.push_back(n);
        g.outputs = {y};

        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        IOTensor bind;
        bind.name  = "x";
        bind.shape = {1, 1, inH, inW};
        bind.dtype = DType::Float32;
        bind.data.resize(in.size() * sizeof(float));
        std::memcpy(bind.data.data(), in.data(), in.size() * sizeof(float));
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({bind}, outs), Status::Ok);
        return {outs[0].f32(), outs[0].f32() + outH * outW};
    }
} // namespace

// The kernel must reproduce onnxruntime's Resize-cubic element for element. A 4x4 ramp to 6x6 under
// the ONNX defaults (half_pixel, cubic_coeff_a=-0.75, exclude_outside=0) exercises both the
// interior taps and the border clamp, and the negative lobe is visible in the -0.43 corner.
TEST(ResizeCubic, MatchesOnnxRuntimeOnAKnownCase) {
    std::vector<float> in(16);
    for (int i = 0; i < 16; ++i)
    {
        in[(size_t) i] = (float) i;
    }
    const std::vector<float> want = {-0.434028f, 0.059027f,  0.865741f,  1.439815f,  2.246527f,  2.739583f,  1.538194f,  2.031250f,  2.837963f,
                                     3.412037f,  4.218750f,  4.711805f,  4.765048f,  5.258102f,  6.064816f,  6.638890f,  7.445603f,  7.938658f,
                                     7.061345f,  7.554400f,  8.361114f,  8.935187f,  9.741899f,  10.234955f, 10.288195f, 10.781250f, 11.587964f,
                                     12.162037f, 12.968750f, 13.461805f, 12.260417f, 12.753471f, 13.560185f, 14.134258f, 14.940971f, 15.434027f};
    const std::vector<float> got  = runResize(in, 4, 4, 6, 6, "cubic");
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < want.size(); ++i)
    {
        EXPECT_NEAR(got[i], want[i], 1e-4f) << "element " << i;
    }
}

// The four weights are a partition of unity for every fractional position and every coefficient, so
// a constant image resizes to itself -- the invariant a wrong sign or a dropped term breaks first.
TEST(ResizeCubic, WeightsArePartitionOfUnity) {
    for (float a: {-0.75f, -0.5f, -1.0f})
    {
        for (int k = 0; k <= 20; ++k)
        {
            float w[kResizeCubicTaps];
            vxResizeCubicWeights((float) k / 20.f, a, w);
            EXPECT_NEAR(w[0] + w[1] + w[2] + w[3], 1.0f, 1e-6f) << "a=" << a << " t=" << (float) k / 20.f;
        }
    }
}

TEST(ResizeCubic, ConstantImageResizesToItself) {
    const std::vector<float> got = runResize(std::vector<float>(36, 2.5f), 6, 6, 11, 11, "cubic");
    for (size_t i = 0; i < got.size(); ++i)
    {
        EXPECT_NEAR(got[i], 2.5f, 1e-5f) << "element " << i;
    }
}

// cubic_coeff_a is honoured rather than hard-coded: a different coefficient must give a different
// picture (the constant case above stays identical either way, so it cannot catch this).
TEST(ResizeCubic, CubicCoeffAttributeChangesTheResult) {
    std::vector<float> in(16);
    for (int i = 0; i < 16; ++i)
    {
        in[(size_t) i] = (float) ((i * 7) % 5);
    }
    Attributes half;
    half.map["cubic_coeff_a"]            = floatAttr(-0.5f);
    const std::vector<float> withDefault = runResize(in, 4, 4, 8, 8, "cubic");
    const std::vector<float> withHalf    = runResize(in, 4, 4, 8, 8, "cubic", half);
    ASSERT_EQ(withDefault.size(), withHalf.size());
    bool differs = false;
    for (size_t i = 0; i < withDefault.size(); ++i)
    {
        differs = differs || std::abs(withDefault[i] - withHalf[i]) > 1e-4f;
    }
    EXPECT_TRUE(differs) << "cubic_coeff_a is being ignored";
}

// exclude_outside renormalizes the axis once the outside taps drop, so a constant image still
// resizes to itself -- and the border pixels differ from the clamping default.
TEST(ResizeCubic, ExcludeOutsideKeepsPartitionOfUnity) {
    Attributes excl;
    Attr       one;
    one.kind                     = Attr::Int;
    one.i                        = 1;
    excl.map["exclude_outside"]  = one;
    const std::vector<float> got = runResize(std::vector<float>(36, -1.25f), 6, 6, 9, 9, "cubic", excl);
    for (size_t i = 0; i < got.size(); ++i)
    {
        EXPECT_NEAR(got[i], -1.25f, 1e-5f) << "element " << i;
    }
}

// An interpolation mode with no kernel is refused by name. It used to become nearest.
TEST(ResizeCubic, UnknownModeIsRefusedRatherThanSilentlyNearest) {
    EXPECT_EQ(vxResizeMode("nearest"), kResizeModeNearest);
    EXPECT_EQ(vxResizeMode("linear"), kResizeModeLinear);
    EXPECT_EQ(vxResizeMode("cubic"), kResizeModeCubic);
    EXPECT_THROW(vxResizeMode("lanczos"), Error);
    EXPECT_THROW(vxResizeMode("area"), Error);
}
