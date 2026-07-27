// Nearest Resize: the source index is an exact integer function of the shape.
//
// An integer downsample factor under half_pixel puts EVERY output pixel exactly on
// round_prefer_floor's tie (fy = 2d + 0.5 for a halving), so the whole result hinges on the tie
// going down. Evaluating the coordinate in floats does not decide it reliably -- the GPU computes
// the divide through a reciprocal, and one ulp there moved every sample a whole pixel (the GPU read
// x[2d+1] where the CPU oracle read x[2d], on every pixel of every channel, at cosine ~0 against the
// oracle). These tests pin the integer rule and the values the kernel must produce.
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace vknn {
    int vxResizeNearestSrc(int d, int outS, int inS, int coordMode);
}

namespace {
    constexpr int kCoordHalfPixel        = 0;
    constexpr int kCoordAlignCorners     = 1;
    constexpr int kCoordAsymmetric       = 2;
    constexpr int kCoordPytorchHalfPixel = 3;

    Attr strAttr(std::string s) {
        Attr a;
        a.kind = Attr::String;
        a.str  = std::move(s);
        return a;
    }

    // Run a nearest Resize of `in` (shape [1,1,inH,inW]) by `scale` on the CPU backend, through the
    // same graph path a model takes. The input desc is left unresolved so the geometry comes from the
    // bound runtime shape times `scales`, exactly as a real Resize node resolves it.
    std::vector<float> runNearest(const std::vector<float> &in, int64_t inH, int64_t inW, float scale, const std::string &coordMode) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs.push_back(x);

        TensorDesc sd;
        sd.name          = "scales";
        sd.shape         = {4};
        sd.isInitializer = true;
        TensorId   s     = g.addTensor(sd);
        HostBuffer hb;
        hb.resizeElems(4, DType::Float32);
        hb.f32()[0]       = 1.f;
        hb.f32()[1]       = 1.f;
        hb.f32()[2]       = scale;
        hb.f32()[3]       = scale;
        g.initializers[s] = hb;

        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     n;
        n.type                                       = OpType::Resize;
        n.name                                       = "rs";
        n.inputs                                     = {x, kNoTensor, s}; // X, roi (absent), scales
        n.outputs                                    = {y};
        n.attr.map["mode"]                           = strAttr("nearest");
        n.attr.map["coordinate_transformation_mode"] = strAttr(coordMode);
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
        return {outs[0].f32(), outs[0].f32() + numElements(outs[0].shape)};
    }
} // namespace

// Halving under half_pixel: fy = 2d + 0.5 on every output pixel, and round_prefer_floor keeps the
// even source column. Taking the tie the other way returns 2d+1 everywhere -- a whole-pixel shift.
TEST(ResizeNearest, HalvingKeepsTheEvenSourcePixel) {
    for (int outS: {1, 2, 3, 4, 8, 37, 129, 1031})
    {
        for (int d = 0; d < outS; ++d)
        {
            EXPECT_EQ(vxResizeNearestSrc(d, outS, outS * 2, kCoordHalfPixel), 2 * d) << "outS=" << outS << " d=" << d;
        }
    }
}

// Two axes halved together, at extents large enough that the float coordinate is far from where
// its mantissa is exact: every index must still be even on both.
TEST(ResizeNearest, LargeHalvingIsExactOnBothAxes) {
    constexpr int kTallOut = 517, kWideOut = 1031; // coprime, neither a power of two
    for (int d = 0; d < kTallOut; ++d)
    {
        EXPECT_EQ(vxResizeNearestSrc(d, kTallOut, 2 * kTallOut, kCoordHalfPixel), 2 * d) << "row " << d;
    }
    for (int d = 0; d < kWideOut; ++d)
    {
        EXPECT_EQ(vxResizeNearestSrc(d, kWideOut, 2 * kWideOut, kCoordHalfPixel), 2 * d) << "col " << d;
    }
}

// The integer rule must agree with the float rule wherever the float rule is unambiguous, i.e.
// everywhere the fractional coordinate is not exactly .5. Sweeping a wide range of ratios in every
// coordinate mode is what keeps this from being a rewrite that changes unrelated results.
TEST(ResizeNearest, AgreesWithTheFloatRuleAwayFromTheTie) {
    for (int coordMode: {kCoordHalfPixel, kCoordAlignCorners, kCoordAsymmetric, kCoordPytorchHalfPixel})
    {
        for (int inS: {1, 2, 3, 5, 7, 8, 16, 33, 64, 100})
        {
            for (int outS: {1, 2, 3, 5, 7, 8, 16, 33, 64, 100})
            {
                for (int d = 0; d < outS; ++d)
                {
                    const double scale = (double) outS / (double) inS;
                    double       fy;
                    if (coordMode == kCoordAlignCorners)
                    {
                        fy = outS > 1 ? (double) d * (inS - 1) / (outS - 1) : 0.0;
                    } else if (coordMode == kCoordAsymmetric)
                    {
                        fy = (double) d / scale;
                    } else if (coordMode == kCoordPytorchHalfPixel)
                    {
                        fy = outS > 1 ? ((double) d + 0.5) / scale - 0.5 : 0.0;
                    } else
                    {
                        fy = ((double) d + 0.5) / scale - 0.5;
                    }
                    double floorY = std::floor(fy), frac = fy - floorY;
                    if (std::abs(frac - 0.5) < 1e-9)
                    {
                        // The tie itself: the rule keeps the floor, which is what the float form
                        // could not be trusted to do.
                        EXPECT_EQ(vxResizeNearestSrc(d, outS, inS, coordMode), (int) floorY) << "mode=" << coordMode << " in=" << inS << " out=" << outS << " d=" << d;
                        continue;
                    }
                    const int want = (int) floorY + (frac > 0.5 ? 1 : 0);
                    EXPECT_EQ(vxResizeNearestSrc(d, outS, inS, coordMode), want) << "mode=" << coordMode << " in=" << inS << " out=" << outS << " d=" << d;
                }
            }
        }
    }
}

// End to end through the kernel: a 4x4 ramp halved must return the even-indexed corners.
TEST(ResizeNearest, KernelHalvingPicksTheEvenGrid) {
    std::vector<float> in(16);
    for (int i = 0; i < 16; ++i)
    {
        in[(size_t) i] = (float) i;
    }
    const std::vector<float> got = runNearest(in, 4, 4, 0.5f, "half_pixel");
    EXPECT_EQ(got, (std::vector<float> {0, 2, 8, 10}));
}

// Upsampling is unaffected: 2x under half_pixel duplicates each source pixel.
TEST(ResizeNearest, KernelDoublingDuplicatesEachPixel) {
    const std::vector<float> got = runNearest({1, 2, 3, 4}, 2, 2, 2.0f, "half_pixel");
    EXPECT_EQ(got, (std::vector<float> {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}));
}
