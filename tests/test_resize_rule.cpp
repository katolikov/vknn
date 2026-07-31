// The Resize geometry rule, on both sides of the CPU/GPU contract.
//
// The nearest source index is an exact integer function of the shape, and the four GPU kernels
// evaluate it in 32-bit signed arithmetic because GLSL has no guaranteed 64-bit integer.
// vxResizeNearestSrcNarrow is that 32-bit form, host-callable, and the shaders carry a
// line-for-line copy of it; these tests pin it to the 64-bit reference over the whole set of
// extents both ops accept, so a source pixel the GPU resolves is the source pixel the oracle
// resolves. They also pin the degenerate geometry each op has to refuse instead of sampling: an
// input axis of zero extent has no pixel to read, and an extent past kResizeMaxSpatialExtent has
// no exact 32-bit form.
#include "backend/vulkan/resize_race_scratch.h"
#include "core/resize_rule.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    // Every coordinate transform and every nearest rounding, so a sweep covers the whole rule
    // rather than the default corner of it.
    constexpr int kCoordModes[]   = {kResizeCoordHalfPixel, kResizeCoordAlignCorners, kResizeCoordAsymmetric, kResizeCoordPytorchHalfPixel};
    constexpr int kNearestModes[] = {kResizeNearestPreferFloor, kResizeNearestPreferCeil, kResizeNearestFloor, kResizeNearestCeil};

    Attr strAttr(std::string s) {
        Attr a;
        a.kind = Attr::String;
        a.str  = std::move(s);
        return a;
    }

    // A Resize graph whose output geometry comes from an explicit `sizes` initializer, run on the
    // CPU backend against a runtime input shape. Returns the session's status so a test can assert
    // a refusal as well as a result; `out` receives the single output tensor on success. An op that
    // refuses surfaces as Status::RuntimeError, which is what Session::run maps a thrown Error to.
    Status runResizeWithSizes(const std::string &mode, const Shape &inputShape, const std::vector<int64_t> &sizes, IOTensor &out) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {}; // unresolved: the geometry comes from the bound runtime shape plus `sizes`
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs.push_back(x);

        TensorDesc sd;
        sd.name          = "sizes";
        sd.shape         = {(int64_t) sizes.size()};
        sd.dtype         = DType::Int64;
        sd.isInitializer = true;
        TensorId   s     = g.addTensor(sd);
        HostBuffer hb;
        hb.resizeElems((int64_t) sizes.size(), DType::Int64);
        for (size_t k = 0; k < sizes.size(); ++k)
        {
            hb.i64()[k] = sizes[k];
        }
        g.initializers[s] = hb;

        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     n;
        n.type             = OpType::Resize;
        n.name             = "rs";
        n.inputs           = {x, kNoTensor, kNoTensor, s}; // X, roi (absent), scales (absent), sizes
        n.outputs          = {y};
        n.attr.map["mode"] = strAttr(mode);
        g.nodes.push_back(n);
        g.outputs = {y};

        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        int64_t elems = 1;
        for (int64_t dim: inputShape)
        {
            elems *= dim;
        }
        IOTensor bind;
        bind.name  = "x";
        bind.shape = inputShape;
        bind.dtype = DType::Float32;
        bind.data.assign((size_t) elems * sizeof(float), 0);
        for (int64_t k = 0; k < elems; ++k)
        {
            reinterpret_cast<float *>(bind.data.data())[k] = (float) k;
        }
        std::vector<IOTensor> outs;
        Status                st = sess->run({bind}, outs);
        if (st == Status::Ok && !outs.empty())
        {
            out = outs[0];
        }
        return st;
    }

} // namespace

// The 32-bit form the shaders execute must resolve the same pixel as the 64-bit reference, on
// every ratio class: exact integer up/downscales (where the half_pixel tie decides the whole
// picture), coprime ratios, and single-pixel axes.
TEST(ResizeRule, NarrowMirrorMatchesTheOracleOnOrdinaryExtents) {
    for (int inS: {1, 2, 3, 5, 8, 17, 64, 127, 256, 1031})
    {
        for (int outS: {1, 2, 3, 5, 8, 17, 64, 127, 256, 1031})
        {
            for (int coordMode: kCoordModes)
            {
                for (int nearestMode: kNearestModes)
                {
                    for (int d = 0; d < outS; ++d)
                    {
                        EXPECT_EQ(vxResizeNearestSrcNarrow(d, outS, inS, coordMode, nearestMode), vxResizeNearestSrc(d, outS, inS, coordMode, nearestMode)) << "inS=" << inS << " outS=" << outS << " cm=" << coordMode << " nm=" << nearestMode << " d=" << d;
                    }
                }
            }
        }
    }
}

// Past outS * inS = 2^30 the 32-bit numerator of the half_pixel rule no longer holds the product.
// Both tensors here are well under a megabyte, so nothing upstream refuses the shape: the rule
// itself has to stay exact. Sampling near the tail is where the product is largest.
TEST(ResizeRule, NarrowMirrorStaysExactPastTheThirtyBitProduct) {
    constexpr int kTallIn = 40000, kTallOut = 30000; // outS * inS = 1.2e9, past 2^30
    for (int coordMode: kCoordModes)
    {
        for (int nearestMode: kNearestModes)
        {
            for (int d: {0, 1, 12345, 26843, 26844, 29998, 29999})
            {
                EXPECT_EQ(vxResizeNearestSrcNarrow(d, kTallOut, kTallIn, coordMode, nearestMode), vxResizeNearestSrc(d, kTallOut, kTallIn, coordMode, nearestMode)) << "cm=" << coordMode << " nm=" << nearestMode << " d=" << d;
            }
        }
    }
    // The concrete pixel the 32-bit product wraps on: (2*29999+1) * 40000 - 30000 is 2.4e9, which
    // a signed 32-bit numerator carries as a large negative and clamps to source column 0.
    EXPECT_EQ(vxResizeNearestSrcNarrow(29999, kTallOut, kTallIn, kResizeCoordHalfPixel, kResizeNearestPreferFloor), 39999);
}

// The bit-scanned path and the direct-product path are two spellings of one rule, so the boundary
// between them must not be observable. Extents straddling kResizeNarrowFactorMax exercise both
// with the same ratios.
TEST(ResizeRule, NarrowMirrorAgreesAcrossTheDirectProductBoundary) {
    for (int inS: {kResizeNarrowFactorMax - 1, kResizeNarrowFactorMax, kResizeNarrowFactorMax + 1, 2 * kResizeNarrowFactorMax})
    {
        for (int outS: {kResizeNarrowFactorMax - 1, kResizeNarrowFactorMax, kResizeNarrowFactorMax + 1, 2 * kResizeNarrowFactorMax})
        {
            for (int coordMode: kCoordModes)
            {
                for (int nearestMode: kNearestModes)
                {
                    for (int d: {0, 1, outS / 3, outS / 2, outS - 2, outS - 1})
                    {
                        EXPECT_EQ(vxResizeNearestSrcNarrow(d, outS, inS, coordMode, nearestMode), vxResizeNearestSrc(d, outS, inS, coordMode, nearestMode)) << "inS=" << inS << " outS=" << outS << " cm=" << coordMode << " nm=" << nearestMode << " d=" << d;
                    }
                }
            }
        }
    }
}

// The largest extents either op admits. Both the doubled denominator (2 * outS) and the doubled
// output position (2 * d + 1) sit at the top of the 32-bit range here, which is the bound
// kResizeMaxSpatialExtent names.
TEST(ResizeRule, NarrowMirrorMatchesTheOracleAtTheAcceptedExtentBound) {
    const int kBound = (int) kResizeMaxSpatialExtent;
    for (int inS: {1, 7, kBound - 1, kBound})
    {
        for (int outS: {1, 7, kBound - 1, kBound})
        {
            for (int coordMode: kCoordModes)
            {
                for (int nearestMode: kNearestModes)
                {
                    for (int d: {0, 1, outS / 2, outS - 1})
                    {
                        EXPECT_EQ(vxResizeNearestSrcNarrow(d, outS, inS, coordMode, nearestMode), vxResizeNearestSrc(d, outS, inS, coordMode, nearestMode)) << "inS=" << inS << " outS=" << outS << " cm=" << coordMode << " nm=" << nearestMode << " d=" << d;
                    }
                }
            }
        }
    }
}

// An output axis longer than 2^30 makes 2 * d + 1 exceed a 32-bit int. The reference form carries
// it in 64 bits, so the value stays the exact ratio and the cast order inside the rule is not free
// to overflow first.
TEST(ResizeRule, ReferenceRuleSurvivesAnOutputPositionPastTheThirtyBitMark) {
    constexpr int kHugeOut = 1 << 30, kSmallIn = 4;
    // asymmetric: d * inS / outS. At d = outS - 1 the exact quotient is just under inS.
    EXPECT_EQ(vxResizeNearestSrc(kHugeOut - 1, kHugeOut, kSmallIn, kResizeCoordAsymmetric, kResizeNearestFloor), kSmallIn - 1);
    // half_pixel at the same position: ((2d+1) * inS - outS) / (2 * outS) is (2^32-... ) / 2^31,
    // which is inS - 1 plus a fraction under a half, so prefer_floor keeps inS - 1.
    EXPECT_EQ(vxResizeNearestSrc(kHugeOut - 1, kHugeOut, kSmallIn, kResizeCoordHalfPixel, kResizeNearestPreferFloor), kSmallIn - 1);
    // 2 * d + 1 overflows a 32-bit int at exactly this position; the reference must not.
    EXPECT_EQ(vxResizeNearestSrc(kHugeOut, kHugeOut + 1, kSmallIn, kResizeCoordHalfPixel, kResizeNearestPreferFloor), kSmallIn - 1);
}

// An input axis of zero extent has no source pixel at all. Nothing in the rule can invent one, so
// it resolves to 0 rather than to the -1 an unguarded floor produces.
TEST(ResizeRule, ZeroExtentInputAxisResolvesToZeroInBothForms) {
    for (int coordMode: kCoordModes)
    {
        for (int nearestMode: kNearestModes)
        {
            for (int d: {0, 1, 7})
            {
                EXPECT_EQ(vxResizeNearestSrc(d, 8, 0, coordMode, nearestMode), 0) << "cm=" << coordMode << " nm=" << nearestMode << " d=" << d;
                EXPECT_EQ(vxResizeNearestSrcNarrow(d, 8, 0, coordMode, nearestMode), 0) << "cm=" << coordMode << " nm=" << nearestMode << " d=" << d;
            }
        }
    }
}

// A zero-extent input spatial axis with an explicit nonzero output size asks every arm to sample a
// plane that holds no elements. Each arm clamps its tap indices into [0, extent-1], which is an
// empty range here, so sampling would read off the front of an empty buffer. The op refuses.
TEST(ResizeCpuOp, ZeroExtentInputWithNonzeroSizesIsRefusedInEveryArm) {
    for (const char *mode: {"nearest", "linear", "cubic"})
    {
        IOTensor out;
        EXPECT_EQ(runResizeWithSizes(mode, {1, 3, 0, 5}, {1, 3, 4, 4}, out), Status::RuntimeError) << "mode=" << mode << " zero H";
        EXPECT_EQ(runResizeWithSizes(mode, {1, 3, 5, 0}, {1, 3, 4, 4}, out), Status::RuntimeError) << "mode=" << mode << " zero W";
    }
}

// A `sizes` tensor that asks for a zero-extent output axis is a request for an empty tensor, not a
// missing parameter: the pre-opset-10 Upsample fallback must not step in and substitute the graph
// desc's shape. The output is empty and its shape is the one that was asked for.
TEST(ResizeCpuOp, ZeroExtentOutputSizeProducesAnEmptyTensorOfThatShape) {
    for (const char *mode: {"nearest", "linear", "cubic"})
    {
        IOTensor out;
        ASSERT_EQ(runResizeWithSizes(mode, {1, 2, 4, 4}, {1, 2, 0, 3}, out), Status::Ok) << "mode=" << mode;
        EXPECT_EQ(out.shape, (Shape {1, 2, 0, 3})) << "mode=" << mode;
        EXPECT_EQ(numElements(out.shape), 0) << "mode=" << mode;
    }
}

// Both spatial axes empty on both sides is the identity of the empty case: nothing to read and
// nothing to write, and it must run rather than refuse.
TEST(ResizeCpuOp, ZeroExtentInputWithZeroExtentOutputRuns) {
    IOTensor out;
    ASSERT_EQ(runResizeWithSizes("nearest", {1, 2, 0, 4}, {1, 2, 0, 4}, out), Status::Ok);
    EXPECT_EQ(numElements(out.shape), 0);
}

// The extent bound is a refusal, not a truncation: an output axis past kResizeMaxSpatialExtent has
// no exact 32-bit form for the GPU kernels to evaluate, and casting it into the rule's `int`
// parameters would silently resample a different picture.
TEST(ResizeCpuOp, SpatialExtentPastTheAcceptedBoundIsRefused) {
    IOTensor out;
    EXPECT_EQ(runResizeWithSizes("nearest", {1, 1, 2, 2}, {1, 1, kResizeMaxSpatialExtent + 1, 2}, out), Status::RuntimeError);
    EXPECT_EQ(runResizeWithSizes("nearest", {1, 1, 2, 2}, {1, 1, 2, kResizeMaxSpatialExtent + 1}, out), Status::RuntimeError);
}

// A negative `sizes` entry is not a shape. It must be refused rather than folded into an
// allocation whose element count silently reads as something else.
TEST(ResizeCpuOp, NegativeOutputSizeIsRefused) {
    IOTensor out;
    EXPECT_EQ(runResizeWithSizes("nearest", {1, 1, 4, 4}, {1, 1, -4, 4}, out), Status::RuntimeError);
}

// The kernel race allocates buffers sized like the node's real NC4HW4 tensors: source-sized plus
// destination-sized, and a second destination-sized one when an epilogue is fused. On an upscale
// the destination-sized pair dominates, so the footprint grows with the square of the scale.
TEST(ResizeRaceScratch, FootprintIsTheSumOfTheTensorsTheRaceAllocates) {
    constexpr int64_t kBatch = 1, kChannels = 32, kInHeight = 270, kInWidth = 480;
    constexpr int64_t kOutHeight = 1080, kOutWidth = 1920;
    const size_t      source = resizeNc4TensorBytes(kBatch, kChannels, kInHeight, kInWidth, /*fp16Storage=*/true);
    const size_t      dest   = resizeNc4TensorBytes(kBatch, kChannels, kOutHeight, kOutWidth, /*fp16Storage=*/true);
    EXPECT_EQ(resizeRaceScratchBytes(kBatch, kChannels, kInHeight, kInWidth, kOutHeight, kOutWidth, /*epilogueActive=*/false, /*fp16Storage=*/true), source + dest);
    EXPECT_EQ(resizeRaceScratchBytes(kBatch, kChannels, kInHeight, kInWidth, kOutHeight, kOutWidth, /*epilogueActive=*/true, /*fp16Storage=*/true), source + 2 * dest);
    // Channels round up to whole blocks of kNC4Block, so a partial block is paid for in full.
    EXPECT_EQ(resizeNc4TensorBytes(kBatch, /*channels=*/1, kInHeight, kInWidth, /*fp16Storage=*/true), resizeNc4TensorBytes(kBatch, /*channels=*/kNC4Block, kInHeight, kInWidth, /*fp16Storage=*/true));
}

// The budget is what keeps a load-time speed measurement from becoming the largest allocation in
// the session. A four-times upscale of a mid-size feature map is over it and must not race; the
// small maps a race is worth running on stay inside it.
TEST(ResizeRaceScratch, LargeUpscaleIsOverTheBudgetAndOrdinaryMapsAreNot) {
    EXPECT_FALSE(resizeRaceScratchFits(1, 32, 270, 480, 1080, 1920, /*epilogueActive=*/true, /*fp16Storage=*/true));
    EXPECT_FALSE(resizeRaceScratchFits(1, 32, 270, 480, 1080, 1920, /*epilogueActive=*/false, /*fp16Storage=*/true));
    EXPECT_TRUE(resizeRaceScratchFits(1, 32, 28, 28, 56, 56, /*epilogueActive=*/true, /*fp16Storage=*/true));
    EXPECT_TRUE(resizeRaceScratchFits(1, 256, 14, 14, 28, 28, /*epilogueActive=*/true, /*fp16Storage=*/false));
    // fp32 storage doubles the footprint, so the fit rule has to move with the precision.
    EXPECT_LT(resizeRaceScratchBytes(1, 64, 64, 64, 128, 128, /*epilogueActive=*/true, /*fp16Storage=*/true), resizeRaceScratchBytes(1, 64, 64, 64, 128, 128, /*epilogueActive=*/true, /*fp16Storage=*/false));
    // The predicate is exactly the budget comparison, on both sides of the boundary.
    EXPECT_TRUE(resizeRaceScratchBytes(1, 4, 1, 1, 1, 1, /*epilogueActive=*/false, /*fp16Storage=*/false) <= kResizeRaceScratchBudgetBytes);
    EXPECT_FALSE(resizeRaceScratchFits(1, 4, 1, 1, kResizeRaceScratchBudgetBytes, 1, /*epilogueActive=*/false, /*fp16Storage=*/false));
}
