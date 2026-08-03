// GridSample sampler contracts (src/backend/vulkan/ops/gridsample_rule.h), the host definition the
// gridsample*.comp shaders mirror.
//
// Two classes are pinned here:
//
//   - PRECISION. The warp fold multiplies the flow operand by a scalar baked into the push
//     constant. The flow's storage precision is NOT the node's: pinSampleCoordFp32 pins a
//     coordinate cone to fp32 inside an fp16 session, and the kernel then decodes full-precision
//     flow words (FLOW_FP32). A scalar narrowed to fp16 there would put back exactly the
//     coordinate error the pin removes, and the CPU oracle multiplies by the exact fp32 attribute.
//   - EMPTY SOURCE PLANES. A source with Hin == 0 or Win == 0 has no pixel to sample. The
//     border/reflection tap resolvers clamp into [0, extent-1], an EMPTY range, so without a class
//     guard they hand the tap read a negative offset into an empty buffer.
#include "backend/vulkan/ops/gridsample_rule.h"
#include "import/passes.h"
#include "vknn/graph.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    // Padding modes a tap can be resolved under, so every case below closes the whole class rather
    // than the one mode it was first seen in.
    const uint32_t kAllPadModes[] = {kGridSamplePadZeros, kGridSamplePadBorder, kGridSamplePadReflection};

    // data[1,4,8,8] (NC4) + a warp-fused GridSample: flow (a Binary Add's output, coordinate
    // transparent) at input 1, a constant base grid at input 2. Mirrors what fuseGridSampleWarp
    // leaves behind, so pinSampleCoordFp32 sees the shape it pins in a real session.
    Graph makeWarpFlowGraph() {
        Graph g;
        auto  addTensor = [&](const char *name, Shape shape, bool flat, bool isInput = false) {
            TensorDesc d;
            d.name    = name;
            d.shape   = std::move(shape);
            d.gpuFlat = flat;
            d.isInput = isInput;
            return g.addTensor(d);
        };
        TensorId data     = addTensor("data", {1, 4, 8, 8}, false, true);
        TensorId rawFlow  = addTensor("raw_flow", {1, 2, 8, 8}, false, true);
        TensorId flowBias = addTensor("flow_bias", {1, 2, 8, 8}, false);
        {
            TensorDesc &bd   = g.desc(flowBias);
            bd.isInitializer = true;
            HostBuffer hb;
            hb.resizeElems(128, DType::Float32);
            for (int i = 0; i < 128; ++i)
            {
                hb.f32()[i] = 0.f;
            }
            g.initializers[flowBias] = hb;
        }
        TensorId flow = addTensor("flow", {1, 2, 8, 8}, false);
        TensorId base = addTensor("base", {1, 8, 8, 2}, true);
        {
            TensorDesc &bd   = g.desc(base);
            bd.isInitializer = true;
            HostBuffer hb;
            hb.resizeElems(128, DType::Float32);
            for (int i = 0; i < 128; ++i)
            {
                hb.f32()[i] = 0.f;
            }
            g.initializers[base] = hb;
        }
        TensorId y         = addTensor("y", {1, 4, 8, 8}, false);
        g.desc(y).isOutput = true;
        g.inputs           = {data, rawFlow};
        Node add;
        add.type    = OpType::Binary;
        add.subOp   = (int) BinaryType::Add;
        add.name    = "flow_add";
        add.inputs  = {rawFlow, flowBias};
        add.outputs = {flow};
        Node warp;
        warp.type    = OpType::GridSample;
        warp.name    = "warp";
        warp.inputs  = {data, flow, base};
        warp.outputs = {y};
        g.nodes      = {add, warp};
        g.outputs    = {y};
        return g;
    }

} // namespace

// --- The attribute spellings map onto the shader's MODE / PADMODE selectors, and an unknown
// spelling falls back to the ONNX default (bilinear / zeros) instead of a neighbouring branch. ---
TEST(GridSampleRule, AttributeStringsMapToShaderSelectors) {
    EXPECT_EQ(gridSampleModeCode("bilinear"), kGridSampleModeBilinear);
    EXPECT_EQ(gridSampleModeCode("linear"), kGridSampleModeBilinear);
    EXPECT_EQ(gridSampleModeCode("nearest"), kGridSampleModeNearest);
    EXPECT_EQ(gridSampleModeCode("cubic"), kGridSampleModeCubic);
    EXPECT_EQ(gridSampleModeCode("bicubic"), kGridSampleModeCubic);
    EXPECT_EQ(gridSampleModeCode(""), kGridSampleModeBilinear);
    EXPECT_EQ(gridSamplePadCode("zeros"), kGridSamplePadZeros);
    EXPECT_EQ(gridSamplePadCode("border"), kGridSamplePadBorder);
    EXPECT_EQ(gridSamplePadCode("reflection"), kGridSamplePadReflection);
    EXPECT_EQ(gridSamplePadCode(""), kGridSamplePadZeros);
}

// --- The coordinate operand's decode precision: a constant grid is uploaded fp32 by the op, a
// runtime grid rides its own storage; the warp flow is always its own activation buffer, so only a
// pin makes it fp32. ---
TEST(GridSampleRule, CoordinateOperandDecodePrecision) {
    EXPECT_TRUE(gridSampleGridWordsFp32(/*gridIsInitializer=*/true, /*gridStoreFp32=*/false)) << "a constant grid uploads fp32";
    EXPECT_TRUE(gridSampleGridWordsFp32(false, true)) << "a pinned runtime grid decodes fp32";
    EXPECT_FALSE(gridSampleGridWordsFp32(false, false)) << "an unpinned runtime grid decodes at its fp16 storage";
    EXPECT_FALSE(gridSampleFlowWordsFp32(/*flowIsInitializer=*/true, /*flowStoreFp32=*/false)) << "the flow is never uploaded by the op";
    EXPECT_TRUE(gridSampleFlowWordsFp32(false, true)) << "a pinned runtime flow decodes fp32";
    EXPECT_FALSE(gridSampleFlowWordsFp32(false, false));
}

// --- The warp scalar follows the FLOW's precision, not the session's. A pinned flow decodes full
// fp32 words, so the scalar stays exact: narrowing it there reintroduces the ~2^-11 relative
// coordinate error the pin exists to remove, and the fp32 CPU oracle multiplies by the exact
// attribute value. ---
TEST(GridSampleRule, WarpScaleFollowsFlowPrecisionNotSessionPrecision) {
    const float scale = 0.1f; // not representable in fp16
    EXPECT_NE(scale, halfToFloat(floatToHalfSat(scale))) << "the case needs a scale fp16 cannot hold";
    EXPECT_FLOAT_EQ(gridSampleWarpScale(scale, /*sessionFp16=*/true, /*flowWordsFp32=*/true), scale) << "a pinned fp32 flow multiplies by the exact scalar";
    EXPECT_EQ(gridSampleWarpScale(scale, /*sessionFp16=*/false, /*flowWordsFp32=*/true), scale) << "an fp32 session keeps the scalar exact";
    // An UNPINNED fp16 flow keeps the narrowing: there the kernel reproduces the standalone Mul,
    // whose scalar operand was itself stored fp16.
    EXPECT_EQ(gridSampleWarpScale(scale, /*sessionFp16=*/true, /*flowWordsFp32=*/false), halfToFloat(floatToHalfSat(scale)));
}

// --- The pin really does fire on a warp flow, so the rule above is not hypothetical:
// pinSampleCoordFp32 walks every GridSample coordinate operand, and a warp flow produced by a
// coordinate-transparent op is pinned fp32 — which is exactly the state that must keep the scalar
// exact. ---
TEST(GridSampleRule, PinnedWarpFlowKeepsTheScalarExact) {
    Graph g = makeWarpFlowGraph();
    pinSampleCoordFp32(g);
    const Node &warp = g.nodes[1];
    ASSERT_EQ(warp.type, OpType::GridSample);
    const TensorId flow = warp.inputs[1];
    ASSERT_TRUE(g.desc(flow).storeFp32) << "pinSampleCoordFp32 must pin the warp flow";
    const bool flowFp32 = gridSampleFlowWordsFp32(g.isInitializer(flow), g.desc(flow).storeFp32);
    EXPECT_TRUE(flowFp32) << "a pinned runtime flow decodes fp32 words";
    const float scale = 2.0f / 1023.0f; // a normalized pixel step, 2 / (width - 1)
    EXPECT_EQ(gridSampleWarpScale(scale, /*sessionFp16=*/true, flowFp32), scale) << "fp32 flow words multiply by an exact scalar";
}

// --- A source plane with a zero extent holds no pixel: every tap is out of range in EVERY padding
// mode. Border and reflection clamp into [0, extent-1] — an empty range whose result is undefined
// in GLSL — and the tap read then indexes an empty source buffer. ---
TEST(GridSampleRule, EmptySourcePlaneTapsAreOutOfRangeInEveryPadMode) {
    const int emptyPlanes[][2] = {{0, 8}, {8, 0}, {0, 0}}; // {Hin, Win}
    for (uint32_t pad: kAllPadModes)
    {
        for (const auto &plane: emptyPlanes)
        {
            const int hin = plane[0], win = plane[1];
            for (int align = 0; align <= 1; ++align)
            {
                SCOPED_TRACE("pad=" + std::to_string(pad) + " Hin=" + std::to_string(hin) + " Win=" + std::to_string(win) + " align=" + std::to_string(align));
                EXPECT_TRUE(gridSampleTapsCanBeOob(pad, hin, win)) << "the tap read must test for the out-of-range marker";
                // Walk the CUBIC fan (the widest: floor-1 .. floor+2 per axis) around coordinates
                // that map below, inside and above the extent the plane would have had.
                const int mappedFloors[] = {-1, 0, 2};
                for (int mappedFloor: mappedFloors)
                {
                    for (int t = 0; t < kGridSampleCubicTaps; ++t)
                    {
                        const int tap    = mappedFloor - 1 + t;
                        const int column = gridSampleResolveTapColumn(tap, pad, win, align);
                        const int row    = gridSampleResolveTapRow(tap, pad, hin, win, align);
                        if (win == 0)
                        {
                            EXPECT_EQ(column, kGridSampleTapOob) << "no column exists on an empty plane";
                        }
                        if (hin == 0)
                        {
                            EXPECT_EQ(row, kGridSampleTapOob) << "no row exists on an empty plane";
                        }
                        EXPECT_EQ(gridSampleTapOffset(pad, hin, win, column, row), kGridSampleTapOob)
                            << "an empty plane must contribute zero, never a source offset";
                    }
                }
            }
        }
    }
}

// --- A non-empty plane keeps the padding semantics exactly: zeros reports out-of-range taps,
// border clamps to the edge pixel, reflection bounces off the edges. Hand-computed for a 4-wide
// axis so a regression in the empty-plane guard cannot pass by widening the OOB report. ---
TEST(GridSampleRule, NonEmptyPlaneTapResolutionIsUnchanged) {
    constexpr int kWidth = 4, kHeight = 4;
    const int     taps[] = {-2, -1, 0, 1, 2, 3, 4, 5};
    // zeros: only the in-range taps read a column.
    const int expectZeros[] = {kGridSampleTapOob, kGridSampleTapOob, 0, 1, 2, 3, kGridSampleTapOob, kGridSampleTapOob};
    // border: clamp onto the edge pixels.
    const int expectBorder[] = {0, 0, 0, 1, 2, 3, 3, 3};
    // reflection (align_corners=0, axis edges at -0.5 and 3.5): a tap d past an edge lands d inside.
    const int expectReflect0[] = {1, 0, 0, 1, 2, 3, 3, 2};
    // reflection (align_corners=1, axis edges at 0 and 3): the edge pixel is the mirror line.
    const int expectReflect1[] = {2, 1, 0, 1, 2, 3, 2, 1};
    for (size_t i = 0; i < sizeof(taps) / sizeof(taps[0]); ++i)
    {
        SCOPED_TRACE("tap=" + std::to_string(taps[i]));
        EXPECT_FALSE(gridSampleTapsCanBeOob(kGridSamplePadBorder, kHeight, kWidth)) << "a non-empty border plane never reports an out-of-range tap";
        EXPECT_FALSE(gridSampleTapsCanBeOob(kGridSamplePadReflection, kHeight, kWidth));
        EXPECT_TRUE(gridSampleTapsCanBeOob(kGridSamplePadZeros, kHeight, kWidth));
        EXPECT_EQ(gridSampleResolveTapColumn(taps[i], kGridSamplePadZeros, kWidth, 0), expectZeros[i]);
        EXPECT_EQ(gridSampleResolveTapColumn(taps[i], kGridSamplePadBorder, kWidth, 0), expectBorder[i]);
        EXPECT_EQ(gridSampleResolveTapColumn(taps[i], kGridSamplePadReflection, kWidth, 0), expectReflect0[i]);
        EXPECT_EQ(gridSampleResolveTapColumn(taps[i], kGridSamplePadReflection, kWidth, 1), expectReflect1[i]);
        // The row resolver returns a row OFFSET: the same row index scaled by the plane width.
        const int expectedRow = expectZeros[i] == kGridSampleTapOob ? kGridSampleTapOob : expectZeros[i] * kWidth;
        EXPECT_EQ(gridSampleResolveTapRow(taps[i], kGridSamplePadZeros, kHeight, kWidth, 0), expectedRow);
        EXPECT_EQ(gridSampleResolveTapRow(taps[i], kGridSamplePadBorder, kHeight, kWidth, 0), expectBorder[i] * kWidth);
    }
}
