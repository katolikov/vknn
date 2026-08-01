// A Cast to an integer type must truncate the value the oracle sees, not an fp16 rounding of it.
//
// Truncation is discontinuous: it maps a whole interval to one integer and jumps at the boundary,
// so ANY storage rounding that crosses a boundary becomes a full-unit output error, not a small
// one. An input just under 1.0 -- 0.9998311 is inside the range a normalized activation occupies
// -- rounds to exactly 1.0 in fp16 (the gap between representable values near 1.0 is 2^-11, about
// 0.00049), and trunc then yields 1 where the fp32 oracle yields 0. The error is 100 percent of
// the value, and it lands on every input within one fp16 ulp below an integer.
//
// This is the class pinGatherIndexFp32 and pinSampleCoordFp32 already close for index and
// coordinate tensors. These tests pin the same contract for the Cast operand cone.
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/unary_type.h"
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    // ONNX TensorProto dtype codes the Cast attribute carries.
    constexpr int64_t kOnnxFloat = 1;
    constexpr int64_t kOnnxInt32 = 6;
    constexpr int64_t kOnnxInt64 = 7;
    constexpr int64_t kOnnxUint8 = 2;

    TensorId addNamed(Graph &g, const char *name, bool isInput = false) {
        TensorDesc d;
        d.name    = name;
        d.shape   = {1, 4, 8, 8};
        d.isInput = isInput;
        return g.addTensor(d);
    }

    // x -> Cast(to) -> y, optionally with a pure movement hop in front of the Cast so the walk has
    // something to follow back toward the boundary input.
    Graph buildCastGraph(int64_t to, bool withMovementHop) {
        Graph    g;
        TensorId x      = addNamed(g, "x", /*isInput=*/true);
        g.inputs        = {x};
        TensorId castIn = x;
        if (withMovementHop)
        {
            TensorId hop = addNamed(g, "hop");
            Node     mv;
            mv.type    = OpType::ConvertLayout;
            mv.name    = "movement";
            mv.inputs  = {x};
            mv.outputs = {hop};
            g.nodes.push_back(mv);
            castIn = hop;
        }
        TensorId y = addNamed(g, "y");
        Node     c;
        c.type    = OpType::Cast;
        c.name    = "cast";
        c.inputs  = {castIn};
        c.outputs = {y};
        Attr toAttr;
        toAttr.kind      = Attr::Int;
        toAttr.i         = to;
        c.attr.map["to"] = toAttr;
        g.nodes.push_back(c);
        g.outputs = {y};
        return g;
    }

    // x -> Unary(step) -> y, the shape foldIntRoundtripCast leaves behind.
    Graph buildUnaryGraph(UnaryType step) {
        Graph    g;
        TensorId x = addNamed(g, "x", /*isInput=*/true);
        g.inputs   = {x};
        TensorId y = addNamed(g, "y");
        Node     u;
        u.type    = OpType::Unary;
        u.name    = "step";
        u.subOp   = (int32_t) step;
        u.inputs  = {x};
        u.outputs = {y};
        g.nodes.push_back(u);
        g.outputs = {y};
        return g;
    }

} // namespace

// The reason the pin is needed, stated as arithmetic rather than as a claim: an fp32 value inside
// the last fp16 step below 1.0 truncates differently once it is stored as fp16.
TEST(Fp32PinDiscontinuousStep, Fp16StorageMovesTheTruncationBoundary) {
    const float justUnderOne = 0.9998311f;
    // Round-trip through fp16 the way the flat storage path does.
    const auto asHalfBits = [](float v) {
        // Values in [0.5, 1.0) carry an 11-bit significand in fp16; rounding to nearest even at
        // that width is what the store performs.
        const float scale = 2048.0f; // 2^11, the significand step across [0.5, 1.0)
        return std::nearbyint(v * scale) / scale;
    };
    EXPECT_LT(justUnderOne, 1.0f);
    EXPECT_FLOAT_EQ(asHalfBits(justUnderOne), 1.0f) << "fp16 rounds this input up to exactly 1.0";
    EXPECT_EQ((int) justUnderOne, 0) << "the oracle truncates to 0";
    EXPECT_EQ((int) asHalfBits(justUnderOne), 1) << "the fp16 store truncates to 1 -- a full unit off";
}

TEST(Fp32PinDiscontinuousStep, IntegerTargetPinsTheCastOperand) {
    for (int64_t to: {kOnnxInt32, kOnnxInt64, kOnnxUint8})
    {
        Graph g = buildCastGraph(to, /*withMovementHop=*/false);
        pinDiscontinuousStepFp32(g);
        EXPECT_TRUE(g.desc(g.nodes[0].inputs[0]).storeFp32) << "the operand of a Cast to ONNX dtype " << to << " must carry full precision";
        EXPECT_TRUE(g.desc(g.nodes[0].outputs[0]).storeFp32) << "the truncated integer must survive storage, not round again";
    }
}

// A float target is a same-precision copy: nothing truncates, so nothing needs pinning and the
// session keeps the cheaper fp16 store.
TEST(Fp32PinDiscontinuousStep, FloatTargetIsLeftAtSessionPrecision) {
    Graph g = buildCastGraph(kOnnxFloat, /*withMovementHop=*/false);
    pinDiscontinuousStepFp32(g);
    EXPECT_FALSE(g.desc(g.nodes[0].inputs[0]).storeFp32);
    EXPECT_FALSE(g.desc(g.nodes[0].outputs[0]).storeFp32);
}

// The value reaching the Cast is what matters, so the pin follows pure movement producers back
// toward the boundary -- a rounding one hop upstream is the same full-unit error.
TEST(Fp32PinDiscontinuousStep, PinWalksBackThroughAMovementProducer) {
    Graph g = buildCastGraph(kOnnxInt32, /*withMovementHop=*/true);
    pinDiscontinuousStepFp32(g);
    const Node &movement = g.nodes[0];
    const Node &cast     = g.nodes[1];
    EXPECT_TRUE(g.desc(cast.inputs[0]).storeFp32) << "the movement output feeding the Cast";
    EXPECT_TRUE(g.desc(movement.inputs[0]).storeFp32) << "and the value before the movement hop";
}

TEST(Fp32PinDiscontinuousStep, PassIsIdempotent) {
    Graph g = buildCastGraph(kOnnxInt32, /*withMovementHop=*/true);
    pinDiscontinuousStepFp32(g);
    const bool inPinned  = g.desc(g.nodes[1].inputs[0]).storeFp32;
    const bool outPinned = g.desc(g.nodes[1].outputs[0]).storeFp32;
    pinDiscontinuousStepFp32(g);
    EXPECT_EQ(g.desc(g.nodes[1].inputs[0]).storeFp32, inPinned);
    EXPECT_EQ(g.desc(g.nodes[1].outputs[0]).storeFp32, outPinned);
}

// foldIntRoundtripCast collapses Cast(float -> wide int) -> Cast(wide int -> float) into one
// Unary(Trunc), so the graph that reaches the session carries the truncation with NO Cast node in
// it. Pinning only Cast would leave exactly the folded form — the common one — unprotected.
TEST(Fp32PinDiscontinuousStep, SteppingUnariesArePinnedToo) {
    for (UnaryType step: {UnaryType::Trunc, UnaryType::Floor, UnaryType::Ceil, UnaryType::Round, UnaryType::Sign})
    {
        Graph g = buildUnaryGraph(step);
        pinDiscontinuousStepFp32(g);
        EXPECT_TRUE(g.desc(g.nodes[0].inputs[0]).storeFp32) << "operand of Unary subOp " << (int) step << " must carry full precision";
        EXPECT_TRUE(g.desc(g.nodes[0].outputs[0]).storeFp32) << "result of Unary subOp " << (int) step << " must not round again";
    }
}

// A smooth unary carries a storage rounding proportionally, so it keeps the cheaper fp16 store.
TEST(Fp32PinDiscontinuousStep, SmoothUnariesAreLeftAtSessionPrecision) {
    for (UnaryType smooth: {UnaryType::Sigmoid, UnaryType::Tanh, UnaryType::Exp, UnaryType::Sqrt, UnaryType::Abs})
    {
        Graph g = buildUnaryGraph(smooth);
        pinDiscontinuousStepFp32(g);
        EXPECT_FALSE(g.desc(g.nodes[0].inputs[0]).storeFp32) << "Unary subOp " << (int) smooth << " does not need the pin";
        EXPECT_FALSE(g.desc(g.nodes[0].outputs[0]).storeFp32);
    }
}
