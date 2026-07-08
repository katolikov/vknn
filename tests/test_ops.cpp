// vknn operator unit tests (host, CPU backend). One self-contained graph per op, checked against a
// reference (hand-computed, or an onnxruntime golden noted inline). The CPU op is the correctness
// oracle the Vulkan path is diffed against on-device, so a regression here surfaces before the GPU.
//
// ConvTranspose covers the explicit-pad, auto_pad SAME, and output_shape paths -- SAME and
// output_shape yield the same output size here but different values, so a regression in either
// attribute is caught.
#include "core/matmul_tile.h"
#include "import/dim_expr.h"
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    Attr ints(std::vector<int64_t> v) {
        Attr a;
        a.kind = Attr::Ints;
        a.ints = std::move(v);
        return a;
    }
    Attr str(std::string s) {
        Attr a;
        a.kind = Attr::String;
        a.str  = std::move(s);
        return a;
    }

    struct Init {
        std::vector<int64_t> shape;
        std::vector<float>   data;
    };
    struct OpOut {
        std::vector<float>   data;
        std::vector<int64_t> shape;
    };

    // Build a single-op graph: one float input "x" + N constant initializers, run on CPU, return
    // the output values and shape. subOp carries the UnaryType/BinaryType code for kUnary/kBinary.
    OpOut runOp(OpType type, int subOp, const Attributes &attr, const std::vector<int64_t> &xshape, const std::vector<float> &xdata, const std::vector<Init> &inits) {
        Graph      g;
        TensorDesc xi;
        xi.name                 = "x";
        xi.shape                = xshape;
        xi.isInput              = true;
        TensorId              x = g.addTensor(xi);
        std::vector<TensorId> ids {x};
        g.inputs.push_back(x);
        for (size_t k = 0; k < inits.size(); ++k)
        {
            TensorDesc d;
            d.name          = "i" + std::to_string(k);
            d.shape         = inits[k].shape;
            d.isInitializer = true;
            TensorId   id   = g.addTensor(d);
            HostBuffer hb;
            hb.resizeElems(inits[k].data.size(), DType::Float32);
            for (size_t i = 0; i < inits[k].data.size(); ++i)
            {
                hb.f32()[i] = inits[k].data[i];
            }
            g.initializers[id] = hb;
            ids.push_back(id);
        }
        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     n;
        n.type    = type;
        n.name    = "op";
        n.subOp   = subOp;
        n.inputs  = ids;
        n.outputs = {y};
        n.attr    = attr;
        g.nodes.push_back(n);
        g.outputs = {y};

        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = "x";
        in.shape = xshape;
        in.dtype = DType::Float32;
        in.data.resize(xdata.size() * 4);
        for (size_t i = 0; i < xdata.size(); ++i)
        {
            reinterpret_cast<float *>(in.data.data())[i] = xdata[i];
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        EXPECT_FALSE(outs.empty());
        if (outs.empty())
        {
            return {};
        }
        const float *o = outs[0].f32();
        return {std::vector<float>(o, o + numElements(outs[0].shape)), outs[0].shape};
    }

    void expectNear(const std::vector<float> &got, const std::vector<float> &ref, float tol = 1e-4f) {
        ASSERT_EQ(got.size(), ref.size());
        for (size_t i = 0; i < ref.size(); ++i)
        {
            EXPECT_NEAR(got[i], ref[i], tol) << "i=" << i;
        }
    }

    // Build a FusedPointwise graph: primary input "x" + N constant operand tensors, run the
    // pw_steps/pw_params unit on CPU, return the output values and shape.
    //
    // `steps` is 8 ints per step [kind, code, srcA, srcB, srcC, dst, bcast, bcastSrc] and `params`
    // is 2 floats per step [p0, p1] (see src/backend/cpu/ops/fused_pointwise.cpp). Sources reference
    // the accumulator (kPwRefAcc), the entry value (kPwRefEntry), a register (kPwRefReg0 - r), or an
    // operand tensor (kPwRefOp0 - i, i indexing node.inputs where the primary sits at 0).
    OpOut runFusedPw(const std::vector<int64_t> &xshape, const std::vector<float> &xdata, const std::vector<Init> &operands, const std::vector<int64_t> &steps, const std::vector<float> &params) {
        Graph      g;
        TensorDesc xi;
        xi.name                 = "x";
        xi.shape                = xshape;
        xi.isInput              = true;
        TensorId              x = g.addTensor(xi);
        g.inputs.push_back(x);
        std::vector<TensorId> ids {x};
        for (size_t k = 0; k < operands.size(); ++k)
        {
            TensorDesc d;
            d.name          = "o" + std::to_string(k);
            d.shape         = operands[k].shape;
            d.isInitializer = true;
            TensorId   id   = g.addTensor(d);
            HostBuffer hb;
            hb.resizeElems(operands[k].data.size(), DType::Float32);
            for (size_t i = 0; i < operands[k].data.size(); ++i)
            {
                hb.f32()[i] = operands[k].data[i];
            }
            g.initializers[id] = hb;
            ids.push_back(id);
        }
        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     n;
        n.type    = OpType::FusedPointwise;
        n.name    = "pw";
        n.inputs  = ids;
        n.outputs = {y};
        {
            Attr a;
            a.kind                 = Attr::Ints;
            a.ints                 = steps;
            n.attr.map["pw_steps"] = a;
        }
        {
            Attr a;
            a.kind                  = Attr::Floats;
            a.floats                = params;
            n.attr.map["pw_params"] = a;
        }
        {
            Attr a;
            a.kind                = Attr::Int;
            a.i                   = 1;
            n.attr.map["pw_flat"] = a;
        }
        g.nodes.push_back(n);
        g.outputs = {y};

        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = "x";
        in.shape = xshape;
        in.dtype = DType::Float32;
        in.data.resize(xdata.size() * 4);
        for (size_t i = 0; i < xdata.size(); ++i)
        {
            reinterpret_cast<float *>(in.data.data())[i] = xdata[i];
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        EXPECT_FALSE(outs.empty());
        if (outs.empty())
        {
            return {};
        }
        const float *o = outs[0].f32();
        return {std::vector<float>(o, o + numElements(outs[0].shape)), outs[0].shape};
    }

} // namespace

// --- Conv (1x1, identity*2 weight + bias) feeding Relu: a two-op chain. ---
TEST(Ops, Conv1x1Relu) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 2, 1, 1};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorDesc wi;
    wi.name          = "w";
    wi.shape         = {2, 2, 1, 1};
    wi.isInitializer = true;
    TensorId   w     = g.addTensor(wi);
    HostBuffer wb;
    wb.resizeElems(4, DType::Float32);
    wb.f32()[0]       = 2;
    wb.f32()[1]       = 0;
    wb.f32()[2]       = 0;
    wb.f32()[3]       = 2;
    g.initializers[w] = wb;
    TensorDesc bi;
    bi.name          = "b";
    bi.shape         = {2};
    bi.isInitializer = true;
    TensorId   b     = g.addTensor(bi);
    HostBuffer bb;
    bb.resizeElems(2, DType::Float32);
    bb.f32()[0]       = -3;
    bb.f32()[1]       = 0;
    g.initializers[b] = bb;
    TensorDesc yo;
    yo.name    = "y";
    TensorId y = g.addTensor(yo);
    Node     conv;
    conv.type    = OpType::Conv;
    conv.name    = "conv";
    conv.inputs  = {x, w, b};
    conv.outputs = {y};
    g.nodes.push_back(conv);
    TensorDesc y2o;
    y2o.name     = "y2";
    y2o.isOutput = true;
    TensorId y2  = g.addTensor(y2o);
    Node     relu;
    relu.type    = OpType::Relu;
    relu.name    = "relu";
    relu.inputs  = {y};
    relu.outputs = {y2};
    g.nodes.push_back(relu);
    g.outputs = {y2};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);
    IOTensor in;
    in.name  = "x";
    in.shape = {1, 2, 1, 1};
    in.dtype = DType::Float32;
    in.data.resize(2 * 4);
    reinterpret_cast<float *>(in.data.data())[0] = 1.0f; // 2*1-3 = -1 -> relu 0
    reinterpret_cast<float *>(in.data.data())[1] = 5.0f; // 2*5+0 = 10 -> relu 10
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_FALSE(outs.empty());
    EXPECT_NEAR(outs[0].f32()[0], 0.0f, 1e-5);
    EXPECT_NEAR(outs[0].f32()[1], 10.0f, 1e-5);
}

// --- Conv auto_pad SAME_UPPER, 3x3 stride 1, all-ones weight: a 3x3 box filter over the
// zero-padded 5x5 input 1..25. out = ceil(5/1) = 5; total_pad = (5-1)*1 + 3 - 5 = 2, split 1/1,
// so each output is the sum of the input's in-bounds 3x3 neighborhood (e.g. out[0][0] =
// 1+2+6+7 = 16, out[2][2] = (7+8+9)+(12+13+14)+(17+18+19) = 117). ---
TEST(Ops, ConvAutoPadSameUpper) {
    Attributes attr;
    attr.map["auto_pad"] = str("SAME_UPPER");
    std::vector<float> in(25);
    for (int i = 0; i < 25; ++i)
    {
        in[i] = (float) (i + 1);
    }
    auto out = runOp(OpType::Conv, 0, attr, {1, 1, 5, 5}, in, {{{1, 1, 3, 3}, std::vector<float>(9, 1.f)}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 1, 5, 5}));
    expectNear(out.data, {16, 27, 33, 39, 28, 39, 63, 72, 81, 57, 69, 108, 117, 126, 87, 99, 153, 162, 171, 117, 76, 117, 123, 129, 88});
}

// --- Conv auto_pad SAME_UPPER, 3x3 stride 2, 4x4 input 1..16: the asymmetric-split case. out =
// ceil(4/2) = 2; total_pad = (2-1)*2 + 3 - 4 = 1, SAME_UPPER puts the odd unit at the END
// (pads = {0,0,1,1}), so window origins are {0,2} and only the bottom/right windows are cropped:
// out[0][0] = rows 0-2 x cols 0-2 = 54, out[1][1] = rows 2-3 x cols 2-3 = 11+12+15+16 = 54. ---
TEST(Ops, ConvAutoPadSameUpperStride2) {
    Attributes attr;
    attr.map["auto_pad"] = str("SAME_UPPER");
    attr.map["strides"]  = ints({2, 2});
    std::vector<float> in(16);
    for (int i = 0; i < 16; ++i)
    {
        in[i] = (float) (i + 1);
    }
    auto out = runOp(OpType::Conv, 0, attr, {1, 1, 4, 4}, in, {{{1, 1, 3, 3}, std::vector<float>(9, 1.f)}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 1, 2, 2}));
    expectNear(out.data, {54, 45, 72, 54});
}

// --- Conv auto_pad SAME_LOWER, 3x3 stride 2, same 4x4 input: the odd pad unit moves to the BEGIN
// (pads = {1,1,0,0}), window origins {-1,1}, so the top/left windows are cropped instead:
// out[0][0] = rows 0-1 x cols 0-1 = 1+2+5+6 = 14 (vs SAME_UPPER's 54). ---
TEST(Ops, ConvAutoPadSameLowerStride2) {
    Attributes attr;
    attr.map["auto_pad"] = str("SAME_LOWER");
    attr.map["strides"]  = ints({2, 2});
    std::vector<float> in(16);
    for (int i = 0; i < 16; ++i)
    {
        in[i] = (float) (i + 1);
    }
    auto out = runOp(OpType::Conv, 0, attr, {1, 1, 4, 4}, in, {{{1, 1, 3, 3}, std::vector<float>(9, 1.f)}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 1, 2, 2}));
    expectNear(out.data, {14, 30, 57, 99});
}

// --- Conv auto_pad VALID: zero padding regardless of an accompanying pads attr (auto_pad wins
// when it is not NOTSET). 5x5 input 1..25, ones 3x3: out[r][c] = 45r + 9c + 63. ---
TEST(Ops, ConvAutoPadValid) {
    Attributes attr;
    attr.map["auto_pad"] = str("VALID");
    attr.map["pads"]     = ints({1, 1, 1, 1});
    std::vector<float> in(25);
    for (int i = 0; i < 25; ++i)
    {
        in[i] = (float) (i + 1);
    }
    auto out = runOp(OpType::Conv, 0, attr, {1, 1, 5, 5}, in, {{{1, 1, 3, 3}, std::vector<float>(9, 1.f)}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 1, 3, 3}));
    expectNear(out.data, {63, 72, 81, 108, 117, 126, 153, 162, 171});
}

// --- Grouped Conv (1 < group < Cin), 1x1: group=2, Cin=4, Cout=4 -> 2 in / 2 out channels per
// group. Output channel oc in group g = oc/2 reads only input channels [2g, 2g+2). Run through the
// full Session pipeline: lowerGroupedConv rewrites it to two group-1 Convs over channel slices joined
// by a Concat, so this exercises the lowering AND that the lowered graph reproduces grouped semantics
// bit-for-bit (the GPU parity path decomposes the same way). ---
TEST(Ops, GroupedConv1x1) {
    Attr grp;
    grp.kind = Attr::Int;
    grp.i    = 2;
    Attributes attr;
    attr.map["group"] = grp;
    // Input [1,4,1,2]: two spatial positions, channels 0..3.
    // ch0 = {1, 2}, ch1 = {3, 4}, ch2 = {5, 6}, ch3 = {7, 8}
    std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8};
    // Weight [4,2,1,1] (Cout=4, inCg=2). oc0=[1,0] oc1=[0,1] (group0 over ch0,ch1);
    // oc2=[1,1] oc3=[2,0] (group1 over ch2,ch3).
    std::vector<float> w = {1, 0, 0, 1, 1, 1, 2, 0};
    auto out = runOp(OpType::Conv, 0, attr, {1, 4, 1, 2}, in, {{{4, 2, 1, 1}, w}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 4, 1, 2}));
    // oc0 = ch0            = {1, 2}
    // oc1 = ch1            = {3, 4}
    // oc2 = ch2 + ch3      = {12, 14}
    // oc3 = 2*ch2          = {10, 12}
    expectNear(out.data, {1, 2, 3, 4, 12, 14, 10, 12});
}

// --- Grouped Conv with a per-group input count that is NOT a multiple of 4: group=2, Cin=6, Cout=2
// -> 3 in-channels per group. The GPU parity lowering slices channels [0,3) and [3,6); the second
// slice starts mid-vec4 (the NC4 edge), which the Slice+Conv decomposition handles cleanly. ---
TEST(Ops, GroupedConvNonMultipleOf4) {
    Attr grp;
    grp.kind = Attr::Int;
    grp.i    = 2;
    Attributes attr;
    attr.map["group"] = grp;
    // Input [1,6,1,1]: one spatial position, channels 0..5 = {1,2,3,4,5,6}.
    std::vector<float> in = {1, 2, 3, 4, 5, 6};
    // Weight [2,3,1,1]: oc0 sums group0 (ch0,1,2) with {1,1,1}; oc1 sums group1 (ch3,4,5) with {1,1,1}.
    std::vector<float> w = {1, 1, 1, 1, 1, 1};
    auto out = runOp(OpType::Conv, 0, attr, {1, 6, 1, 1}, in, {{{2, 3, 1, 1}, w}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 2, 1, 1}));
    // oc0 = ch0+ch1+ch2 = 6 ; oc1 = ch3+ch4+ch5 = 15
    expectNear(out.data, {6, 15});
}

// --- Grouped 3x3 Conv (spatial + grouping): group=2, Cin=2, Cout=2, 3x3 SAME, per-group depthwise-
// like (1 in / 1 out channel per group), a box filter per channel. Verifies the lowered group-1
// Convs carry the strides/pads/dilations attributes through to each part. ---
TEST(Ops, GroupedConv3x3) {
    Attr grp;
    grp.kind = Attr::Int;
    grp.i    = 2;
    Attributes attr;
    attr.map["group"]    = grp;
    attr.map["pads"]     = ints({1, 1, 1, 1});
    attr.map["kernel_shape"] = ints({3, 3});
    // Input [1,2,2,2]: ch0 = 1..4, ch1 = 10,20,30,40.
    std::vector<float> in = {1, 2, 3, 4, 10, 20, 30, 40};
    // Weight [2,1,3,3]: both output channels are an all-ones 3x3 box over their own group's 1 channel.
    std::vector<float> w(2 * 1 * 3 * 3, 1.f);
    auto out = runOp(OpType::Conv, 0, attr, {1, 2, 2, 2}, in, {{{2, 1, 3, 3}, w}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 2, 2, 2}));
    // Each output is the sum over the 3x3 neighbourhood (SAME pad) of its own channel.
    // ch0 sums: [1+2+3+4]=10 at every position (all four pixels fall in every window here).
    // ch1 sums: [10+20+30+40]=100 likewise.
    expectNear(out.data, {10, 10, 10, 10, 100, 100, 100, 100});
}

// --- MaxPool auto_pad SAME_UPPER, 2x2 stride 2 over 5x5 input 1..25: out = ceil(5/2) = 3,
// total_pad = (3-1)*2 + 2 - 5 = 1 at the end, so the last row/col windows are half-size
// (out[0][2] = max(5,10) = 10, out[2][2] = max(25) = 25). ---
TEST(Ops, MaxPoolAutoPadSameUpper) {
    Attributes attr;
    attr.map["auto_pad"]     = str("SAME_UPPER");
    attr.map["kernel_shape"] = ints({2, 2});
    attr.map["strides"]      = ints({2, 2});
    std::vector<float> in(25);
    for (int i = 0; i < 25; ++i)
    {
        in[i] = (float) (i + 1);
    }
    auto out = runOp(OpType::MaxPool, 0, attr, {1, 1, 5, 5}, in, {});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 1, 3, 3}));
    expectNear(out.data, {7, 9, 10, 17, 19, 20, 22, 24, 25});
}

// --- AveragePool auto_pad SAME_LOWER, 2x2 stride 2 over 5x5 input 1..25: the odd pad unit is at
// the begin, window origins {-1,1,3}; the default count_include_pad=0 divides by the in-bounds
// count only (out[0][0] = avg{1} = 1, out[1][0] = avg{6,11} = 8.5, out[1][1] = avg{7,8,12,13} = 10). ---
TEST(Ops, AvgPoolAutoPadSameLower) {
    Attributes attr;
    attr.map["auto_pad"]     = str("SAME_LOWER");
    attr.map["kernel_shape"] = ints({2, 2});
    attr.map["strides"]      = ints({2, 2});
    std::vector<float> in(25);
    for (int i = 0; i < 25; ++i)
    {
        in[i] = (float) (i + 1);
    }
    auto out = runOp(OpType::AvgPool, 0, attr, {1, 1, 5, 5}, in, {});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 1, 3, 3}));
    expectNear(out.data, {1.f, 2.5f, 4.5f, 8.5f, 10.f, 12.f, 18.5f, 20.f, 22.f});
}

// --- Unary family: Sigmoid + HardSwish. ---
TEST(Ops, UnarySigmoidHardSwish) {
    float vals[4] = {-2.f, -0.5f, 0.5f, 3.f};
    for (int sub: {(int) UnaryType::Sigmoid, (int) UnaryType::HardSwish})
    {
        auto               out = runOp(OpType::Unary, sub, {}, {1, 4}, {vals[0], vals[1], vals[2], vals[3]}, {});
        std::vector<float> ref(4);
        for (int i = 0; i < 4; ++i)
        {
            ref[i] = sub == (int) UnaryType::Sigmoid ? 1.f / (1.f + std::exp(-vals[i])) : vals[i] * std::min(std::max(vals[i] + 3.f, 0.f), 6.f) / 6.f;
        }
        expectNear(out.data, ref, 1e-5f);
    }
}

// --- Unary Round: nearest integer with ties to even (ONNX Round == GLSL roundEven), so the CPU
// oracle and the GPU agree bitwise on exact halves. -0.5 rounds to negative zero. ---
TEST(Ops, UnaryRoundHalfToEven) {
    EXPECT_EQ(unaryFromOnnx("Round"), UnaryType::Round);
    std::vector<float> vals {-2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, -1.4f, 1.4f, -2.6f, 2.6f};
    auto               out = runOp(OpType::Unary, (int) UnaryType::Round, {}, {1, 10}, vals, {});
    expectNear(out.data, {-2, -2, 0, 0, 2, 2, -1, 1, -3, 3}, 0.f);
    ASSERT_EQ(out.data.size(), 10u);
    EXPECT_TRUE(std::signbit(out.data[2])); // -0.5 -> -0.0, matching roundEven's zero sign
}

// --- Binary Mul with a broadcast scalar. ---
TEST(Ops, BinaryMulBroadcast) {
    auto out = runOp(OpType::Binary, (int) BinaryType::Mul, {}, {1, 3}, {1, 2, 3}, {{{1}, {3.f}}});
    expectNear(out.data, {3, 6, 9}, 1e-5f);
}

// --- Add with a per-channel broadcast bias. ---
TEST(Ops, AddBroadcast) {
    auto out = runOp(OpType::Add, 0, {}, {1, 3}, {0, 1, 2}, {{{1, 3}, {10, 20, 30}}});
    expectNear(out.data, {10, 21, 32}, 1e-5f);
}

// --- RMSNorm over the last axis: y = x*rsqrt(mean(x^2)+eps)*gamma, no mean subtraction, no bias.
// The CPU oracle the fused Vulkan kernel is diffed against; reference computed here in double. ---
TEST(Ops, RMSNormLastAxis) {
    Attributes attr;
    Attr       eps;
    eps.kind            = Attr::Float;
    eps.f               = 1e-6f;
    attr.map["epsilon"] = eps;
    std::vector<float> x {1.f, -2.f, 3.f, -4.f, 0.5f, 1.5f, -2.5f, 4.f};
    std::vector<float> gamma {2.f, 0.5f, 1.f, -1.f};
    auto               out = runOp(OpType::RMSNorm, 0, attr, {2, 4}, x, {{{4}, gamma}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {2, 4}));
    std::vector<float> ref(8);
    for (int r = 0; r < 2; ++r)
    {
        double sq = 0.0;
        for (int j = 0; j < 4; ++j)
        {
            sq += (double) x[r * 4 + j] * (double) x[r * 4 + j];
        }
        double inv = 1.0 / std::sqrt(sq / 4.0 + 1e-6);
        for (int j = 0; j < 4; ++j)
        {
            ref[r * 4 + j] = (float) ((double) x[r * 4 + j] * inv * (double) gamma[j]);
        }
    }
    expectNear(out.data, ref, 1e-5f);
}

// --- lowerRMSNorm fuses the decomposed Pow/ReduceMean/Add/Sqrt/Reciprocal/Mul/Mul chain into one
// RMSNorm node, and the fused graph reproduces the decomposition's math. ---
TEST(Ops, LowerRMSNormFusesChain) {
    Graph g;
    auto  addInit = [&](const std::string &nm, std::vector<int64_t> shape, std::vector<float> data) {
        TensorDesc d;
        d.name          = nm;
        d.shape         = std::move(shape);
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems(data.size(), DType::Float32);
        for (size_t i = 0; i < data.size(); ++i)
        {
            hb.f32()[i] = data[i];
        }
        g.initializers[id] = hb;
        return id;
    };
    TensorDesc xi;
    xi.name          = "x";
    xi.shape         = {2, 4};
    xi.isInput       = true;
    TensorId x       = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorId           two   = addInit("two", {1}, {2.f});
    TensorId           eps   = addInit("eps", {1}, {1e-5f});
    std::vector<float> gammaData {2.f, 0.5f, 1.f, -1.f};
    TensorId           gamma = addInit("gamma", {4}, gammaData);
    auto               tmp   = [&](const std::string &nm) {
        TensorDesc d;
        d.name = nm;
        return g.addTensor(d);
    };
    TensorId   p = tmp("p"), m = tmp("m"), a = tmp("a"), s = tmp("s"), r = tmp("r"), nrm = tmp("nrm");
    TensorDesc yo;
    yo.name     = "y";
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    auto     node = [&](OpType t, int sub, std::vector<TensorId> in, std::vector<TensorId> out, const std::string &nm) {
        Node n;
        n.type    = t;
        n.subOp   = sub;
        n.inputs  = std::move(in);
        n.outputs = std::move(out);
        n.name    = nm;
        return n;
    };
    Node rmean = node(OpType::Reduce, (int) ReduceType::Mean, {p}, {m}, "rmean");
    rmean.attr.map["axes"] = ints({-1});
    Attr kd;
    kd.kind                    = Attr::Int;
    kd.i                       = 1;
    rmean.attr.map["keepdims"] = kd;
    g.nodes = {node(OpType::Binary, (int) BinaryType::Pow, {x, two}, {p}, "pow"),
               rmean,
               node(OpType::Add, 0, {m, eps}, {a}, "add"),
               node(OpType::Unary, (int) UnaryType::Sqrt, {a}, {s}, "sqrt"),
               node(OpType::Unary, (int) UnaryType::Reciprocal, {s}, {r}, "recip"),
               node(OpType::Binary, (int) BinaryType::Mul, {x, r}, {nrm}, "mulN"),
               node(OpType::Binary, (int) BinaryType::Mul, {gamma, nrm}, {y}, "mulG")};
    g.outputs = {y};

    runStandardPasses(g);

    int rms = 0, pows = 0, reduces = 0, sqrts = 0, recips = 0;
    for (const Node &n: g.nodes)
    {
        if (n.type == OpType::RMSNorm)
        {
            rms++;
            EXPECT_EQ(g.desc(n.inputs[0]).name, "x");
            EXPECT_EQ(g.desc(n.inputs[1]).name, "gamma");
            EXPECT_NEAR(n.attr.getf("epsilon", -1.f), 1e-5f, 1e-9f);
        } else if (n.type == OpType::Binary && n.subOp == (int) BinaryType::Pow)
        {
            pows++;
        } else if (n.type == OpType::Reduce)
        {
            reduces++;
        } else if (n.type == OpType::Unary && n.subOp == (int) UnaryType::Sqrt)
        {
            sqrts++;
        } else if (n.type == OpType::Unary && n.subOp == (int) UnaryType::Reciprocal)
        {
            recips++;
        }
    }
    EXPECT_EQ(rms, 1);
    EXPECT_EQ(pows, 0);
    EXPECT_EQ(reduces, 0);
    EXPECT_EQ(sqrts, 0);
    EXPECT_EQ(recips, 0);

    // Run the fused graph on the CPU oracle and compare to the decomposition's hand-computed output.
    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);
    std::vector<float> xd {1.f, -2.f, 3.f, -4.f, 0.5f, 1.5f, -2.5f, 4.f};
    IOTensor           in;
    in.name  = "x";
    in.shape = {2, 4};
    in.dtype = DType::Float32;
    in.data.resize(xd.size() * 4);
    std::memcpy(in.data.data(), xd.data(), xd.size() * 4);
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_FALSE(outs.empty());
    std::vector<float> ref(8);
    for (int rw = 0; rw < 2; ++rw)
    {
        double sq = 0.0;
        for (int j = 0; j < 4; ++j)
        {
            sq += (double) xd[rw * 4 + j] * (double) xd[rw * 4 + j];
        }
        double inv = 1.0 / std::sqrt(sq / 4.0 + 1e-5);
        for (int j = 0; j < 4; ++j)
        {
            ref[rw * 4 + j] = (float) ((double) xd[rw * 4 + j] * inv * (double) gammaData[j]);
        }
    }
    std::vector<float> got(outs[0].f32(), outs[0].f32() + 8);
    expectNear(got, ref, 1e-5f);
}

// --- GlobalAveragePool over HxW. ---
TEST(Ops, GlobalAveragePool) {
    auto out = runOp(OpType::GlobalAvgPool, 0, {}, {1, 1, 2, 2}, {1, 2, 3, 4}, {});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 1, 1, 1}));
    expectNear(out.data, {2.5f}, 1e-5f);
}

// --- 1-D Conv (rank-3 input/weight, 1-length strides, 2-length pads): normalized at import to the
// canonical 2-D geometry (kH=k, kW=1) with rank-3 output. Values from onnxruntime. ---
TEST(Ops, Conv1dNormalizedGeometry) {
    Attributes attr;
    attr.map["pads"]    = ints({1, 1});
    attr.map["strides"] = ints({2});
    std::vector<float> x(16);
    for (int i = 0; i < 16; ++i)
    {
        x[i] = (float) i;
    }
    auto out = runOp(OpType::Conv, 0, attr, {1, 2, 8}, x,
                     {{{3, 2, 3}, {1, 0, -1, 2, 1, 0, 0, 1, 0, 1, 1, 1, -1, 2, -1, 0, 0, 3}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 3, 4}));
    expectNear(out.data, {7, 26, 32, 38, 17, 32, 40, 48, 26, 33, 39, 45}, 1e-4f);
}

// --- Reduce with a caller-bound runtime shape that differs from the compiled desc. CPU ops derive
// geometry from runtime shapes (the dynamic-shape contract); Reduce must size its accumulator bins
// and output from the bound shape, or a divergent shape indexes past the desc-sized accumulator. ---
TEST(Ops, ReduceGeometryFollowsRuntimeShape) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {2, 3, 4};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorDesc yo;
    yo.name     = "y";
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     n;
    n.type             = OpType::Reduce;
    n.subOp            = (int32_t) ReduceType::Mean;
    n.name             = "rm";
    n.inputs           = {x};
    n.outputs          = {y};
    n.attr.map["axes"] = ints({-1});
    {
        Attr a;
        a.kind                 = Attr::Int;
        a.i                    = 0;
        n.attr.map["keepdims"] = a;
    }
    g.nodes.push_back(n);
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg); // static desc for y: [2,3]
    ASSERT_TRUE(sess);

    // Bind the same 24 values as [4,3,2]: the reduced (last) axis extent becomes 2.
    IOTensor in;
    in.name  = "x";
    in.shape = {4, 3, 2};
    in.dtype = DType::Float32;
    in.data.resize(24 * sizeof(float));
    for (int i = 0; i < 24; ++i)
    {
        reinterpret_cast<float *>(in.data.data())[i] = (float) i;
    }
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_EQ(outs[0].shape, (std::vector<int64_t> {4, 3}));
    const float *o = outs[0].f32();
    for (int i = 0; i < 12; ++i)
    {
        EXPECT_FLOAT_EQ(o[i], 2.f * i + 0.5f) << "bin " << i; // mean of {2i, 2i+1}
    }
}

// --- Split with a caller-bound runtime shape that differs from the compiled desc: segment sizes
// derive from the runtime axis extent (equal split, no `split` param), not the static desc. ---
TEST(Ops, SplitGeometryFollowsRuntimeShape) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {2, 6};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorDesc ad, bd;
    ad.name     = "a";
    ad.isOutput = true;
    bd.name     = "b";
    bd.isOutput = true;
    TensorId a = g.addTensor(ad), b = g.addTensor(bd);
    Node     n;
    n.type             = OpType::Split;
    n.name             = "sp";
    n.inputs           = {x};
    n.outputs          = {a, b};
    n.attr.map["axis"] = [] { Attr t; t.kind = Attr::Int; t.i = 1; return t; }();
    g.nodes.push_back(n);
    g.outputs = {a, b};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg); // static descs: [2,3] each
    ASSERT_TRUE(sess);

    // Bind 16 values as [2,8]: each half of the runtime axis has extent 4.
    IOTensor in;
    in.name  = "x";
    in.shape = {2, 8};
    in.dtype = DType::Float32;
    in.data.resize(16 * sizeof(float));
    for (int i = 0; i < 16; ++i)
    {
        reinterpret_cast<float *>(in.data.data())[i] = (float) i;
    }
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_EQ(outs.size(), 2u);
    ASSERT_EQ(outs[0].shape, (std::vector<int64_t> {2, 4}));
    ASSERT_EQ(outs[1].shape, (std::vector<int64_t> {2, 4}));
    const float *oa = outs[0].f32(), *ob = outs[1].f32();
    const float  ea[8] = {0, 1, 2, 3, 8, 9, 10, 11}, eb[8] = {4, 5, 6, 7, 12, 13, 14, 15};
    for (int i = 0; i < 8; ++i)
    {
        EXPECT_FLOAT_EQ(oa[i], ea[i]) << "a[" << i << "]";
        EXPECT_FLOAT_EQ(ob[i], eb[i]) << "b[" << i << "]";
    }
}

// --- Pad under an unresolved input desc (the session adopts the caller-bound runtime shape): the
// output shape derives from the runtime input shape plus the pads parameter, not the static desc. ---
TEST(Ops, PadGeometryFollowsRuntimeShape) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {}; // unresolved: the caller-bound runtime shape is authoritative
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorDesc yo;
    yo.name     = "y";
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     n;
    n.type             = OpType::Pad;
    n.name             = "pd";
    n.inputs           = {x};
    n.outputs          = {y};
    n.attr.map["pads"] = ints({0, 1, 0, 1}); // one column of zeros on each side of the last axis
    g.nodes.push_back(n);
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg); // y stays unresolved with x
    ASSERT_TRUE(sess);

    // Bind 4 values as [2,2]: the padded runtime output is [2,4].
    IOTensor in;
    in.name  = "x";
    in.shape = {2, 2};
    in.dtype = DType::Float32;
    in.data.resize(4 * sizeof(float));
    for (int i = 0; i < 4; ++i)
    {
        reinterpret_cast<float *>(in.data.data())[i] = (float) (i + 1);
    }
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_EQ(outs[0].shape, (std::vector<int64_t> {2, 4}));
    expectNear({outs[0].f32(), outs[0].f32() + 8}, {0, 1, 2, 0, 0, 3, 4, 0}, 1e-6f);
}

// --- Resize under an unresolved input desc (the session adopts the caller-bound runtime shape):
// the output H/W derive from the runtime input shape times the scales parameter, not the static
// desc. ---
TEST(Ops, ResizeGeometryFollowsRuntimeShape) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {}; // unresolved: the caller-bound runtime shape is authoritative
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
    hb.f32()[0] = 1.f;
    hb.f32()[1] = 1.f;
    hb.f32()[2] = 2.f;
    hb.f32()[3] = 2.f;
    g.initializers[s] = hb;
    TensorDesc yo;
    yo.name     = "y";
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     n;
    n.type             = OpType::Resize;
    n.name             = "rs";
    n.inputs           = {x, kNoTensor, s}; // X, roi (absent), scales
    n.outputs          = {y};
    n.attr.map["mode"] = str("nearest");
    g.nodes.push_back(n);
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg); // y stays unresolved with x
    ASSERT_TRUE(sess);

    // Bind 9 values as [1,1,3,3]: the 2x-scaled runtime output is [1,1,6,6].
    IOTensor in;
    in.name  = "x";
    in.shape = {1, 1, 3, 3};
    in.dtype = DType::Float32;
    in.data.resize(9 * sizeof(float));
    for (int i = 0; i < 9; ++i)
    {
        reinterpret_cast<float *>(in.data.data())[i] = (float) (i + 1);
    }
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_EQ(outs[0].shape, (std::vector<int64_t> {1, 1, 6, 6}));
    std::vector<float> want;
    for (int r: {0, 0, 1, 1, 2, 2})
    {
        for (int c: {0, 0, 1, 1, 2, 2})
        {
            want.push_back((float) (r * 3 + c + 1));
        }
    }
    expectNear({outs[0].f32(), outs[0].f32() + 36}, want, 1e-6f);
}

// --- MatMul A[2,3] @ B[3,2] (B constant). ---
TEST(Ops, MatMul) {
    auto out = runOp(OpType::MatMul, 0, {}, {2, 3}, {1, 2, 3, 4, 5, 6}, {{{3, 2}, {1, 0, 0, 1, 1, 1}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {2, 2}));
    expectNear(out.data, {4, 5, 10, 11}, 1e-4f);
}

// --- ConvTranspose stride 2, kernel 2, all-ones weight: each pixel tiles a 2x2 block. ---
TEST(Ops, ConvTransposeBasicStride2) {
    Attributes attr;
    attr.map["strides"] = ints({2, 2});
    auto out            = runOp(OpType::ConvTranspose, 0, attr, {1, 1, 2, 2}, {1, 2, 3, 4}, {{{1, 1, 2, 2}, {1, 1, 1, 1}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 1, 4, 4}));
    expectNear(out.data, {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4});
}

// --- ConvTranspose auto_pad SAME_UPPER, stride 2, kernel 3: output size is in*stride = 4 (not the
// explicit-pad formula's 5); pad_begin = 0. ORT golden. ---
TEST(Ops, ConvTransposeAutoPadSameUpper) {
    Attributes attr;
    attr.map["strides"]  = ints({2, 2});
    attr.map["auto_pad"] = str("SAME_UPPER");
    auto out             = runOp(OpType::ConvTranspose, 0, attr, {1, 1, 2, 2}, {1, 2, 3, 4}, {{{1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 1, 4, 4}));
    expectNear(out.data, {1, 2, 5, 4, 4, 5, 14, 10, 10, 14, 36, 24, 12, 15, 34, 20});
}

// --- ConvTranspose output_shape [4,4], stride 2, kernel 3: same size as SAME_UPPER but pad_begin = 1
// (default split puts the larger half at the begin), so the values differ. ORT golden. ---
TEST(Ops, ConvTransposeOutputShape) {
    Attributes attr;
    attr.map["strides"]      = ints({2, 2});
    attr.map["output_shape"] = ints({4, 4});
    auto out                 = runOp(OpType::ConvTranspose, 0, attr, {1, 1, 2, 2}, {1, 2, 3, 4}, {{{1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 1, 4, 4}));
    expectNear(out.data, {5, 14, 10, 12, 14, 36, 24, 30, 15, 34, 20, 24, 24, 55, 32, 36});
}

// --- ConvTranspose with a per-output-channel bias (Cin=1, Cout=2). ORT golden. ---
TEST(Ops, ConvTransposeBias) {
    Attributes attr;
    attr.map["strides"] = ints({2, 2});
    // weight [1,2,2,2]: channel 0 = [[1,0],[0,1]], channel 1 = [[1,1],[1,1]]; bias [2] = {10, -5}.
    auto out = runOp(OpType::ConvTranspose, 0, attr, {1, 1, 2, 2}, {1, 2, 3, 4}, {{{1, 2, 2, 2}, {1, 0, 0, 1, 1, 1, 1, 1}}, {{2}, {10, -5}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 2, 4, 4}));
    expectNear(out.data, {11, 10, 12, 10, 10, 11, 10, 12, 13, 10, 14, 10, 10, 13, 10, 14, -4, -4, -3, -3, -4, -4, -3, -3, -2, -2, -1, -1, -2, -2, -1, -1});
}

// --- Greater vs a scalar: strict >, ties are 0. ---
TEST(Ops, GreaterScalar) {
    auto out = runOp(OpType::Greater, 0, {}, {2, 3}, {1, 2, 3, 4, 5, 6}, {{{1}, {3.f}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {2, 3}));
    EXPECT_EQ(out.data, (std::vector<float> {0, 0, 0, 1, 1, 1}));
}

// --- GreaterOrEqual vs a scalar: ties are 1 (the only difference from Greater on this input). ---
TEST(Ops, GreaterEqualScalarTies) {
    auto out = runOp(OpType::GreaterEqual, 0, {}, {2, 3}, {1, 2, 3, 4, 5, 6}, {{{1}, {3.f}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {2, 3}));
    EXPECT_EQ(out.data, (std::vector<float> {0, 0, 1, 1, 1, 1}));
}

// --- Greater with NumPy broadcasting: [2,3] vs a [3] row. ---
TEST(Ops, GreaterBroadcastRow) {
    auto out = runOp(OpType::Greater, 0, {}, {2, 3}, {1, 5, 0, 3, 4, 2}, {{{3}, {2, 4, 1}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {2, 3}));
    // row0 {1,5,0} vs {2,4,1} -> {0,1,0}; row1 {3,4,2} vs {2,4,1} -> {1,0,1}
    EXPECT_EQ(out.data, (std::vector<float> {0, 1, 0, 1, 0, 1}));
}

// --- Greater with rank-4 broadcasting: [1,2,2,2] vs a per-channel [1,2,1,1] threshold. ---
TEST(Ops, GreaterBroadcastPerChannel) {
    auto out = runOp(OpType::Greater, 0, {}, {1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {{{1, 2, 1, 1}, {2.5f, 6.5f}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 2, 2, 2}));
    // ch0 {1,2,3,4} > 2.5 -> {0,0,1,1}; ch1 {5,6,7,8} > 6.5 -> {0,0,1,1}
    EXPECT_EQ(out.data, (std::vector<float> {0, 0, 1, 1, 0, 0, 1, 1}));
}

// --- Less vs a scalar: strict <, ties are 0. ---
TEST(Ops, LessScalar) {
    auto out = runOp(OpType::Less, 0, {}, {2, 3}, {1, 2, 3, 4, 5, 6}, {{{1}, {3.f}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {2, 3}));
    EXPECT_EQ(out.data, (std::vector<float> {1, 1, 0, 0, 0, 0}));
}

// --- LessOrEqual vs a scalar: ties are 1 (the only difference from Less on this input). ---
TEST(Ops, LessEqualScalarTies) {
    auto out = runOp(OpType::LessEqual, 0, {}, {2, 3}, {1, 2, 3, 4, 5, 6}, {{{1}, {3.f}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {2, 3}));
    EXPECT_EQ(out.data, (std::vector<float> {1, 1, 1, 0, 0, 0}));
}

// --- Less with NumPy broadcasting: [2,3] vs a [3] row. ---
TEST(Ops, LessBroadcastRow) {
    auto out = runOp(OpType::Less, 0, {}, {2, 3}, {1, 5, 0, 3, 4, 2}, {{{3}, {2, 4, 1}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {2, 3}));
    // row0 {1,5,0} vs {2,4,1} -> {1,0,1}; row1 {3,4,2} vs {2,4,1} -> {0,0,0}
    EXPECT_EQ(out.data, (std::vector<float> {1, 0, 1, 0, 0, 0}));
}

// --- Less with rank-4 broadcasting: [1,2,2,2] vs a per-channel [1,2,1,1] threshold. ---
TEST(Ops, LessBroadcastPerChannel) {
    auto out = runOp(OpType::Less, 0, {}, {1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {{{1, 2, 1, 1}, {2.5f, 6.5f}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 2, 2, 2}));
    // ch0 {1,2,3,4} < 2.5 -> {1,1,0,0}; ch1 {5,6,7,8} < 6.5 -> {1,1,0,0}
    EXPECT_EQ(out.data, (std::vector<float> {1, 1, 0, 0, 1, 1, 0, 0}));
}

// --- And, same shape: bool operands read as (x != 0), output 1.0/0.0. ---
TEST(Ops, AndSameShape) {
    auto out = runOp(OpType::And, 0, {}, {1, 4}, {1, 1, 0, 0}, {{{1, 4}, {1, 0, 1, 0}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 4}));
    EXPECT_EQ(out.data, (std::vector<float> {1, 0, 0, 0}));
}

// --- And with NumPy broadcasting: a [2,3] mask AND a [3] row (the causal+padding mask combine). ---
TEST(Ops, AndBroadcastRow) {
    auto out = runOp(OpType::And, 0, {}, {2, 3}, {1, 0, 1, 0, 1, 0}, {{{3}, {1, 1, 0}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {2, 3}));
    // row0 {1,0,1} & {1,1,0} -> {1,0,0}; row1 {0,1,0} & {1,1,0} -> {0,1,0}
    EXPECT_EQ(out.data, (std::vector<float> {1, 0, 0, 0, 1, 0}));
}

// --- And treats any nonzero as true (not just exactly 1). ---
TEST(Ops, AndNonzeroIsTrue) {
    auto out = runOp(OpType::And, 0, {}, {1, 3}, {2.5f, 0.f, -1.f}, {{{1, 3}, {-3.f, 4.f, 0.f}}});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 3}));
    EXPECT_EQ(out.data, (std::vector<float> {1, 0, 0}));
}

// --- IsNaN: only NaN maps to 1; finite values and +/-inf map to 0. ---
TEST(Ops, IsNaN) {
    auto out = runOp(OpType::IsNaN, 0, {}, {1, 5}, {1.0f, NAN, -2.0f, INFINITY, -INFINITY}, {});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 5}));
    EXPECT_EQ(out.data, (std::vector<float> {0, 1, 0, 0, 0}));
}

namespace {
    struct TopKOut {
        std::vector<float>   values;
        std::vector<int64_t> indices;
        Shape                shape;  // shared by both outputs
        DType                iDtype; // declared dtype of the indices output
    };

    // Build a two-output TopK graph: float input "x", k as a const int64 initializer input (opset
    // 10+ form) or the `k` attribute (opset <10 form) when kAsAttr is set. Runs on CPU and returns
    // both outputs.
    TopKOut runTopK(const Shape &xshape, const std::vector<float> &xdata, int64_t k, int64_t axis, int64_t largest, int64_t sorted, bool kAsAttr = false) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = xshape;
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};
        Node n;
        n.type   = OpType::TopK;
        n.name   = "topk";
        n.inputs = {x};
        if (kAsAttr)
        {
            Attr a;
            a.kind         = Attr::Int;
            a.i            = k;
            n.attr.map["k"] = a;
        } else
        {
            TensorDesc kd;
            kd.name          = "k";
            kd.shape         = {1};
            kd.isInitializer = true;
            kd.dtype         = DType::Int64;
            TensorId   kid   = g.addTensor(kd);
            HostBuffer hb;
            hb.resizeElems(1, DType::Int64);
            hb.i64()[0]        = k;
            g.initializers[kid] = hb;
            n.inputs.push_back(kid);
        }
        auto seti = [&](const char *name, int64_t v) {
            Attr a;
            a.kind           = Attr::Int;
            a.i              = v;
            n.attr.map[name] = a;
        };
        seti("axis", axis);
        seti("largest", largest);
        seti("sorted", sorted);
        TensorDesc vo;
        vo.name     = "v";
        vo.isOutput = true;
        TensorId v  = g.addTensor(vo);
        TensorDesc ixd;
        ixd.name     = "i";
        ixd.isOutput = true;
        ixd.dtype    = DType::Int64; // ONNX declares the indices output int64 (graph-output value_info)
        TensorId ix  = g.addTensor(ixd);
        n.outputs   = {v, ix};
        g.nodes     = {n};
        g.outputs   = {v, ix};

        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = "x";
        in.shape = xshape;
        in.dtype = DType::Float32;
        in.data.resize(xdata.size() * 4);
        std::memcpy(in.data.data(), xdata.data(), xdata.size() * 4);
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        if (outs.size() != 2)
        {
            return {};
        }
        TopKOut       r;
        r.shape          = outs[0].shape;
        r.iDtype         = outs[1].dtype;
        const float   *vp    = reinterpret_cast<const float *>(outs[0].data.data());
        const int64_t *ip    = reinterpret_cast<const int64_t *>(outs[1].data.data());
        int64_t        elems = numElements(outs[0].shape);
        r.values.assign(vp, vp + elems);
        r.indices.assign(ip, ip + elems);
        return r;
    }
} // namespace

// --- TopK defaults (axis=-1, largest=1, sorted=1): per-row top-2 of a [2,4] input, values
//     descending, indices int64. ---
TEST(Ops, TopKLargestLastAxis) {
    auto r = runTopK({2, 4}, {1, 3, 2, 4, 7, 5, 8, 6}, 2, -1, 1, 1);
    ASSERT_EQ(r.shape, (Shape {2, 2}));
    EXPECT_EQ(r.iDtype, DType::Int64);
    EXPECT_EQ(r.values, (std::vector<float> {4, 3, 8, 7}));
    EXPECT_EQ(r.indices, (std::vector<int64_t> {3, 1, 2, 0}));
}

// --- TopK largest=0: the k smallest, ascending. ---
TEST(Ops, TopKSmallest) {
    auto r = runTopK({1, 4}, {3, 1, 4, 2}, 2, -1, 0, 1);
    ASSERT_EQ(r.shape, (Shape {1, 2}));
    EXPECT_EQ(r.values, (std::vector<float> {1, 2}));
    EXPECT_EQ(r.indices, (std::vector<int64_t> {1, 3}));
}

// --- TopK on a non-default axis: axis=0 selects down the columns of a [3,2]. ---
TEST(Ops, TopKAxis0) {
    // col0 {5,2,4} -> top2 {5,4} at rows {0,2}; col1 {1,6,3} -> {6,3} at rows {1,2}
    auto r = runTopK({3, 2}, {5, 1, 2, 6, 4, 3}, 2, 0, 1, 1);
    ASSERT_EQ(r.shape, (Shape {2, 2}));
    EXPECT_EQ(r.values, (std::vector<float> {5, 6, 4, 3}));
    EXPECT_EQ(r.indices, (std::vector<int64_t> {0, 1, 2, 2}));
}

// --- TopK tie ordering: equal values keep ascending source indices (ONNX), for both directions. ---
TEST(Ops, TopKTiesAscendingIndices) {
    auto lg = runTopK({1, 5}, {2, 7, 7, 7, 1}, 3, -1, 1, 1);
    EXPECT_EQ(lg.values, (std::vector<float> {7, 7, 7}));
    EXPECT_EQ(lg.indices, (std::vector<int64_t> {1, 2, 3}));
    auto sm = runTopK({1, 5}, {3, 1, 1, 1, 5}, 2, -1, 0, 1);
    EXPECT_EQ(sm.values, (std::vector<float> {1, 1}));
    EXPECT_EQ(sm.indices, (std::vector<int64_t> {1, 2}));
}

// --- TopK sorted=0: element order is unspecified by ONNX; the kernel emits the same sorted order
//     as sorted=1 (a valid instance, and deterministic run to run). ---
TEST(Ops, TopKUnsortedIsDeterministic) {
    auto r = runTopK({1, 4}, {1, 3, 2, 4}, 2, -1, 1, 0);
    ASSERT_EQ(r.shape, (Shape {1, 2}));
    EXPECT_EQ(r.values, (std::vector<float> {4, 3}));
    EXPECT_EQ(r.indices, (std::vector<int64_t> {3, 1}));
}

// --- TopK with k as the opset-9 attribute (no second input). ---
TEST(Ops, TopKKAttr) {
    auto r = runTopK({1, 4}, {1, 3, 2, 4}, 2, -1, 1, 1, /*kAsAttr=*/true);
    ASSERT_EQ(r.shape, (Shape {1, 2}));
    EXPECT_EQ(r.values, (std::vector<float> {4, 3}));
    EXPECT_EQ(r.indices, (std::vector<int64_t> {3, 1}));
}

// --- TopK with k past the axis length clamps to the axis length (shape rule and kernel agree). ---
TEST(Ops, TopKClampsOversizedK) {
    auto r = runTopK({1, 3}, {2, 3, 1}, 9, -1, 1, 1);
    ASSERT_EQ(r.shape, (Shape {1, 3}));
    EXPECT_EQ(r.values, (std::vector<float> {3, 2, 1}));
    EXPECT_EQ(r.indices, (std::vector<int64_t> {1, 0, 2}));
}

// --- ConvertLayout on the CPU backend: NC4HW4 is a device-only storage format and every host
//     residency is canonical NCHW, so both directions are value- and shape-preserving copies. A
//     flat->NC4HW4 -> NC4HW4->flat pair must return the input exactly, including the odd channel
//     counts (1, 3, 5) whose device form carries zero-filled pad lanes. ---
TEST(Ops, ConvertLayoutRoundTripIdentity) {
    for (int64_t C: {1, 3, 4, 5, 8})
    {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {1, C, 2, 3};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs.push_back(x);
        TensorDesc ti;
        ti.name    = "t";
        TensorId t = g.addTensor(ti);
        Node     toNc4;
        toNc4.type    = OpType::ConvertLayout;
        toNc4.name    = "to_nc4";
        toNc4.subOp   = 1; // flat -> NC4HW4
        toNc4.inputs  = {x};
        toNc4.outputs = {t};
        g.nodes.push_back(toNc4);
        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     toFlat;
        toFlat.type    = OpType::ConvertLayout;
        toFlat.name    = "to_flat";
        toFlat.subOp   = 0; // NC4HW4 -> flat
        toFlat.inputs  = {t};
        toFlat.outputs = {y};
        g.nodes.push_back(toFlat);
        g.outputs = {y};

        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        ASSERT_TRUE(sess) << "C=" << C;
        std::vector<float> vals(1 * C * 2 * 3);
        for (size_t i = 0; i < vals.size(); ++i)
        {
            vals[i] = (float) i * 1.25f - 3.f;
        }
        IOTensor in;
        in.name  = "x";
        in.shape = {1, C, 2, 3};
        in.dtype = DType::Float32;
        in.data.resize(vals.size() * 4);
        std::memcpy(in.data.data(), vals.data(), in.data.size());
        std::vector<IOTensor> outs;
        ASSERT_EQ(sess->run({in}, outs), Status::Ok) << "C=" << C;
        ASSERT_EQ(outs[0].shape, (std::vector<int64_t> {1, C, 2, 3})) << "C=" << C;
        const float *o = outs[0].f32();
        for (size_t i = 0; i < vals.size(); ++i)
        {
            EXPECT_EQ(o[i], vals[i]) << "C=" << C << " i=" << i; // identity copy: exact, not near
        }
    }
}

// --- A single ConvertLayout in either direction is a pass-through on the host: same shape, same
//     values (the physical repack happens at the backend boundary, never in this kernel). ---
TEST(Ops, ConvertLayoutPassThroughBothDirections) {
    const std::vector<float> vals {2, -1, 0.5f, 7, -3.25f, 4, 9, -8, 1.5f, 6};
    for (int dir: {0, 1})
    {
        auto out = runOp(OpType::ConvertLayout, dir, {}, {1, 5, 1, 2}, vals, {});
        ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 5, 1, 2})) << "dir=" << dir;
        EXPECT_EQ(out.data, vals) << "dir=" << dir;
    }
}

// --- A CPU consumer downstream of ConvertLayout reads canonical NCHW: with C=5 (odd, pad lanes on
//     the device side) the per-channel means stay per-channel. A kernel that physically packed the
//     host buffer would scatter channels into pad lanes and shift every mean. ---
TEST(Ops, ConvertLayoutConsumerReadsNchw) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 5, 1, 2};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorDesc ti;
    ti.name    = "t";
    TensorId t = g.addTensor(ti);
    Node     cv;
    cv.type    = OpType::ConvertLayout;
    cv.name    = "cv";
    cv.subOp   = 1; // flat -> NC4HW4
    cv.inputs  = {x};
    cv.outputs = {t};
    g.nodes.push_back(cv);
    TensorDesc yo;
    yo.name     = "y";
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     gap;
    gap.type    = OpType::GlobalAvgPool;
    gap.name    = "gap";
    gap.inputs  = {t};
    gap.outputs = {y};
    g.nodes.push_back(gap);
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);
    // channel c holds {10c, 10c+2} -> mean 10c+1
    std::vector<float> vals {0, 2, 10, 12, 20, 22, 30, 32, 40, 42};
    IOTensor           in;
    in.name  = "x";
    in.shape = {1, 5, 1, 2};
    in.dtype = DType::Float32;
    in.data.resize(vals.size() * 4);
    std::memcpy(in.data.data(), vals.data(), in.data.size());
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_EQ(outs[0].shape, (std::vector<int64_t> {1, 5, 1, 1}));
    const float *o = outs[0].f32();
    for (int c = 0; c < 5; ++c)
    {
        EXPECT_FLOAT_EQ(o[c], 10.f * c + 1.f) << "c=" << c;
    }
}

// --- Non-fp32 model I/O: a UINT8 input flows through the fp32 compute path (Cast->Mul) and back out as
//     UINT8 (Cast truncates toward zero + clamps to [0,255]); the boundary keeps the declared dtypes. ---
TEST(Ops, DtypeUint8RoundTrip) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 1, 2, 2};
    xi.isInput = true;
    xi.dtype   = DType::UInt8;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorDesc si;
    si.name          = "s";
    si.shape         = {1};
    si.isInitializer = true;
    TensorId   s     = g.addTensor(si);
    HostBuffer sb;
    sb.resizeElems(1, DType::Float32);
    sb.f32()[0]       = 0.5f;
    g.initializers[s] = sb;
    TensorDesc t0;
    t0.name     = "xf";
    TensorId xf = g.addTensor(t0);
    Node     c0;
    c0.type       = OpType::Cast;
    c0.name       = "castin";
    c0.inputs     = {x};
    c0.outputs    = {xf};
    c0.attr.map["to"].i = 1; // FLOAT
    g.nodes.push_back(c0);
    TensorDesc t1;
    t1.name    = "m";
    TensorId m = g.addTensor(t1);
    Node     mul;
    mul.type    = OpType::Binary;
    mul.subOp   = (int) BinaryType::Mul;
    mul.name    = "mul";
    mul.inputs  = {xf, s};
    mul.outputs = {m};
    g.nodes.push_back(mul);
    TensorDesc yo;
    yo.name     = "y";
    yo.isOutput = true;
    yo.dtype    = DType::UInt8;
    TensorId y  = g.addTensor(yo);
    Node     c1;
    c1.type             = OpType::Cast;
    c1.name             = "castout";
    c1.inputs           = {m};
    c1.outputs          = {y};
    c1.attr.map["to"].i = 2; // UINT8
    g.nodes.push_back(c1);
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);
    IOTensor in;
    in.name  = "x";
    in.shape = {1, 1, 2, 2};
    in.dtype = DType::UInt8;
    in.data  = {10, 101, 200, 255}; // raw uint8 bytes
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_EQ(outs[0].dtype, DType::UInt8);
    ASSERT_EQ(outs[0].data.size(), 4u);
    // 10*.5=5, 101*.5=50.5->50 (trunc), 200*.5=100, 255*.5=127.5->127
    EXPECT_EQ(outs[0].data, (std::vector<uint8_t> {5, 50, 100, 127}));
}

// --- Cast float->INT64 truncates toward zero and the output boundary emits int64 bytes. ---
TEST(Ops, DtypeCastToInt64) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {3};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorDesc yo;
    yo.name     = "y";
    yo.isOutput = true;
    yo.dtype    = DType::Int64;
    TensorId y  = g.addTensor(yo);
    Node     c;
    c.type             = OpType::Cast;
    c.name             = "cast";
    c.inputs           = {x};
    c.outputs          = {y};
    c.attr.map["to"].i = 7; // INT64
    g.nodes.push_back(c);
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);
    IOTensor in;
    in.name  = "x";
    in.shape = {3};
    in.dtype = DType::Float32;
    in.data.resize(3 * 4);
    float vals[3] = {1.9f, -2.1f, 3.5f};
    std::memcpy(in.data.data(), vals, sizeof(vals));
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_EQ(outs[0].dtype, DType::Int64);
    ASSERT_EQ(outs[0].data.size(), 3u * 8u);
    const int64_t *o = reinterpret_cast<const int64_t *>(outs[0].data.data());
    EXPECT_EQ(o[0], 1);  // 1.9 -> 1
    EXPECT_EQ(o[1], -2); // -2.1 -> -2 (toward zero)
    EXPECT_EQ(o[2], 3);  // 3.5 -> 3
}

// --- FusedPointwise (x*a + b, clipped to [0,10]): same-shape binary steps + a Clip act step. ---
TEST(Ops, FusedPwMulAddClip) {
    std::vector<int64_t> steps {kPwKindBinary, (int) BinaryType::Mul, kPwRefAcc, kPwRefOp0 - 1, kPwRefNone, kPwRefNone, 0, 2,
                                kPwKindBinary, (int) BinaryType::Add, kPwRefAcc, kPwRefOp0 - 2, kPwRefNone, kPwRefNone, 0, 2,
                                kPwKindAct,    (int) ActType::Clip,   kPwRefAcc, kPwRefNone,    kPwRefNone, kPwRefNone, 0, 0};
    std::vector<float>   params {0, 0, 0, 0, 0.f, 10.f};
    auto                 got = runFusedPw({1, 1, 2, 2}, {1, 2, 3, 4}, {{{1, 1, 2, 2}, {2, 2, 2, 2}}, {{1, 1, 2, 2}, {1, 1, 1, 1}}}, steps, params);
    expectNear(got.data, {3, 5, 7, 9});
}

// --- FusedPointwise (sigmoid(x) * channel-broadcast scale): a Unary step then a channel-bcast binary step. ---
TEST(Ops, FusedPwUnaryChannelMul) {
    std::vector<int64_t> steps {kPwKindUnary,  (int) UnaryType::Sigmoid, kPwRefAcc, kPwRefNone,    kPwRefNone, kPwRefNone, 0, 0,
                                kPwKindBinary, (int) BinaryType::Mul,    kPwRefAcc, kPwRefOp0 - 1, kPwRefNone, kPwRefNone, 1, 2};
    std::vector<float>   params {0, 0, 0, 0};
    auto                 got = runFusedPw({1, 3, 1, 2}, {0, 0, 0, 0, 0, 0}, {{{1, 3, 1, 1}, {10, 20, 30}}}, steps, params);
    expectNear(got.data, {5, 5, 10, 10, 15, 15});
}

// --- FusedPointwise unary Round step: the pw-VM ties-to-even path matches the standalone op. ---
TEST(Ops, FusedPwUnaryRound) {
    std::vector<int64_t> steps {kPwKindUnary, (int) UnaryType::Round, kPwRefAcc, kPwRefNone, kPwRefNone, kPwRefNone, 0, 0};
    std::vector<float>   params {0, 0};
    auto                 got = runFusedPw({1, 4}, {-1.5f, -0.5f, 0.5f, 2.5f}, {}, steps, params);
    expectNear(got.data, {-2, 0, 0, 2}, 0.f);
}

// --- FusedPointwise diamond (x * sigmoid(x), the SiLU shape): the entry value feeds two steps,
// so the unit re-reads kPwRefEntry after the accumulator moved past it. ---
TEST(Ops, FusedPwEntryDiamond) {
    std::vector<int64_t> steps {kPwKindUnary,  (int) UnaryType::Sigmoid, kPwRefEntry, kPwRefNone, kPwRefNone, kPwRefNone, 0, 0,
                                kPwKindBinary, (int) BinaryType::Mul,    kPwRefEntry, kPwRefAcc,  kPwRefNone, kPwRefNone, 0, 0};
    std::vector<float>   params {0, 0, 0, 0};
    auto                 got = runFusedPw({1, 4}, {-1, 0, 1, 2}, {}, steps, params);
    std::vector<float>   ref;
    for (float x: {-1.f, 0.f, 1.f, 2.f})
    {
        ref.push_back(x / (1.f + std::exp(-x)));
    }
    expectNear(got.data, ref);
}

// --- FusedPointwise register reuse: r0 = x + a is consumed by a later step after the accumulator
// was overwritten by an unrelated one. y = (x * b) + (x + a). ---
TEST(Ops, FusedPwRegister) {
    std::vector<int64_t> steps {kPwKindBinary, (int) BinaryType::Add, kPwRefEntry, kPwRefOp0 - 1, kPwRefNone, 0,          0, 2,
                                kPwKindBinary, (int) BinaryType::Mul, kPwRefEntry, kPwRefOp0 - 2, kPwRefNone, kPwRefNone, 0, 2,
                                kPwKindBinary, (int) BinaryType::Add, kPwRefAcc,   kPwRefReg0,    kPwRefNone, kPwRefNone, 0, 0};
    std::vector<float>   params {0, 0, 0, 0, 0, 0};
    auto                 got = runFusedPw({1, 3}, {1, 2, 3}, {{{1, 3}, {10, 10, 10}}, {{1, 3}, {2, 2, 2}}}, steps, params);
    expectNear(got.data, {13, 16, 19}); // x*2 + (x+10)
}

// --- FusedPointwise select (the Where shape): mask ? a : x with a tensor condition. ---
TEST(Ops, FusedPwSelect) {
    std::vector<int64_t> steps {kPwKindSelect, 0, kPwRefOp0 - 1, kPwRefOp0 - 2, kPwRefEntry, kPwRefNone, 0, 1};
    std::vector<float>   params {0, 0};
    auto                 got = runFusedPw({1, 4}, {1, 2, 3, 4}, {{{1, 4}, {1, 0, 1, 0}}, {{1, 4}, {-9, -9, -9, -9}}}, steps, params);
    expectNear(got.data, {-9, 2, -9, 4});
}

// --- FusedPointwise load step: replace the accumulator with an operand mid-unit, keeping the
// entry reachable. y = c - x via load(c) then Sub(acc, entry). ---
TEST(Ops, FusedPwLoad) {
    std::vector<int64_t> steps {kPwKindLoad,   0,                     kPwRefOp0 - 1, kPwRefNone,  kPwRefNone, kPwRefNone, 0, 1,
                                kPwKindBinary, (int) BinaryType::Sub, kPwRefAcc,     kPwRefEntry, kPwRefNone, kPwRefNone, 0, 0};
    std::vector<float>   params {0, 0, 0, 0};
    auto                 got = runFusedPw({1, 3}, {1, 2, 3}, {{{1, 3}, {10, 10, 10}}}, steps, params);
    expectNear(got.data, {9, 8, 7});
}

// --- FusedPointwise multi-output (pw_outs): a fanned-out intermediate step value is exported as
// a second graph output while the unit continues to the final result. y = (x+c)^2, z = x+c. ---
TEST(Ops, FusedPwMultiOutput) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 4};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs   = {x};
    TensorDesc ci;
    ci.name          = "c";
    ci.shape         = {1, 4};
    ci.isInitializer = true;
    TensorId   c     = g.addTensor(ci);
    HostBuffer hb;
    hb.resizeElems(4, DType::Float32);
    for (int i = 0; i < 4; ++i)
    {
        hb.f32()[i] = 10.f;
    }
    g.initializers[c] = hb;
    TensorDesc yo;
    yo.name     = "y";
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    TensorDesc zo;
    zo.name     = "z";
    zo.isOutput = true;
    TensorId z  = g.addTensor(zo);
    Node     n;
    n.type    = OpType::FusedPointwise;
    n.name    = "pw";
    n.inputs  = {x, c};
    n.outputs = {y, z};
    {
        Attr a;
        a.kind                 = Attr::Ints;
        a.ints                 = {kPwKindBinary, (int) BinaryType::Add, kPwRefAcc, kPwRefOp0 - 1, kPwRefNone, kPwRefNone, 0, 2,
                                  kPwKindBinary, (int) BinaryType::Mul, kPwRefAcc, kPwRefAcc,     kPwRefNone, kPwRefNone, 0, 0};
        n.attr.map["pw_steps"] = a;
    }
    {
        Attr a;
        a.kind                  = Attr::Floats;
        a.floats                = {0, 0, 0, 0};
        n.attr.map["pw_params"] = a;
    }
    {
        Attr a;
        a.kind                = Attr::Ints;
        a.ints                = {0}; // z stores step 0 (x+c)
        n.attr.map["pw_outs"] = a;
    }
    {
        Attr a;
        a.kind                = Attr::Int;
        a.i                   = 1;
        n.attr.map["pw_flat"] = a;
    }
    g.nodes   = {n};
    g.outputs = {y, z};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);
    IOTensor in;
    in.name  = "x";
    in.shape = {1, 4};
    in.dtype = DType::Float32;
    in.data.resize(4 * 4);
    float xv[4] = {1, 2, 3, 4};
    std::memcpy(in.data.data(), xv, sizeof(xv));
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_EQ(outs.size(), 2u);
    const float *yv = reinterpret_cast<const float *>(outs[0].data.data());
    const float *zv = reinterpret_cast<const float *>(outs[1].data.data());
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(zv[i], xv[i] + 10.f, 1e-5f) << "z i=" << i;
        EXPECT_NEAR(yv[i], (xv[i] + 10.f) * (xv[i] + 10.f), 1e-4f) << "y i=" << i;
    }
}

// --- A non-pointwise producer (Softmax) carrying an attached pw_steps epilogue (Mul by a
// per-tensor constant) must have the chain applied in place by the central CPU executor hook,
// not just by the standalone FusedPointwise op. ---
TEST(Ops, CpuEpilogueHookOnProducer) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 4};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs   = {x};
    TensorDesc ci;
    ci.name          = "c";
    ci.shape         = {1};
    ci.isInitializer = true;
    TensorId  c       = g.addTensor(ci);
    HostBuffer hb;
    hb.resizeElems(1, DType::Float32);
    hb.f32()[0]      = 3.0f;
    g.initializers[c] = hb;
    TensorDesc yo;
    yo.name     = "y";
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node n;
    n.type    = OpType::Softmax;
    n.name    = "sm";
    n.inputs  = {x, c}; // inputs[1]=c is the epilogue operand
    n.outputs = {y};
    {
        Attr a;
        a.kind                 = Attr::Ints;
        // Mul by inputs[1] (a scalar constant, bcast mode 3 on srcB)
        a.ints                 = {kPwKindBinary, (int) BinaryType::Mul, kPwRefAcc, kPwRefOp0 - 1, kPwRefNone, kPwRefNone, 3, 2};
        n.attr.map["pw_steps"] = a;
    }
    {
        Attr a;
        a.kind                  = Attr::Floats;
        a.floats                = {0, 0};
        n.attr.map["pw_params"] = a;
    }
    {
        Attr a;
        a.kind                  = Attr::Int;
        a.i                     = 1;
        n.attr.map["pw_opbase"] = a;
    }
    {
        Attr a;
        a.kind             = Attr::Int;
        a.i                = -1;
        n.attr.map["axis"] = a; // softmax axis=-1 (SoftmaxCpu reads attr.geti("axis", -1))
    }
    g.nodes   = {n};
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);

    IOTensor in;
    in.name  = "x";
    in.shape = {1, 4};
    in.dtype = DType::Float32;
    in.data.resize(4 * 4);
    float xv[4] = {1, 2, 3, 4};
    for (int i = 0; i < 4; ++i)
    {
        reinterpret_cast<float *>(in.data.data())[i] = xv[i];
    }
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);

    // reference: softmax([1,2,3,4]) * 3
    float e[4], sum = 0;
    for (int i = 0; i < 4; ++i)
    {
        e[i] = std::exp(xv[i] - 4.f);
        sum += e[i];
    }
    std::vector<float> ref(4);
    for (int i = 0; i < 4; ++i)
    {
        ref[i] = e[i] / sum * 3.0f;
    }
    const float *o = outs[0].f32();
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(o[i], ref[i], 1e-4f) << "i=" << i;
    }
}

namespace {

    // Chain: y = Clip(Mul(x, s) + b, 0, +inf) -- a Binary(Mul), an Add, then a Clip (ReLU via min=0),
    // each single-consumer, all same shape and all GPU-flat (a Binary/Add with a constant operand and
    // a Clip are both always-flat ops, so the pass's layout-agreement check passes across all three).
    // fusePointwiseChains should merge all three into one standalone FusedPointwise node.
    Graph makeChainGraph() {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {1, 1, 2, 2};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};
        auto k     = [&](const char *nm, std::vector<int64_t> sh, std::vector<float> d) {
            TensorDesc t;
            t.name          = nm;
            t.shape         = sh;
            t.isInitializer = true;
            TensorId   id   = g.addTensor(t);
            HostBuffer hb;
            hb.resizeElems(d.size(), DType::Float32);
            for (size_t i = 0; i < d.size(); ++i)
            {
                hb.f32()[i] = d[i];
            }
            g.initializers[id] = hb;
            return id;
        };
        TensorId   s = k("s", {1, 1, 2, 2}, {2, 2, 2, 2}), b = k("b", {1, 1, 2, 2}, {-5, -5, -5, -5});
        TensorDesc t0d;
        t0d.name      = "t0";
        TensorId   t0 = g.addTensor(t0d);
        TensorDesc t1d;
        t1d.name      = "t1";
        TensorId   t1 = g.addTensor(t1d);
        TensorDesc yd;
        yd.name            = "y";
        TensorId y         = g.addTensor(yd);
        g.desc(y).isOutput = true;
        Node m;
        m.type    = OpType::Binary;
        m.subOp   = (int) BinaryType::Mul;
        m.name    = "mul";
        m.inputs  = {x, s};
        m.outputs = {t0};
        Node a;
        a.type    = OpType::Add;
        a.name    = "add";
        a.inputs  = {t0, b};
        a.outputs = {t1};
        Node r;
        r.type            = OpType::Clip;
        r.name            = "clip";
        r.inputs          = {t1};
        r.outputs         = {y};
        r.attr.map["min"] = [] {
            Attr a;
            a.kind = Attr::Float;
            a.f    = 0.f;
            return a;
        }();
        g.nodes   = {m, a, r};
        g.outputs = {y};
        return g;
    }

    std::vector<float> runGraphCpu(Graph g, const std::vector<float> &xd) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = "x";
        in.shape = {1, 1, 2, 2};
        in.dtype = DType::Float32;
        in.data.resize(xd.size() * 4);
        for (size_t i = 0; i < xd.size(); ++i)
        {
            reinterpret_cast<float *>(in.data.data())[i] = xd[i];
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        if (outs.empty())
        {
            return {};
        }
        const float *o = outs[0].f32();
        return {o, o + numElements(outs[0].shape)};
    }

} // namespace

// --- fusePointwiseChains merges Mul->Add->Clip (single-consumer, same-shape, same GPU layout) into
// one standalone FusedPointwise node; the fused graph must be bit-exact vs. the unfused one. ---
TEST(Passes, FusePointwiseBitExact) {
    std::vector<float> xd {1, 2, 3, 4};
    auto               unfused = runGraphCpu(makeChainGraph(), xd);

    Graph fg = makeChainGraph();
    inferShapes(fg, 1);
    fusePointwiseChains(fg, true);
    int fused = 0;
    for (auto &n: fg.nodes)
    {
        if (n.type == OpType::FusedPointwise)
        {
            fused++;
        }
    }
    EXPECT_EQ(fused, 1);
    EXPECT_LE(fg.nodes.size(), 1u);

    auto got = runGraphCpu(std::move(fg), xd);
    ASSERT_EQ(got.size(), unfused.size());
    for (size_t i = 0; i < got.size(); ++i)
    {
        EXPECT_FLOAT_EQ(got[i], unfused[i]);
    }
}

namespace {

    // A graph with a fully-symbolic input [-1,-1,-1,-1] feeding a 3x3 Conv (16 out-channels, pad 1,
    // stride 1). Shape inference must resolve the input before it can propagate an output shape, so
    // this exercises exactly the declared-shape / batch-fallback / hard-error paths. The named input
    // is "pixel_values" to mirror the vit case (all four axes dynamic).
    Graph makeDynamicInputConvGraph() {
        Graph      g;
        TensorDesc xi;
        xi.name    = "pixel_values";
        xi.shape   = {-1, -1, -1, -1};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs.push_back(x);

        TensorDesc wi;
        wi.name          = "w";
        wi.shape         = {16, 3, 3, 3};
        wi.isInitializer = true;
        TensorId   w     = g.addTensor(wi);
        HostBuffer hb;
        hb.resizeElems(16 * 3 * 3 * 3, DType::Float32);
        for (size_t i = 0; i < 16 * 3 * 3 * 3; ++i)
        {
            hb.f32()[i] = 0.f;
        }
        g.initializers[w] = hb;

        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);

        Node n;
        n.type                     = OpType::Conv;
        n.name                     = "conv";
        n.inputs                   = {x, w};
        n.outputs                  = {y};
        n.attr.map["strides"]      = ints({1, 1});
        n.attr.map["pads"]         = ints({1, 1, 1, 1});
        n.attr.map["dilations"]    = ints({1, 1});
        n.attr.map["kernel_shape"] = ints({3, 3});
        g.nodes.push_back(n);
        g.outputs = {y};
        return g;
    }

} // namespace

// --- Declared input shapes: a symbolic leading (batch) axis still falls back to `batch` (=1) with no
// declaration, so a dynamic-batch model resolves exactly as before. Here only axis 0 is dynamic. ---
TEST(Passes, InferShapesBatchFallback) {
    Graph g;
    // input [-1,3,8,8]: only the batch axis is dynamic (the resnet18 "data [N,3,224,224]" case).
    TensorDesc xi;
    xi.name    = "data";
    xi.shape   = {-1, 3, 8, 8};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);

    TensorDesc wi;
    wi.name          = "w";
    wi.shape         = {16, 3, 3, 3};
    wi.isInitializer = true;
    TensorId   w     = g.addTensor(wi);
    HostBuffer hb;
    hb.resizeElems(16 * 3 * 3 * 3, DType::Float32);
    g.initializers[w] = hb;

    TensorDesc yo;
    yo.name     = "y";
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     n;
    n.type                     = OpType::Conv;
    n.name                     = "conv";
    n.inputs                   = {x, w};
    n.outputs                  = {y};
    n.attr.map["strides"]      = ints({1, 1});
    n.attr.map["pads"]         = ints({1, 1, 1, 1});
    n.attr.map["dilations"]    = ints({1, 1});
    n.attr.map["kernel_shape"] = ints({3, 3});
    g.nodes.push_back(n);
    g.outputs = {y};

    inferShapes(g, 1); // no declared map: batch axis -> 1, no error (no other dynamic axis)
    EXPECT_EQ(g.desc(x).shape, (Shape {1, 3, 8, 8}));
    EXPECT_EQ(g.desc(y).shape, (Shape {1, 16, 8, 8}));
}

// --- A declared spatial shape flows through inferShapes to the output: pixel_values 1x3x224x224 with
// pad-1 3x3 stride-1 conv -> y [1,16,224,224]. Without the declaration this input is all -1. ---
TEST(Passes, InferShapesDeclaredSpatial) {
    Graph                        g = makeDynamicInputConvGraph();
    std::map<std::string, Shape> declared;
    declared["pixel_values"] = {1, 3, 224, 224};
    inferShapes(g, 1, &declared);
    EXPECT_EQ(g.desc(g.inputs[0]).shape, (Shape {1, 3, 224, 224}));
    EXPECT_EQ(g.desc(g.outputs[0]).shape, (Shape {1, 16, 224, 224}));
}

// --- An UNDECLARED dynamic non-batch axis is a hard error (Status::InvalidArgument), never a silent
// substitution to 1 (the vit_b16_q8 0.32-cosine bug: spatial axes froze to a 1x1 plan). The message
// names the input and the offending axis. ---
TEST(Passes, InferShapesUndeclaredDynamicAxisErrors) {
    Graph g = makeDynamicInputConvGraph(); // pixel_values [-1,-1,-1,-1], no declaration
    bool  threw = false;
    try
    {
        inferShapes(g, 1); // axis 0 -> batch, but axes 1..3 are dynamic non-batch with no declaration
    } catch (const Error &e)
    {
        threw = true;
        EXPECT_EQ(e.status(), Status::InvalidArgument);
        EXPECT_NE(std::string(e.what()).find("pixel_values"), std::string::npos) << e.what();
    }
    EXPECT_TRUE(threw) << "undeclared dynamic non-batch axis must hard-error, not silently become 1";
}

// --- A declared shape whose rank disagrees with the input's rank is a hard error: it cannot resolve
// the input's dynamic dims and would silently mis-shape the plan. ---
TEST(Passes, InferShapesDeclaredRankMismatchErrors) {
    Graph                        g = makeDynamicInputConvGraph(); // rank-4 input
    std::map<std::string, Shape> declared;
    declared["pixel_values"] = {1, 3, 224}; // rank 3 != 4
    bool threw               = false;
    try
    {
        inferShapes(g, 1, &declared);
    } catch (const Error &e)
    {
        threw = true;
        EXPECT_EQ(e.status(), Status::InvalidArgument);
    }
    EXPECT_TRUE(threw) << "declared shape of the wrong rank must hard-error";
}

namespace {
    // A single graph input with the given shape and per-axis dim_param symbols, no nodes -- exercises
    // just the input dim-resolution path of inferShapes (bindings / batch fallback / aggregated error).
    Graph makeSymbolicInputGraph(const Shape &shape, const std::vector<std::string> &dimParams) {
        Graph      g;
        TensorDesc xi;
        xi.name      = "x";
        xi.shape     = shape;
        xi.dimParams = dimParams;
        xi.isInput   = true;
        TensorId x   = g.addTensor(xi);
        g.inputs.push_back(x);
        return g;
    }
} // namespace

// --- The dim-expression evaluator: a bare symbol, an integer literal, a compound sum, and a product,
// each resolved from bound symbols; an unbound symbol reports itself as free. ---
TEST(DimExpr, EvaluatesGrammar) {
    std::map<std::string, int64_t> b = {{"past_sequence_length", 256}, {"sequence_length", 1}};
    EXPECT_TRUE(evalDimExpr("past_sequence_length", b).ok);
    EXPECT_EQ(evalDimExpr("past_sequence_length", b).value, 256);
    EXPECT_EQ(evalDimExpr("7", b).value, 7);
    EXPECT_EQ(evalDimExpr("past_sequence_length + sequence_length", b).value, 257);
    EXPECT_EQ(evalDimExpr("2 * sequence_length + past_sequence_length", b).value, 258);
    DimEval free = evalDimExpr("num_heads + sequence_length", b);
    EXPECT_FALSE(free.ok);
    ASSERT_EQ(free.freeSymbols.size(), 1u);
    EXPECT_EQ(free.freeSymbols[0], "num_heads");
}

// --- A symbolic input axis resolves from a --dim binding of its dim_param name; batch_size (axis 0)
// falls back to `batch` while sequence_length (axis 1) is bound. ---
TEST(Passes, InferShapesDimBindingResolves) {
    Graph                          g     = makeSymbolicInputGraph({-1, -1}, {"batch_size", "sequence_length"});
    std::map<std::string, int64_t> binds = {{"sequence_length", 5}};
    inferShapes(g, 1, nullptr, &binds);
    EXPECT_EQ(g.desc(g.inputs[0]).shape, (Shape {1, 5}));
}

// --- A bound --dim of the leading (batch) symbol wins over the --batch fallback. ---
TEST(Passes, InferShapesDimBindingBatchSymbol) {
    Graph                          g     = makeSymbolicInputGraph({-1, -1}, {"batch_size", "sequence_length"});
    std::map<std::string, int64_t> binds = {{"batch_size", 4}, {"sequence_length", 7}};
    inferShapes(g, 1, nullptr, &binds);
    EXPECT_EQ(g.desc(g.inputs[0]).shape, (Shape {4, 7}));
}

// --- A compound dim_param expression evaluates from its two bound symbols (the optimum attention_mask
// axis "past_sequence_length + sequence_length" at decode C=256 -> 257). ---
TEST(Passes, InferShapesDimBindingCompound) {
    Graph                          g     = makeSymbolicInputGraph({-1, -1}, {"batch_size", "past_sequence_length + sequence_length"});
    std::map<std::string, int64_t> binds = {{"past_sequence_length", 256}, {"sequence_length", 1}};
    inferShapes(g, 1, nullptr, &binds);
    EXPECT_EQ(g.desc(g.inputs[0]).shape, (Shape {1, 257}));
}

// --- An unbound symbolic non-batch axis is an aggregated hard error that NAMES the free symbol, so the
// caller knows exactly what to bind (not a per-tensor "pass --shape" message). ---
TEST(Passes, InferShapesUnboundSymbolErrors) {
    Graph g     = makeSymbolicInputGraph({-1, -1}, {"batch_size", "sequence_length"});
    bool  threw = false;
    try
    {
        inferShapes(g, 1); // no bindings
    } catch (const Error &e)
    {
        threw = true;
        EXPECT_EQ(e.status(), Status::InvalidArgument);
        EXPECT_NE(std::string(e.what()).find("sequence_length"), std::string::npos) << e.what();
    }
    EXPECT_TRUE(threw) << "an unbound symbolic non-batch axis must hard-error naming the symbol";
}

// --- A per-tensor --shape declaration overrides a --dim binding for that tensor (declared wins). ---
TEST(Passes, InferShapesShapeOverridesDim) {
    Graph                          g        = makeSymbolicInputGraph({-1, -1}, {"batch_size", "sequence_length"});
    std::map<std::string, Shape>   declared = {{"x", {2, 9}}};
    std::map<std::string, int64_t> binds    = {{"sequence_length", 5}};
    inferShapes(g, 1, &declared, &binds);
    EXPECT_EQ(g.desc(g.inputs[0]).shape, (Shape {2, 9}));
}

namespace {

    // MatMul(x [n,n], W const [n,n]) -> t0, Mul(t0, c const [n,n]) -> y. The Mul is a one-step
    // pointwise chain whose producer is the MatMul.
    Graph makeMatMulMulGraph(int64_t n) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {n, n};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};
        auto konst = [&](const char *nm) {
            TensorDesc t;
            t.name          = nm;
            t.shape         = {n, n};
            t.isInitializer = true;
            TensorId   id   = g.addTensor(t);
            HostBuffer hb;
            hb.resizeElems((size_t) (n * n), DType::Float32);
            for (int64_t i = 0; i < n * n; ++i)
            {
                hb.f32()[i] = 1.f;
            }
            g.initializers[id] = hb;
            return id;
        };
        TensorId w  = konst("w"), c = konst("c");
        TensorId t0 = g.addTensor({.name = "t0"}), y = g.addTensor({.name = "y"});
        g.desc(y).isOutput = true;
        Node mm;
        mm.type    = OpType::MatMul;
        mm.name    = "mm";
        mm.inputs  = {x, w};
        mm.outputs = {t0};
        Node ml;
        ml.type    = OpType::Binary;
        ml.subOp   = (int) BinaryType::Mul;
        ml.name    = "mul";
        ml.inputs  = {t0, c};
        ml.outputs = {y};
        g.nodes    = {mm, ml};
        g.outputs  = {y};
        return g;
    }

} // namespace

// --- The pass attaches a chain to the naive MatMul kernel's epilogue but refuses the register-
// tiled one (M,N,K all >= kTiledMatMulMin — the same shared constant matmul.cpp selects the
// tiled kernel by): the register-blocked GEMM has no headroom for the VM at its store loop. ---
TEST(Passes, FusePointwiseMatMulTiledRefusal) {
    {
        Graph g = makeMatMulMulGraph(kTiledMatMulMin); // at the threshold: tiled kernel, refuse
        inferShapes(g, 1);
        fusePointwiseChains(g, true);
        for (const auto &nd: g.nodes)
        {
            if (nd.type == OpType::MatMul)
            {
                EXPECT_FALSE(nd.attr.has("pw_steps")) << "tiled-shaped MatMul must not host a pw unit";
            }
        }
    }
    {
        Graph g = makeMatMulMulGraph(kTiledMatMulMin / 2); // below: naive kernel hosts the unit
        inferShapes(g, 1);
        fusePointwiseChains(g, true);
        bool hosted = false;
        for (const auto &nd: g.nodes)
        {
            if (nd.type == OpType::MatMul)
            {
                hosted = nd.attr.has("pw_steps");
            }
        }
        EXPECT_TRUE(hosted) << "small MatMul should host the pointwise epilogue";
    }
}

// The chain primary must be the full-size runtime stream, never a constant: a constant has no GPU
// activation buffer, so a constant primary null-derefs the fused kernel on device. Here the constant
// is inputs[0] of the Mul (as in yonosplat's Mul(const,x)), so the pass must pick x as the primary.
TEST(Passes, FusePointwiseRuntimePrimary) {
    Graph      g;
    TensorDesc xi;
    xi.name       = "x";
    xi.shape      = {1, 2, 2, 2};
    xi.isInput    = true;
    TensorId x    = g.addTensor(xi);
    g.inputs      = {x};
    auto konst    = [&](const char *nm, float v) {
        TensorDesc t;
        t.name          = nm;
        t.shape         = {1, 2, 2, 2};
        t.isInitializer = true;
        TensorId   id   = g.addTensor(t);
        HostBuffer hb;
        hb.resizeElems(8, DType::Float32);
        for (int i = 0; i < 8; ++i)
        {
            hb.f32()[i] = v;
        }
        g.initializers[id] = hb;
        return id;
    };
    TensorId s  = konst("s", 2.f), b = konst("b", 1.f);
    TensorId t0 = g.addTensor({.name = "t0"}), y = g.addTensor({.name = "y"});
    g.desc(y).isOutput = true;
    Node m;
    m.type    = OpType::Binary;
    m.subOp   = (int) BinaryType::Mul;
    m.name    = "mul";
    m.inputs  = {s, x}; // constant first
    m.outputs = {t0};
    Node a;
    a.type    = OpType::Add;
    a.name    = "add";
    a.inputs  = {t0, b};
    a.outputs = {y};
    g.nodes   = {m, a};
    g.outputs = {y};
    inferShapes(g, 1);
    fusePointwiseChains(g, true);
    int fused = -1;
    for (size_t i = 0; i < g.nodes.size(); ++i)
    {
        if (g.nodes[i].type == OpType::FusedPointwise)
        {
            fused = (int) i;
        }
    }
    ASSERT_GE(fused, 0);
    EXPECT_FALSE(g.isInitializer(g.nodes[fused].inputs[0])) << "primary must be the runtime tensor, not a constant";
}

namespace {

    // Graph: Range(start, limit, delta) -> r, then Add(x, r) -> y. Scalars are initializers, so
    // inferShapes resolves r's [n] and constFold bakes the range away before the session runs.
    Graph makeRangeAddGraph(float start, float limit, float delta, int64_t xlen) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {1, 1, 1, xlen};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};
        auto scal  = [&](const char *nm, float v) {
            TensorDesc t;
            t.name          = nm;
            t.shape         = {1};
            t.isInitializer = true;
            TensorId   id   = g.addTensor(t);
            HostBuffer hb;
            hb.resizeElems(1, DType::Float32);
            hb.f32()[0]        = v;
            g.initializers[id] = hb;
            return id;
        };
        TensorId s = scal("start", start), l = scal("limit", limit), d = scal("delta", delta);
        TensorId r = g.addTensor({.name = "r"}), y = g.addTensor({.name = "y"});
        g.desc(y).isOutput = true;
        Node rg;
        rg.type    = OpType::Range;
        rg.name    = "range";
        rg.inputs  = {s, l, d};
        rg.outputs = {r};
        Node a;
        a.type    = OpType::Add;
        a.name    = "add";
        a.inputs  = {x, r};
        a.outputs = {y};
        g.nodes   = {rg, a};
        g.outputs = {y};
        return g;
    }

} // namespace

// --- Range: float scalars fold to a [n] initializer that a downstream Add consumes. ---
TEST(Passes, RangeConstFoldFloat) {
    Graph g = makeRangeAddGraph(1.f, 9.f, 2.f, 4); // arange = {1,3,5,7}
    inferShapes(g, 1);
    constFold(g);
    ASSERT_EQ(g.nodes.size(), 1u) << "Range must fold away";
    EXPECT_EQ(g.nodes[0].type, OpType::Add);
    TensorId r = g.nodes[0].inputs[1];
    ASSERT_TRUE(g.isInitializer(r));
    ASSERT_EQ(g.desc(r).shape, (Shape {4}));

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);
    IOTensor in;
    in.name                 = "x";
    in.shape                = {1, 1, 1, 4};
    in.dtype                = DType::Float32;
    std::vector<float> xd = {10, 20, 30, 40};
    in.data.resize(xd.size() * 4);
    for (size_t i = 0; i < xd.size(); ++i)
    {
        reinterpret_cast<float *>(in.data.data())[i] = xd[i];
    }
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_FALSE(outs.empty());
    const float *o = outs[0].f32();
    expectNear({o, o + 4}, {11, 23, 35, 47});
}

// --- GridSample mode=cubic vs onnxruntime goldens (opset 20, alpha=-0.75 cubic convolution),
// all six padding_mode x align_corners combinations on a 4x4 ramp and a fixed random grid. ---
TEST(Ops, GridSampleCubicVsOrt) {
    std::vector<float> xd(16);
    for (int i = 0; i < 16; ++i)
    {
        xd[i] = (float) i;
    }
    const std::vector<float> grid = {
        -8.4738344e-01f, 5.5983758e-01f, -1.2318152e-01f, 4.4693041e-01f, 9.5597899e-01f, 7.6991796e-02f,
        2.2408962e-03f, -8.5589772e-01f, -4.6312201e-01f, -2.3502111e-04f, 3.5845995e-01f, 6.0747802e-01f,
        -2.3811775e-01f, -8.6812729e-01f, -4.2370880e-01f, 8.1918705e-01f, -5.7322931e-01f, -9.5752060e-02f};
    struct Case {
        const char        *pad;
        int                align;
        std::vector<float> gold;
    };
    const std::vector<Case> cases = {
        {"zeros", 0, {9.943107f, 12.197111f, 5.780506f, 0.826377f, 6.945103f, 15.411776f, 0.40153f, 12.044308f, 6.271687f}},
        {"zeros", 1, {11.665485f, 10.5366f, 9.707059f, 2.310742f, 6.870825f, 13.14908f, 1.900998f, 13.508641f, 6.448456f}},
        {"border", 0, {10.709298f, 11.25916f, 9.583391f, 1.108617f, 6.476125f, 13.414345f, 0.621387f, 12.8671f, 5.673596f}},
        {"border", 1, {9.906149f, 10.29533f, 9.303031f, 2.161082f, 6.733497f, 12.085019f, 1.787719f, 11.976208f, 6.102964f}},
        {"reflection", 0, {10.68639f, 11.25916f, 9.658238f, 1.002547f, 6.476125f, 13.414345f, 0.4935f, 12.916595f, 5.673596f}},
        {"reflection", 1, {10.032711f, 10.367593f, 9.346231f, 1.762658f, 6.710605f, 12.384731f, 1.405842f, 12.396446f, 6.040795f}},
    };
    for (const auto &c: cases)
    {
        Attributes attr;
        attr.map["mode"]         = str("cubic");
        attr.map["padding_mode"] = str(c.pad);
        Attr al;
        al.kind                   = Attr::Int;
        al.i                      = c.align;
        attr.map["align_corners"] = al;
        auto out                  = runOp(OpType::GridSample, 0, attr, {1, 1, 4, 4}, xd, {{{1, 3, 3, 2}, grid}});
        ASSERT_EQ(out.shape, (Shape {1, 1, 3, 3})) << c.pad << " a" << c.align;
        for (size_t i = 0; i < c.gold.size(); ++i)
        {
            EXPECT_NEAR(out.data[i], c.gold[i], 2e-4f) << c.pad << " a" << c.align << " i=" << i;
        }
    }
}

// --- Zero-element constants broadcast per NumPy: a 0 dim propagates (0 x 1 -> 0), it is not
// max'ed into 1. Regression: constFold ran BinaryCpu on an empty folded constant (arange(0)),
// the max-based broadcast fabricated n=1 and read element 0 of a null buffer (SIGSEGV). ---
TEST(Passes, ConstFoldEmptyRangeBinary) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 4};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs   = {x};
    auto scal  = [&](const char *nm, float v) {
        TensorDesc t;
        t.name          = nm;
        t.shape         = {1};
        t.isInitializer = true;
        TensorId   id   = g.addTensor(t);
        HostBuffer hb;
        hb.resizeElems(1, DType::Float32);
        hb.f32()[0]        = v;
        g.initializers[id] = hb;
        return id;
    };
    TensorId s = scal("start", 0.f), l = scal("limit", 0.f), d = scal("delta", 1.f), two = scal("two", 2.f);
    TensorId r = g.addTensor({.name = "r"}), m = g.addTensor({.name = "m"});
    TensorId y = g.addTensor({.name = "y"});
    g.desc(y).isOutput = true;
    Node rg;
    rg.type    = OpType::Range; // folds to an EMPTY [0] constant
    rg.name    = "range";
    rg.inputs  = {s, l, d};
    rg.outputs = {r};
    Node mul;
    mul.type    = OpType::Binary;
    mul.subOp   = (int) BinaryType::Mul;
    mul.name    = "mul"; // Mul(empty, scalar) must fold to an empty [0], not read past a null buffer
    mul.inputs  = {r, two};
    mul.outputs = {m};
    Node add;
    add.type    = OpType::Add;
    add.name    = "add"; // keeps m alive; never folds (x is runtime)
    add.inputs  = {x, m};
    add.outputs = {y};
    g.nodes     = {rg, mul, add};
    g.outputs   = {y};
    inferShapes(g, 1);
    EXPECT_EQ(g.desc(r).shape, (Shape {0}));
    constFold(g);
    ASSERT_TRUE(g.isInitializer(m));
    EXPECT_EQ(g.desc(m).shape, (Shape {0}));
    EXPECT_TRUE(g.initializers[m].bytes.empty());
}

// --- inferShapes: an UNRESOLVED operand (empty shape on a produced tensor) must leave the
// output unresolved -- only a true rank-0 initializer scalar may adopt the other operand's shape.
// Regression: Add(unresolved MatMul, bias initializer) resolved to the bias's own shape, poisoning
// every downstream rank in transformer imports (yonosplat_v2 attention blocks). ---
TEST(Passes, BinaryUnresolvedOperandStaysUnresolved) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {2, 8};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    TensorDesc si;
    si.name     = "s"; // runtime reshape target: keeps the Reshape output unresolved
    si.shape    = {3};
    si.isInput  = true;
    si.dtype    = DType::Int64;
    TensorId sh = g.addTensor(si);
    g.inputs    = {x, sh};

    TensorDesc bd;
    bd.name          = "bias";
    bd.shape         = {16};
    bd.isInitializer = true;
    TensorId   b     = g.addTensor(bd);
    HostBuffer hb;
    hb.resizeElems(16, DType::Float32);
    g.initializers[b] = hb;

    TensorId r = g.addTensor({.name = "r"}), y0 = g.addTensor({.name = "y0"});
    TensorId y1 = g.addTensor({.name = "y1"}), y2 = g.addTensor({.name = "y2"});
    Node     rs;
    rs.type    = OpType::Reshape;
    rs.name    = "reshape";
    rs.inputs  = {x, sh};
    rs.outputs = {r};
    Node add;
    add.type    = OpType::Add;
    add.name    = "add";
    add.inputs  = {r, b};
    add.outputs = {y0};
    Node gt;
    gt.type    = OpType::Greater;
    gt.name    = "gt";
    gt.inputs  = {r, b};
    gt.outputs = {y1};
    Node wh;
    wh.type    = OpType::Where;
    wh.name    = "where";
    wh.inputs  = {y1, r, b};
    wh.outputs = {y2};

    TensorDesc fd;
    fd.name          = "flat"; // const [-1] target: needs the (unresolved) input's element count
    fd.shape         = {1};
    fd.isInitializer = true;
    fd.dtype         = DType::Int64;
    TensorId   f     = g.addTensor(fd);
    HostBuffer fb;
    fb.resizeElems(1, DType::Int64);
    fb.i64()[0]       = -1;
    g.initializers[f] = fb;
    TensorId y3       = g.addTensor({.name = "y3"});
    Node     rs2;
    rs2.type    = OpType::Reshape;
    rs2.name    = "reshape_flat";
    rs2.inputs  = {r, f};
    rs2.outputs = {y3};
    g.nodes     = {rs, add, gt, wh, rs2};
    g.outputs   = {y0, y1, y2, y3};

    inferShapes(g, 1);
    EXPECT_TRUE(g.desc(r).shape.empty()) << "runtime reshape target must stay unresolved";
    EXPECT_TRUE(g.desc(y0).shape.empty()) << "Add with an unresolved operand must stay unresolved";
    EXPECT_TRUE(g.desc(y1).shape.empty()) << "Greater with an unresolved operand must stay unresolved";
    EXPECT_TRUE(g.desc(y2).shape.empty()) << "Where with an unresolved operand must stay unresolved";
    EXPECT_TRUE(g.desc(y3).shape.empty()) << "Reshape[-1] of an unresolved input must not fabricate [0]";
}

// --- inferShapes: Slice must not resolve while any present bound/axes/steps param is still
// runtime — copying the input shape fabricates an unsliced dim that a downstream Shape() fold
// freezes (the RoPE half-slice: [.,64] resolved full, corrected to [.,32] a round later). ---
TEST(Passes, SliceRuntimeBoundsStayUnresolved) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {2, 16, 64};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    TensorDesc si;
    si.name     = "bound"; // runtime scalar bound (e.g. head_dim/2 computed from Shape() arith)
    si.shape    = {1};
    si.dtype    = DType::Int64;
    si.isInput  = true;
    TensorId bd = g.addTensor(si);
    g.inputs    = {x, bd};
    auto vec    = [&](const char *nm, std::vector<int64_t> v) {
        TensorDesc t;
        t.name          = nm;
        t.shape         = {(int64_t) v.size()};
        t.dtype         = DType::Int64;
        t.isInitializer = true;
        TensorId   id   = g.addTensor(t);
        HostBuffer hb;
        hb.resizeElems(v.size(), DType::Int64);
        for (size_t i = 0; i < v.size(); ++i)
        {
            hb.i64()[i] = v[i];
        }
        g.initializers[id] = hb;
        return id;
    };
    TensorId ax = vec("axes", {2}), sp = vec("steps", {1});
    // Both bounds flow through a Reshape with a runtime target, so their descs stay empty —
    // the shape-computed start/end pattern (head_dim/2 from Shape() arithmetic).
    TensorId e0 = g.addTensor({.name = "e0"}), e1 = g.addTensor({.name = "e1"});
    Node     r0;
    r0.type    = OpType::Reshape;
    r0.name    = "r0";
    r0.inputs  = {bd, bd};
    r0.outputs = {e0};
    Node r1;
    r1.type    = OpType::Reshape;
    r1.name    = "r1";
    r1.inputs  = {bd, bd};
    r1.outputs = {e1};
    TensorId y = g.addTensor({.name = "y"});
    Node     sl;
    sl.type    = OpType::Slice;
    sl.name    = "slice";
    sl.inputs  = {x, e0, e1, ax, sp}; // starts AND ends still unresolved
    sl.outputs = {y};
    g.nodes    = {r0, r1, sl};
    g.outputs  = {y};
    inferShapes(g, 1);
    EXPECT_TRUE(g.desc(y).shape.empty()) << "Slice with runtime bounds must stay unresolved, not copy the input shape";
}
TEST(Passes, BinaryScalarInitializerBroadcasts) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {2, 8};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs   = {x};
    TensorDesc sd;
    sd.name          = "k"; // rank-0 scalar constant (empty shape, 1 payload element)
    sd.isInitializer = true;
    TensorId   k     = g.addTensor(sd);
    HostBuffer hb;
    hb.resizeElems(1, DType::Float32);
    hb.f32()[0]        = 2.f;
    g.initializers[k]  = hb;
    TensorId y         = g.addTensor({.name = "y"});
    g.desc(y).isOutput = true;
    Node m;
    m.type    = OpType::Binary;
    m.subOp   = (int) BinaryType::Mul;
    m.name    = "mul";
    m.inputs  = {x, k};
    m.outputs = {y};
    g.nodes   = {m};
    g.outputs = {y};
    inferShapes(g, 1);
    EXPECT_EQ(g.desc(y).shape, (Shape {2, 8}));
}

// --- Add of two rank-0 (shape []) scalar graph inputs runs live on the CPU (inputs are never
// const-folded), so its operands are materialized by bindInput. numElements([]) is 0, so sizing the
// host buffer by the shape count alone yields an empty (null-data) buffer whose read null-derefs;
// a rank-0 scalar must keep its one element so the operand read is in-bounds. ---
TEST(Ops, CpuAddTwoRank0ScalarInputs) {
    Graph      g;
    TensorDesc ai;
    ai.name    = "a";
    ai.shape   = {}; // rank-0 scalar input
    ai.isInput = true;
    TensorId a = g.addTensor(ai);
    TensorDesc bi;
    bi.name    = "b";
    bi.shape   = {}; // rank-0 scalar input
    bi.isInput = true;
    TensorId b = g.addTensor(bi);
    g.inputs   = {a, b};
    TensorId y = g.addTensor({.name = "y"});
    g.desc(y).isOutput = true;
    Node n;
    n.type    = OpType::Add;
    n.name    = "add";
    n.inputs  = {a, b};
    n.outputs = {y};
    g.nodes   = {n};
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);

    auto scalarIn = [](const char *nm, float v) {
        IOTensor in;
        in.name  = nm;
        in.shape = {}; // rank-0
        in.dtype = DType::Float32;
        in.data.resize(sizeof(float));
        std::memcpy(in.data.data(), &v, sizeof(float));
        return in;
    };
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({scalarIn("a", 3.0f), scalarIn("b", 4.0f)}, outs), Status::Ok);
    ASSERT_FALSE(outs.empty());
    const float *o = outs[0].f32();
    ASSERT_NE(o, nullptr);
    EXPECT_EQ(outs[0].shape, (Shape {}));
    EXPECT_NEAR(o[0], 7.0f, 1e-6f);
}

// --- A rank-0 (shape []) scalar output declared in a non-fp32 dtype goes through readbackOutput's
// conversion path, sized by the output element count. numElements([]) is 0, so counting the scalar by
// the shape alone would emit an empty output and silently drop the value; the scalar's one element
// must be counted so the converted output carries it. ---
TEST(Ops, CpuRank0ScalarOutputFp16Readback) {
    Graph      g;
    TensorDesc ai;
    ai.name    = "a";
    ai.shape   = {}; // rank-0 scalar input
    ai.isInput = true;
    TensorId a = g.addTensor(ai);
    TensorDesc bi;
    bi.name    = "b";
    bi.shape   = {}; // rank-0 scalar input
    bi.isInput = true;
    TensorId b = g.addTensor(bi);
    g.inputs   = {a, b};
    TensorDesc yo;
    yo.name     = "y";
    yo.shape    = {};                 // rank-0 scalar output
    yo.dtype    = DType::Float16;      // declared FLOAT16 -> readback converts from internal fp32
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node n;
    n.type    = OpType::Add;
    n.name    = "add";
    n.inputs  = {a, b};
    n.outputs = {y};
    g.nodes   = {n};
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);

    auto scalarIn = [](const char *nm, float v) {
        IOTensor in;
        in.name  = nm;
        in.shape = {};
        in.dtype = DType::Float32;
        in.data.resize(sizeof(float));
        std::memcpy(in.data.data(), &v, sizeof(float));
        return in;
    };
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({scalarIn("a", 2.0f), scalarIn("b", 5.0f)}, outs), Status::Ok);
    ASSERT_FALSE(outs.empty());
    EXPECT_EQ(outs[0].dtype, DType::Float16);
    ASSERT_EQ(outs[0].data.size(), sizeof(fp16_t)); // one fp16 element, not an empty buffer
    fp16_t h;
    std::memcpy(&h, outs[0].data.data(), sizeof(fp16_t));
    EXPECT_NEAR(halfToFloat(h), 7.0f, 1e-3f);
}

// --- pruneDeadInitializers: a folded chain's intermediate constants keep no payload; only the
// final constant a live node consumes survives (folded meshgrids/Cast-copied weights otherwise
// serialize gigabytes of orphans into the .vxm). ---
TEST(Passes, PruneDeadInitializers) {
    Graph g = makeRangeAddGraph(1.f, 9.f, 2.f, 4); // Range -> r (folds), Add(x, r) stays
    inferShapes(g, 1);
    constFold(g);
    TensorId r = g.nodes[0].inputs[1];
    ASSERT_TRUE(g.isInitializer(r));
    EXPECT_EQ(g.initializers.size(), 4u); // start/limit/delta + the folded r
    // start/limit/delta fed only the folded Range: their payloads are now orphaned
    pruneDeadInitializers(g);
    EXPECT_TRUE(g.isInitializer(r)) << "the live folded constant must survive";
    EXPECT_EQ(g.initializers.size(), 1u) << "orphaned fold inputs must be dropped";
}

// --- Range: int64 scalars with a negative delta emit an exact int64 vector. ---
TEST(Passes, RangeConstFoldInt64) {
    Graph g    = makeRangeAddGraph(0.f, 0.f, 1.f, 4); // scalars replaced with int64 below
    auto  seti = [&](TensorId id, int64_t v) {
        g.desc(id).dtype = DType::Int64;
        HostBuffer hb;
        hb.resizeElems(1, DType::Int64);
        hb.i64()[0]        = v;
        g.initializers[id] = hb;
    };
    seti(g.nodes[0].inputs[0], 10);
    seti(g.nodes[0].inputs[1], -3);
    seti(g.nodes[0].inputs[2], -3); // n = ceil(13/3) = 5
    TensorId r = g.nodes[0].outputs[0];
    inferShapes(g, 1);
    ASSERT_EQ(g.desc(r).shape, (Shape {5}));
    constFold(g);
    ASSERT_TRUE(g.isInitializer(r));
    EXPECT_EQ(g.desc(r).dtype, DType::Int64);
    const int64_t ref[5] = {10, 7, 4, 1, -2};
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(g.initializers[r].i64()[i], ref[i]) << "i=" << i;
    }
}

// --- Less/LessOrEqual over constant int64 shape tensors fold to an exact fp32 mask. The compare
// reads each operand through its own dtype, so values past fp32's 24-bit mantissa (2^25 vs 2^25+1
// would tie in fp32) still order correctly, and only <= counts the exact tie as 1. ---
TEST(Passes, LessConstFoldInt64) {
    Graph g;
    auto  addI64 = [&](const char *name, const Shape &shape, const std::vector<int64_t> &vals) {
        TensorDesc d;
        d.name          = name;
        d.shape         = shape;
        d.isInitializer = true;
        d.dtype         = DType::Int64;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) vals.size(), DType::Int64);
        for (size_t i = 0; i < vals.size(); ++i)
        {
            hb.i64()[i] = vals[i];
        }
        g.initializers[id] = hb;
        return id;
    };
    TensorId a  = addI64("a", {4}, {0, 1ll << 25, (1ll << 25) + 1, 7});
    TensorId b  = addI64("b", {1}, {1ll << 25});
    TensorId y0 = g.addTensor({.name = "lt"});
    TensorId y1 = g.addTensor({.name = "le"});
    Node     lt;
    lt.type    = OpType::Less;
    lt.name    = "less";
    lt.inputs  = {a, b};
    lt.outputs = {y0};
    Node le;
    le.type    = OpType::LessEqual;
    le.name    = "lessequal";
    le.inputs  = {a, b};
    le.outputs = {y1};
    g.nodes    = {lt, le};
    g.outputs  = {y0, y1};

    inferShapes(g, 1);
    ASSERT_EQ(g.desc(y0).shape, (Shape {4}));
    ASSERT_EQ(g.desc(y1).shape, (Shape {4}));
    constFold(g);
    EXPECT_TRUE(g.nodes.empty()) << "all-constant compares must fold away";
    ASSERT_TRUE(g.isInitializer(y0));
    ASSERT_TRUE(g.isInitializer(y1));
    EXPECT_EQ(g.desc(y0).dtype, DType::Float32); // canonical fp32 mask
    EXPECT_EQ(g.desc(y1).dtype, DType::Float32);
    const float refLt[4] = {1, 0, 0, 1}; // a < 2^25
    const float refLe[4] = {1, 1, 0, 1}; // a <= 2^25: the exact tie flips to 1
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(g.initializers[y0].f32()[i], refLt[i]) << "i=" << i;
        EXPECT_EQ(g.initializers[y1].f32()[i], refLe[i]) << "i=" << i;
    }
}

// --- TopK shape rule: both outputs get the input shape with the axis dim replaced by k (the const
// int64 input[1]), the values output carries the data dtype, and the indices output is Int64 — the
// declared dtype the session readback emits for a graph-output index tensor. ---
TEST(Passes, TopKInferShapes) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {2, 3, 5};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs   = {x};
    TensorDesc kd;
    kd.name          = "k";
    kd.shape         = {1};
    kd.isInitializer = true;
    kd.dtype         = DType::Int64;
    TensorId   kid   = g.addTensor(kd);
    HostBuffer hb;
    hb.resizeElems(1, DType::Int64);
    hb.i64()[0]         = 2;
    g.initializers[kid] = hb;
    TensorId v          = g.addTensor({.name = "v"});
    TensorId ix         = g.addTensor({.name = "i"});
    Node     n;
    n.type    = OpType::TopK;
    n.name    = "topk";
    n.inputs  = {x, kid};
    n.outputs = {v, ix};
    {
        Attr a;
        a.kind            = Attr::Int;
        a.i               = 1;
        n.attr.map["axis"] = a;
    }
    g.nodes   = {n};
    g.outputs = {v, ix};

    inferShapes(g, 1);
    EXPECT_EQ(g.desc(v).shape, (Shape {2, 2, 5}));
    EXPECT_EQ(g.desc(ix).shape, (Shape {2, 2, 5}));
    EXPECT_EQ(g.desc(v).dtype, DType::Float32);
    EXPECT_EQ(g.desc(ix).dtype, DType::Int64);
}

// A graph output declared FLOAT16 but produced (after Identity elimination) by a node whose inferred
// dtype is Float32 must keep FLOAT16 through runStandardPasses. Otherwise inferShapes overwrites the
// value_info dtype and the output-rewiring passes repoint the output to an fp32-default tensor, so the
// session readback emits fp32 bytes for a declared-fp16 (or uint8) output. Regression guard.
TEST(Passes, PreservesDeclaredOutputDtype) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 8};
    xi.isInput = true;
    xi.dtype   = DType::Float32;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorDesc td;
    td.name    = "t";
    td.shape   = {1, 8};
    td.dtype   = DType::Float32;
    TensorId t = g.addTensor(td);
    TensorDesc yd;
    yd.name     = "y";
    yd.shape    = {1, 8};
    yd.isOutput = true;
    yd.dtype    = DType::Float16; // ONNX value_info declares the output FLOAT16
    TensorId y  = g.addTensor(yd);
    Node     relu;
    relu.type    = OpType::Relu;
    relu.name    = "relu";
    relu.inputs  = {x};
    relu.outputs = {t};
    Node ident;
    ident.type    = OpType::Identity;
    ident.name    = "id";
    ident.inputs  = {t};
    ident.outputs = {y};
    g.nodes.push_back(relu);
    g.nodes.push_back(ident);
    g.outputs = {y};

    runStandardPasses(g);
    ASSERT_EQ(g.outputs.size(), 1u);
    ASSERT_NE(g.outputs[0], kNoTensor);
    EXPECT_EQ(g.desc(g.outputs[0]).dtype, DType::Float16);
}

// --- lowerGroupedConv: a general grouped Conv (1 < group < Cin) with a constant weight is rewritten
// to `group` group-1 Convs over per-group channel slices, joined by a Concat. Verifies the structure
// (no grouped Conv survives; group parts + slices + one concat appear) so the GPU parity path runs on
// the dense Conv kernel. A grouped Conv with a runtime (non-initializer) weight is left untouched for
// the CPU op. ---
TEST(Passes, LowerGroupedConvStructure) {
    auto build = [](bool constWeight) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {1, 8, 4, 4}; // Cin=8
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs.push_back(x);
        TensorDesc wi;
        wi.name          = "w";
        wi.shape         = {8, 2, 3, 3}; // group=4 -> inCg=2, outCg=2
        wi.dtype         = DType::Float32;
        wi.isInitializer = constWeight;
        TensorId w       = g.addTensor(wi);
        if (constWeight)
        {
            HostBuffer wb;
            wb.resizeElems(8 * 2 * 3 * 3, DType::Float32);
            g.initializers[w] = wb;
        }
        TensorDesc yd;
        yd.name     = "y";
        yd.shape    = {1, 8, 4, 4};
        yd.isOutput = true;
        TensorId   y = g.addTensor(yd);
        Node       c;
        c.type    = OpType::Conv;
        c.name    = "gconv";
        c.inputs  = {x, w};
        c.outputs = {y};
        Attr grp;
        grp.kind          = Attr::Int;
        grp.i             = 4;
        c.attr.map["group"]        = grp;
        Attr ks;
        ks.kind           = Attr::Ints;
        ks.ints           = {3, 3};
        c.attr.map["kernel_shape"] = ks;
        Attr pd;
        pd.kind           = Attr::Ints;
        pd.ints           = {1, 1, 1, 1};
        c.attr.map["pads"]         = pd;
        g.nodes.push_back(c);
        g.outputs = {y};
        return g;
    };

    // constant weight -> lowered
    {
        Graph g = build(true);
        runStandardPasses(g);
        int convs = 0, slices = 0, concats = 0, grouped = 0;
        for (const Node &n: g.nodes)
        {
            if (n.type == OpType::Conv)
            {
                convs++;
                if (n.attr.geti("group", 1) > 1)
                {
                    grouped++;
                }
            } else if (n.type == OpType::Slice)
            {
                slices++;
            } else if (n.type == OpType::Concat)
            {
                concats++;
            }
        }
        EXPECT_EQ(grouped, 0);  // no grouped Conv survives
        EXPECT_EQ(convs, 4);    // one group-1 Conv per group
        EXPECT_EQ(slices, 4);   // one channel Slice per group
        EXPECT_EQ(concats, 1);  // parts rejoined once
    }
    // runtime weight -> untouched (stays a grouped Conv for the CPU op)
    {
        Graph g = build(false);
        runStandardPasses(g);
        int grouped = 0;
        for (const Node &n: g.nodes)
        {
            if (n.type == OpType::Conv && n.attr.geti("group", 1) > 1)
            {
                grouped++;
            }
        }
        EXPECT_EQ(grouped, 1);
    }
}

// --- Dropout: inference-mode Dropout is an identity on its data input. eliminateDropout removes a
// Dropout whose training_mode input is absent or a constant false and whose mask output is absent
// or unconsumed, rewiring consumers to the producer. A Dropout with a consumed mask or a
// non-constant/true training_mode survives untouched (the pass never fabricates a mask; the node
// then fails backend planning as unsupported). ---
namespace {

    // Registers a float activation tensor named `name`.
    TensorId addAct(Graph &g, const char *name) {
        TensorDesc d;
        d.name  = name;
        d.shape = {1, 8};
        return g.addTensor(d);
    }

    // Registers a rank-0 scalar initializer of dtype `dt` holding `v` (BOOL imports as UInt8 0/1).
    TensorId addScalarInit(Graph &g, const char *name, DType dt, double v) {
        TensorDesc d;
        d.name          = name;
        d.isInitializer = true;
        d.dtype         = dt;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems(1, dt);
        switch (dt)
        {
            case DType::Float32:
                hb.f32()[0] = (float) v;
                break;
            case DType::Int64:
                hb.i64()[0] = (int64_t) v;
                break;
            default:
                hb.bytes[0] = (uint8_t) v;
                break;
        }
        g.initializers[id] = hb;
        return id;
    }

} // namespace

namespace {

    // Single-Conv graph over a 1x2x6x6 input: weight [2,2,3,3] with a fixed non-uniform pattern.
    // `attr` selects the padding form (auto_pad vs explicit pads) under test.
    Graph convPadGraph(const Attributes &attr) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {1, 2, 6, 6};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};
        TensorDesc wi;
        wi.name          = "w";
        wi.shape         = {2, 2, 3, 3};
        wi.isInitializer = true;
        TensorId   w     = g.addTensor(wi);
        HostBuffer wb;
        wb.resizeElems(36, DType::Float32);
        for (int i = 0; i < 36; ++i)
        {
            wb.f32()[i] = (float) (i % 5) - 2.f;
        }
        g.initializers[w] = wb;
        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     c;
        c.type    = OpType::Conv;
        c.name    = "conv";
        c.inputs  = {x, w};
        c.outputs = {y};
        c.attr    = attr;
        g.nodes.push_back(c);
        g.outputs = {y};
        return g;
    }

    // Runs an already-passed graph on the CPU backend with input x = i * 0.25 - 4.
    std::vector<float> runConvPadGraph(Graph g) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = "x";
        in.shape = {1, 2, 6, 6};
        in.dtype = DType::Float32;
        in.data.resize(72 * 4);
        for (int i = 0; i < 72; ++i)
        {
            reinterpret_cast<float *>(in.data.data())[i] = (float) i * 0.25f - 4.f;
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        if (outs.empty())
        {
            return {};
        }
        const float *o = outs[0].f32();
        return std::vector<float>(o, o + numElements(outs[0].shape));
    }

} // namespace

TEST(Passes, DropoutInferenceModeEliminated) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 8};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs   = {x};
    TensorId r     = addAct(g, "r");
    TensorId d     = addAct(g, "d");
    TensorId m     = addAct(g, "m"); // mask requested but never consumed
    TensorId ratio = addScalarInit(g, "ratio", DType::Float32, 0.5);
    TensorId tm    = addScalarInit(g, "tm", DType::UInt8, 0); // constant-false training_mode
    TensorId k     = addScalarInit(g, "k", DType::Float32, 2.0);
    TensorId y     = addAct(g, "y");
    g.desc(y).isOutput = true;
    g.outputs          = {y};
    Node relu;
    relu.type    = OpType::Relu;
    relu.name    = "relu";
    relu.inputs  = {x};
    relu.outputs = {r};
    Node dp;
    dp.type    = OpType::Dropout;
    dp.name    = "dp";
    dp.inputs  = {r, ratio, tm};
    dp.outputs = {d, m};
    Node mul;
    mul.type    = OpType::Binary;
    mul.subOp   = (int) BinaryType::Mul;
    mul.name    = "mul";
    mul.inputs  = {d, k};
    mul.outputs = {y};
    g.nodes = {relu, dp, mul};

    eliminateDropout(g);
    ASSERT_EQ(g.nodes.size(), 2u);
    EXPECT_EQ(g.nodes[0].type, OpType::Relu);
    EXPECT_EQ(g.nodes[1].type, OpType::Binary);
    EXPECT_EQ(g.nodes[1].inputs[0], r) << "the Mul must read the Relu output directly";
}

TEST(Passes, DropoutGraphOutputRewired) {
    // Opset-7 form: data input and output only, the output is a graph output.
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 8};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs   = {x};
    TensorId r = addAct(g, "r");
    TensorId d = addAct(g, "d");
    g.desc(d).isOutput = true;
    g.outputs          = {d};
    Node relu;
    relu.type    = OpType::Relu;
    relu.name    = "relu";
    relu.inputs  = {x};
    relu.outputs = {r};
    Node dp;
    dp.type    = OpType::Dropout;
    dp.name    = "dp";
    dp.inputs  = {r};
    dp.outputs = {d};
    g.nodes = {relu, dp};

    eliminateDropout(g);
    ASSERT_EQ(g.nodes.size(), 1u);
    EXPECT_EQ(g.nodes[0].type, OpType::Relu);
    ASSERT_EQ(g.outputs.size(), 1u);
    EXPECT_EQ(g.outputs[0], r) << "the graph output must be rewired to the Relu output";
}

TEST(Passes, DropoutConsumedMaskKept) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 8};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs   = {x};
    TensorId d = addAct(g, "d");
    TensorId m = addAct(g, "m");
    TensorId k = addScalarInit(g, "k", DType::Float32, 2.0);
    TensorId y = addAct(g, "y");
    g.desc(d).isOutput = true;
    g.desc(y).isOutput = true;
    g.outputs          = {d, y};
    Node dp;
    dp.type    = OpType::Dropout;
    dp.name    = "dp";
    dp.inputs  = {x};
    dp.outputs = {d, m};
    Node mul; // consumes the mask: the Dropout must not be eliminated (no fabricated mask)
    mul.type    = OpType::Binary;
    mul.subOp   = (int) BinaryType::Mul;
    mul.name    = "mul";
    mul.inputs  = {m, k};
    mul.outputs = {y};
    g.nodes = {dp, mul};

    eliminateDropout(g);
    ASSERT_EQ(g.nodes.size(), 2u);
    EXPECT_EQ(g.nodes[0].type, OpType::Dropout);
}

TEST(Passes, DropoutTrainingModeConstTrueKept) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 8};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs   = {x};
    TensorId d     = addAct(g, "d");
    TensorId ratio = addScalarInit(g, "ratio", DType::Float32, 0.5);
    TensorId tm    = addScalarInit(g, "tm", DType::UInt8, 1); // constant-true training_mode
    g.desc(d).isOutput = true;
    g.outputs          = {d};
    Node dp;
    dp.type    = OpType::Dropout;
    dp.name    = "dp";
    dp.inputs  = {x, ratio, tm};
    dp.outputs = {d};
    g.nodes = {dp};

    eliminateDropout(g);
    ASSERT_EQ(g.nodes.size(), 1u);
    EXPECT_EQ(g.nodes[0].type, OpType::Dropout);
}

TEST(Passes, DropoutTrainingModeRuntimeKept) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 8};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    TensorDesc ti; // training_mode fed at run time: not provably false
    ti.name     = "tm";
    ti.isInput  = true;
    ti.dtype    = DType::UInt8;
    TensorId tm = g.addTensor(ti);
    g.inputs    = {x, tm};
    TensorId d     = addAct(g, "d");
    TensorId ratio = addScalarInit(g, "ratio", DType::Float32, 0.5);
    g.desc(d).isOutput = true;
    g.outputs          = {d};
    Node dp;
    dp.type    = OpType::Dropout;
    dp.name    = "dp";
    dp.inputs  = {x, ratio, tm};
    dp.outputs = {d};
    g.nodes = {dp};

    eliminateDropout(g);
    ASSERT_EQ(g.nodes.size(), 1u);
    EXPECT_EQ(g.nodes[0].type, OpType::Dropout);
}

TEST(Passes, DropoutConstantNodeTrainingModeFalse) {
    // training_mode produced by a not-yet-folded Constant node (the pass runs before constFold).
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 8};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs   = {x};
    TensorId tm    = addAct(g, "tm");
    TensorId d     = addAct(g, "d");
    TensorId ratio = addScalarInit(g, "ratio", DType::Float32, 0.5);
    g.desc(d).isOutput = true;
    g.outputs          = {d};
    Node cn;
    cn.type    = OpType::Constant;
    cn.name    = "tm_const";
    cn.outputs = {tm};
    Attr v;
    v.kind = Attr::Ints;
    v.ints = {0};
    cn.attr.map["value"] = v;
    Node dp;
    dp.type    = OpType::Dropout;
    dp.name    = "dp";
    dp.inputs  = {x, ratio, tm};
    dp.outputs = {d};
    g.nodes = {cn, dp};

    eliminateDropout(g);
    ASSERT_EQ(g.nodes.size(), 1u);
    EXPECT_EQ(g.nodes[0].type, OpType::Constant) << "only the Dropout is removed; DCE owns the dead Constant";
    ASSERT_EQ(g.outputs.size(), 1u);
    EXPECT_EQ(g.outputs[0], x);
}

TEST(Passes, DropoutRunStandardPassesRewires) {
    EXPECT_EQ(opTypeFromOnnx("Dropout"), OpType::Dropout);
    // x -> MatMul(w) -> Dropout -> Softmax -> y imports to MatMul -> Softmax directly connected.
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 8};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs   = {x};
    TensorDesc wd;
    wd.name          = "w";
    wd.shape         = {8, 8};
    wd.isInitializer = true;
    TensorId   w     = g.addTensor(wd);
    HostBuffer hb;
    hb.resizeElems(64, DType::Float32);
    g.initializers[w] = hb;
    TensorId t        = addAct(g, "t");
    TensorId d        = addAct(g, "d");
    TensorId y        = addAct(g, "y");
    g.desc(y).isOutput = true;
    g.outputs          = {y};
    Node mm;
    mm.type    = OpType::MatMul;
    mm.name    = "mm";
    mm.inputs  = {x, w};
    mm.outputs = {t};
    Node dp;
    dp.type    = OpType::Dropout;
    dp.name    = "dp";
    dp.inputs  = {t};
    dp.outputs = {d};
    Node sm;
    sm.type    = OpType::Softmax;
    sm.name    = "sm";
    sm.inputs  = {d};
    sm.outputs = {y};
    g.nodes = {mm, dp, sm};

    runStandardPasses(g);
    const Node *matmul = nullptr, *softmax = nullptr;
    for (const Node &n: g.nodes)
    {
        EXPECT_NE(n.type, OpType::Dropout);
        if (n.type == OpType::MatMul)
        {
            matmul = &n;
        }
        if (n.type == OpType::Softmax)
        {
            softmax = &n;
        }
    }
    ASSERT_NE(matmul, nullptr);
    ASSERT_NE(softmax, nullptr);
    EXPECT_EQ(softmax->inputs[0], matmul->outputs[0]) << "Softmax must read the MatMul output directly";
}

// --- lowerConv x auto_pad parity: a SAME_UPPER conv is skipped by lowerConv (stays Conv, resolved
// through convGeom), while the same conv with the explicit resolved pads {1,1,1,1} lowers to
// ConvGemm; both paths must produce identical values. ---
TEST(Passes, LowerConvSkipsAutoPadAndMatchesExplicit) {
    Attributes same;
    same.map["auto_pad"] = str("SAME_UPPER");
    Attributes expl;
    expl.map["pads"] = ints({1, 1, 1, 1}); // the resolved SAME_UPPER pads for 3x3/s1
    PassOptions opt;
    opt.lowerConv = true;

    Graph gs = convPadGraph(same);
    runStandardPasses(gs, opt);
    ASSERT_EQ(gs.nodes.size(), 1u);
    EXPECT_EQ(gs.nodes[0].type, OpType::Conv) << "lowerConv must skip auto_pad convs";

    Graph ge = convPadGraph(expl);
    runStandardPasses(ge, opt);
    ASSERT_EQ(ge.nodes.size(), 1u);
    EXPECT_EQ(ge.nodes[0].type, OpType::ConvGemm) << "explicit-pad twin must lower";

    std::vector<float> outSame = runConvPadGraph(std::move(gs));
    std::vector<float> outExpl = runConvPadGraph(std::move(ge));
    ASSERT_EQ(outSame.size(), (size_t) (2 * 6 * 6));
    expectNear(outSame, outExpl, 0.f); // identical fp32 loop order: bit-equal
}

// --- fuseDwPw x auto_pad: the fused node inherits the depthwise conv's attrs, auto_pad included,
// so the fused and unfused paths must resolve the SAME_UPPER pads identically. ---
TEST(Passes, FuseDwPwAutoPadSameUpper) {
    auto build = [] {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {1, 2, 5, 5};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};
        TensorDesc di;
        di.name          = "dw";
        di.shape         = {2, 1, 3, 3};
        di.isInitializer = true;
        TensorId   dw    = g.addTensor(di);
        HostBuffer db;
        db.resizeElems(18, DType::Float32);
        for (int i = 0; i < 18; ++i)
        {
            db.f32()[i] = (float) (i % 3) - 1.f;
        }
        g.initializers[dw] = db;
        TensorDesc pi;
        pi.name          = "pw";
        pi.shape         = {2, 2, 1, 1};
        pi.isInitializer = true;
        TensorId   pw    = g.addTensor(pi);
        HostBuffer pb;
        pb.resizeElems(4, DType::Float32);
        for (int i = 0; i < 4; ++i)
        {
            pb.f32()[i] = (float) i - 1.5f;
        }
        g.initializers[pw] = pb;
        TensorDesc t0;
        t0.name     = "t";
        TensorId t  = g.addTensor(t0);
        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     d;
        d.type    = OpType::Conv;
        d.name    = "dconv";
        d.inputs  = {x, dw};
        d.outputs = {t};
        d.attr.map["auto_pad"] = str("SAME_UPPER");
        {
            Attr a;
            a.kind             = Attr::Int;
            a.i                = 2;
            d.attr.map["group"] = a;
        }
        Node p;
        p.type    = OpType::Conv;
        p.name    = "pconv";
        p.inputs  = {t, pw};
        p.outputs = {y};
        g.nodes.push_back(d);
        g.nodes.push_back(p);
        g.outputs = {y};
        return g;
    };
    auto run = [](Graph g) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        IOTensor in;
        in.name  = "x";
        in.shape = {1, 2, 5, 5};
        in.dtype = DType::Float32;
        in.data.resize(50 * 4);
        for (int i = 0; i < 50; ++i)
        {
            reinterpret_cast<float *>(in.data.data())[i] = (float) i * 0.5f - 12.f;
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        const float *o = outs[0].f32();
        return std::vector<float>(o, o + numElements(outs[0].shape));
    };

    Graph       gf = build();
    PassOptions opt;
    opt.fuseDwPw = true;
    runStandardPasses(gf, opt);
    bool fused = false;
    for (const Node &n: gf.nodes)
    {
        fused = fused || n.type == OpType::FusedDwPw;
    }
    ASSERT_TRUE(fused) << "SAME_UPPER depthwise + 1x1 project must still fuse";
    ASSERT_EQ(gf.desc(gf.outputs[0]).shape, (Shape {1, 2, 5, 5})); // SAME keeps the extent

    Graph gu = build();
    runStandardPasses(gu, {}); // unfused reference
    std::vector<float> outFused   = run(std::move(gf));
    std::vector<float> outUnfused = run(std::move(gu));
    ASSERT_EQ(outFused.size(), (size_t) 50);
    expectNear(outFused, outUnfused, 0.f); // identical fp32 loop order: bit-equal
}

// --- InstanceNormalization: imported as OpType::InstanceNorm, then lowerInstanceNorm decomposes it
// into existing ops (spatial ReduceMean, Sub, Mul, Add-eps, Sqrt, Div, per-channel Mul/Add) so the
// normalization runs on ops both backends already implement — there is no InstanceNorm kernel. A
// node whose scale/B are not fp32 initializers stays opaque and fails planning as unsupported. ---
namespace {

    // Single-InstanceNormalization graph: input x of `xshape`, 1-D [C] fp32 scale/B initializers,
    // `epsilon` attribute. `scaleIsInput` swaps the scale initializer for a runtime graph input.
    Graph instanceNormGraph(const std::vector<int64_t> &xshape, const std::vector<float> &scale, const std::vector<float> &bias, float eps, bool scaleIsInput = false) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = xshape;
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};
        auto addInit = [&](const char *name, const std::vector<float> &v) {
            TensorDesc d;
            d.name          = name;
            d.shape         = {(int64_t) v.size()};
            d.isInitializer = true;
            TensorId   id   = g.addTensor(d);
            HostBuffer hb;
            hb.resizeElems(v.size(), DType::Float32);
            for (size_t i = 0; i < v.size(); ++i)
            {
                hb.f32()[i] = v[i];
            }
            g.initializers[id] = hb;
            return id;
        };
        TensorId sc;
        if (scaleIsInput)
        {
            TensorDesc d;
            d.name    = "scale";
            d.shape   = {(int64_t) scale.size()};
            d.isInput = true;
            sc        = g.addTensor(d);
            g.inputs.push_back(sc);
        } else
        {
            sc = addInit("scale", scale);
        }
        TensorId   bi = addInit("bias", bias);
        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     n;
        n.type    = OpType::InstanceNorm;
        n.name    = "inorm";
        n.inputs  = {x, sc, bi};
        n.outputs = {y};
        Attr e;
        e.kind                = Attr::Float;
        e.f                   = eps;
        n.attr.map["epsilon"] = e;
        g.nodes.push_back(n);
        g.outputs = {y};
        return g;
    }

    // Reference in double: per (n, c) over the trailing spatial dims,
    // y = scale[c] * (x - mean) / sqrt(var + eps) + bias[c] with the population (biased) variance.
    std::vector<float> instanceNormRef(const std::vector<int64_t> &xshape, const std::vector<float> &x, const std::vector<float> &scale, const std::vector<float> &bias, double eps) {
        int64_t N = xshape[0], C = xshape[1], S = 1;
        for (size_t d = 2; d < xshape.size(); ++d)
        {
            S *= xshape[d];
        }
        std::vector<float> y(x.size());
        for (int64_t n = 0; n < N; ++n)
        {
            for (int64_t c = 0; c < C; ++c)
            {
                const float *xs = x.data() + (n * C + c) * S;
                double       m = 0, v = 0;
                for (int64_t i = 0; i < S; ++i)
                {
                    m += xs[i];
                }
                m /= (double) S;
                for (int64_t i = 0; i < S; ++i)
                {
                    v += (xs[i] - m) * (xs[i] - m);
                }
                v /= (double) S;
                double inv = 1.0 / std::sqrt(v + eps);
                float *ys  = y.data() + (n * C + c) * S;
                for (int64_t i = 0; i < S; ++i)
                {
                    ys[i] = (float) ((double) scale[c] * (xs[i] - m) * inv + (double) bias[c]);
                }
            }
        }
        return y;
    }

    // Runs an InstanceNorm graph on the CPU backend (Session::plan applies the standard passes).
    std::vector<float> runInstanceNorm(Graph g, const std::vector<int64_t> &xshape, const std::vector<float> &xdata) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = "x";
        in.shape = xshape;
        in.dtype = DType::Float32;
        in.data.resize(xdata.size() * 4);
        for (size_t i = 0; i < xdata.size(); ++i)
        {
            reinterpret_cast<float *>(in.data.data())[i] = xdata[i];
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        if (outs.empty())
        {
            return {};
        }
        const float *o = outs[0].f32();
        return std::vector<float>(o, o + numElements(outs[0].shape));
    }

    // Deterministic pseudo-random fill in [-2, 2).
    std::vector<float> pseudoRandom(size_t n) {
        std::vector<float> v(n);
        uint32_t           s = 0x9e3779b9u;
        for (size_t i = 0; i < n; ++i)
        {
            s    = s * 1664525u + 1013904223u;
            v[i] = (float) (s >> 8) / (float) (1u << 24) * 4.f - 2.f;
        }
        return v;
    }

} // namespace

TEST(Ops, InstanceNormNchwMatchesReference) {
    std::vector<int64_t> xshape {2, 3, 8, 8};
    std::vector<float>   x     = pseudoRandom(2 * 3 * 8 * 8);
    std::vector<float>   scale {0.5f, 1.5f, -2.0f};
    std::vector<float>   bias {0.1f, -0.3f, 0.7f};
    std::vector<float>   got = runInstanceNorm(instanceNormGraph(xshape, scale, bias, 1e-5f), xshape, x);
    expectNear(got, instanceNormRef(xshape, x, scale, bias, 1e-5), 1e-5f);
}

TEST(Ops, InstanceNormRank3NormalizesOverL) {
    // [N,C,L] (whisper-class): the mean/variance reduce over L only. The non-default epsilon
    // separates the attribute-read path from the 1e-5 fallback.
    std::vector<int64_t> xshape {2, 4, 16};
    std::vector<float>   x     = pseudoRandom(2 * 4 * 16);
    std::vector<float>   scale {1.0f, 0.25f, -1.5f, 3.0f};
    std::vector<float>   bias {0.0f, 0.5f, -0.25f, 1.0f};
    std::vector<float>   got = runInstanceNorm(instanceNormGraph(xshape, scale, bias, 1e-3f), xshape, x);
    expectNear(got, instanceNormRef(xshape, x, scale, bias, 1e-3), 1e-5f);
}

TEST(Passes, InstanceNormLoweredToPerChannelOps) {
    EXPECT_EQ(opTypeFromOnnx("InstanceNormalization"), OpType::InstanceNorm);
    Graph       g = instanceNormGraph({1, 3, 8, 8}, {1.f, 2.f, 3.f}, {0.f, 1.f, -1.f}, 1e-5f);
    TensorId    y = g.outputs[0];
    PassOptions opt;
    opt.fusePointwiseChains = false; // keep the decomposition visible for the structure check
    runStandardPasses(g, opt);
    int gap = 0, sub = 0, mul = 0, div = 0, sqrt = 0, add = 0;
    for (const Node &n: g.nodes)
    {
        EXPECT_NE(n.type, OpType::InstanceNorm) << "the pass must lower every eligible node";
        gap += n.type == OpType::GlobalAvgPool;
        add += n.type == OpType::Add;
        sub += n.type == OpType::Binary && (BinaryType) n.subOp == BinaryType::Sub;
        mul += n.type == OpType::Binary && (BinaryType) n.subOp == BinaryType::Mul;
        div += n.type == OpType::Binary && (BinaryType) n.subOp == BinaryType::Div;
        sqrt += n.type == OpType::Unary && (UnaryType) n.subOp == UnaryType::Sqrt;
    }
    EXPECT_EQ(gap, 2) << "both rank-4 spatial ReduceMeans must recover as GlobalAvgPool";
    EXPECT_EQ(sub, 1);
    EXPECT_EQ(mul, 2) << "squared-diff Mul + per-channel scale Mul";
    EXPECT_EQ(div, 1);
    EXPECT_EQ(sqrt, 1);
    EXPECT_EQ(add, 2) << "epsilon Add + per-channel bias Add";
    ASSERT_EQ(g.outputs.size(), 1u);
    EXPECT_EQ(g.outputs[0], y) << "consumers keep reading the InstanceNormalization output tensor";
}

TEST(Passes, InstanceNormRuntimeScaleStaysOpaque) {
    // A runtime (non-initializer) scale cannot be reshaped at import: the node keeps its opaque op
    // (planning then rejects it as unsupported — the pass never fabricates parameters).
    Graph g = instanceNormGraph({1, 3, 8, 8}, {1.f, 1.f, 1.f}, {0.f, 0.f, 0.f}, 1e-5f, /*scaleIsInput=*/true);
    runStandardPasses(g);
    int kept = 0;
    for (const Node &n: g.nodes)
    {
        kept += n.type == OpType::InstanceNorm;
    }
    EXPECT_EQ(kept, 1);
}

// --- dequantizeGraph: QDQ-format quantized checkpoints collapse to plain float graphs at import.
// A DequantizeLinear over all-initializer inputs folds to an fp32 initializer computed in double
// as (x - zero_point) * scale, per-tensor or per-axis (axis attribute); a QuantizeLinear ->
// DequantizeLinear activation sandwich with matching scale/zero_point drops, consumers rewired to
// the float producer (dequantized execution skips the sandwich's re-rounding by definition).
// Residual activation q/dq nodes (mismatched sandwiches, boundary q/dq) stay in place. ---
namespace {

    // Registers a named tensor with no shape (an activation whose shape inference has not run).
    TensorId addUnshaped(Graph &g, const char *name) {
        TensorDesc d;
        d.name = name;
        return g.addTensor(d);
    }

    // Registers a Float32 initializer holding `v` (integer payloads arrive widened to fp32 by the
    // importer, so quantized-value tensors are built the same way here).
    TensorId addFloatInit(Graph &g, const char *name, const Shape &shape, const std::vector<float> &v) {
        TensorDesc d;
        d.name          = name;
        d.shape         = shape;
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) v.size(), DType::Float32);
        for (size_t i = 0; i < v.size(); ++i)
        {
            hb.f32()[i] = v[i];
        }
        g.initializers[id] = hb;
        return id;
    }

    // Registers a quantize zero_point initializer: fp32 host storage (as the importer widens int8/
    // uint8 payloads) but a descriptor labeled Int8/UInt8, so the dequantize pass can recover the
    // quantize dtype's saturation range from it -- exactly what the ONNX importer produces.
    TensorId addZeroPointInit(Graph &g, const char *name, const Shape &shape, const std::vector<float> &v, DType quantDt) {
        TensorId id      = addFloatInit(g, name, shape, v);
        g.desc(id).dtype = quantDt;
        return id;
    }

    // DequantizeLinear(w, s, zp?) -> wd feeding Add(x, wd) -> y, over an input x of `shape`.
    // `axis` adds the per-axis attribute when `withAxis` is set. Returns the DQ output id via *wd.
    Graph dequantWeightGraph(const Shape &shape, const std::vector<float> &wq, const Shape &sshape, const std::vector<float> &scale, const std::vector<float> &zp, bool withAxis, int64_t axis, TensorId *wdOut) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = shape;
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};
        TensorId w = addFloatInit(g, "w", shape, wq);
        TensorId s = addFloatInit(g, "s", sshape, scale);
        TensorId z = zp.empty() ? kNoTensor : addFloatInit(g, "z", sshape, zp);
        TensorId wd = addUnshaped(g, "wd");
        TensorId y  = addUnshaped(g, "y");
        g.desc(y).isOutput = true;
        g.outputs          = {y};
        Node dq;
        dq.type    = OpType::DequantizeLinear;
        dq.name    = "dq";
        dq.inputs  = z == kNoTensor ? std::vector<TensorId> {w, s} : std::vector<TensorId> {w, s, z};
        dq.outputs = {wd};
        if (withAxis)
        {
            Attr a;
            a.kind              = Attr::Int;
            a.i                 = axis;
            dq.attr.map["axis"] = a;
        }
        Node add;
        add.type    = OpType::Binary;
        add.subOp   = (int) BinaryType::Add;
        add.name    = "add";
        add.inputs  = {x, wd};
        add.outputs = {y};
        g.nodes = {dq, add};
        if (wdOut)
        {
            *wdOut = wd;
        }
        return g;
    }

    // x -> Relu -> QuantizeLinear(s, zp) -> DequantizeLinear(s2, zp2) -> Mul(k) -> y. `shared`
    // reuses the Q node's scale/zp ids on the DQ; otherwise the DQ gets its own initializers
    // holding `dqScale`/`dqZp` (value matching is then what decides the collapse). `quantDt` labels
    // the zero_point dtype (Int8/UInt8), the range the collapse's inserted saturation Clip uses.
    Graph qdqSandwichGraph(bool shared, float qScale, float qZp, float dqScale, float dqZp, bool dqHasZp = true, DType quantDt = DType::Int8) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {1, 8};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};
        TensorId r  = addUnshaped(g, "r");
        TensorId q  = addUnshaped(g, "q");
        TensorId d  = addUnshaped(g, "d");
        TensorId y  = addUnshaped(g, "y");
        TensorId s1 = addFloatInit(g, "s1", {}, {qScale});
        TensorId z1 = addZeroPointInit(g, "z1", {}, {qZp}, quantDt);
        TensorId s2 = shared ? s1 : addFloatInit(g, "s2", {}, {dqScale});
        TensorId z2 = shared ? z1 : (dqHasZp ? addZeroPointInit(g, "z2", {}, {dqZp}, quantDt) : kNoTensor);
        TensorId k  = addFloatInit(g, "k", {1}, {2.f});
        g.desc(y).isOutput = true;
        g.outputs          = {y};
        Node relu;
        relu.type    = OpType::Relu;
        relu.name    = "relu";
        relu.inputs  = {x};
        relu.outputs = {r};
        Node qn;
        qn.type    = OpType::QuantizeLinear;
        qn.name    = "q";
        qn.inputs  = {r, s1, z1};
        qn.outputs = {q};
        Node dn;
        dn.type    = OpType::DequantizeLinear;
        dn.name    = "dq";
        dn.inputs  = z2 == kNoTensor ? std::vector<TensorId> {q, s2} : std::vector<TensorId> {q, s2, z2};
        dn.outputs = {d};
        Node mul;
        mul.type    = OpType::Binary;
        mul.subOp   = (int) BinaryType::Mul;
        mul.name    = "mul";
        mul.inputs  = {d, k};
        mul.outputs = {y};
        g.nodes = {relu, qn, dn, mul};
        return g;
    }

    // Runs a [1,8]-input graph on the CPU backend and returns the fp32 output values.
    std::vector<float> runQdqGraph(Graph g, const std::vector<float> &xd) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = "x";
        in.shape = {1, 8};
        in.dtype = DType::Float32;
        in.data.resize(xd.size() * 4);
        for (size_t i = 0; i < xd.size(); ++i)
        {
            reinterpret_cast<float *>(in.data.data())[i] = xd[i];
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        if (outs.empty())
        {
            return {};
        }
        const float *o = outs[0].f32();
        return std::vector<float>(o, o + numElements(outs[0].shape));
    }

    // Number of QuantizeLinear/DequantizeLinear nodes left in the graph.
    int countQdq(const Graph &g) {
        int n = 0;
        for (const Node &nd: g.nodes)
        {
            n += nd.type == OpType::QuantizeLinear || nd.type == OpType::DequantizeLinear;
        }
        return n;
    }

} // namespace

TEST(Passes, DequantizeLinearPerTensorFoldsExact) {
    std::vector<float> wq {-128.f, -1.f, 0.f, 1.f, 100.f, 127.f}; // int8 values, widened
    TensorId           wd = kNoTensor;
    Graph              g  = dequantWeightGraph({2, 3}, wq, {}, {0.0123f}, {-5.f}, false, 0, &wd);
    dequantizeGraph(g);
    ASSERT_EQ(g.nodes.size(), 1u);
    EXPECT_EQ(g.nodes[0].type, OpType::Binary);
    ASSERT_TRUE(g.isInitializer(wd));
    EXPECT_EQ(g.desc(wd).dtype, DType::Float32);
    EXPECT_EQ(g.desc(wd).shape, (Shape {2, 3}));
    const float *f = g.initializers.at(wd).f32();
    for (size_t i = 0; i < wq.size(); ++i)
    {
        EXPECT_EQ(f[i], (float) (((double) wq[i] - (double) -5.f) * (double) 0.0123f)) << "i=" << i;
    }
    dequantizeGraph(g); // idempotent: the folded graph is a fixpoint
    ASSERT_EQ(g.nodes.size(), 1u);
}

TEST(Passes, DequantizeLinearPerAxisFoldsExact) {
    // Conv-weight form: [8,4,3,3] with axis=0 scales/zero-points of length 8.
    const Shape        shape {8, 4, 3, 3};
    std::vector<float> wq(8 * 4 * 3 * 3);
    for (size_t i = 0; i < wq.size(); ++i)
    {
        wq[i] = (float) ((int) ((i * 7) % 256) - 128);
    }
    std::vector<float> scale(8), zp(8);
    for (int c = 0; c < 8; ++c)
    {
        scale[c] = 0.001f * (float) (c + 1);
        zp[c]    = (float) (c % 5 - 2);
    }
    TensorId wd = kNoTensor;
    Graph    g  = dequantWeightGraph(shape, wq, {8}, scale, zp, true, 0, &wd);
    dequantizeGraph(g);
    ASSERT_EQ(g.nodes.size(), 1u);
    ASSERT_TRUE(g.isInitializer(wd));
    EXPECT_EQ(g.desc(wd).shape, shape);
    const float *f = g.initializers.at(wd).f32();
    for (size_t i = 0; i < wq.size(); ++i)
    {
        size_t c = i / 36; // inner stride = 4*3*3 / ... = C/g*Kh*Kw = 36
        ASSERT_EQ(f[i], (float) (((double) wq[i] - (double) zp[c]) * (double) scale[c])) << "i=" << i;
    }
}

TEST(Passes, DequantizeLinearNegativeAxisInnerStride) {
    // axis=-1 on a [2,6] tensor: normalizes to axis 1, channel stride 1.
    std::vector<float> wq {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
    std::vector<float> scale {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<float> zp {0.f, -1.f, 1.f, -2.f, 2.f, 3.f};
    TensorId           wd = kNoTensor;
    Graph              g  = dequantWeightGraph({2, 6}, wq, {6}, scale, zp, true, -1, &wd);
    dequantizeGraph(g);
    ASSERT_EQ(g.nodes.size(), 1u);
    ASSERT_TRUE(g.isInitializer(wd));
    const float *f = g.initializers.at(wd).f32();
    for (size_t i = 0; i < wq.size(); ++i)
    {
        size_t c = i % 6;
        EXPECT_EQ(f[i], (float) (((double) wq[i] - (double) zp[c]) * (double) scale[c])) << "i=" << i;
    }
}

TEST(Passes, DequantizeLinearUint8NonzeroZeroPoint) {
    // uint8 values with the asymmetric zero point 128; scale as a length-1 1-D tensor.
    std::vector<float> wq {0.f, 1.f, 127.f, 128.f, 200.f, 255.f};
    TensorId           wd = kNoTensor;
    Graph              g  = dequantWeightGraph({6}, wq, {1}, {0.5f}, {128.f}, false, 0, &wd);
    dequantizeGraph(g);
    ASSERT_EQ(g.nodes.size(), 1u);
    ASSERT_TRUE(g.isInitializer(wd));
    const float *f = g.initializers.at(wd).f32();
    for (size_t i = 0; i < wq.size(); ++i)
    {
        EXPECT_EQ(f[i], (float) (((double) wq[i] - 128.0) * (double) 0.5f)) << "i=" << i;
    }
}

// A boundary DequantizeLinear that survives to the CPU kernel at run time (its input is a graph
// activation, not an all-initializer chain the pass would fold) with a rank-0 SCALAR scale and
// zero_point. A rank-0 shape has zero numElements, so counting the scale by elems() would make the
// per-tensor channel index compute (i / inner) % 0 -- a divide-by-zero. The kernel must treat a
// rank-0 scalar as one channel (the per-tensor form).
TEST(Ops, DequantizeLinearRank0ScalarScaleRuns) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {2, 3};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorId   s = addFloatInit(g, "s", {}, {0.5f}); // rank-0 scalar scale
    TensorId   z = addFloatInit(g, "z", {}, {3.0f}); // rank-0 scalar zero_point
    TensorDesc yo;
    yo.name     = "y";
    yo.shape    = {2, 3};
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     n;
    n.type    = OpType::DequantizeLinear;
    n.name    = "dq";
    n.inputs  = {x, s, z};
    n.outputs = {y};
    g.nodes.push_back(n);
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);
    IOTensor in;
    in.name  = "x";
    in.shape = {2, 3};
    in.dtype = DType::Float32;
    in.data.resize(6 * sizeof(float));
    const float xv[6] = {0.f, 1.f, 3.f, 5.f, 7.f, 255.f}; // integer-valued quantized inputs
    for (int i = 0; i < 6; ++i)
    {
        reinterpret_cast<float *>(in.data.data())[i] = xv[i];
    }
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_EQ(outs.size(), 1u);
    const float *o = outs[0].f32();
    for (int i = 0; i < 6; ++i)
    {
        EXPECT_FLOAT_EQ(o[i], (xv[i] - 3.0f) * 0.5f) << "i=" << i; // per-tensor (x - zp) * scale
    }
}

TEST(Passes, QdqSandwichCollapsesByteEqual) {
    // int8 zp=3, scale=0.05 -> the inserted clamp is [-6.55, 6.2]. The post-ReLU values below all
    // land inside it, so the Clip is a genuine no-op and byte-equality with the plain float graph
    // (which has no clamp) still holds -- the point of this test.
    Graph g = qdqSandwichGraph(/*shared=*/true, 0.05f, 3.f, 0.05f, 3.f);
    dequantizeGraph(g);
    ASSERT_EQ(g.nodes.size(), 3u);
    EXPECT_EQ(g.nodes[0].type, OpType::Relu);
    EXPECT_EQ(g.nodes[1].type, OpType::Clip) << "the collapse leaves a saturation Clip in place of the round-trip";
    EXPECT_EQ(g.nodes[2].type, OpType::Binary);
    EXPECT_EQ(g.nodes[1].inputs[0], g.nodes[0].outputs[0]) << "the Clip must read the Relu output directly";
    EXPECT_EQ(g.nodes[2].inputs[0], g.nodes[1].outputs[0]) << "the Mul must read the Clip output";
    EXPECT_EQ(countQdq(g), 0);

    // The collapsed graph runs byte-identically to the same graph built with an explicit in-range
    // Clip. Every post-ReLU value is within [-6.55, 6.2], so the clamp does not bind.
    std::vector<float> xd {-3.5f, -1.f, -0.25f, 0.f, 0.6f, 1.5f, 2.5f, 6.f};
    Graph              plain;
    {
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {1, 8};
        xi.isInput = true;
        TensorId x = plain.addTensor(xi);
        plain.inputs = {x};
        TensorId r   = addUnshaped(plain, "r");
        TensorId c   = addUnshaped(plain, "c");
        TensorId y   = addUnshaped(plain, "y");
        TensorId lo  = addFloatInit(plain, "lo", {}, {(-128.f - 3.f) * 0.05f}); // (qmin - zp) * scale
        TensorId hi  = addFloatInit(plain, "hi", {}, {(127.f - 3.f) * 0.05f});  // (qmax - zp) * scale
        TensorId k   = addFloatInit(plain, "k", {1}, {2.f});
        plain.desc(y).isOutput = true;
        plain.outputs          = {y};
        Node relu;
        relu.type    = OpType::Relu;
        relu.name    = "relu";
        relu.inputs  = {x};
        relu.outputs = {r};
        Node clip;
        clip.type    = OpType::Clip;
        clip.name    = "clip";
        clip.inputs  = {r, lo, hi};
        clip.outputs = {c};
        Node mul;
        mul.type    = OpType::Binary;
        mul.subOp   = (int) BinaryType::Mul;
        mul.name    = "mul";
        mul.inputs  = {c, k};
        mul.outputs = {y};
        plain.nodes = {relu, clip, mul};
    }
    std::vector<float> got = runQdqGraph(std::move(g), xd);
    std::vector<float> ref = runQdqGraph(std::move(plain), xd);
    ASSERT_EQ(got.size(), ref.size());
    ASSERT_FALSE(got.empty());
    EXPECT_EQ(memcmp(got.data(), ref.data(), got.size() * 4), 0) << "collapsed QDQ must be byte-equal to the float+clamp graph";
}

TEST(Passes, QdqSandwichPreservesClamp) {
    // uint8 zp=0, scale=0.05 -> clamp [0, 12.75]: the lower bound is a ReLU. ORT's QDQ quantizer
    // folds a preceding activation into this range, so collapsing the sandwich to raw float would
    // silently drop the clamp; the inserted Clip recovers it. Negative inputs must come back as 0.
    Graph g = qdqSandwichGraph(/*shared=*/true, 0.05f, 0.f, 0.05f, 0.f, /*dqHasZp=*/true, DType::UInt8);
    // Drop the Relu so the clamp is the only lower bound under test (rewire the Q to read x directly).
    ASSERT_EQ(g.nodes[0].type, OpType::Relu);
    TensorId reluIn  = g.nodes[0].inputs[0];
    TensorId reluOut = g.nodes[0].outputs[0];
    g.nodes[1].inputs[0] = reluIn; // Q now reads x
    g.nodes.erase(g.nodes.begin());
    (void) reluOut;
    dequantizeGraph(g);
    EXPECT_EQ(countQdq(g), 0);

    std::vector<float> xd {-3.5f, -1.f, -0.25f, 0.f, 0.6f, 1.5f, 20.f, 7.f};
    std::vector<float> got = runQdqGraph(std::move(g), xd);
    ASSERT_EQ(got.size(), 8u);
    // Mul(k=2) of clamp(x, 0, 12.75): negatives -> 0, 20 saturates to 12.75, mid values pass.
    const float expect[8] = {0.f, 0.f, 0.f, 0.f, 1.2f, 3.f, 25.5f, 14.f};
    for (int i = 0; i < 8; ++i)
    {
        EXPECT_NEAR(got[i], expect[i], 1e-4f) << "i=" << i << " (clamp must recover the ReLU and saturate the top)";
    }
}

TEST(Passes, QdqSandwichInt8ClampSaturates) {
    // int8 zp=-40, scale=0.1 -> clamp [(-128+40)*0.1, (127+40)*0.1] = [-8.8, 16.7]. A per-tensor
    // int8 saturating case: values past either bound clamp, in-range values pass.
    Graph g = qdqSandwichGraph(/*shared=*/true, 0.1f, -40.f, 0.1f, -40.f, /*dqHasZp=*/true, DType::Int8);
    ASSERT_EQ(g.nodes[0].type, OpType::Relu);
    TensorId reluIn = g.nodes[0].inputs[0];
    g.nodes[1].inputs[0] = reluIn; // Q reads x directly, no ReLU floor
    g.nodes.erase(g.nodes.begin());
    dequantizeGraph(g);
    EXPECT_EQ(countQdq(g), 0);

    std::vector<float> xd {-20.f, -8.8f, -2.f, 0.f, 5.f, 16.7f, 30.f, 16.699f};
    std::vector<float> got = runQdqGraph(std::move(g), xd);
    ASSERT_EQ(got.size(), 8u);
    // Mul(k=2) of clamp(x, -8.8, 16.7).
    const float expect[8] = {-17.6f, -17.6f, -4.f, 0.f, 10.f, 33.4f, 33.4f, 33.398f};
    for (int i = 0; i < 8; ++i)
    {
        EXPECT_NEAR(got[i], expect[i], 1e-3f) << "i=" << i << " (int8 clamp must saturate both bounds)";
    }
}

TEST(Passes, QdqSandwichValueMatchCollapses) {
    // Distinct scale ids with equal payloads; the Q carries an explicit zero point of 0 and the DQ
    // none (absent matches an all-zero payload). The collapse leaves Relu -> Clip -> Mul.
    Graph g = qdqSandwichGraph(/*shared=*/false, 0.05f, 0.f, 0.05f, 0.f, /*dqHasZp=*/false);
    dequantizeGraph(g);
    EXPECT_EQ(countQdq(g), 0);
    ASSERT_EQ(g.nodes.size(), 3u);
    EXPECT_EQ(g.nodes[0].type, OpType::Relu);
    EXPECT_EQ(g.nodes[1].type, OpType::Clip);
    EXPECT_EQ(g.nodes[2].type, OpType::Binary);
    EXPECT_EQ(g.nodes[1].inputs[0], g.nodes[0].outputs[0]) << "Clip reads the Relu output";
    EXPECT_EQ(g.nodes[2].inputs[0], g.nodes[1].outputs[0]) << "Mul reads the Clip output";
}

TEST(Passes, QdqMismatchedSandwichKept) {
    // A requant edge (different scales) is not a removable sandwich; both nodes stay for a later
    // lowering stage, and a second run changes nothing.
    Graph g = qdqSandwichGraph(/*shared=*/false, 0.05f, 3.f, 0.07f, 3.f);
    dequantizeGraph(g);
    EXPECT_EQ(countQdq(g), 2);
    ASSERT_EQ(g.nodes.size(), 4u);
    dequantizeGraph(g);
    EXPECT_EQ(countQdq(g), 2);
}

TEST(Passes, NoDequantizeKeepsQuantizedNodes) {
    Graph       off = qdqSandwichGraph(true, 0.05f, 3.f, 0.05f, 3.f);
    PassOptions opt;
    opt.dequantize = false;
    runStandardPasses(off, opt);
    EXPECT_EQ(countQdq(off), 2) << "--no-dequantize keeps the quantized ops";

    Graph on = qdqSandwichGraph(true, 0.05f, 3.f, 0.05f, 3.f);
    runStandardPasses(on, {});
    EXPECT_EQ(countQdq(on), 0) << "the default pipeline collapses the sandwich";
}

// --- C.2 QLinear decomposition + CPU Quantize/DequantizeLinear kernels. --------------------------
// The QLinear family (QLinearConv/QLinearMatMul/QGemm/QLinearAdd/QLinearGlobalAveragePool) lowers at
// import to a plain float op plus an output-range saturation Clip; the CPU Quantize/DequantizeLinear
// kernels are the graph-boundary quant hops the decomposition keeps as nodes. These tests pin the
// numeric behavior against hand-computed references and the ONNX saturation/round-half-even rules.
namespace {

    // QuantizeLinear(x, scale, zp) with a scalar or per-axis scale, run on CPU. `quantDt` sets the
    // output integer dtype (its saturation range).
    std::vector<float> runQuantize(const Shape &xshape, const std::vector<float> &xd, const Shape &sshape, const std::vector<float> &scale, const std::vector<float> &zp, DType quantDt, int64_t axis = 1) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = xshape;
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};
        TensorId s = addFloatInit(g, "s", sshape, scale);
        TensorId z = zp.empty() ? kNoTensor : addZeroPointInit(g, "z", sshape, zp, quantDt);
        TensorId y = addUnshaped(g, "y");
        g.desc(y).isOutput = true;
        g.desc(y).dtype    = quantDt; // the declared graph-output quant type
        g.outputs          = {y};
        Node q;
        q.type    = OpType::QuantizeLinear;
        q.name    = "q";
        q.inputs  = z == kNoTensor ? std::vector<TensorId> {x, s} : std::vector<TensorId> {x, s, z};
        q.outputs = {y};
        if (scale.size() > 1)
        {
            Attr a;
            a.kind            = Attr::Int;
            a.i               = axis;
            q.attr.map["axis"] = a;
        }
        g.nodes = {q};
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = "x";
        in.shape = xshape;
        in.dtype = DType::Float32;
        in.data.resize(xd.size() * 4);
        for (size_t i = 0; i < xd.size(); ++i)
        {
            reinterpret_cast<float *>(in.data.data())[i] = xd[i];
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        if (outs.empty())
        {
            return {};
        }
        const float *o = outs[0].f32();
        return std::vector<float>(o, o + numElements(outs[0].shape));
    }

    // DequantizeLinear(x, scale, zp) reading a float "x" input, run on CPU. The pass keeps a
    // DequantizeLinear whose data input is not an initializer as a node, so the CPU kernel runs.
    std::vector<float> runDequantize(const Shape &xshape, const std::vector<float> &xd, const Shape &sshape, const std::vector<float> &scale, const std::vector<float> &zp, int64_t axis = 1) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = xshape;
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};
        TensorId s = addFloatInit(g, "s", sshape, scale);
        TensorId z = zp.empty() ? kNoTensor : addFloatInit(g, "z", sshape, zp);
        TensorId y = addUnshaped(g, "y");
        g.desc(y).isOutput = true;
        g.outputs          = {y};
        Node dq;
        dq.type    = OpType::DequantizeLinear;
        dq.name    = "dq";
        dq.inputs  = z == kNoTensor ? std::vector<TensorId> {x, s} : std::vector<TensorId> {x, s, z};
        dq.outputs = {y};
        if (scale.size() > 1)
        {
            Attr a;
            a.kind             = Attr::Int;
            a.i                = axis;
            dq.attr.map["axis"] = a;
        }
        g.nodes = {dq};
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = "x";
        in.shape = xshape;
        in.dtype = DType::Float32;
        in.data.resize(xd.size() * 4);
        for (size_t i = 0; i < xd.size(); ++i)
        {
            reinterpret_cast<float *>(in.data.data())[i] = xd[i];
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        if (outs.empty())
        {
            return {};
        }
        const float *o = outs[0].f32();
        return std::vector<float>(o, o + numElements(outs[0].shape));
    }

} // namespace

TEST(Ops, DequantizeLinearPerTensor) {
    // y = (x - zp) * scale, scalar scale/zp.
    auto y = runDequantize({1, 4}, {10, 0, -5, 130}, {}, {0.5f}, {8});
    expectNear(y, {(10 - 8) * 0.5f, (0 - 8) * 0.5f, (-5 - 8) * 0.5f, (130 - 8) * 0.5f});
}

TEST(Ops, DequantizeLinearPerAxis) {
    // Per-axis over axis 1 (channel) of a [1,3,2] tensor: each channel gets its own scale/zp.
    Shape              xs = {1, 3, 2};
    std::vector<float> xd = {10, 12, 20, 22, 30, 32};
    std::vector<float> sc = {0.1f, 0.2f, 0.5f};
    std::vector<float> zp = {2, 4, 6};
    auto               y  = runDequantize(xs, xd, {3}, sc, zp, /*axis=*/1);
    std::vector<float> ref;
    for (int c = 0; c < 3; ++c)
    {
        for (int j = 0; j < 2; ++j)
        {
            ref.push_back((xd[c * 2 + j] - zp[c]) * sc[c]);
        }
    }
    expectNear(y, ref);
}

TEST(Ops, QuantizeLinearSaturatesAndRoundsHalfEven) {
    // uint8 range [0,255]; round-half-to-even; saturation on both ends. x/scale with scale=1, zp=0:
    //   0.5 -> 0 (even), 1.5 -> 2 (even), 2.5 -> 2 (even), 3.5 -> 4 (even)  [banker's rounding]
    //   -3 -> 0 (saturate low), 300 -> 255 (saturate high).
    auto y = runQuantize({1, 6}, {0.5f, 1.5f, 2.5f, 3.5f, -3.f, 300.f}, {}, {1.0f}, {0.f}, DType::UInt8);
    expectNear(y, {0, 2, 2, 4, 0, 255});
}

TEST(Ops, QuantizeLinearInt8ZeroPoint) {
    // int8 range [-128,127] with a nonzero zero_point and a real scale.
    // q = round(x/scale) + zp, saturated. scale=0.25, zp=-10.
    //   x=0    -> 0   + -10 = -10
    //   x=1    -> 4   + -10 = -6
    //   x=-40  -> -160 + -10 = -170 -> saturate -128
    //   x=40   -> 160  + -10 = 150  -> saturate 127
    auto y = runQuantize({1, 4}, {0.f, 1.f, -40.f, 40.f}, {}, {0.25f}, {-10.f}, DType::Int8);
    expectNear(y, {-10, -6, -128, 127});
}

namespace {

    // Adds an int32-labeled bias initializer (fp32 host storage, Int32 dtype label), as the ONNX
    // importer produces for a QLinear op's int32 bias.
    TensorId addInt32Init(Graph &g, const char *name, const Shape &shape, const std::vector<float> &v) {
        TensorId id      = addFloatInit(g, name, shape, v);
        g.desc(id).dtype = DType::Int32;
        return id;
    }

    // Builds and runs a QLinearConv over a float "x" input:
    //   QLinearConv(x_q, x_s, x_zp, w_q, w_s, w_zp, y_s, y_zp, [B_i32]) -> y_q
    // then a trailing DequantizeLinear(y_q, y_s, y_zp) -> out so the graph output is real float that
    // the decomposition's output clamp is visible in. The pass folds it to Conv + output Clip; the
    // trailing DQ over the decomposed (float) output drops. `x` is fed directly as the pre-dequant
    // float activation value (a standalone leading dequant is not modeled here -- the decomposition
    // reads the producer's float, and here the "producer" is the graph input). Returns the fp32 out.
    std::vector<float> runQLinearConv(const Shape &xshape, const std::vector<float> &xd,
                                      const Shape &wshape, const std::vector<float> &wq,
                                      const Shape &wsshape, const std::vector<float> &ws, const std::vector<float> &wzp, int64_t wAxis,
                                      float xs, float xzp, float ys, float yzp, DType actDt,
                                      const Shape &bshape, const std::vector<float> &bq, const Attributes &convAttr) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = xshape;
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};
        TensorId xsI  = addFloatInit(g, "xs", {}, {xs});
        TensorId xzpI = addZeroPointInit(g, "xzp", {}, {xzp}, actDt);
        TensorId wI   = addFloatInit(g, "w", wshape, wq);
        TensorId wsI  = addFloatInit(g, "ws", wsshape, ws);
        TensorId wzpI = addZeroPointInit(g, "wzp", wsshape, wzp, DType::Int8);
        TensorId ysI  = addFloatInit(g, "ys", {}, {ys});
        TensorId yzpI = addZeroPointInit(g, "yzp", {}, {yzp}, actDt);
        std::vector<TensorId> ins = {x, xsI, xzpI, wI, wsI, wzpI, ysI, yzpI};
        if (!bq.empty())
        {
            ins.push_back(addInt32Init(g, "b", bshape, bq));
        }
        TensorId yq  = addUnshaped(g, "yq");
        TensorId out = addUnshaped(g, "out");
        g.desc(out).isOutput = true;
        g.outputs            = {out};
        Node conv;
        conv.type    = OpType::QLinearConv;
        conv.name    = "qconv";
        conv.inputs  = ins;
        conv.outputs = {yq};
        conv.attr    = convAttr;
        if (ws.size() > 1)
        {
            Attr a;
            a.kind                = Attr::Int;
            a.i                   = wAxis;
            conv.attr.map["axis"] = a; // per-axis weight quant axis (default 0 for weights)
        }
        Node dq;
        dq.type    = OpType::DequantizeLinear;
        dq.name    = "outdq";
        dq.inputs  = {yq, ysI, yzpI};
        dq.outputs = {out};
        g.nodes = {conv, dq};
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = "x";
        in.shape = xshape;
        in.dtype = DType::Float32;
        in.data.resize(xd.size() * 4);
        for (size_t i = 0; i < xd.size(); ++i)
        {
            reinterpret_cast<float *>(in.data.data())[i] = xd[i];
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        if (outs.empty())
        {
            return {};
        }
        const float *o = outs[0].f32();
        return std::vector<float>(o, o + numElements(outs[0].shape));
    }

    // The saturation range a y-quant (ys, yzp, dtype) encodes, in real float: [(qmin-zp)*s,(qmax-zp)*s].
    void yClampRange(float ys, float yzp, DType dt, float &lo, float &hi) {
        float qmin = dt == DType::Int8 ? -128.f : 0.f;
        float qmax = dt == DType::Int8 ? 127.f : 255.f;
        lo = (qmin - yzp) * ys;
        hi = (qmax - yzp) * ys;
    }

} // namespace

TEST(Ops, QLinearConvDecomposedNumericExactReluRecovered) {
    // 1x1 conv, 2 in / 2 out channels, spatial 1x1, per-tensor weight quant. The y-quant range is a
    // uint8 range with zp=0 (lo=0): the output Clip recovers a ReLU. Verify against a hand-computed
    // float conv + clamp, and confirm a negative conv result is clamped to 0 (ReLU recovered).
    //   x (real float, fed directly): channel0=1, channel1=-2
    //   W_f = W_q * ws (wzp=0): [[3,0],[0,4]] -> out0 = 3*1 = 3 ; out1 = 4*(-2) = -8
    //   B_f = B_i32 * (xs*ws): xs=1, ws=1 -> B_f = B_i32 = [-1, +1]
    //   pre-clamp: out0 = 3 + -1 = 2 ; out1 = -8 + 1 = -7
    //   y-clamp (uint8, ys=0.1, yzp=0): lo=0, hi=25.5 -> out0=2, out1=0 (ReLU!)
    Attributes attr;
    attr.map["kernel_shape"] = ints({1, 1});
    std::vector<float> y = runQLinearConv(
        /*xshape*/ {1, 2, 1, 1}, /*xd*/ {1.f, -2.f},
        /*wshape*/ {2, 2, 1, 1}, /*wq*/ {3, 0, 0, 4},
        /*wsshape*/ {}, /*ws*/ {1.0f}, /*wzp*/ {0.f}, /*wAxis*/ 0,
        /*xs*/ 1.0f, /*xzp*/ 0.f, /*ys*/ 0.1f, /*yzp*/ 0.f, /*actDt*/ DType::UInt8,
        /*bshape*/ {2}, /*bq*/ {-1.f, 1.f}, attr);
    float lo, hi;
    yClampRange(0.1f, 0.f, DType::UInt8, lo, hi);
    auto clamp = [&](float v) { return v < lo ? lo : (v > hi ? hi : v); };
    expectNear(y, {clamp(2.f), clamp(-7.f)});
    EXPECT_NEAR(y[1], 0.f, 1e-5f) << "y-quant range with zp=0 recovers the fused ReLU (no ReLU deleted)";
}

TEST(Ops, QLinearConvPerAxisWeightScale) {
    // Per-axis weight scale over output-channel axis 0: each of the 2 output channels gets its own
    // ws. W_q = [[2,0],[0,2]], ws=[0.5, 3.0] -> W_f=[[1,0],[0,6]]. x=[4, 5], no bias.
    //   out0 = 1*4 = 4 ; out1 = 6*5 = 30. y-quant int8, ys=1, yzp=0: lo=-128, hi=127 (no clamp).
    Attributes attr;
    attr.map["kernel_shape"] = ints({1, 1});
    std::vector<float> y = runQLinearConv(
        /*xshape*/ {1, 2, 1, 1}, /*xd*/ {4.f, 5.f},
        /*wshape*/ {2, 2, 1, 1}, /*wq*/ {2, 0, 0, 2},
        /*wsshape*/ {2}, /*ws*/ {0.5f, 3.0f}, /*wzp*/ {0.f, 0.f}, /*wAxis*/ 0,
        /*xs*/ 1.0f, /*xzp*/ 0.f, /*ys*/ 1.0f, /*yzp*/ 0.f, /*actDt*/ DType::Int8,
        /*bshape*/ {}, /*bq*/ {}, attr);
    expectNear(y, {4.f, 30.f});
}

namespace {

    // Builds and runs a QGemm(A, a_s, a_zp, B, b_s, b_zp, [C_i32], y_s, y_zp) over a float "A" input,
    // with a trailing DequantizeLinear to expose the real-float output. transB selects B layout.
    std::vector<float> runQGemm(const Shape &ashape, const std::vector<float> &ad,
                                const Shape &bshape, const std::vector<float> &bq, const std::vector<float> &bs, const std::vector<float> &bzp,
                                float as, float azp, float ys, float yzp, DType actDt,
                                const std::vector<float> &cq, int64_t transB) {
        Graph      g;
        TensorDesc ai;
        ai.name    = "A";
        ai.shape   = ashape;
        ai.isInput = true;
        TensorId a = g.addTensor(ai);
        g.inputs   = {a};
        TensorId asI  = addFloatInit(g, "as", {}, {as});
        TensorId azpI = addZeroPointInit(g, "azp", {}, {azp}, actDt);
        TensorId bI   = addFloatInit(g, "B", bshape, bq);
        TensorId bsI  = addFloatInit(g, "bs", bs.size() > 1 ? Shape {(int64_t) bs.size()} : Shape {}, bs);
        TensorId bzpI = addZeroPointInit(g, "bzp", bzp.size() > 1 ? Shape {(int64_t) bzp.size()} : Shape {}, bzp, DType::Int8);
        TensorId ysI  = addFloatInit(g, "ys", {}, {ys});
        TensorId yzpI = addZeroPointInit(g, "yzp", {}, {yzp}, actDt);
        std::vector<TensorId> ins = {a, asI, azpI, bI, bsI, bzpI};
        if (!cq.empty())
        {
            ins.push_back(addInt32Init(g, "C", {(int64_t) cq.size()}, cq));
        }
        ins.push_back(ysI);
        ins.push_back(yzpI);
        TensorId yq  = addUnshaped(g, "yq");
        TensorId out = addUnshaped(g, "out");
        g.desc(out).isOutput = true;
        g.outputs            = {out};
        Node gemm;
        gemm.type    = OpType::QGemm;
        gemm.name    = "qgemm";
        gemm.inputs  = ins;
        gemm.outputs = {yq};
        if (transB)
        {
            gemm.attr.map["transB"] = ints({}); // set below as Int
            Attr t;
            t.kind                  = Attr::Int;
            t.i                     = 1;
            gemm.attr.map["transB"] = t;
        }
        Node dq;
        dq.type    = OpType::DequantizeLinear;
        dq.name    = "outdq";
        dq.inputs  = {yq, ysI, yzpI};
        dq.outputs = {out};
        g.nodes = {gemm, dq};
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = "A";
        in.shape = ashape;
        in.dtype = DType::Float32;
        in.data.resize(ad.size() * 4);
        for (size_t i = 0; i < ad.size(); ++i)
        {
            reinterpret_cast<float *>(in.data.data())[i] = ad[i];
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        if (outs.empty())
        {
            return {};
        }
        const float *o = outs[0].f32();
        return std::vector<float>(o, o + numElements(outs[0].shape));
    }

} // namespace

TEST(Ops, QGemmWithBiasDecomposed) {
    // A [1,2] * B [2,2] + C bias. B_q = [[2,0],[1,3]], bs=0.5 -> B_f=[[1,0],[0.5,1.5]].
    //   A = [4, 6]  -> A*B_f = [4*1 + 6*0.5, 4*0 + 6*1.5] = [7, 9]
    //   C_f = C_i32 * (as*bs): as=1, bs=0.5 -> C_f = C_i32 * 0.5 = [2, -4]*0.5? use C_i32=[2,-4] -> [1,-2]
    //   pre-clamp: [8, 7]. y int8 ys=1 yzp=0: no clamp.
    auto y = runQGemm(
        /*ashape*/ {1, 2}, /*ad*/ {4.f, 6.f},
        /*bshape*/ {2, 2}, /*bq*/ {2, 0, 1, 3}, /*bs*/ {0.5f}, /*bzp*/ {0.f},
        /*as*/ 1.0f, /*azp*/ 0.f, /*ys*/ 1.0f, /*yzp*/ 0.f, /*actDt*/ DType::Int8,
        /*cq*/ {2.f, -4.f}, /*transB*/ 0);
    expectNear(y, {8.f, 7.f});
}

namespace {

    // Builds and runs QLinearAdd(A, a_s, a_zp, B, b_s, b_zp, y_s, y_zp) where both A and B are float
    // graph inputs (real activation values), with a trailing DequantizeLinear to expose real float.
    std::vector<float> runQLinearAdd(const Shape &shape, const std::vector<float> &ad, const std::vector<float> &bd,
                                     float as, float azp, float bs, float bzp, float ys, float yzp, DType actDt) {
        Graph      g;
        TensorDesc ai;
        ai.name    = "A";
        ai.shape   = shape;
        ai.isInput = true;
        TensorDesc bi;
        bi.name    = "B";
        bi.shape   = shape;
        bi.isInput = true;
        TensorId a = g.addTensor(ai);
        TensorId b = g.addTensor(bi);
        g.inputs   = {a, b};
        TensorId asI  = addFloatInit(g, "as", {}, {as});
        TensorId azpI = addZeroPointInit(g, "azp", {}, {azp}, actDt);
        TensorId bsI  = addFloatInit(g, "bs", {}, {bs});
        TensorId bzpI = addZeroPointInit(g, "bzp", {}, {bzp}, actDt);
        TensorId ysI  = addFloatInit(g, "ys", {}, {ys});
        TensorId yzpI = addZeroPointInit(g, "yzp", {}, {yzp}, actDt);
        TensorId yq   = addUnshaped(g, "yq");
        TensorId out  = addUnshaped(g, "out");
        g.desc(out).isOutput = true;
        g.outputs            = {out};
        Node add;
        add.type    = OpType::QLinearAdd;
        add.name    = "qadd";
        add.inputs  = {a, asI, azpI, b, bsI, bzpI, ysI, yzpI};
        add.outputs = {yq};
        Node dq;
        dq.type    = OpType::DequantizeLinear;
        dq.name    = "outdq";
        dq.inputs  = {yq, ysI, yzpI};
        dq.outputs = {out};
        g.nodes = {add, dq};
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        auto mkIn = [&](const char *nm, const std::vector<float> &d) {
            IOTensor in;
            in.name  = nm;
            in.shape = shape;
            in.dtype = DType::Float32;
            in.data.resize(d.size() * 4);
            for (size_t i = 0; i < d.size(); ++i)
            {
                reinterpret_cast<float *>(in.data.data())[i] = d[i];
            }
            return in;
        };
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({mkIn("A", ad), mkIn("B", bd)}, outs), Status::Ok);
        if (outs.empty())
        {
            return {};
        }
        const float *o = outs[0].f32();
        return std::vector<float>(o, o + numElements(outs[0].shape));
    }

} // namespace

TEST(Ops, QLinearAddChainDecomposed) {
    // Add of two real-float activation tensors, then output clamp. y-quant int8 zp=0 ys=0.5:
    // lo=-64, hi=63.5. A=[10, 100, -100], B=[5, 50, -50] -> [15, 150, -150] -> clamp [15, 63.5, -64].
    auto y = runQLinearAdd({1, 3}, {10.f, 100.f, -100.f}, {5.f, 50.f, -50.f},
                           /*as*/ 1.f, 0.f, /*bs*/ 1.f, 0.f, /*ys*/ 0.5f, 0.f, DType::Int8);
    float lo, hi;
    yClampRange(0.5f, 0.f, DType::Int8, lo, hi);
    auto clamp = [&](float v) { return v < lo ? lo : (v > hi ? hi : v); };
    expectNear(y, {clamp(15.f), clamp(150.f), clamp(-150.f)});
}

// --- C.4 dynamic-quant cluster lowering. --------------------------------------------------------
// A DynamicQuantizeLinear feeding a MatMulInteger/ConvInteger (then Cast + a scale Mul, and an
// optional separate bias Add) is the shape an ONNX dynamic-quantization export emits. The pass
// lowers the whole cluster to a float MatMul/Conv reading the DynamicQuantizeLinear's unquantized
// float input and a folded fp32 weight W_f=(w_q-w_zp)*w_s -- NO output clamp, because MatMulInteger
// and ConvInteger emit full-range int32 with no output quant range to saturate. These tests build
// the cluster by hand, run the folded graph on the CPU backend, and compare against a plain fp32
// MatMul/Conv on the same folded weights (the float function the dynamic-quant graph approximates).
namespace {

    // Builds a dynamic-quant MatMul cluster over a float "x" input of shape [M,K]:
    //   DynamicQuantizeLinear(x) -> x_q, x_s, x_zp
    //   Mul(x_s, w_s[init]) -> comb_s
    //   MatMulInteger(x_q, w_q[init KxN], x_zp, w_zp[init]) -> y_i32
    //   Cast(y_i32 -> float) -> y_cast
    //   Mul(y_cast, comb_s) -> y_scaled  (graph output, or the bias Add's operand when withBias)
    //   [withBias] Add(y_scaled, bias[init 1xN]) -> y
    // `ws` is the weight scale (length 1 per-tensor, or N per-column); `wzp` the weight zero point
    // (int8 payload widened to fp32, Int8-labeled), empty for none. Returns the assembled graph.
    Graph dynQuantMatMulGraph(const Shape &xshape, int64_t N, const std::vector<float> &wq,
                              const std::vector<float> &ws, const std::vector<float> &wzp,
                              bool withBias, const std::vector<float> &bias) {
        Graph      g;
        int64_t    K = xshape.back();
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = xshape;
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};
        TensorId xq   = addUnshaped(g, "x_q");
        TensorId xs   = addUnshaped(g, "x_s");
        TensorId xzp  = addUnshaped(g, "x_zp");
        TensorId wI   = addFloatInit(g, "w", {K, N}, wq);
        TensorId wsI  = addFloatInit(g, "ws", ws.size() == 1 ? Shape {} : Shape {N}, ws);
        TensorId wzpI = wzp.empty() ? kNoTensor : addZeroPointInit(g, "wzp", {}, wzp, DType::Int8);
        TensorId combS = addUnshaped(g, "comb_s");
        TensorId yi    = addUnshaped(g, "y_i32");
        TensorId ycast = addUnshaped(g, "y_cast");
        TensorId yscl  = addUnshaped(g, "y_scaled");
        Node dql;
        dql.type    = OpType::DynamicQuantizeLinear;
        dql.name    = "dql";
        dql.inputs  = {x};
        dql.outputs = {xq, xs, xzp};
        Node scMul;
        scMul.type    = OpType::Binary;
        scMul.subOp   = (int) BinaryType::Mul;
        scMul.name    = "scales_mul";
        scMul.inputs  = {xs, wsI};
        scMul.outputs = {combS};
        Node mmi;
        mmi.type    = OpType::MatMulInteger;
        mmi.name    = "mmi";
        mmi.inputs  = wzpI == kNoTensor ? std::vector<TensorId> {xq, wI, xzp} : std::vector<TensorId> {xq, wI, xzp, wzpI};
        mmi.outputs = {yi};
        Node cast;
        cast.type    = OpType::Cast;
        cast.name    = "cast";
        cast.inputs  = {yi};
        cast.outputs = {ycast};
        {
            Attr a;
            a.kind             = Attr::Int;
            a.i                = 1; // to=FLOAT
            cast.attr.map["to"] = a;
        }
        Node outMul;
        outMul.type    = OpType::Binary;
        outMul.subOp   = (int) BinaryType::Mul;
        outMul.name    = "out_scale_mul";
        outMul.inputs  = {ycast, combS};
        outMul.outputs = {yscl};
        g.nodes = {dql, scMul, mmi, cast, outMul};
        TensorId out = yscl;
        if (withBias)
        {
            TensorId bI = addFloatInit(g, "bias", {1, N}, bias);
            TensorId yb = addUnshaped(g, "y");
            Node     add;
            add.type    = OpType::Binary;
            add.subOp   = (int) BinaryType::Add;
            add.name    = "bias_add";
            add.inputs  = {yscl, bI};
            add.outputs = {yb};
            g.nodes.push_back(add);
            out = yb;
        }
        g.desc(out).isOutput = true;
        g.outputs            = {out};
        return g;
    }

    // Runs an [M,K]-input graph on the CPU backend and returns the fp32 output values.
    std::vector<float> runFloatGraph(Graph g, const char *inName, const Shape &xshape, const std::vector<float> &xd) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = inName;
        in.shape = xshape;
        in.dtype = DType::Float32;
        in.data.resize(xd.size() * 4);
        for (size_t i = 0; i < xd.size(); ++i)
        {
            reinterpret_cast<float *>(in.data.data())[i] = xd[i];
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        if (outs.empty())
        {
            return {};
        }
        const float *o = outs[0].f32();
        return std::vector<float>(o, o + numElements(outs[0].shape));
    }

    // The float reference the dynamic-quant MatMul cluster approximates: y = x @ ((w_q - w_zp) * w_s)
    // (+ bias), all in double. w_s is scalar or per-column of length N.
    std::vector<float> refMatMul(const Shape &xshape, int64_t N, const std::vector<float> &xd,
                                 const std::vector<float> &wq, const std::vector<float> &ws,
                                 const std::vector<float> &wzp, const std::vector<float> &bias) {
        int64_t            M = xshape[0], K = xshape[1];
        double             zp = wzp.empty() ? 0.0 : (double) wzp[0];
        std::vector<float> out((size_t) (M * N));
        for (int64_t m = 0; m < M; ++m)
        {
            for (int64_t n = 0; n < N; ++n)
            {
                double acc = 0.0;
                double s   = ws.size() == 1 ? ws[0] : ws[(size_t) n];
                for (int64_t k = 0; k < K; ++k)
                {
                    double wf = ((double) wq[(size_t) (k * N + n)] - zp) * s;
                    acc += (double) xd[(size_t) (m * K + k)] * wf;
                }
                if (!bias.empty())
                {
                    acc += (double) bias[(size_t) n];
                }
                out[(size_t) (m * N + n)] = (float) acc;
            }
        }
        return out;
    }

    // Counts DynamicQuantizeLinear / MatMulInteger / ConvInteger nodes left in the graph.
    int countDynQuant(const Graph &g) {
        int n = 0;
        for (const Node &nd: g.nodes)
        {
            n += nd.type == OpType::DynamicQuantizeLinear || nd.type == OpType::MatMulInteger || nd.type == OpType::ConvInteger;
        }
        return n;
    }

} // namespace

TEST(Passes, DynamicQuantMatMulClusterLowersToFloatMatMul) {
    // K=3, N=2 per-tensor weight scale, zp=0. The cluster collapses to MatMul(x, W_f) with no clamp.
    const Shape        xshape {2, 3};
    std::vector<float> wq {1, -2, 3, 4, -5, 6}; // [K=3, N=2] row-major int8 values
    std::vector<float> ws {0.5f};
    Graph              g = dynQuantMatMulGraph(xshape, 2, wq, ws, {}, false, {});
    dequantizeGraph(g);
    EXPECT_EQ(countDynQuant(g), 0) << "the whole cluster lowers -- no residual dynamic-quant nodes";
    int matmuls = 0;
    for (const Node &n: g.nodes)
    {
        matmuls += n.type == OpType::MatMul;
        EXPECT_NE(n.type, OpType::Clip) << "dynamic-quant lowering inserts no output clamp";
    }
    EXPECT_EQ(matmuls, 1);
    std::vector<float> xd {1, 2, 3, -1, -2, -3};
    auto               y = runFloatGraph(std::move(g), "x", xshape, xd);
    expectNear(y, refMatMul(xshape, 2, xd, wq, ws, {}, {}));
}

TEST(Passes, DynamicQuantMatMulPerColumnScaleAndZeroPoint) {
    // Per-column weight scale (length N) and a nonzero weight zero_point -- W_f=(w_q-wzp)*w_s[col].
    const Shape        xshape {1, 4};
    int64_t            N = 3;
    std::vector<float> wq {10, 20, 30, 11, 21, 31, 12, 22, 32, 13, 23, 33}; // [K=4, N=3]
    std::vector<float> ws {0.1f, 0.2f, 0.05f};
    std::vector<float> wzp {7};
    Graph              g = dynQuantMatMulGraph(xshape, N, wq, ws, wzp, false, {});
    dequantizeGraph(g);
    EXPECT_EQ(countDynQuant(g), 0);
    std::vector<float> xd {2, -1, 0.5f, 3};
    auto               y = runFloatGraph(std::move(g), "x", xshape, xd);
    expectNear(y, refMatMul(xshape, N, xd, wq, ws, wzp, {}));
}

TEST(Passes, DynamicQuantMatMulWithBiasAddSurvives) {
    // A separate downstream bias Add reads the lowered MatMul output unchanged.
    const Shape        xshape {2, 2};
    int64_t            N = 2;
    std::vector<float> wq {2, -1, 1, 3}; // [K=2, N=2]
    std::vector<float> ws {0.25f};
    std::vector<float> bias {0.5f, -1.5f};
    Graph              g = dynQuantMatMulGraph(xshape, N, wq, ws, {}, true, bias);
    dequantizeGraph(g);
    EXPECT_EQ(countDynQuant(g), 0);
    int adds = 0, matmuls = 0;
    for (const Node &n: g.nodes)
    {
        adds += n.type == OpType::Binary && n.subOp == (int) BinaryType::Add;
        matmuls += n.type == OpType::MatMul;
    }
    EXPECT_EQ(matmuls, 1);
    EXPECT_EQ(adds, 1) << "the bias Add is preserved";
    std::vector<float> xd {1, 2, -1, 0.5f};
    auto               y = runFloatGraph(std::move(g), "x", xshape, xd);
    expectNear(y, refMatMul(xshape, N, xd, wq, ws, {}, bias));
}

namespace {

    // Builds a dynamic-quant ConvInteger cluster over a float "x" NCHW input:
    //   DynamicQuantizeLinear(x) -> x_q, x_s, x_zp
    //   Mul(x_s, w_s[init]) -> comb_s
    //   ConvInteger(x_q, w_q[init O,C,kh,kw], x_zp, w_zp[init]) -> y_i32
    //   Cast(y_i32 -> float) -> Mul(comb_s) -> y   (graph output)
    // w_s is scalar (per-tensor) or per-output-channel of length O. Conv attrs come from `convAttr`.
    Graph dynQuantConvGraph(const Shape &xshape, const Shape &wshape, const std::vector<float> &wq,
                            const std::vector<float> &ws, const std::vector<float> &wzp, const Attributes &convAttr) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = xshape;
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};
        int64_t  O    = wshape[0];
        TensorId xq   = addUnshaped(g, "x_q");
        TensorId xs   = addUnshaped(g, "x_s");
        TensorId xzp  = addUnshaped(g, "x_zp");
        TensorId wI   = addFloatInit(g, "w", wshape, wq);
        TensorId wsI  = addFloatInit(g, "ws", ws.size() == 1 ? Shape {} : Shape {O}, ws);
        TensorId wzpI = wzp.empty() ? kNoTensor : addZeroPointInit(g, "wzp", {}, wzp, DType::UInt8);
        TensorId combS = addUnshaped(g, "comb_s");
        TensorId yi    = addUnshaped(g, "y_i32");
        TensorId ycast = addUnshaped(g, "y_cast");
        TensorId y     = addUnshaped(g, "y");
        Node dql;
        dql.type    = OpType::DynamicQuantizeLinear;
        dql.name    = "dql";
        dql.inputs  = {x};
        dql.outputs = {xq, xs, xzp};
        Node scMul;
        scMul.type    = OpType::Binary;
        scMul.subOp   = (int) BinaryType::Mul;
        scMul.name    = "scales_mul";
        scMul.inputs  = {xs, wsI};
        scMul.outputs = {combS};
        Node ci;
        ci.type    = OpType::ConvInteger;
        ci.name    = "ci";
        ci.inputs  = wzpI == kNoTensor ? std::vector<TensorId> {xq, wI, xzp} : std::vector<TensorId> {xq, wI, xzp, wzpI};
        ci.outputs = {yi};
        ci.attr    = convAttr;
        Node cast;
        cast.type    = OpType::Cast;
        cast.name    = "cast";
        cast.inputs  = {yi};
        cast.outputs = {ycast};
        {
            Attr a;
            a.kind              = Attr::Int;
            a.i                 = 1;
            cast.attr.map["to"] = a;
        }
        Node outMul;
        outMul.type    = OpType::Binary;
        outMul.subOp   = (int) BinaryType::Mul;
        outMul.name    = "out_scale_mul";
        outMul.inputs  = {ycast, combS};
        outMul.outputs = {y};
        g.nodes            = {dql, scMul, ci, cast, outMul};
        g.desc(y).isOutput = true;
        g.outputs          = {y};
        return g;
    }

} // namespace

TEST(Passes, DynamicQuantConvClusterLowersToFloatConv) {
    // 1x1x3x3 input, one 1x1x2x2 filter (O=1,C=1), stride 1, no pad, per-tensor scale, uint8 wzp.
    // The cluster collapses to Conv(x, W_f) with no clamp; compare against a hand-computed 2x2 output.
    Attributes attr;
    {
        Attr ks;
        ks.kind = Attr::Ints;
        ks.ints = {2, 2};
        attr.map["kernel_shape"] = ks;
        Attr st;
        st.kind = Attr::Ints;
        st.ints = {1, 1};
        attr.map["strides"] = st;
        Attr gp;
        gp.kind          = Attr::Int;
        gp.i             = 1;
        attr.map["group"] = gp;
    }
    std::vector<float> wq {2, 0, 1, 3}; // [O=1,C=1,2,2]
    std::vector<float> ws {0.5f};
    std::vector<float> wzp {1};
    Graph              g = dynQuantConvGraph({1, 1, 3, 3}, {1, 1, 2, 2}, wq, ws, wzp, attr);
    dequantizeGraph(g);
    EXPECT_EQ(countDynQuant(g), 0) << "the ConvInteger cluster lowers fully";
    int convs = 0;
    for (const Node &n: g.nodes)
    {
        convs += n.type == OpType::Conv;
        EXPECT_NE(n.type, OpType::Clip) << "dynamic-quant lowering inserts no output clamp";
    }
    EXPECT_EQ(convs, 1);
    // x = 3x3 grid 1..9. W_f = (wq - 1) * 0.5 = [[0.5,-0.5],[0,1]].
    std::vector<float> xd {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto               y = runFloatGraph(std::move(g), "x", {1, 1, 3, 3}, xd);
    // Reference 2x2 valid conv with W_f over each 2x2 window: out = 0.5*a - 0.5*b + 0*c + 1*d.
    auto win = [&](int r, int c) {
        double a = xd[r * 3 + c], b = xd[r * 3 + c + 1], cc = xd[(r + 1) * 3 + c], d = xd[(r + 1) * 3 + c + 1];
        return (float) (0.5 * a - 0.5 * b + 0.0 * cc + 1.0 * d);
    };
    expectNear(y, {win(0, 0), win(0, 1), win(1, 0), win(1, 1)});
}

TEST(Passes, DynamicQuantBareMatMulIntegerNotLowered) {
    // A MatMulInteger whose activation operand is NOT a DynamicQuantizeLinear output (here a plain
    // int8 graph input) does not match the cluster and is left standing for the support report.
    Graph      g;
    TensorDesc xi;
    xi.name    = "xq";
    xi.shape   = {2, 3};
    xi.dtype   = DType::Int8;
    xi.isInput = true;
    TensorId xq = g.addTensor(xi);
    g.inputs    = {xq};
    TensorId wI = addFloatInit(g, "w", {3, 2}, {1, 2, 3, 4, 5, 6});
    g.desc(wI).dtype = DType::Int8;
    TensorId y  = addUnshaped(g, "y");
    g.desc(y).isOutput = true;
    g.desc(y).dtype    = DType::Int32;
    g.outputs          = {y};
    Node mmi;
    mmi.type    = OpType::MatMulInteger;
    mmi.name    = "mmi";
    mmi.inputs  = {xq, wI};
    mmi.outputs = {y};
    g.nodes     = {mmi};
    dequantizeGraph(g);
    EXPECT_EQ(countDynQuant(g), 1) << "a bare MatMulInteger with an escaping int32 output is not lowered";
    EXPECT_EQ(g.nodes[0].type, OpType::MatMulInteger);
}
