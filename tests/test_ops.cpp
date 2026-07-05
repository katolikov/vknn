// vknn operator unit tests (host, CPU backend). One self-contained graph per op, checked against a
// reference (hand-computed, or an onnxruntime golden noted inline). The CPU op is the correctness
// oracle the Vulkan path is diffed against on-device, so a regression here surfaces before the GPU.
//
// ConvTranspose covers the explicit-pad, auto_pad SAME, and output_shape paths -- SAME and
// output_shape yield the same output size here but different values, so a regression in either
// attribute is caught.
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
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

// --- GlobalAveragePool over HxW. ---
TEST(Ops, GlobalAveragePool) {
    auto out = runOp(OpType::GlobalAvgPool, 0, {}, {1, 1, 2, 2}, {1, 2, 3, 4}, {});
    ASSERT_EQ(out.shape, (std::vector<int64_t> {1, 1, 1, 1}));
    expectNear(out.data, {2.5f}, 1e-5f);
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
