// fp16-compiled initializer payloads on the CPU path. vknn_compile --fp16 rewrites every fp32
// initializer as Float16 (convertInitializersFp16), including rank-0 scalar constants that are
// tensor operands of pointwise chains (a normalization head like (x - 127.5) / 255). These tests
// pin the whole pipeline: the conversion keeps a rank-0 payload, and the CPU runtime decodes fp16
// payloads for every reader — the fused pointwise VM, the unfused Binary ops, and a legacy
// fusedResidual edge that references an initializer outside node.inputs.
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    // x[1,1,2,2] -> Sub(c0=127.5 rank-0) -> Div(c1=255 rank-0) -> y, the uint8-normalization head
    // shape. Both constants are rank-0 (shape []), matching how ONNX exporters emit scalar operands.
    Graph normHeadGraph() {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {1, 1, 2, 2};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs.push_back(x);
        auto scalar = [&](const char *name, float v) {
            TensorDesc d;
            d.name          = name;
            d.shape         = {}; // rank-0
            d.isInitializer = true;
            TensorId   id   = g.addTensor(d);
            HostBuffer hb;
            hb.resizeElems(1, DType::Float32);
            hb.f32()[0]        = v;
            g.initializers[id] = hb;
            return id;
        };
        TensorId   c0 = scalar("c0", 127.5f);
        TensorId   c1 = scalar("c1", 255.f);
        TensorDesc ti;
        ti.name    = "t";
        TensorId t = g.addTensor(ti);
        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     sub;
        sub.type    = OpType::Binary;
        sub.name    = "sub";
        sub.subOp   = (int) BinaryType::Sub;
        sub.inputs  = {x, c0};
        sub.outputs = {t};
        g.nodes.push_back(sub);
        Node div;
        div.type    = OpType::Binary;
        div.name    = "div";
        div.subOp   = (int) BinaryType::Div;
        div.inputs  = {t, c1};
        div.outputs = {y};
        g.nodes.push_back(div);
        g.outputs = {y};
        return g;
    }

    // Compile the graph the way vknn_compile --fp16 does (standard passes, then the fp16 initializer
    // conversion), round-trip it through a .vxm, and run it on the CPU backend.
    std::vector<float> compileFp16AndRunCpu(Graph g, bool fusePointwise, const std::vector<float> &xdata) {
        PassOptions opt;
        opt.fusePointwiseChains = fusePointwise;
        runStandardPasses(g, opt);
        convertInitializersFp16(g);
        std::string path = testing::TempDir() + "vknn_fp16_payloads_" + (fusePointwise ? "fused" : "unfused") + ".vxm";
        EXPECT_TRUE(saveGraphBin(g, path));
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::createFromVxm(path, cfg);
        std::remove(path.c_str());
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = "x";
        in.shape = {1, 1, 2, 2};
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
        return std::vector<float>(o, o + numElements(outs[0].shape));
    }

    // Compile --fp16 (standard passes + fp16 conversion), round-trip through a .vxm, and CPU-run with
    // a single named fp32 input of the given shape. A general twin of compileFp16AndRunCpu for graphs
    // whose input is not the fixed x[1,1,2,2].
    std::vector<float> compileFp16AndRunCpu2(Graph g, const std::vector<float> &xdata, const std::string &inName, const Shape &inShape) {
        runStandardPasses(g);
        convertInitializersFp16(g);
        std::string path = testing::TempDir() + "vknn_fp16_payloads_e2e.vxm";
        EXPECT_TRUE(saveGraphBin(g, path));
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::createFromVxm(path, cfg);
        std::remove(path.c_str());
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = inName;
        in.shape = inShape;
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
        return std::vector<float>(o, o + numElements(outs[0].shape));
    }

} // namespace

// The fp16 conversion keeps a rank-0 scalar's single element: the payload shrinks to one fp16
// value, not to zero bytes (numElements() is 0 for an empty shape, so a shape-derived element
// count would drop the value entirely).
TEST(Fp16Payloads, ScalarInitializerSurvivesConversion) {
    Graph      g;
    TensorDesc d;
    d.name          = "c";
    d.shape         = {}; // rank-0
    d.isInitializer = true;
    TensorId   id   = g.addTensor(d);
    HostBuffer hb;
    hb.resizeElems(1, DType::Float32);
    hb.f32()[0]        = 127.5f;
    g.initializers[id] = hb;

    Fp16ConvertStats st = convertInitializersFp16(g);
    EXPECT_EQ(st.converted, 1);
    ASSERT_EQ(g.desc(id).dtype, DType::Float16);
    ASSERT_EQ(g.initializers[id].bytes.size(), 2u); // one fp16 element
    fp16_t h = *reinterpret_cast<const fp16_t *>(g.initializers[id].bytes.data());
    EXPECT_EQ(halfToFloat(h), 127.5f); // exactly representable in fp16
}

// Constant values beyond the fp16 range clamp to the finite extreme (+/-65504), not to +/-inf. The
// attention-mask idiom (1 - mask) * -3.4028235e38 relies on this: an fp16 -inf multiplied by the
// non-pad 0 yields 0*-inf = NaN, poisoning the softmax and the whole output. Clamping the constant
// to -65504 keeps 0*-65504 = 0, while a -65504 additive bias still drives exp() to 0 for pad tokens.
// A pre-existing +/-inf in the fp32 payload reproduces the same 0*inf hazard, so it clamps too; a
// normal value is untouched.
TEST(Fp16Payloads, OutOfRangeConstantClampsToFiniteExtreme) {
    Graph      g;
    TensorDesc d;
    d.name          = "c";
    d.shape         = {5};
    d.isInitializer = true;
    TensorId   id   = g.addTensor(d);
    HostBuffer hb;
    hb.resizeElems(5, DType::Float32);
    hb.f32()[0] = -3.4028235e38f;      // fp32 lowest, overflows fp16 -> would be -inf
    hb.f32()[1] = 3.4028235e38f;       // fp32 max, overflows fp16 -> would be +inf
    hb.f32()[2] = 1.5f;                // normal, exactly representable
    hb.f32()[3] = -std::numeric_limits<float>::infinity(); // already -inf
    hb.f32()[4] = std::numeric_limits<float>::infinity();  // already +inf
    g.initializers[id] = hb;

    Fp16ConvertStats st = convertInitializersFp16(g);
    EXPECT_EQ(st.converted, 1);
    ASSERT_EQ(g.desc(id).dtype, DType::Float16);
    ASSERT_EQ(g.initializers[id].bytes.size(), 10u); // five fp16 elements
    const fp16_t *h = reinterpret_cast<const fp16_t *>(g.initializers[id].bytes.data());

    // The clamped values are the finite fp16 extremes, never inf.
    EXPECT_EQ(h[0], 0xFBFFu);              // -65504
    EXPECT_EQ(h[1], 0x7BFFu);              // +65504
    EXPECT_EQ(halfToFloat(h[0]), -65504.0f);
    EXPECT_EQ(halfToFloat(h[1]), 65504.0f);
    EXPECT_FALSE(std::isinf(halfToFloat(h[0])));
    EXPECT_FALSE(std::isinf(halfToFloat(h[1])));

    // The normal value is unchanged.
    EXPECT_EQ(halfToFloat(h[2]), 1.5f);

    // Pre-existing infinities clamp to the finite extreme as well.
    EXPECT_EQ(h[3], 0xFBFFu);
    EXPECT_EQ(h[4], 0x7BFFu);
    EXPECT_FALSE(std::isinf(halfToFloat(h[3])));
    EXPECT_FALSE(std::isinf(halfToFloat(h[4])));
}

// End-to-end mechanism: the attention-mask extended-bias idiom (1 - mask) * -3.4028235e38 compiled
// --fp16 and run on the CPU backend. For a non-pad token (mask=1) the bias must be a finite 0, never
// NaN. Before the clamp the -3.4e38 constant narrowed to fp16 -inf, so 0 * -inf produced NaN; with
// the constant clamped to -65504, 0 * -65504 = 0. This is the runtime poison the fix removes.
TEST(Fp16Payloads, MaskBiasIdiomNoNaNFp16Cpu) {
    Graph      g;
    TensorDesc mi;
    mi.name    = "mask";
    mi.shape   = {1, 4};
    mi.isInput = true;
    TensorId mask = g.addTensor(mi);
    g.inputs.push_back(mask);
    auto scalar = [&](const char *name, float v) {
        TensorDesc d;
        d.name          = name;
        d.shape         = {};
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems(1, DType::Float32);
        hb.f32()[0]        = v;
        g.initializers[id] = hb;
        return id;
    };
    TensorId one = scalar("one", 1.0f);
    TensorId neg = scalar("neg", -3.4028235e38f); // fp32 lowest, the mask-bias constant
    TensorDesc si;
    si.name    = "sub_out";
    TensorId subOut = g.addTensor(si);
    TensorDesc yo;
    yo.name     = "y";
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     sub; // 1 - mask
    sub.type    = OpType::Binary;
    sub.name    = "sub";
    sub.subOp   = (int) BinaryType::Sub;
    sub.inputs  = {one, mask};
    sub.outputs = {subOut};
    g.nodes.push_back(sub);
    Node mul; // (1 - mask) * -3.4e38
    mul.type    = OpType::Binary;
    mul.name    = "mul";
    mul.subOp   = (int) BinaryType::Mul;
    mul.inputs  = {subOut, neg};
    mul.outputs = {y};
    g.nodes.push_back(mul);
    g.outputs = {y};

    // mask = [1, 1, 0, 0]: two non-pad tokens (expect finite 0) and two pad tokens (expect the
    // large-negative bias, finite, never inf/NaN).
    auto out = compileFp16AndRunCpu2(g, {1.f, 1.f, 0.f, 0.f}, "mask", {1, 4});
    ASSERT_EQ(out.size(), 4u);
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_FALSE(std::isnan(out[i])) << "i=" << i;
        EXPECT_FALSE(std::isinf(out[i])) << "i=" << i;
    }
    EXPECT_EQ(out[0], 0.0f); // non-pad: 0 * -65504 = 0, not NaN
    EXPECT_EQ(out[1], 0.0f);
    EXPECT_LT(out[2], -1000.0f); // pad: the large-negative bias
    EXPECT_LT(out[3], -1000.0f);
}

// Fused: the head becomes one standalone FusedPointwise unit whose Sub/Div operands are the fp16
// rank-0 constants. The CPU pointwise VM must see them decoded to fp32, not as empty/garbage
// payloads that silently read as 0 (which turns the unit into (x-0)/0 = inf).
TEST(Fp16Payloads, FusedPointwiseScalarOperandsCpu) {
    auto out = compileFp16AndRunCpu(normHeadGraph(), true, {0.f, 127.5f, 255.f, 510.f});
    ASSERT_EQ(out.size(), 4u);
    const float ref[4] = {-0.5f, 0.f, 0.5f, 1.5f};
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(out[i], ref[i], 1e-6f) << "i=" << i;
    }
}

// Unfused twin (fusion off): the same constants feed plain Binary Sub/Div nodes. The CPU Binary op
// reads the operand via the session pool, so a rank-0 fp16 initializer that decodes to an empty
// host buffer is a null-pointer read (the --fp16 --no-fuse-pointwise segfault class).
TEST(Fp16Payloads, UnfusedScalarBinaryCpu) {
    auto out = compileFp16AndRunCpu(normHeadGraph(), false, {0.f, 127.5f, 255.f, 510.f});
    ASSERT_EQ(out.size(), 4u);
    const float ref[4] = {-0.5f, 0.f, 0.5f, 1.5f};
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(out[i], ref[i], 1e-6f) << "i=" << i;
    }
}

// A fusedResidual edge can reference an initializer without that tensor appearing in node.inputs
// (legacy .vxm encoding; model_io round-trips the edge). The session's CPU pool load must
// materialize it like any other CPU-consumed initializer, fp16-decoded.
TEST(Fp16Payloads, FusedResidualInitializerCpu) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 1, 2, 2};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorDesc wi;
    wi.name          = "w";
    wi.shape         = {1, 1, 1, 1};
    wi.isInitializer = true;
    TensorId   w     = g.addTensor(wi);
    HostBuffer wb;
    wb.resizeElems(1, DType::Float32);
    wb.f32()[0]       = 2.f;
    g.initializers[w] = wb;
    TensorDesc ri;
    ri.name          = "r";
    ri.shape         = {1, 1, 2, 2};
    ri.isInitializer = true;
    TensorId   r     = g.addTensor(ri);
    HostBuffer rb;
    rb.resizeElems(4, DType::Float32);
    for (int i = 0; i < 4; ++i)
    {
        rb.f32()[i] = 10.f * (float) (i + 1);
    }
    g.initializers[r] = rb;
    TensorDesc yo;
    yo.name     = "y";
    yo.shape    = {1, 1, 2, 2};
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     conv;
    conv.type          = OpType::Conv;
    conv.name          = "conv";
    conv.inputs        = {x, w};
    conv.outputs       = {y};
    conv.fusedResidual = r; // not in inputs: the legacy residual encoding
    g.nodes.push_back(conv);
    g.outputs = {y};
    convertInitializersFp16(g);

    std::string path = testing::TempDir() + "vknn_fp16_payloads_residual.vxm";
    ASSERT_TRUE(saveGraphBin(g, path));
    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::createFromVxm(path, cfg);
    std::remove(path.c_str());
    ASSERT_TRUE(sess);
    IOTensor in;
    in.name  = "x";
    in.shape = {1, 1, 2, 2};
    in.dtype = DType::Float32;
    in.data.resize(4 * 4);
    for (int i = 0; i < 4; ++i)
    {
        reinterpret_cast<float *>(in.data.data())[i] = (float) (i + 1);
    }
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_FALSE(outs.empty());
    for (int i = 0; i < 4; ++i)
    {
        // conv(1x1, w=2) + residual: 2*(i+1) + 10*(i+1)
        EXPECT_NEAR(outs[0].f32()[i], 2.f * (float) (i + 1) + 10.f * (float) (i + 1), 1e-6f) << "i=" << i;
    }
}

// A rank-0 (shape []) scalar operand carried through an elementwise CPU op during const-fold: the
// op allocates and iterates by numElements(shape), which is 0 for an empty shape, so the folded
// output is an empty (null-data) host buffer. A later const-folded consumer of that output reads
// its host pointer and null-derefs. This is the detection-head (YOLO box-decode) crash class:
// Cast(scalar) -> Div, both foldable, feeds a SIGSEGV inside the Binary kernel. The op must treat
// a rank-0 tensor as its one element, so the folded scalar carries a value, not zero bytes.
TEST(Fp16Payloads, ConstFoldScalarCastThenBinary) {
    Graph      g;
    auto       init = [&](const char *name, const Shape &shape, float v) {
        TensorDesc d;
        d.name          = name;
        d.shape         = shape;
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems(1, DType::Float32);
        hb.f32()[0]        = v;
        g.initializers[id] = hb;
        return id;
    };
    // Numerator carries a [1] shape and the denominator a rank-0 [] shape: broadcasting [1] against
    // the empty-scalar operand yields a [1] output the Binary loop actually iterates, so a null-data
    // denominator is dereferenced (the tiny-yolov3 arange box-decode divide).
    TensorId   num   = init("num", {1}, 6.f);   // numerator, shape [1]
    TensorId   denIn = init("den_in", {}, 3.f); // pre-cast denominator scalar, rank-0
    // Cast(den_in) -> den : a float->float cast of a rank-0 scalar, foldable by const-fold.
    TensorDesc dc;
    dc.name    = "den";
    TensorId den = g.addTensor(dc);
    Node       cast;
    cast.type            = OpType::Cast;
    cast.name            = "cast_den";
    cast.inputs          = {denIn};
    cast.outputs         = {den};
    cast.attr.map["to"]  = [] { Attr a; a.kind = Attr::Int; a.i = 1 /*FLOAT*/; return a; }();
    g.nodes.push_back(cast);
    // Div(num, den) -> y : both operands are now known constants, folded on the CPU.
    TensorDesc yo;
    yo.name     = "y";
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     div;
    div.type    = OpType::Binary;
    div.name    = "div";
    div.subOp   = (int) BinaryType::Div;
    div.inputs  = {num, den};
    div.outputs = {y};
    g.nodes.push_back(div);
    g.outputs = {y};

    // runStandardPasses const-folds the whole chain; before the fix this SIGSEGVs in the Binary kernel.
    ASSERT_NO_FATAL_FAILURE(runStandardPasses(g));
    // The chain folds to the scalar 6/3 = 2, kept as a one-element initializer (not zero bytes).
    ASSERT_TRUE(g.isInitializer(y));
    ASSERT_GE(g.initializers[y].bytes.size(), 4u);
    EXPECT_NEAR(g.initializers[y].f32()[0], 2.f, 1e-6f);
}

// The producer half in isolation: an elementwise CPU op (Cast) given a rank-0 scalar must fold to a
// one-element initializer, never an empty payload. Guards the class at the source so no consumer can
// inherit a null-data scalar.
TEST(Fp16Payloads, ConstFoldScalarCastKeepsElement) {
    Graph      g;
    TensorDesc si;
    si.name          = "s";
    si.shape         = {}; // rank-0
    si.isInitializer = true;
    TensorId   s     = g.addTensor(si);
    HostBuffer sb;
    sb.resizeElems(1, DType::Float32);
    sb.f32()[0]       = 42.f;
    g.initializers[s] = sb;
    TensorDesc yo;
    yo.name     = "y";
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     cast;
    cast.type           = OpType::Cast;
    cast.name           = "cast_s";
    cast.inputs         = {s};
    cast.outputs        = {y};
    cast.attr.map["to"] = [] { Attr a; a.kind = Attr::Int; a.i = 1 /*FLOAT*/; return a; }();
    g.nodes.push_back(cast);
    g.outputs = {y};

    ASSERT_EQ(constFold(g), 1); // the Cast folds
    ASSERT_TRUE(g.isInitializer(y));
    ASSERT_GE(g.initializers[y].bytes.size(), 4u); // one fp32 element, not empty
    EXPECT_NEAR(g.initializers[y].f32()[0], 42.f, 1e-6f);
}
