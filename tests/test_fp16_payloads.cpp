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
