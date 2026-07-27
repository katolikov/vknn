// An fp32-pinned tensor read as a fused-pointwise OPERAND.
//
// markFp32 gives a pinned tensor an fp32 store and bridges every reader that runs fp16 with a
// narrowing ConvertDtype. A pointwise unit's operand slots are reads like any other, but the unit
// reaches them through pwOperandBuf, which hands the kernel the tensor's device buffer directly --
// so if the pin survives to the operand slot, an fp16 kernel decodes fp32 words as float16_t and
// the result is not approximately wrong, it is garbage.
//
// The pin is exactly what a caller reaches for to RAISE accuracy, so silently destroying it is the
// worst possible failure. These tests pin the contract on the graph the engine actually builds.
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {
    constexpr int64_t kC = 8, kH = 12, kW = 16;

    // Conv -> Sigmoid -> (mask) multiplied into the conv's own output: the per-pixel weighting an
    // image pipeline is built from, with the mask named to match the built-in selective-fp32 preset
    // so Precision::Normal pins it. The Mul is a pointwise unit whose operand IS the pinned tensor.
    Graph buildMaskedGraph() {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {1, kC, kH, kW};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};

        TensorDesc wi;
        wi.name          = "w";
        wi.shape         = {1, kC, 1, 1};
        wi.isInitializer = true;
        TensorId   w     = g.addTensor(wi);
        HostBuffer hb;
        hb.resizeElems((size_t) kC, DType::Float32);
        for (int64_t i = 0; i < kC; ++i)
        {
            hb.f32()[i] = 0.25f + 0.1f * (float) i;
        }
        g.initializers[w] = hb;

        TensorId logit = g.addTensor({.name = "logit"});
        Node     conv;
        conv.type    = OpType::Conv;
        conv.name    = "maskconv";
        conv.inputs  = {x, w};
        conv.outputs = {logit};
        g.nodes.push_back(conv);

        // "/enc/Mul_" is in mixedPrecisionFp32Tensors(), so this tensor is both kept materialized by
        // the pointwise fusion and pinned to fp32 storage by markFp32 under Precision::Normal.
        TensorId mask = g.addTensor({.name = "/enc/Mul_mask"});
        Node     sig;
        sig.type    = OpType::Unary;
        sig.name    = "sigmoid";
        sig.subOp   = (int) UnaryType::Sigmoid;
        sig.inputs  = {logit};
        sig.outputs = {mask};
        g.nodes.push_back(sig);

        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     mul;
        mul.type    = OpType::Binary;
        mul.name    = "weight";
        mul.subOp   = (int) BinaryType::Mul;
        mul.inputs  = {x, mask};
        mul.outputs = {y};
        g.nodes.push_back(mul);
        g.outputs = {y};
        return g;
    }

    std::vector<float> inputValues() {
        std::vector<float> v((size_t) (kC * kH * kW));
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = 0.5f + 0.25f * (float) (i % 7);
        }
        return v;
    }

    void bind(std::vector<IOTensor> &in, const std::vector<float> &src) {
        in.resize(1);
        in[0].name  = "x";
        in[0].shape = {1, kC, kH, kW};
        in[0].dtype = DType::Float32;
        in[0].data.resize(src.size() * sizeof(float));
        std::memcpy(in[0].data.data(), src.data(), src.size() * sizeof(float));
    }
} // namespace

// The pin must not change what the graph computes. A pinned operand read at the wrong precision
// shows up here as a gross mismatch, not a rounding difference.
TEST(Fp32PinPwOperand, PinnedOperandStillMatchesTheCpuReference) {
    const std::vector<float> src = inputValues();

    Config gpu;
    gpu.backend                 = BackendKind::Vulkan;
    gpu.allowCpuFallback        = false;
    gpu.precision               = Precision::Normal; // selective fp32: the built-in preset pins /enc/Mul_*
    Graph                    gg = buildMaskedGraph();
    std::unique_ptr<Session> gsess;
    try
    { gsess = Session::create(std::move(gg), gpu); } catch (const std::exception &)
    { gsess.reset(); }
    if (!gsess)
    {
        GTEST_SKIP() << "no Vulkan device";
    }
    std::vector<IOTensor> gin, gout;
    bind(gin, src);
    ASSERT_EQ(gsess->run(gin, gout), Status::Ok);
    ASSERT_EQ(gout.size(), 1u);

    Config cpu;
    cpu.backend = BackendKind::Cpu;
    Graph cg    = buildMaskedGraph();
    auto  csess = Session::create(std::move(cg), cpu);
    ASSERT_NE(csess, nullptr);
    std::vector<IOTensor> cin, cout;
    bind(cin, src);
    ASSERT_EQ(csess->run(cin, cout), Status::Ok);

    const float *got      = gout[0].f32();
    const float *ref      = cout[0].f32();
    size_t       reported = 0;
    for (size_t i = 0; i < src.size() && reported < 8; ++i)
    {
        if (std::abs(got[i] - ref[i]) > 2e-2f)
        {
            ++reported;
            ADD_FAILURE() << "element " << i << ": gpu=" << got[i] << " cpu=" << ref[i] << " -- an fp32-pinned pointwise operand is being decoded at the wrong precision";
        }
    }
}

// The same graph with NO pin is the control: if this one fails too, the fault is not the pin.
TEST(Fp32PinPwOperand, UnpinnedControlMatchesTheCpuReference) {
    const std::vector<float> src = inputValues();

    Config gpu;
    gpu.backend                 = BackendKind::Vulkan;
    gpu.allowCpuFallback        = false;
    gpu.precision               = Precision::Low; // no selective-fp32 set at all
    Graph                    gg = buildMaskedGraph();
    std::unique_ptr<Session> gsess;
    try
    { gsess = Session::create(std::move(gg), gpu); } catch (const std::exception &)
    { gsess.reset(); }
    if (!gsess)
    {
        GTEST_SKIP() << "no Vulkan device";
    }
    std::vector<IOTensor> gin, gout;
    bind(gin, src);
    ASSERT_EQ(gsess->run(gin, gout), Status::Ok);

    Config cpu;
    cpu.backend = BackendKind::Cpu;
    Graph cg    = buildMaskedGraph();
    auto  csess = Session::create(std::move(cg), cpu);
    ASSERT_NE(csess, nullptr);
    std::vector<IOTensor> cin, cout;
    bind(cin, src);
    ASSERT_EQ(csess->run(cin, cout), Status::Ok);

    const float *got = gout[0].f32();
    const float *ref = cout[0].f32();
    for (size_t i = 0; i < src.size(); ++i)
    {
        ASSERT_NEAR(got[i], ref[i], 2e-2f) << "element " << i << " (unpinned control)";
    }
}
