// Pinned host memory (include/vknn/pinned_host_memory.h) at the API boundary: the block's layout
// contract, and the copy fallback every backend without host import takes — a pinned input reads
// like a payload input, a pinned output lands in the caller's block with the IOTensor's `data`
// left empty, through both the Session and the Model API. The GPU binding itself needs a device
// and is gated on it (gate_cpu_gpu.sh with a pinned run).
#include "vknn/graph.h"
#include "vknn/model.h"
#include "vknn/pinned_host_memory.h"
#include "vknn/session.h"
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>

using namespace vknn;

namespace {
    constexpr int64_t kN = 1, kC = 4, kH = 2, kW = 3;
    constexpr int64_t kElems = kN * kC * kH * kW;

    // x -> Relu -> y
    Graph reluGraph() {
        Graph      g;
        TensorDesc dx;
        dx.name    = "x";
        dx.shape   = {kN, kC, kH, kW};
        dx.isInput = true;
        TensorId x = g.addTensor(dx);
        TensorDesc dy;
        dy.name    = "y";
        dy.shape   = {kN, kC, kH, kW};
        TensorId y = g.addTensor(dy);
        Node relu;
        relu.type    = OpType::Relu;
        relu.name    = "relu";
        relu.inputs  = {x};
        relu.outputs = {y};
        g.nodes      = {relu};
        g.inputs     = {x};
        g.outputs    = {y};
        return g;
    }
    std::vector<float> sample() {
        std::vector<float> v((size_t) kElems);
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = (float) i - 10.5f;
        }
        return v;
    }
    Config cpuConfig() {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        return cfg;
    }
} // namespace

TEST(PinnedHostMemory, BlocksArePageAlignedZeroedAndGranuleSized) {
    auto block = PinnedHostMemory::alloc(1000);
    ASSERT_NE(block, nullptr);
    EXPECT_EQ(block->bytes(), (size_t) 1000);
    EXPECT_EQ(block->capacity(), kPinnedHostAlignment);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(block->data()) % kPinnedHostAlignment, (uintptr_t) 0);
    for (size_t i = 0; i < block->capacity(); ++i)
    {
        ASSERT_EQ(block->data()[i], 0) << "byte " << i;
    }
    auto exact = PinnedHostMemory::alloc(2 * kPinnedHostAlignment);
    ASSERT_NE(exact, nullptr);
    EXPECT_EQ(exact->capacity(), 2 * kPinnedHostAlignment) << "a multiple of the granule is not rounded up";
    auto empty = PinnedHostMemory::alloc(0);
    ASSERT_NE(empty, nullptr);
    EXPECT_EQ(empty->capacity(), kPinnedHostAlignment) << "a zero request still allocates one granule";
}

TEST(PinnedHostMemory, SessionReadsAPinnedInputAndWritesAPinnedOutput) {
    auto sess = Session::create(reluGraph(), cpuConfig());
    ASSERT_NE(sess, nullptr);
    const std::vector<float> x = sample();
    // Reference: payload input, payload output.
    std::vector<IOTensor> in(1), out;
    in[0].name  = "x";
    in[0].shape = {kN, kC, kH, kW};
    in[0].dtype = DType::Float32;
    in[0].data.resize(x.size() * sizeof(float));
    std::memcpy(in[0].data.data(), x.data(), in[0].data.size());
    ASSERT_EQ(sess->run(in, out), Status::Ok);
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].data.size(), x.size() * sizeof(float));
    // Pinned input, pinned output.
    std::vector<IOTensor> pin(1), pout(1);
    pin[0].name   = "x";
    pin[0].shape  = {kN, kC, kH, kW};
    pin[0].dtype  = DType::Float32;
    pin[0].pinned = PinnedHostMemory::alloc(x.size() * sizeof(float));
    std::memcpy(pin[0].pinned->data(), x.data(), x.size() * sizeof(float));
    pout[0].name   = "y";
    pout[0].pinned = PinnedHostMemory::alloc(x.size() * sizeof(float));
    auto outBlock  = pout[0].pinned;
    for (int run = 0; run < 2; ++run) // the second run reuses both blocks
    {
        ASSERT_EQ(sess->run(pin, pout), Status::Ok);
        ASSERT_EQ(pout.size(), 1u);
        EXPECT_TRUE(pout[0].data.empty()) << "a pinned output delivers into its block, not into data";
        ASSERT_EQ(pout[0].pinned, outBlock) << "the block travels back on the IOTensor";
        EXPECT_EQ(pout[0].dtype, DType::Float32);
        EXPECT_EQ(pout[0].shape, (Shape {kN, kC, kH, kW}));
        EXPECT_EQ(std::memcmp(outBlock->data(), out[0].data.data(), out[0].data.size()), 0) << "run " << run;
        const float *y = reinterpret_cast<const float *>(outBlock->data());
        for (size_t i = 0; i < x.size(); ++i)
        {
            ASSERT_FLOAT_EQ(y[i], x[i] > 0.f ? x[i] : 0.f) << "element " << i;
        }
    }
    // A pinned input with a payload output.
    std::vector<IOTensor> mixedOut;
    ASSERT_EQ(sess->run(pin, mixedOut), Status::Ok);
    ASSERT_EQ(mixedOut.size(), 1u);
    EXPECT_EQ(mixedOut[0].pinned, nullptr);
    ASSERT_EQ(mixedOut[0].data.size(), out[0].data.size());
    EXPECT_EQ(std::memcmp(mixedOut[0].data.data(), out[0].data.data(), out[0].data.size()), 0);
}

TEST(PinnedHostMemory, ModelApiPinnedTensorsAliasTheirBlocks) {
    // The Model API loads from a file: round-trip the graph through the compiled format.
    Config cfg  = cpuConfig();
    auto   sess = Session::create(reluGraph(), cfg);
    ASSERT_NE(sess, nullptr);
    const std::string path = (std::filesystem::temp_directory_path() / "vknn_pinned_relu.vxm").string();
    ASSERT_TRUE(sess->saveOptimized(path));
    Model model = Model::load(path, cfg);
    ASSERT_FALSE(model.inputs().empty());
    const std::vector<float> x  = sample();
    Tensor                   in = Tensor::pinned({kN, kC, kH, kW}, "x");
    ASSERT_NE(in.pinnedBlock(), nullptr);
    EXPECT_EQ(in.size(), kElems) << "a pinned tensor counts its shape";
    EXPECT_FALSE(in.empty());
    EXPECT_TRUE(in.values().empty()) << "the values live in the block, not in a host vector";
    std::memcpy(in.data(), x.data(), x.size() * sizeof(float));
    Tensor outBinding = Tensor::toPinned({kN, kC, kH, kW}, "y");
    auto   outBlock   = outBinding.pinnedBlock();
    ASSERT_NE(outBlock, nullptr);
    std::vector<Tensor> outs = model.run({in}, {outBinding});
    ASSERT_EQ(outs.size(), 1u);
    EXPECT_EQ(outs[0].pinnedBlock(), outBlock) << "the returned output aliases the caller's block";
    EXPECT_EQ(outs[0].data(), reinterpret_cast<const float *>(outBlock->data()));
    ASSERT_EQ(outs[0].size(), kElems);
    for (int64_t i = 0; i < kElems; ++i)
    {
        ASSERT_FLOAT_EQ(outs[0][i], x[(size_t) i] > 0.f ? x[(size_t) i] : 0.f) << "element " << i;
    }
    // Without an output binding the same pinned input yields an ordinary host tensor.
    std::vector<Tensor> plain = model.run({in});
    ASSERT_EQ(plain.size(), 1u);
    EXPECT_EQ(plain[0].pinnedBlock(), nullptr);
    ASSERT_EQ(plain[0].size(), kElems);
    EXPECT_EQ(std::memcmp(plain[0].data(), outs[0].data(), (size_t) kElems * sizeof(float)), 0);
}
