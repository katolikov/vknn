// Engine-side output argmax (Session::setOutputArgMax / readOutputArgMax): a registered output
// stays engine-resident (its run() entry carries metadata but no data) and the engine reduces it
// to {index, value} with first-occurrence tie semantics — the element a left-to-right host scan
// with a strictly-greater test selects. These host tests (CPU backend, the host reduction path)
// pin the contract:
//   - the reduced index/value equal the host scan over the same run's full output,
//   - equal maxima resolve to the lowest index; NaN elements are never selected,
//   - the registered output's run() entry carries no data; an unregistered session is unchanged,
//   - validation rejects unknown names, non-vector shapes, and out-of-range buckets,
//   - registration is idempotent and reading an unregistered name is NotFound.
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    IOTensor f32Tensor(const std::string &name, const Shape &shape, const std::vector<float> &values) {
        IOTensor io;
        io.name  = name;
        io.shape = shape;
        io.dtype = DType::Float32;
        io.data.resize(values.size() * sizeof(float));
        std::memcpy(io.data.data(), values.data(), io.data.size());
        return io;
    }

    const IOTensor *outByName(const std::vector<IOTensor> &outs, const std::string &name) {
        for (const IOTensor &o: outs)
        {
            if (o.name == name)
            {
                return &o;
            }
        }
        return nullptr;
    }

    // Identity-shaped vector graph: x [1, elems] -> Relu -> scores [1, elems].
    Graph makeVectorGraph(int64_t elems) {
        Graph      g;
        TensorDesc xDesc;
        xDesc.name    = "x";
        xDesc.shape   = {1, elems};
        xDesc.isInput = true;
        TensorId x    = g.addTensor(xDesc);
        g.inputs      = {x};

        TensorDesc scoresDesc;
        scoresDesc.name     = "scores";
        scoresDesc.isOutput = true;
        TensorId scores     = g.addTensor(scoresDesc);

        Node relu;
        relu.type    = OpType::Relu;
        relu.name    = "relu";
        relu.inputs  = {x};
        relu.outputs = {scores};
        g.nodes.push_back(relu);
        g.outputs = {scores};
        return g;
    }

    // Matrix graph whose output is not effectively one-dimensional: x [2, elems] -> Relu.
    Graph makeMatrixGraph(int64_t rows, int64_t elems) {
        Graph      g;
        TensorDesc xDesc;
        xDesc.name    = "x";
        xDesc.shape   = {rows, elems};
        xDesc.isInput = true;
        TensorId x    = g.addTensor(xDesc);
        g.inputs      = {x};
        TensorDesc outDesc;
        outDesc.name     = "scores";
        outDesc.isOutput = true;
        TensorId out     = g.addTensor(outDesc);
        Node     relu;
        relu.type    = OpType::Relu;
        relu.name    = "relu";
        relu.inputs  = {x};
        relu.outputs = {out};
        g.nodes.push_back(relu);
        g.outputs = {out};
        return g;
    }

    std::unique_ptr<Session> cpuSession(Graph &&g) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        return Session::create(std::move(g), cfg);
    }

} // namespace

TEST(OutputArgMax, MatchesHostScanAndSuppressesData) {
    auto sess = cpuSession(makeVectorGraph(16));
    ASSERT_TRUE(sess);
    ASSERT_EQ(sess->setOutputArgMax(0, "scores"), Status::Ok);

    std::vector<float> values(16, 0.25f);
    values[11] = 7.5f;
    values[3]  = 6.0f;
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({f32Tensor("x", {1, 16}, values)}, outs), Status::Ok);

    const IOTensor *scores = outByName(outs, "scores");
    ASSERT_NE(scores, nullptr);
    EXPECT_TRUE(scores->data.empty()); // engine-resident: metadata only

    int64_t index = -1;
    float   value = 0.0f;
    ASSERT_EQ(sess->readOutputArgMax("scores", index, value), Status::Ok);
    EXPECT_EQ(index, 11);
    EXPECT_FLOAT_EQ(value, 7.5f);
}

TEST(OutputArgMax, FirstOccurrenceWinsOnTies) {
    auto sess = cpuSession(makeVectorGraph(8));
    ASSERT_TRUE(sess);
    ASSERT_EQ(sess->setOutputArgMax(0, "scores"), Status::Ok);

    // Two equal maxima: the host scan's strictly-greater test keeps the first.
    std::vector<float>    values {0.0f, 4.0f, 1.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({f32Tensor("x", {1, 8}, values)}, outs), Status::Ok);

    int64_t index = -1;
    float   value = 0.0f;
    ASSERT_EQ(sess->readOutputArgMax("scores", index, value), Status::Ok);
    EXPECT_EQ(index, 1);
    EXPECT_FLOAT_EQ(value, 4.0f);
}

TEST(OutputArgMax, NanElementsAreNeverSelected) {
    auto sess = cpuSession(makeVectorGraph(4));
    ASSERT_TRUE(sess);
    ASSERT_EQ(sess->setOutputArgMax(0, "scores"), Status::Ok);

    // Relu(NaN) stays NaN (max(NaN, 0) semantics differ per backend), so feed the NaN through and
    // rely on the documented rule only: a NaN never replaces the running maximum.
    const float           nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float>    values {1.0f, nan, 5.0f, 5.0f};
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({f32Tensor("x", {1, 4}, values)}, outs), Status::Ok);

    int64_t index = -1;
    float   value = 0.0f;
    ASSERT_EQ(sess->readOutputArgMax("scores", index, value), Status::Ok);
    EXPECT_EQ(index, 2);
    EXPECT_FLOAT_EQ(value, 5.0f);
}

TEST(OutputArgMax, UnregisteredSessionKeepsFullOutputs) {
    auto sess = cpuSession(makeVectorGraph(8));
    ASSERT_TRUE(sess);
    std::vector<float>    values {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({f32Tensor("x", {1, 8}, values)}, outs), Status::Ok);
    const IOTensor *scores = outByName(outs, "scores");
    ASSERT_NE(scores, nullptr);
    EXPECT_EQ(scores->data.size(), 8 * sizeof(float)); // the unused-feature guarantee

    int64_t index = -1;
    float   value = 0.0f;
    EXPECT_EQ(sess->readOutputArgMax("scores", index, value), Status::NotFound);
}

TEST(OutputArgMax, ValidationAndIdempotence) {
    auto sess = cpuSession(makeVectorGraph(8));
    ASSERT_TRUE(sess);
    EXPECT_EQ(sess->setOutputArgMax(0, "no_such_output"), Status::NotFound);
    EXPECT_EQ(sess->setOutputArgMax(0, "x"), Status::NotFound); // an input is not an output
    EXPECT_EQ(sess->setOutputArgMax(3, "scores"), Status::InvalidArgument);
    EXPECT_EQ(sess->setOutputArgMax(0, "scores"), Status::Ok);
    EXPECT_EQ(sess->setOutputArgMax(0, "scores"), Status::Ok); // idempotent

    auto matrix = cpuSession(makeMatrixGraph(2, 8));
    ASSERT_TRUE(matrix);
    EXPECT_EQ(matrix->setOutputArgMax(0, "scores"), Status::InvalidArgument); // [2,8] is not a vector
}

// Decode-chain surface on the host reduction path: the step-indexed read validates its bounds
// against Config::decodeChainSteps, step 0 equals the legacy read, and any later step needs the
// device reduction path (the CPU backend holds one host-scanned result only).
TEST(OutputArgMax, ChainStepReadBoundsOnTheHostPath) {
    Config cfg;
    cfg.backend          = BackendKind::Cpu;
    cfg.decodeChainSteps = 4;
    auto sess            = Session::create(makeVectorGraph(8), cfg);
    ASSERT_TRUE(sess);
    ASSERT_EQ(sess->setOutputArgMax(0, "scores"), Status::Ok);
    std::vector<float> values(8, 0.5f);
    values[5] = 3.0f;
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({f32Tensor("x", {1, 8}, values)}, outs), Status::Ok);

    int64_t index = -1, stepIndex = -1;
    float   value = 0.0f, stepValue = 0.0f;
    ASSERT_EQ(sess->readOutputArgMax("scores", index, value), Status::Ok);
    ASSERT_EQ(sess->readOutputArgMax("scores", 0, stepIndex, stepValue), Status::Ok);
    EXPECT_EQ(stepIndex, index);
    EXPECT_FLOAT_EQ(stepValue, value);
    EXPECT_EQ(sess->readOutputArgMax("scores", -1, stepIndex, stepValue), Status::InvalidArgument);
    EXPECT_EQ(sess->readOutputArgMax("scores", 4, stepIndex, stepValue), Status::InvalidArgument);
    EXPECT_EQ(sess->readOutputArgMax("scores", 1, stepIndex, stepValue), Status::Unsupported); // no device slots on the host path
}

// configureDecodeChain validation: the output must be argmax-registered first, and the chain needs
// the device reduction path — the CPU backend's host scan cannot chain. setDecodeChainWindow
// without a configured chain is NotFound.
TEST(OutputArgMax, ConfigureDecodeChainValidation) {
    Config cfg;
    cfg.backend          = BackendKind::Cpu;
    cfg.decodeChainSteps = 4;
    auto sess            = Session::create(makeVectorGraph(8), cfg);
    ASSERT_TRUE(sess);
    EXPECT_EQ(sess->configureDecodeChain(3, "x", "x", "x", "scores"), Status::InvalidArgument);  // bucket out of range
    EXPECT_EQ(sess->configureDecodeChain(0, "x", "x", "x", "scores"), Status::InvalidArgument);  // not argmax-registered yet
    ASSERT_EQ(sess->setOutputArgMax(0, "scores"), Status::Ok);
    EXPECT_EQ(sess->configureDecodeChain(0, "x", "x", "x", "scores"), Status::Unsupported); // host reduction path: no device chain
    EXPECT_EQ(sess->setDecodeChainWindow(0, 0, 1), Status::NotFound);                       // nothing configured
}
