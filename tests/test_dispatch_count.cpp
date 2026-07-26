// Recorded-dispatch accounting: the attribution arithmetic (src/core/dispatch_tally.h) the Vulkan
// segment drives around each op's record(), and the profiler surfaces that carry the result
// (OpRecord::dispatches -> the Config::profile table total and the result JSON).
//
// Scope note: the counting SITE is Vulkan-only. src/backend/vulkan is compiled only when
// VKNN_ENABLE_VULKAN is on (Android), so no host test can prove that a real split-K GEMM records
// two dispatches or that a Winograd conv records three — that is a device gate. What is host-
// testable, and what this file covers, is everything between the counter and the report: the span
// attribution (including the accumulate-across-iterations behavior a recorded decode chain needs),
// the node-vs-unattributed split that makes the segment total differ from the per-op table, and the
// profiler's aggregation and JSON schema.
#include "core/dispatch_tally.h"
#include "vknn/profiler.h"
#include <gtest/gtest.h>

using namespace vknn;

// One op == one dispatch is the exception, not the rule: the tally attributes whatever a span
// recorded, so a multi-pass kernel reports its real count.
TEST(DispatchTally, AttributesEachSpanToItsNode) {
    DispatchTally tally;
    tally.beginRun(3);
    tally.openNode(0); // a plain elementwise op
    tally.note();
    tally.closeNode();
    tally.openNode(1); // a split-K GEMM: partial + reduce
    tally.note();
    tally.note();
    tally.closeNode();
    tally.openNode(2); // a Winograd conv: input transform, GEMM, output transform
    tally.note();
    tally.note();
    tally.note();
    tally.closeNode();

    EXPECT_EQ(tally.nodeDispatches(0), 1u);
    EXPECT_EQ(tally.nodeDispatches(1), 2u);
    EXPECT_EQ(tally.nodeDispatches(2), 3u);
    EXPECT_EQ(tally.nodeTotal(), 6u);
    EXPECT_EQ(tally.runTotal(), 6u);
    // The whole point of the counter: the node count understates the dispatch count.
    EXPECT_GT(tally.runTotal(), tally.nodeCount());
}

// A node the planner elided to a zero-copy view, or lowered to a buffer copy, records no dispatch.
TEST(DispatchTally, ElidedNodeCountsZero) {
    DispatchTally tally;
    tally.beginRun(2);
    tally.openNode(0);
    tally.closeNode(); // recorded nothing
    tally.openNode(1);
    tally.note();
    tally.closeNode();

    EXPECT_EQ(tally.nodeDispatches(0), 0u);
    EXPECT_EQ(tally.nodeDispatches(1), 1u);
    EXPECT_EQ(tally.nodeTotal(), 1u);
}

// Boundary converts, resident-link copies, chain feedback, and argmax epilogues dispatch with no
// span open. They must land in the segment total and stay out of the per-node table, because the
// gap between the two is exactly the per-run overhead the per-op profile cannot show.
TEST(DispatchTally, UnattributedDispatchesCountOnlyInTheRunTotal) {
    DispatchTally tally;
    tally.beginRun(1);
    tally.note(); // input boundary convert, before any node
    tally.openNode(0);
    tally.note();
    tally.closeNode();
    tally.note(); // output argmax epilogue, after the last node

    EXPECT_EQ(tally.nodeDispatches(0), 1u);
    EXPECT_EQ(tally.nodeTotal(), 1u);
    EXPECT_EQ(tally.runTotal(), 3u);
    EXPECT_EQ(tally.runTotal() - tally.nodeTotal(), 2u);
}

// A decode chain records its whole body once per iteration into one command stream, so a node's
// count is the sum over the recorded iterations, not the last one.
TEST(DispatchTally, RepeatedSpansAccumulate) {
    DispatchTally tally;
    constexpr int kRecordedIterations = 4;
    tally.beginRun(1);
    for (int iteration = 0; iteration < kRecordedIterations; ++iteration)
    {
        tally.openNode(0);
        tally.note();
        tally.note();
        tally.closeNode();
    }
    EXPECT_EQ(tally.nodeDispatches(0), 2u * kRecordedIterations);
    EXPECT_EQ(tally.runTotal(), 2u * kRecordedIterations);
}

// A re-record (a changed boundary buffer, a reconfigured chain) reports the NEW command stream, and
// the lifetime counter keeps running across both.
TEST(DispatchTally, BeginRunResetsButLifetimeAccumulates) {
    DispatchTally tally;
    tally.beginRun(1);
    tally.openNode(0);
    tally.note();
    tally.note();
    tally.closeNode();
    EXPECT_EQ(tally.runTotal(), 2u);

    tally.beginRun(1); // re-record
    tally.openNode(0);
    tally.note();
    tally.closeNode();
    EXPECT_EQ(tally.nodeDispatches(0), 1u);
    EXPECT_EQ(tally.runTotal(), 1u);
    EXPECT_EQ(tally.lifetime(), 3u);
}

// Out-of-range and unbalanced calls must not corrupt the table: the segment loop is the only
// caller, but a table that can be poisoned is a table nobody can trust as a measurement.
TEST(DispatchTally, OutOfRangeAndUnbalancedCallsAreHarmless) {
    DispatchTally tally;
    tally.beginRun(1);
    tally.openNode(7); // past the node count
    tally.note();
    tally.closeNode();
    tally.closeNode(); // no span open
    EXPECT_EQ(tally.nodeDispatches(0), 0u);
    EXPECT_EQ(tally.nodeDispatches(7), 0u);
    EXPECT_EQ(tally.nodeTotal(), 0u);
    EXPECT_EQ(tally.runTotal(), 1u); // still counted for the segment
}

// The reporting half: a model's dispatch count is readable off the profiler without summing a
// table by hand, and the per-record count reaches the result JSON.
TEST(Profiler, CarriesDispatchCounts) {
    Profiler prof;
    prof.setEnabled(true);

    OpRecord conv;
    conv.name       = "/features/Conv";
    conv.type       = OpType::Conv;
    conv.backend    = "Vulkan";
    conv.gpuMs      = 1.5;
    conv.dispatches = 3; // Winograd: transform, GEMM, output
    prof.add(conv);

    OpRecord matmul;
    matmul.name       = "/attn/MatMul";
    matmul.type       = OpType::MatMul;
    matmul.backend    = "Vulkan";
    matmul.gpuMs      = 0.5;
    matmul.dispatches = 2; // split-K: partial + reduce
    prof.add(matmul);

    OpRecord onCpu;
    onCpu.name    = "/head/TopK";
    onCpu.type    = OpType::TopK;
    onCpu.backend = "CPU";
    onCpu.cpuMs   = 0.25;
    prof.add(onCpu); // dispatches stays 0: the CPU oracle dispatches nothing

    EXPECT_EQ(prof.totalDispatches(), 5u);
    EXPECT_EQ(prof.records().size(), 3u);
    // 3 records, 5 dispatches: the node count understates by more than half here.
    EXPECT_GT(prof.totalDispatches(), prof.records().size());

    const std::string json = prof.toJson();
    EXPECT_NE(json.find("\"dispatches\":3"), std::string::npos);
    EXPECT_NE(json.find("\"dispatches\":2"), std::string::npos);
    EXPECT_NE(json.find("\"dispatches\":0"), std::string::npos);
}
