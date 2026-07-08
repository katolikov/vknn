// Engine-resident output->input links (Session::linkOutputToInput): a linked input keeps its
// engine-side values across runs and receives caller-declared element ranges from a linked output
// at the start of each run, so recurrent state (a decoder's KV cache) never round-trips through the
// caller. These host tests (CPU backend, the byte oracle) pin the contract:
//   - a linked session produces byte-identical values to the unlinked loop that folds the output
//     back into the input by hand, across several sequential runs,
//   - a linked output's run() entry carries metadata but no data; readResident() fetches the state,
//   - binding host data for a linked input reinitializes the resident state,
//   - linkage validation rejects unknown names, non-boundary names, out-of-bounds and overlapping
//     ranges, and dtype-mismatched pairs — naming the offending tensors,
//   - clearLinks() restores the unlinked behavior,
//   - a session that never links runs exactly as before (the unused-feature guarantee).
#include "vknn/session.h"
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    // Fill an IOTensor's payload from fp32 values.
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

    std::vector<float> f32Values(const IOTensor &io) {
        std::vector<float> v(io.data.size() / sizeof(float));
        std::memcpy(v.data(), io.data.data(), io.data.size());
        return v;
    }

    // Accumulator graph: state_next = state + x (output), y = state_next + x (output). Linking
    // state_next -> state turns it into a recurrent accumulator the caller feeds only `x`.
    Graph makeAccumulatorGraph(int64_t elems) {
        Graph      g;
        TensorDesc stateDesc;
        stateDesc.name    = "state";
        stateDesc.shape   = {1, elems};
        stateDesc.isInput = true;
        TensorId   state  = g.addTensor(stateDesc);
        TensorDesc xDesc;
        xDesc.name    = "x";
        xDesc.shape   = {1, elems};
        xDesc.isInput = true;
        TensorId x    = g.addTensor(xDesc);
        g.inputs      = {state, x};

        TensorDesc nextDesc;
        nextDesc.name     = "state_next";
        nextDesc.isOutput = true;
        TensorId   next   = g.addTensor(nextDesc);
        TensorDesc yDesc;
        yDesc.name     = "y";
        yDesc.isOutput = true;
        TensorId y     = g.addTensor(yDesc);

        Node add1;
        add1.type    = OpType::Add;
        add1.name    = "add_state";
        add1.inputs  = {state, x};
        add1.outputs = {next};
        g.nodes.push_back(add1);
        Node add2;
        add2.type    = OpType::Add;
        add2.name    = "add_y";
        add2.inputs  = {next, x};
        add2.outputs = {y};
        g.nodes.push_back(add2);
        g.outputs = {next, y};
        return g;
    }

    // History graph mirroring a KV cache: hist [1, heads, slots, width] concatenated with obs
    // [1, heads, 1, width] along the slot axis -> hist_next [1, heads, slots+1, width]. The new
    // row sits at slot index `slots` of each head, exactly like a with-past decoder's present.
    Graph makeHistoryGraph(int64_t heads, int64_t slots, int64_t width) {
        Graph      g;
        TensorDesc histDesc;
        histDesc.name    = "hist";
        histDesc.shape   = {1, heads, slots, width};
        histDesc.isInput = true;
        TensorId   hist  = g.addTensor(histDesc);
        TensorDesc obsDesc;
        obsDesc.name    = "obs";
        obsDesc.shape   = {1, heads, 1, width};
        obsDesc.isInput = true;
        TensorId obs    = g.addTensor(obsDesc);
        g.inputs        = {hist, obs};

        TensorDesc nextDesc;
        nextDesc.name     = "hist_next";
        nextDesc.isOutput = true;
        TensorId next     = g.addTensor(nextDesc);

        Node cat;
        cat.type             = OpType::Concat;
        cat.name             = "append";
        cat.inputs           = {hist, obs};
        cat.outputs          = {next};
        cat.attr.map["axis"] = [] {
            Attr a;
            a.kind = Attr::Int;
            a.i    = 2;
            return a;
        }();
        g.nodes.push_back(cat);
        g.outputs = {next};
        return g;
    }

    // Int64 pair graph for the dtype-mismatch validation: ids (int64 input) casts to a float output
    // and to an int64 output.
    Graph makeCastGraph(int64_t elems) {
        Graph      g;
        TensorDesc idsDesc;
        idsDesc.name    = "ids";
        idsDesc.shape   = {elems};
        idsDesc.dtype   = DType::Int64;
        idsDesc.isInput = true;
        TensorId ids    = g.addTensor(idsDesc);
        g.inputs        = {ids};

        TensorDesc floatDesc;
        floatDesc.name     = "ids_float";
        floatDesc.isOutput = true;
        TensorId idsFloat  = g.addTensor(floatDesc);
        Node     toFloat;
        toFloat.type           = OpType::Cast;
        toFloat.name           = "to_float";
        toFloat.inputs         = {ids};
        toFloat.outputs        = {idsFloat};
        toFloat.attr.map["to"] = [] {
            Attr a;
            a.kind = Attr::Int;
            a.i    = 1; // ONNX TensorProto FLOAT
            return a;
        }();
        g.nodes.push_back(toFloat);

        TensorDesc i64Desc;
        i64Desc.name     = "ids_i64";
        i64Desc.dtype    = DType::Int64;
        i64Desc.isOutput = true;
        TensorId idsI64  = g.addTensor(i64Desc);
        Node     toI64;
        toI64.type           = OpType::Cast;
        toI64.name           = "to_i64";
        toI64.inputs         = {ids};
        toI64.outputs        = {idsI64};
        toI64.attr.map["to"] = [] {
            Attr a;
            a.kind = Attr::Int;
            a.i    = 7; // ONNX TensorProto INT64
            return a;
        }();
        g.nodes.push_back(toI64);
        g.outputs = {idsFloat, idsI64};
        return g;
    }

    std::unique_ptr<Session> makeCpuSession(Graph &&g) {
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        return Session::create(std::move(g), cfg);
    }

} // namespace

// The linked accumulator produces byte-identical state and outputs to the unlinked loop that binds
// the folded state by hand, across several sequential runs.
TEST(ResidentLink, LinkedMatchesUnlinkedAcrossRuns) {
    const int64_t elems    = 16;
    auto          linked   = makeCpuSession(makeAccumulatorGraph(elems));
    auto          unlinked = makeCpuSession(makeAccumulatorGraph(elems));
    ASSERT_TRUE(linked && unlinked);

    // Full-tensor fold: state <- state_next after every run.
    ASSERT_EQ(linked->linkOutputToInput("state_next", "state", {{0, 0, elems}}), Status::Ok);

    std::vector<float> stateRef(elems, 0.f);
    for (int step = 0; step < 4; ++step)
    {
        std::vector<float> x(elems);
        for (int64_t i = 0; i < elems; ++i)
        {
            x[i] = (float) (step * 100 + i) * 0.25f;
        }
        // Linked: bind x only; the engine folds state_next into state itself.
        std::vector<IOTensor> linkedOuts;
        ASSERT_EQ(linked->run({f32Tensor("x", {1, elems}, x)}, linkedOuts), Status::Ok);
        // Unlinked: bind the hand-held state and x, then fold on the host.
        std::vector<IOTensor> plainOuts;
        ASSERT_EQ(unlinked->run({f32Tensor("state", {1, elems}, stateRef), f32Tensor("x", {1, elems}, x)}, plainOuts), Status::Ok);
        const IOTensor *plainNext = outByName(plainOuts, "state_next");
        ASSERT_NE(plainNext, nullptr);
        stateRef = f32Values(*plainNext);

        // y (an unlinked output) must match byte-for-byte.
        const IOTensor *linkedY = outByName(linkedOuts, "y");
        const IOTensor *plainY  = outByName(plainOuts, "y");
        ASSERT_NE(linkedY, nullptr);
        ASSERT_NE(plainY, nullptr);
        EXPECT_EQ(linkedY->data, plainY->data) << "step " << step;
        // The linked output carries metadata but no payload.
        const IOTensor *linkedNext = outByName(linkedOuts, "state_next");
        ASSERT_NE(linkedNext, nullptr);
        EXPECT_TRUE(linkedNext->data.empty());
        EXPECT_EQ(linkedNext->shape, (Shape {1, elems}));

        // The resident state matches the hand-folded state ONE RUN LATER (the fold applies at the
        // start of the next run), so compare through readResident after the fold lands below.
    }
    // Force the pending fold to land, then compare the resident state with the reference.
    std::vector<IOTensor> outs;
    ASSERT_EQ(linked->run({f32Tensor("x", {1, elems}, std::vector<float>(elems, 0.f))}, outs), Status::Ok);
    IOTensor resident;
    ASSERT_EQ(linked->readResident("state", resident), Status::Ok);
    EXPECT_EQ(f32Values(resident), stateRef);
}

// The ranged fold (one new row per head into a caller-chosen slot, the KV-cache pattern) matches
// the unlinked host fold byte-for-byte across runs.
TEST(ResidentLink, RangedHistoryFoldMatchesManual) {
    const int64_t heads = 2, slots = 3, width = 4;
    auto          linked   = makeCpuSession(makeHistoryGraph(heads, slots, width));
    auto          unlinked = makeCpuSession(makeHistoryGraph(heads, slots, width));
    ASSERT_TRUE(linked && unlinked);
    ASSERT_EQ(linked->linkOutputToInput("hist_next", "hist", {}), Status::Ok);

    std::vector<float> histRef(heads * slots * width, 0.f);
    for (int step = 0; step < 3; ++step)
    {
        // Before the run: fold the PREVIOUS run's new row (present slot `slots`) into slot step-1.
        if (step > 0)
        {
            const int64_t          slot = step - 1;
            std::vector<LinkRange> ranges;
            for (int64_t h = 0; h < heads; ++h)
            {
                ranges.push_back({(h * (slots + 1) + slots) * width, (h * slots + slot) * width, width});
            }
            ASSERT_EQ(linked->linkOutputToInput("hist_next", "hist", ranges), Status::Ok);
        }
        std::vector<float> obs(heads * width);
        for (size_t i = 0; i < obs.size(); ++i)
        {
            obs[i] = (float) (step + 1) * 10.f + (float) i;
        }
        std::vector<IOTensor> linkedOuts, plainOuts;
        ASSERT_EQ(linked->run({f32Tensor("obs", {1, heads, 1, width}, obs)}, linkedOuts), Status::Ok);
        ASSERT_EQ(unlinked->run({f32Tensor("hist", {1, heads, slots, width}, histRef), f32Tensor("obs", {1, heads, 1, width}, obs)}, plainOuts), Status::Ok);

        // Host fold for the reference: present row -> hist slot `step`.
        const IOTensor *plainNext = outByName(plainOuts, "hist_next");
        ASSERT_NE(plainNext, nullptr);
        std::vector<float> next = f32Values(*plainNext);
        const int64_t      slot = step < slots ? step : slots - 1;
        for (int64_t h = 0; h < heads; ++h)
        {
            std::memcpy(histRef.data() + (h * slots + slot) * width, next.data() + (h * (slots + 1) + slots) * width, (size_t) width * sizeof(float));
        }
    }
    // Land the last pending fold and compare the resident history against the host-folded one.
    {
        const int64_t          slot = 2;
        std::vector<LinkRange> ranges;
        for (int64_t h = 0; h < heads; ++h)
        {
            ranges.push_back({(h * (slots + 1) + slots) * width, (h * slots + slot) * width, width});
        }
        ASSERT_EQ(linked->linkOutputToInput("hist_next", "hist", ranges), Status::Ok);
        std::vector<IOTensor> outs;
        std::vector<float>    zeroObs(heads * width, 0.f);
        ASSERT_EQ(linked->run({f32Tensor("obs", {1, heads, 1, width}, zeroObs)}, outs), Status::Ok);
    }
    IOTensor resident;
    ASSERT_EQ(linked->readResident("hist", resident), Status::Ok);
    EXPECT_EQ(f32Values(resident), histRef);
}

// Binding host data for a linked input reinitializes the resident state.
TEST(ResidentLink, BindingLinkedInputReinitializesState) {
    const int64_t elems = 8;
    auto          s     = makeCpuSession(makeAccumulatorGraph(elems));
    ASSERT_TRUE(s);
    ASSERT_EQ(s->linkOutputToInput("state_next", "state", {{0, 0, elems}}), Status::Ok);

    std::vector<float>    ones(elems, 1.f);
    std::vector<IOTensor> outs;
    ASSERT_EQ(s->run({f32Tensor("x", {1, elems}, ones)}, outs), Status::Ok); // state 0 -> next 1
    ASSERT_EQ(s->run({f32Tensor("x", {1, elems}, ones)}, outs), Status::Ok); // state 1 -> next 2

    // Reinitialize state to zero (empty ranges suppress the pending fold), then run once: the
    // outputs must equal a fresh session's first step.
    ASSERT_EQ(s->linkOutputToInput("state_next", "state", {}), Status::Ok);
    std::vector<float> zeros(elems, 0.f);
    ASSERT_EQ(s->run({f32Tensor("state", {1, elems}, zeros), f32Tensor("x", {1, elems}, ones)}, outs), Status::Ok);
    const IOTensor *y = outByName(outs, "y");
    ASSERT_NE(y, nullptr);
    EXPECT_EQ(f32Values(*y), std::vector<float>(elems, 2.f)); // (0+1)+1
}

// clearLinks() restores the unlinked contract: outputs carry data again.
TEST(ResidentLink, ClearLinksRestoresOutputData) {
    const int64_t elems = 8;
    auto          s     = makeCpuSession(makeAccumulatorGraph(elems));
    ASSERT_TRUE(s);
    ASSERT_EQ(s->linkOutputToInput("state_next", "state", {{0, 0, elems}}), Status::Ok);
    std::vector<float>    ones(elems, 1.f);
    std::vector<IOTensor> outs;
    ASSERT_EQ(s->run({f32Tensor("x", {1, elems}, ones)}, outs), Status::Ok);
    EXPECT_TRUE(outByName(outs, "state_next")->data.empty());

    s->clearLinks();
    ASSERT_EQ(s->run({f32Tensor("state", {1, elems}, std::vector<float>(elems, 0.f)), f32Tensor("x", {1, elems}, ones)}, outs), Status::Ok);
    const IOTensor *next = outByName(outs, "state_next");
    ASSERT_NE(next, nullptr);
    EXPECT_EQ(f32Values(*next), ones);
    IOTensor scratch;
    EXPECT_EQ(s->readResident("state", scratch), Status::NotFound); // nothing is linked any more
}

// Linkage validation: wrong names, non-boundary tensors, bad ranges, and dtype mismatches are
// rejected with an error status (the log names the offending tensors).
TEST(ResidentLink, ValidationRejectsBadLinks) {
    const int64_t elems = 8;
    auto          s     = makeCpuSession(makeAccumulatorGraph(elems));
    ASSERT_TRUE(s);
    // Unknown names.
    EXPECT_EQ(s->linkOutputToInput("nope", "state", {}), Status::NotFound);
    EXPECT_EQ(s->linkOutputToInput("state_next", "nope", {}), Status::NotFound);
    // Sides swapped: an input is not an output and vice versa.
    EXPECT_EQ(s->linkOutputToInput("state", "state_next", {}), Status::NotFound);
    // Out-of-bounds and overlapping destination ranges.
    EXPECT_EQ(s->linkOutputToInput("state_next", "state", {{0, 0, elems + 1}}), Status::InvalidArgument);
    EXPECT_EQ(s->linkOutputToInput("state_next", "state", {{0, elems - 1, 2}}), Status::InvalidArgument);
    EXPECT_EQ(s->linkOutputToInput("state_next", "state", {{0, 0, -1}}), Status::InvalidArgument);
    EXPECT_EQ(s->linkOutputToInput("state_next", "state", {{0, 0, 4}, {4, 2, 4}}), Status::InvalidArgument);
    // Bucket index out of range.
    EXPECT_EQ(s->linkOutputToInput(3, "state_next", "state", {}), Status::InvalidArgument);

    // dtype mismatch: an fp32 output cannot feed an int64 input; the int64->int64 pair is accepted.
    auto castSession = makeCpuSession(makeCastGraph(4));
    ASSERT_TRUE(castSession);
    EXPECT_EQ(castSession->linkOutputToInput("ids_float", "ids", {}), Status::InvalidArgument);
    EXPECT_EQ(castSession->linkOutputToInput("ids_i64", "ids", {{0, 0, 4}}), Status::Ok);
}

// The int64 storage class round-trips through a link exactly (the host path copies 8-byte lanes).
TEST(ResidentLink, Int64LinkCopiesExactly) {
    auto s = makeCpuSession(makeCastGraph(4));
    ASSERT_TRUE(s);
    ASSERT_EQ(s->linkOutputToInput("ids_i64", "ids", {}), Status::Ok); // empty ranges: seed first
    IOTensor ids;
    ids.name  = "ids";
    ids.shape = {4};
    ids.dtype = DType::Int64;
    std::vector<int64_t> seed {3, -7, 1LL << 40, 0};
    ids.data.resize(seed.size() * sizeof(int64_t));
    std::memcpy(ids.data.data(), seed.data(), ids.data.size());

    std::vector<IOTensor> outs;
    ASSERT_EQ(s->run({ids}, outs), Status::Ok); // seed the state; produces ids_i64 == seed
    ASSERT_EQ(s->linkOutputToInput("ids_i64", "ids", {{0, 0, 4}}), Status::Ok);
    ASSERT_EQ(s->run(std::vector<IOTensor> {}, outs), Status::Ok); // fold lands; recomputes from folded state
    const IOTensor *idsFloat = outByName(outs, "ids_float");
    ASSERT_NE(idsFloat, nullptr);
    std::vector<float> expect(seed.size());
    for (size_t i = 0; i < seed.size(); ++i)
    {
        expect[i] = (float) seed[i];
    }
    EXPECT_EQ(f32Values(*idsFloat), expect);
}
