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

// kvFoldSlot is the single source of the decode fold-slot rule: slot position-1, clamped to the
// last cache slot once the position runs past the compiled window, none at position 0. A decode
// chain's per-iteration precompute (base position + i) therefore reproduces the single-step
// loop's slot at every position, including across the context edge.
TEST(ResidentLink, KvFoldSlotMatchesTheSingleStepRuleAcrossTheWindowEdge) {
    const int64_t cacheSlots = 8;
    EXPECT_EQ(kvFoldSlot(0, cacheSlots), -1);
    EXPECT_EQ(kvFoldSlot(-3, cacheSlots), -1);
    EXPECT_EQ(kvFoldSlot(1, cacheSlots), 0);
    EXPECT_EQ(kvFoldSlot(8, cacheSlots), 7);
    EXPECT_EQ(kvFoldSlot(9, cacheSlots), 7);   // clamped: the overrun keeps rewriting the newest slot
    EXPECT_EQ(kvFoldSlot(100, cacheSlots), 7); // and stays clamped arbitrarily far past the edge
    // A chain whose base position sits near the edge: iteration i's precomputed slot equals the
    // slot the single-step loop computes at position base+i, for every iteration of the chain.
    for (int64_t basePosition = 5; basePosition <= 10; ++basePosition)
    {
        for (int64_t iteration = 0; iteration < 4; ++iteration)
        {
            const int64_t stepPosition = basePosition + iteration;
            const int64_t singleStep   = stepPosition - 1 < cacheSlots - 1 ? stepPosition - 1 : cacheSlots - 1;
            EXPECT_EQ(kvFoldSlot(stepPosition, cacheSlots), singleStep);
        }
    }
}

// linkOutputToInputChain validation: the set count is bounded by 1..Config::decodeChainSteps,
// every set is bounds-checked like the single-set form, and more than one set needs a
// device-resident link (the CPU host path applies exactly one).
TEST(ResidentLink, ChainRangeSetValidation) {
    const int64_t elems = 8;
    Config        cfg;
    cfg.backend          = BackendKind::Cpu;
    cfg.decodeChainSteps = 3;
    auto s               = Session::create(makeAccumulatorGraph(elems), cfg);
    ASSERT_TRUE(s);
    EXPECT_EQ(s->linkOutputToInputChain(0, "state_next", "state", {}), Status::InvalidArgument); // no sets
    std::vector<std::vector<LinkRange>> fourSets(4);
    EXPECT_EQ(s->linkOutputToInputChain(0, "state_next", "state", fourSets), Status::InvalidArgument); // > decodeChainSteps
    std::vector<std::vector<LinkRange>> badSecondSet {{{0, 0, 4}}, {{0, 0, elems + 1}}};
    EXPECT_EQ(s->linkOutputToInputChain(0, "state_next", "state", badSecondSet), Status::InvalidArgument); // set 1 out of bounds
    std::vector<std::vector<LinkRange>> twoSets {{{0, 0, 4}}, {{0, 4, 4}}};
    EXPECT_EQ(s->linkOutputToInputChain(0, "state_next", "state", twoSets), Status::InvalidArgument); // host link: one set only
    std::vector<std::vector<LinkRange>> oneSet {{{0, 0, elems}}};
    EXPECT_EQ(s->linkOutputToInputChain(0, "state_next", "state", oneSet), Status::Ok); // == the single-set overload
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

// ---- with-past decoder geometry (the on-device VLM shape class) -------------------------------
// A rows-only-present decoder at the real VLM geometry: 32 kv-heads, C=512 cache slots, head_dim
// 64, a 128-row prefill bucket plus an S=1 decode bucket. The present outputs carry ONLY the
// produced rows ([1,32,S,64]) — the OTHER with-past convention from the cache-concat decoder the
// history tests model — so the fold source is the LAST present row, not row C.

namespace {

    Attr singleInt(int64_t v) {
        Attr a;
        a.kind = Attr::Int;
        a.i    = v;
        return a;
    }
    Attr intList(std::vector<int64_t> v) {
        Attr a;
        a.kind = Attr::Ints;
        a.ints = std::move(v);
        return a;
    }

    // past_key_values.0.{key,value} [1,heads,C,HD] + obs [1,heads,S,HD] ->
    //   present.0.key   = Relu(obs)      (rows-only present; obs values are kept positive)
    //   present.0.value = obs + obs      (distinct per part, so a key/value cross-link shows)
    //   logits          = ReduceSum_{heads,HD}(Concat(pastK,obs)) + ReduceSum_{heads,HD}(Concat(pastV,obs))
    // Every cache slot feeds logits, so a wrong fold slot, source row, or timing changes the output.
    Graph makeVlmStyleDecoder(int64_t heads, int64_t cacheSlots, int64_t headDim) {
        Graph      g;
        TensorDesc pkDesc;
        pkDesc.name      = "past_key_values.0.key";
        pkDesc.shape     = {1, heads, cacheSlots, headDim};
        pkDesc.isInput   = true;
        TensorId   pastK = g.addTensor(pkDesc);
        TensorDesc pvDesc;
        pvDesc.name      = "past_key_values.0.value";
        pvDesc.shape     = {1, heads, cacheSlots, headDim};
        pvDesc.isInput   = true;
        TensorId   pastV = g.addTensor(pvDesc);
        TensorDesc obsDesc;
        obsDesc.name    = "obs";
        obsDesc.shape   = {1, heads, -1, headDim}; // S resolved per bucket (Config::inputShapes / prepareShapes)
        obsDesc.isInput = true;
        TensorId obs    = g.addTensor(obsDesc);
        g.inputs        = {pastK, pastV, obs};

        auto addOut = [&](const char *name) {
            TensorDesc d;
            d.name     = name;
            d.isOutput = true;
            TensorId t = g.addTensor(d);
            g.outputs.push_back(t);
            return t;
        };
        auto addMid = [&](const char *name) {
            TensorDesc d;
            d.name = name;
            return g.addTensor(d);
        };
        TensorId presK = addOut("present.0.key");
        TensorId presV = addOut("present.0.value");
        TensorId catK  = addMid("catK");
        TensorId catV  = addMid("catV");
        TensorId sumK  = addMid("sumK");
        TensorId sumV  = addMid("sumV");
        TensorId lg    = addOut("logits");

        auto addNode = [&](OpType t, const char *name, std::vector<TensorId> in, std::vector<TensorId> out) -> Node & {
            Node n;
            n.type    = t;
            n.name    = name;
            n.inputs  = std::move(in);
            n.outputs = std::move(out);
            g.nodes.push_back(std::move(n));
            return g.nodes.back();
        };
        addNode(OpType::Relu, "pres_key", {obs}, {presK});
        addNode(OpType::Add, "pres_value", {obs, obs}, {presV});
        addNode(OpType::Concat, "cat_key", {pastK, obs}, {catK}).attr.map["axis"]   = singleInt(2);
        addNode(OpType::Concat, "cat_value", {pastV, obs}, {catV}).attr.map["axis"] = singleInt(2);
        {
            Node &n                = addNode(OpType::Reduce, "sum_key", {catK}, {sumK});
            n.attr.map["axes"]     = intList({1, 3});
            n.attr.map["keepdims"] = singleInt(0);
            n.subOp                = 1; // ReduceType::Sum
        }
        {
            Node &n                = addNode(OpType::Reduce, "sum_value", {catV}, {sumV});
            n.attr.map["axes"]     = intList({1, 3});
            n.attr.map["keepdims"] = singleInt(0);
            n.subOp                = 1; // ReduceType::Sum
        }
        addNode(OpType::Add, "logits_add", {sumK, sumV}, {lg});
        return g;
    }

    constexpr int64_t kVlmHeads = 32, kVlmSlots = 512, kVlmHeadDim = 64, kVlmPrefill = 128;

    // A decode/prefill two-bucket session: bucket 0 = S=1 (decode), bucket 1 = S=128 (prefill).
    std::unique_ptr<Session> makeVlmStyleSession() {
        Config cfg;
        cfg.backend     = BackendKind::Cpu;
        cfg.inputShapes = {{"obs", {1, kVlmHeads, 1, kVlmHeadDim}}};
        auto s          = Session::create(makeVlmStyleDecoder(kVlmHeads, kVlmSlots, kVlmHeadDim), cfg);
        if (s && s->prepareShapes({{"obs", {1, kVlmHeads, kVlmPrefill, kVlmHeadDim}}}) != Status::Ok)
        {
            return nullptr;
        }
        return s;
    }

    // Deterministic positive obs values (Relu passthrough) that differ per step and per element.
    std::vector<float> obsValues(int64_t rows, int64_t seed) {
        std::vector<float> v((size_t) (kVlmHeads * rows * kVlmHeadDim));
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[(size_t) i] = 1.0f + (float) ((i * 31 + (size_t) seed * 977) % 251) * 0.03125f;
        }
        return v;
    }

    // Host-side fold of a rows-only present ([1,heads,rows,HD]) into a host cache, rows
    // 0..copyRows-1 landing at startSlot.. — the driver's foldPresent.
    void hostFold(std::vector<float> &cache, const std::vector<float> &present, int64_t presentRows, int64_t copyRows, int64_t startSlot) {
        for (int64_t h = 0; h < kVlmHeads; ++h)
        {
            for (int64_t r = 0; r < copyRows; ++r)
            {
                std::memcpy(cache.data() + ((size_t) h * kVlmSlots + startSlot + r) * kVlmHeadDim, present.data() + ((size_t) h * presentRows + r) * kVlmHeadDim, (size_t) kVlmHeadDim * sizeof(float));
            }
        }
    }

    IOTensor pastTensor(const char *name, const std::vector<float> &cache) {
        return f32Tensor(name, {1, kVlmHeads, kVlmSlots, kVlmHeadDim}, cache);
    }

} // namespace

// kvFoldRanges covers both with-past conventions: the cache-concat present (rows = C+1, new row at
// index C) and the rows-only present (rows = 1, new row at index 0). slot < 0 means no fold.
TEST(ResidentLink, KvFoldRangesCoversBothConventions) {
    // qwen-style: [1, 2, C+1, 4] with C = 8.
    auto q = kvFoldRanges(2, 9, 8, 4, 5);
    ASSERT_EQ(q.size(), 2u);
    EXPECT_EQ(q[0].sourceElem, 8 * 4);       // head 0, row C = 8
    EXPECT_EQ(q[0].destElem, 5 * 4);         // head 0, slot 5
    EXPECT_EQ(q[1].sourceElem, (9 + 8) * 4); // head 1, row C
    EXPECT_EQ(q[1].destElem, (8 + 5) * 4);   // head 1, slot 5
    EXPECT_EQ(q[0].count, 4);
    // rows-only style: [1, 32, 1, 64] against a 512-slot cache — the on-device VLM geometry.
    auto r = kvFoldRanges(kVlmHeads, 1, kVlmSlots, kVlmHeadDim, 209);
    ASSERT_EQ(r.size(), (size_t) kVlmHeads);
    EXPECT_EQ(r[0].sourceElem, 0); // head 0, row 0 (the only row)
    EXPECT_EQ(r[0].destElem, 209 * kVlmHeadDim);
    EXPECT_EQ(r[31].sourceElem, 31 * kVlmHeadDim);
    EXPECT_EQ(r[31].destElem, (31 * kVlmSlots + 209) * kVlmHeadDim);
    EXPECT_TRUE(kvFoldRanges(kVlmHeads, 1, kVlmSlots, kVlmHeadDim, -1).empty());
}

// The device failure signature: cache-concat source offsets applied to a rows-only present are out
// of bounds and MUST be rejected (InvalidArgument naming both tensors in the log) — never applied.
TEST(ResidentLink, CacheConcatRangesRejectedOnRowsOnlyPresent) {
    auto s = makeVlmStyleSession();
    ASSERT_TRUE(s);
    ASSERT_EQ(s->linkOutputToInput(0, "present.0.key", "past_key_values.0.key", {}), Status::Ok);
    // The pre-fix fold ranges: source row C of a present assumed to be [1,KV,C+1,HD].
    std::vector<LinkRange> wrong((size_t) kVlmHeads);
    for (int64_t h = 0; h < kVlmHeads; ++h)
    {
        wrong[(size_t) h] = {(h * (kVlmSlots + 1) + kVlmSlots) * kVlmHeadDim, (h * kVlmSlots + 130) * kVlmHeadDim, kVlmHeadDim};
    }
    EXPECT_EQ(s->linkOutputToInput(0, "present.0.key", "past_key_values.0.key", wrong), Status::InvalidArgument);
    // The corrected ranges (last present row = row 0) are accepted at the same slot.
    EXPECT_EQ(s->linkOutputToInput(0, "present.0.key", "past_key_values.0.key", kvFoldRanges(kVlmHeads, 1, kVlmSlots, kVlmHeadDim, 130)), Status::Ok);
}

// The full VLM decode flow at the on-device geometry, linked vs host loop: two turns (prefill 128
// rows, decode; prefill 81 more rows crossing slot 209, decode), byte-identical logits at every
// step — and a MID-STREAM switch to the host loop (the link-failure fallback path: materialize the
// resident cache + pending fold, drop the links, continue host-side) with no divergence.
TEST(ResidentLink, VlmGeometryLinkedMatchesUnlinkedWithMidStreamResync) {
    auto linked = makeVlmStyleSession();
    auto plain  = makeVlmStyleSession();
    ASSERT_TRUE(linked && plain);
    ASSERT_EQ(linked->bucketCount(), 2u);
    for (int part = 0; part < 2; ++part)
    {
        ASSERT_EQ(linked->linkOutputToInput(0, part ? "present.0.value" : "present.0.key", part ? "past_key_values.0.value" : "past_key_values.0.key", {}), Status::Ok);
    }

    const size_t       cacheElems = (size_t) (kVlmHeads * kVlmSlots * kVlmHeadDim);
    std::vector<float> linkedPastK(cacheElems, 0.f), linkedPastV(cacheElems, 0.f); // host mirror (prefill source)
    std::vector<float> plainPastK(cacheElems, 0.f), plainPastV(cacheElems, 0.f);
    int64_t            p = 0;

    std::vector<IOTensor> linkedOuts, plainOuts;
    bool                  linkedMode    = true;  // flips to host mode at the mid-stream resync below
    bool                  cacheOnDevice = false; // engine-resident state is ahead of linkedPast*
    int64_t               pendingSlot   = -1;

    auto runPrefill = [&](int64_t seed, int64_t realRows) {
        std::vector<float> obs  = obsValues(kVlmPrefill, seed);
        IOTensor           obsT = f32Tensor("obs", {1, kVlmHeads, kVlmPrefill, kVlmHeadDim}, obs);
        ASSERT_EQ(plain->run({pastTensor("past_key_values.0.key", plainPastK), pastTensor("past_key_values.0.value", plainPastV), obsT}, plainOuts), Status::Ok);
        ASSERT_EQ(linked->run({pastTensor("past_key_values.0.key", linkedPastK), pastTensor("past_key_values.0.value", linkedPastV), obsT}, linkedOuts), Status::Ok);
        EXPECT_EQ(outByName(linkedOuts, "logits")->data, outByName(plainOuts, "logits")->data) << "prefill p=" << p;
        // Fold the real rows on the host (the prefill bucket is unlinked in both sessions).
        hostFold(plainPastK, f32Values(*outByName(plainOuts, "present.0.key")), kVlmPrefill, realRows, p);
        hostFold(plainPastV, f32Values(*outByName(plainOuts, "present.0.value")), kVlmPrefill, realRows, p);
        hostFold(linkedPastK, f32Values(*outByName(linkedOuts, "present.0.key")), kVlmPrefill, realRows, p);
        hostFold(linkedPastV, f32Values(*outByName(linkedOuts, "present.0.value")), kVlmPrefill, realRows, p);
        p += realRows;
    };

    // Bring the engine-resident cache (plus the pending fold) back into the linked session's host
    // mirror — the driver's materializeDeviceCache / mid-stream resync.
    auto materialize = [&]() {
        for (int part = 0; part < 2; ++part)
        {
            const char         *pastNm = part ? "past_key_values.0.value" : "past_key_values.0.key";
            const char         *presNm = part ? "present.0.value" : "present.0.key";
            std::vector<float> &mirror = part ? linkedPastV : linkedPastK;
            IOTensor            resident;
            ASSERT_EQ(linked->readResident(pastNm, resident), Status::Ok);
            ASSERT_EQ(resident.data.size(), mirror.size() * sizeof(float));
            std::memcpy(mirror.data(), resident.data.data(), resident.data.size());
            if (pendingSlot >= 0)
            {
                IOTensor present;
                ASSERT_EQ(linked->readResident(presNm, present), Status::Ok);
                hostFold(mirror, f32Values(present), 1, 1, pendingSlot);
            }
        }
        pendingSlot = -1;
    };

    auto runDecode = [&](int64_t seed) {
        const int64_t      slot = p < kVlmSlots ? p : kVlmSlots - 1;
        std::vector<float> obs  = obsValues(1, seed);
        IOTensor           obsT = f32Tensor("obs", {1, kVlmHeads, 1, kVlmHeadDim}, obs);
        // Reference: host loop, full binds, host fold of the (only) present row.
        ASSERT_EQ(plain->run({pastTensor("past_key_values.0.key", plainPastK), pastTensor("past_key_values.0.value", plainPastV), obsT}, plainOuts), Status::Ok);
        hostFold(plainPastK, f32Values(*outByName(plainOuts, "present.0.key")), 1, 1, slot);
        hostFold(plainPastV, f32Values(*outByName(plainOuts, "present.0.value")), 1, 1, slot);
        if (linkedMode)
        {
            const bool rebind = !cacheOnDevice;
            for (int part = 0; part < 2; ++part)
            {
                ASSERT_EQ(linked->linkOutputToInput(0, part ? "present.0.value" : "present.0.key", part ? "past_key_values.0.value" : "past_key_values.0.key", kvFoldRanges(kVlmHeads, 1, kVlmSlots, kVlmHeadDim, rebind ? -1 : pendingSlot)), Status::Ok);
            }
            if (rebind)
            {
                ASSERT_EQ(linked->run({pastTensor("past_key_values.0.key", linkedPastK), pastTensor("past_key_values.0.value", linkedPastV), obsT}, linkedOuts), Status::Ok);
            } else
            {
                ASSERT_EQ(linked->run({obsT}, linkedOuts), Status::Ok);
            }
            cacheOnDevice = true;
            pendingSlot   = slot;
            EXPECT_TRUE(outByName(linkedOuts, "present.0.key")->data.empty()); // engine-resident
        } else
        {
            ASSERT_EQ(linked->run({pastTensor("past_key_values.0.key", linkedPastK), pastTensor("past_key_values.0.value", linkedPastV), obsT}, linkedOuts), Status::Ok);
            hostFold(linkedPastK, f32Values(*outByName(linkedOuts, "present.0.key")), 1, 1, slot);
            hostFold(linkedPastV, f32Values(*outByName(linkedOuts, "present.0.value")), 1, 1, slot);
        }
        EXPECT_EQ(outByName(linkedOuts, "logits")->data, outByName(plainOuts, "logits")->data) << "decode p=" << p;
        ++p;
    };

    // Turn 1: prefill 128 rows, decode 6 tokens (slots 128..133).
    runPrefill(/*seed=*/1, /*realRows=*/kVlmPrefill);
    for (int t = 0; t < 6; ++t)
    {
        runDecode(100 + t);
    }
    // Turn boundary: materialize the device cache for the prefill bucket, then 81 more rows —
    // the image-token count — pushing the cache past slot 209 (134..214).
    materialize();
    cacheOnDevice = false;
    runPrefill(/*seed=*/2, /*realRows=*/81);
    // Turn 2 decode: 4 linked steps (slots 215..218), then the MID-STREAM RESYNC to the host loop
    // (the link-failure fallback), then 4 more host-mode steps (219..222).
    for (int t = 0; t < 4; ++t)
    {
        runDecode(200 + t);
    }
    materialize();
    linked->clearLinks();
    linkedMode    = false;
    cacheOnDevice = false;
    for (int t = 4; t < 8; ++t)
    {
        runDecode(200 + t);
    }
    // The resynced host mirror equals the reference cache byte-for-byte.
    EXPECT_EQ(0, std::memcmp(linkedPastK.data(), plainPastK.data(), linkedPastK.size() * sizeof(float)));
    EXPECT_EQ(0, std::memcmp(linkedPastV.data(), plainPastV.data(), linkedPastV.size() * sizeof(float)));
}
