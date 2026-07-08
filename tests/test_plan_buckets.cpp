// Cached per-shape plan buckets and shape-dispatching run(). A Session holds one PlanBucket per
// declared input-shape set; run() selects the bucket by exact input-shape match and dispatches its
// pre-recorded segments. These host tests (CPU backend, the correctness oracle) pin the contract:
//   - a fixed-shape model plans exactly ONE bucket (the fixed-shape path is unchanged),
//   - an ONNX-loaded session adds a bucket at runtime via prepareShapes() and runs it,
//   - a two-bucket session dispatches each input shape to its own plan with correct per-bucket
//     outputs, and rejects an unknown shape with InvalidArgument that lists the available buckets,
//   - repeating a run on a known bucket is byte-stable (the steady-state determinism invariant),
//   - a multi-bucket .vxm loads every stored bucket and dispatches among them (a .vxm session cannot
//     add buckets at runtime — prepareShapes() is refused).
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    Attr ints(std::vector<int64_t> v) {
        Attr a;
        a.kind = Attr::Ints;
        a.ints = std::move(v);
        return a;
    }

    // A dynamic-batch graph: input "data" [-1,C,H,W] -> 1x1 Conv (Cout channels, identity-ish
    // weights) with a fused Relu -> output "y". The batch axis is dynamic so the same graph plans at
    // any N; every other axis is concrete. The 1x1 conv weight is a per-output-channel scale so the
    // reference output is trivial to compute for any N.
    Graph makeDynamicBatchConvRelu(int64_t C, int64_t Cout, int64_t H, int64_t W) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "data";
        xi.shape   = {-1, C, H, W};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs.push_back(x);

        TensorDesc wi;
        wi.name          = "w";
        wi.shape         = {Cout, C, 1, 1};
        wi.isInitializer = true;
        TensorId   w     = g.addTensor(wi);
        HostBuffer hb;
        hb.resizeElems(Cout * C, DType::Float32);
        for (int64_t oc = 0; oc < Cout; ++oc)
        {
            for (int64_t ic = 0; ic < C; ++ic)
            {
                // Output channel oc reads only input channel oc (when oc<C) with weight 2, else 0:
                // y[n,oc,h,w] = relu(2 * x[n,oc,h,w]).
                hb.f32()[oc * C + ic] = (oc == ic) ? 2.f : 0.f;
            }
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
        n.attr.map["pads"]         = ints({0, 0, 0, 0});
        n.attr.map["dilations"]    = ints({1, 1});
        n.attr.map["kernel_shape"] = ints({1, 1});
        n.fusedAct                 = ActType::Relu;
        g.nodes.push_back(n);
        g.outputs = {y};
        return g;
    }

    // Run one input (fp32, shape `shape`) through a CPU session and return the single output values.
    std::vector<float> runShape(Session &s, const Shape &shape, const std::vector<float> &xdata, Status *st = nullptr) {
        IOTensor in;
        in.name  = "data";
        in.shape = shape;
        in.dtype = DType::Float32;
        in.data.resize(xdata.size() * 4);
        for (size_t i = 0; i < xdata.size(); ++i)
        {
            reinterpret_cast<float *>(in.data.data())[i] = xdata[i];
        }
        std::vector<IOTensor> outs;
        Status                r = s.run({in}, outs);
        if (st)
        {
            *st = r;
        }
        if (r != Status::Ok || outs.empty())
        {
            return {};
        }
        const float *o = outs[0].f32();
        return std::vector<float>(o, o + numElements(outs[0].shape));
    }

    // relu(2*x) reference for the makeDynamicBatchConvRelu graph (channel-diagonal weight).
    std::vector<float> reference(const std::vector<float> &x) {
        std::vector<float> y(x.size());
        for (size_t i = 0; i < x.size(); ++i)
        {
            float v = 2.f * x[i];
            y[i]    = v > 0 ? v : 0;
        }
        return y;
    }

    std::vector<float> ramp(size_t n) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i)
        {
            v[i] = (float) ((i * 7 + 3) % 11) - 5.f; // spans negatives so Relu bites
        }
        return v;
    }

} // namespace

// A fixed/dynamic-batch model plans exactly ONE bucket by default. bucketCount() reports it, and the
// default bucket resolves the dynamic batch axis to 1 (the fixed-shape contract).
TEST(PlanBuckets, FixedShapeModelHasExactlyOneBucket) {
    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto s      = Session::create(makeDynamicBatchConvRelu(4, 4, 3, 3), cfg);
    ASSERT_TRUE(s);
    EXPECT_EQ(s->bucketCount(), 1u);
    // The single bucket resolved N=1.
    ASSERT_EQ(s->inputInfo().size(), 1u);
    EXPECT_EQ(s->inputInfo()[0].shape, (Shape{1, 4, 3, 3}));
}

// Running the default (N=1) shape gives relu(2*x); every repeat is byte-identical and the bucket
// count never grows. This is the fixed-shape steady-state invariant: run() reuses the one prebuilt
// plan and re-derives nothing (on the GPU this is the zero-allocation / zero-re-record path the
// device gate asserts via Buffer::liveCount; the CPU host test asserts its observable consequences —
// byte-stable output and a stable bucket count over many iterations).
TEST(PlanBuckets, DefaultBucketRunsAndRepeatsByteIdentical) {
    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto s      = Session::create(makeDynamicBatchConvRelu(4, 4, 3, 3), cfg);
    ASSERT_TRUE(s);
    std::vector<float> x   = ramp(1 * 4 * 3 * 3);
    std::vector<float> ref = reference(x);
    std::vector<float> y0  = runShape(*s, {1, 4, 3, 3}, x);
    ASSERT_EQ(y0, ref);
    for (int iter = 0; iter < 8; ++iter)
    {
        std::vector<float> y = runShape(*s, {1, 4, 3, 3}, x);
        EXPECT_EQ(y, y0) << "iter " << iter;         // steady-state runs are byte-identical
        EXPECT_EQ(s->bucketCount(), 1u) << "iter " << iter; // no bucket is added by running
    }
}

// prepareShapes() adds a bucket for a new declared shape on an ONNX-imported (create-from-Graph)
// session; the new shape then runs and dispatches to its own bucket.
TEST(PlanBuckets, PrepareShapesAddsBucketAndRuns) {
    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto s      = Session::create(makeDynamicBatchConvRelu(4, 4, 3, 3), cfg);
    ASSERT_TRUE(s);
    ASSERT_EQ(s->bucketCount(), 1u);

    // Add a batch-2 bucket at runtime (explicit; never implicit on run()).
    EXPECT_EQ(s->prepareShapes({{"data", {2, 4, 3, 3}}}), Status::Ok);
    EXPECT_EQ(s->bucketCount(), 2u);

    // Both shapes run and each hits its own plan.
    std::vector<float> x1 = ramp(1 * 4 * 3 * 3);
    std::vector<float> x2 = ramp(2 * 4 * 3 * 3);
    Status             s1, s2;
    std::vector<float> y1 = runShape(*s, {1, 4, 3, 3}, x1, &s1);
    std::vector<float> y2 = runShape(*s, {2, 4, 3, 3}, x2, &s2);
    EXPECT_EQ(s1, Status::Ok);
    EXPECT_EQ(s2, Status::Ok);
    EXPECT_EQ(y1, reference(x1));
    EXPECT_EQ(y2, reference(x2));

    // Re-declaring an existing shape is idempotent (no duplicate bucket).
    EXPECT_EQ(s->prepareShapes({{"data", {2, 4, 3, 3}}}), Status::Ok);
    EXPECT_EQ(s->bucketCount(), 2u);
}

// An input whose shape matches no bucket is rejected with InvalidArgument; the message lists the
// shapes that ARE available (the two prepared buckets).
TEST(PlanBuckets, UnknownShapeRejectedListingBuckets) {
    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto s      = Session::create(makeDynamicBatchConvRelu(4, 4, 3, 3), cfg);
    ASSERT_TRUE(s);
    ASSERT_EQ(s->prepareShapes({{"data", {2, 4, 3, 3}}}), Status::Ok);
    ASSERT_EQ(s->bucketCount(), 2u);

    // Batch 3 was never prepared -> hard error, no compute.
    std::vector<float> x3 = ramp(3 * 4 * 3 * 3);
    Status             st;
    std::vector<float> y3 = runShape(*s, {3, 4, 3, 3}, x3, &st);
    EXPECT_EQ(st, Status::InvalidArgument);
    EXPECT_TRUE(y3.empty());
}

// The buckets are independent plans: running one then the other then the first again keeps each
// bucket's output correct (no cross-bucket state bleed through the shared backend/pool).
TEST(PlanBuckets, BucketsAreIndependentAcrossInterleavedRuns) {
    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto s      = Session::create(makeDynamicBatchConvRelu(4, 4, 3, 3), cfg);
    ASSERT_TRUE(s);
    ASSERT_EQ(s->prepareShapes({{"data", {2, 4, 3, 3}}}), Status::Ok);

    std::vector<float> x1 = ramp(1 * 4 * 3 * 3);
    std::vector<float> x2 = ramp(2 * 4 * 3 * 3);
    std::vector<float> a  = runShape(*s, {1, 4, 3, 3}, x1);
    std::vector<float> b  = runShape(*s, {2, 4, 3, 3}, x2);
    std::vector<float> c  = runShape(*s, {1, 4, 3, 3}, x1); // back to bucket 0
    EXPECT_EQ(a, reference(x1));
    EXPECT_EQ(b, reference(x2));
    EXPECT_EQ(c, a);
}

// Every compiled segment must reference its bucket's LIVE graph object, in every bucket, however the
// buckets were built. A segment captures a `Graph &` at compile time and dereferences it at run time
// (the Vulkan boundary-residency path reads tensor descriptors through it); if the graph object
// relocates after the segment is compiled — a PlanBucket move as buildBucket returns by value, or a
// buckets_ reallocation as a second bucket is appended — that reference dangles and run() reads freed
// memory (the mid-graph CPU-island segfault at a Vulkan<->CPU boundary). This pins the graph's address
// stability directly: it fails if PlanBucket::graph is ever held by value again.
TEST(PlanBuckets, SegmentGraphReferencesStayLiveAcrossBucketMoves) {
    Config cfg;
    cfg.backend = BackendKind::Cpu;

    // Single bucket built via buildBucket-returns-by-value: the returned PlanBucket is moved into
    // buckets_, relocating its graph. The segment's captured reference must still point at it.
    auto s = Session::create(makeDynamicBatchConvRelu(4, 4, 3, 3), cfg);
    ASSERT_TRUE(s);
    EXPECT_TRUE(s->segmentGraphsLive()) << "segment graph dangles after buildBucket move";

    // Append a second bucket, forcing the buckets_ vector to grow and relocate bucket 0. Bucket 0's
    // already-compiled segment must not be left pointing at the old (moved-from) graph address.
    ASSERT_EQ(s->prepareShapes({{"data", {2, 4, 3, 3}}}), Status::Ok);
    ASSERT_EQ(s->bucketCount(), 2u);
    EXPECT_TRUE(s->segmentGraphsLive()) << "segment graph dangles after buckets_ reallocation";

    // Bucket 0 still runs correctly through the relocated graph (dereferencing the reference is safe).
    std::vector<float> x = ramp(1 * 4 * 3 * 3);
    EXPECT_EQ(runShape(*s, {1, 4, 3, 3}, x), reference(x));
}

// A multi-bucket .vxm (two graphs resolved at two batch sizes, sharing one weight pool) loads every
// stored bucket and dispatches each input shape to its own plan. A .vxm session cannot add buckets at
// runtime, so prepareShapes() is refused; an unknown shape is InvalidArgument.
TEST(PlanBuckets, MultiBucketVxmLoadsAndDispatches) {
    // Two already-optimized graphs (batch 1 and 2) built by running the passes at each batch — exactly
    // what vknn_compile --bucket produces, minus the file plumbing (exercised in test_vxm_buckets).
    Graph       g1 = makeDynamicBatchConvRelu(4, 4, 3, 3);
    Graph       g2 = makeDynamicBatchConvRelu(4, 4, 3, 3);
    PassOptions o1;
    o1.batch = 1;
    PassOptions o2;
    o2.batch = 2;
    runStandardPasses(g1, o1);
    runStandardPasses(g2, o2);

    std::string path = testing::TempDir() + "plan_buckets_2b.vxm";
    ASSERT_TRUE(saveGraphBinBuckets({g1, g2}, {"b1", "b2"}, path));

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto s      = Session::createFromVxm(path, cfg);
    std::remove(path.c_str());
    ASSERT_TRUE(s);
    EXPECT_EQ(s->bucketCount(), 2u);
    // Loading N buckets push_backs them in a loop, reallocating buckets_ as it grows; each earlier
    // bucket's segments must still reference their (relocated) live graph.
    EXPECT_TRUE(s->segmentGraphsLive());

    std::vector<float> x1 = ramp(1 * 4 * 3 * 3);
    std::vector<float> x2 = ramp(2 * 4 * 3 * 3);
    Status             s1, s2, s3;
    std::vector<float> y1 = runShape(*s, {1, 4, 3, 3}, x1, &s1);
    std::vector<float> y2 = runShape(*s, {2, 4, 3, 3}, x2, &s2);
    EXPECT_EQ(s1, Status::Ok);
    EXPECT_EQ(s2, Status::Ok);
    EXPECT_EQ(y1, reference(x1));
    EXPECT_EQ(y2, reference(x2));

    // A shape with no bucket is rejected; the message lists what is available.
    std::vector<float> x3 = ramp(3 * 4 * 3 * 3);
    runShape(*s, {3, 4, 3, 3}, x3, &s3);
    EXPECT_EQ(s3, Status::InvalidArgument);

    // A .vxm session's buckets are fixed at compile time.
    EXPECT_EQ(s->prepareShapes({{"data", {3, 4, 3, 3}}}), Status::Unsupported);
    EXPECT_EQ(s->bucketCount(), 2u);
}

// A multi-graph .vxm (buckets from DIFFERENT graphs with disjoint input names) dispatches run() by
// the bound input NAME, not just shape: a caller binding one graph's input can never select another
// graph's bucket, whatever the shapes. Each bucket computes with its own weights, and the per-bucket
// IO accessors describe each graph.
TEST(PlanBuckets, MultiGraphVxmDispatchesByInputName) {
    // Graph A: "data" [1,4,3,3] -> conv(diag 2) -> relu -> "y".
    Graph ga = makeDynamicBatchConvRelu(4, 4, 3, 3);
    // Graph B: same structure, but its own names ("pix" -> "emb") and weight scale 3.
    Graph gb = makeDynamicBatchConvRelu(4, 4, 3, 3);
    gb.tensors[gb.inputs[0]].name  = "pix";
    gb.tensors[gb.outputs[0]].name = "emb";
    {
        TensorId w = gb.find("w");
        ASSERT_NE(w, kNoTensor);
        HostBuffer &hb = gb.initializers[w];
        for (int64_t i = 0; i < 16; ++i)
        {
            hb.f32()[i] = (i % 5 == 0) ? 3.f : 0.f; // diag(3) over the 4x4 channel matrix
        }
        gb.tensors[w].name = "wb"; // distinct weight identity across the two graphs
    }

    // Resolve both graphs' dynamic batch to 1 through the pass pipeline and save the optimized
    // buckets into one container — the same product vknn_compile --graph writes.
    PassOptions oa;
    oa.batch = 1;
    PassOptions ob;
    ob.batch = 1;
    runStandardPasses(ga, oa);
    runStandardPasses(gb, ob);
    std::string path = testing::TempDir() + "vxm_multi_graph_session.vxm";
    ASSERT_TRUE(saveGraphBinBuckets({ga, gb}, {"a", "b"}, path));

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto s = Session::createFromVxm(path, cfg);
    std::remove(path.c_str());
    ASSERT_TRUE(s);
    ASSERT_EQ(s->bucketCount(), 2u);
    EXPECT_TRUE(s->segmentGraphsLive());

    // Per-bucket IO: bucket 0 is "data"->"y", bucket 1 is "pix"->"emb".
    ASSERT_EQ(s->inputInfo(0).size(), 1u);
    ASSERT_EQ(s->inputInfo(1).size(), 1u);
    EXPECT_EQ(s->inputInfo(0)[0].name, "data");
    EXPECT_EQ(s->inputInfo(1)[0].name, "pix");
    EXPECT_EQ(s->outputInfo(1)[0].name, "emb");
    ASSERT_EQ(s->bucketKeys().size(), 2u);

    auto runNamed = [&](const char *name, const std::vector<float> &xdata, Status *st) {
        IOTensor in;
        in.name  = name;
        in.shape = Shape {1, 4, 3, 3};
        in.dtype = DType::Float32;
        in.data.resize(xdata.size() * 4);
        std::memcpy(in.data.data(), xdata.data(), in.data.size());
        std::vector<IOTensor> outs;
        Status                r = s->run({in}, outs);
        if (st)
        {
            *st = r;
        }
        if (r != Status::Ok || outs.empty())
        {
            return std::vector<float>();
        }
        const float *o = outs[0].f32();
        return std::vector<float>(o, o + numElements(outs[0].shape));
    };

    std::vector<float> x = ramp(1 * 4 * 3 * 3);
    Status             ra, rb, rc;

    // "data" selects graph A: relu(2x).
    std::vector<float> ya = runNamed("data", x, &ra);
    EXPECT_EQ(ra, Status::Ok);
    EXPECT_EQ(ya, reference(x));

    // "pix" selects graph B: relu(3x) — same input SHAPE, different graph; the name decides.
    std::vector<float> yb = runNamed("pix", x, &rb);
    EXPECT_EQ(rb, Status::Ok);
    std::vector<float> want(x.size());
    for (size_t i = 0; i < x.size(); ++i)
    {
        float v = 3.f * x[i];
        want[i] = v > 0 ? v : 0;
    }
    EXPECT_EQ(yb, want);

    // An input name no bucket has is rejected (never silently served by bucket 0's defaults).
    runNamed("nosuch", x, &rc);
    EXPECT_EQ(rc, Status::InvalidArgument);
}

// An UNNAMED sole entry in a multi-graph session binds positionally to a SINGLE-input graph only:
// its shape picks that graph even when a multi-input bucket precedes it in file order (an
// all-defaults key match on a bucket that binds nothing of the caller is no claim on it), and a
// run binding no bucket at all is rejected.
TEST(PlanBuckets, MultiGraphUnnamedSoleEntryPicksBySingleInputShape) {
    // Bucket 0: TWO inputs (a+b elementwise add). Bucket 1: single-input conv graph "pix".
    Graph g2in;
    {
        TensorDesc ai;
        ai.name    = "a";
        ai.shape   = {1, 4, 3, 3};
        ai.isInput = true;
        TensorId a = g2in.addTensor(ai);
        TensorDesc bi;
        bi.name    = "b";
        bi.shape   = {1, 4, 3, 3};
        bi.isInput = true;
        TensorId b = g2in.addTensor(bi);
        g2in.inputs = {a, b};
        TensorDesc yo;
        yo.name     = "sum";
        yo.isOutput = true;
        TensorId y  = g2in.addTensor(yo);
        g2in.outputs = {y};
        Node n;
        n.type    = OpType::Add;
        n.name    = "add";
        n.inputs  = {a, b};
        n.outputs = {y};
        g2in.nodes.push_back(n);
    }
    Graph g1in = makeDynamicBatchConvRelu(4, 4, 5, 5);
    g1in.tensors[g1in.inputs[0]].name  = "pix";
    g1in.tensors[g1in.outputs[0]].name = "emb";

    PassOptions o;
    o.batch = 1;
    runStandardPasses(g2in, o);
    runStandardPasses(g1in, o);
    std::string path = testing::TempDir() + "vxm_unnamed_dispatch.vxm";
    ASSERT_TRUE(saveGraphBinBuckets({g2in, g1in}, {"add2", "conv1"}, path));

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto s      = Session::createFromVxm(path, cfg);
    std::remove(path.c_str());
    ASSERT_TRUE(s);
    ASSERT_EQ(s->bucketCount(), 2u);

    // Unnamed sole entry with the conv graph's shape: must reach bucket 1 (not the earlier add
    // bucket via its all-defaults key).
    IOTensor in;
    in.shape = Shape {1, 4, 5, 5};
    in.dtype = DType::Float32;
    std::vector<float> x = ramp(1 * 4 * 5 * 5);
    in.data.resize(x.size() * 4);
    std::memcpy(in.data.data(), x.data(), in.data.size());
    std::vector<IOTensor> outs;
    ASSERT_EQ(s->run({in}, outs), Status::Ok);
    ASSERT_FALSE(outs.empty());
    EXPECT_EQ(outs[0].name, "emb");
    EXPECT_EQ(std::vector<float>(outs[0].f32(), outs[0].f32() + x.size()), reference(x));

    // An unnamed entry whose shape matches NO single-input bucket binds nothing anywhere: rejected.
    IOTensor bad;
    bad.shape = Shape {1, 4, 7, 7};
    bad.dtype = DType::Float32;
    bad.data.assign(1 * 4 * 7 * 7 * 4, 0);
    std::vector<IOTensor> outs2;
    EXPECT_EQ(s->run({bad}, outs2), Status::InvalidArgument);
}
