// Fused-edge integrity across the passes that remove nodes or rewire edges once fusion metadata
// exists: a fusedResidual/fusedBias reference lives outside node.inputs (rewireTensor's contract;
// a legacy .vxm can carry it unmirrored, model_io round-trips the field), so every pass that
// deletes tensors, rewires reads, or reorders nodes must follow those edges. Covers topoSort's
// dependency order, eliminateDeadNodes / pruneDeadInitializers liveness, the load-time MatMul
// view fold, and the insertLayoutConverts / markFp32 convert splicing.
#include "core/matmul_view.h"
#include "import/passes.h"
#include "import/passes_internal.h"
#include "vknn/graph.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    TensorId addAct(Graph &g, const std::string &name, Shape shape, bool flat = false) {
        TensorDesc d;
        d.name    = name;
        d.shape   = std::move(shape);
        d.gpuFlat = flat;
        return g.addTensor(d);
    }

    TensorId addInputT(Graph &g, const std::string &name, Shape shape, bool flat = false) {
        TensorId t          = addAct(g, name, std::move(shape), flat);
        g.desc(t).isInput   = true;
        g.inputs.push_back(t);
        return t;
    }

    TensorId addFloatInit(Graph &g, const std::string &name, Shape shape) {
        TensorDesc d;
        d.name          = name;
        d.shape         = shape;
        d.isInitializer = true;
        TensorId   t    = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems(numElements(shape), DType::Float32);
        for (int64_t i = 0; i < numElements(shape); ++i)
        {
            hb.f32()[i] = 1.f;
        }
        g.initializers[t] = hb;
        return t;
    }

    TensorId addI64Init(Graph &g, const std::string &name, const std::vector<int64_t> &vals) {
        TensorDesc d;
        d.name          = name;
        d.shape         = {(int64_t) vals.size()};
        d.dtype         = DType::Int64;
        d.isInitializer = true;
        TensorId   t    = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) vals.size(), DType::Int64);
        std::memcpy(hb.i64(), vals.data(), vals.size() * sizeof(int64_t));
        g.initializers[t] = hb;
        return t;
    }

    void addNode(Graph &g, OpType type, const std::string &name, std::vector<TensorId> ins, TensorId out) {
        Node n;
        n.type    = type;
        n.name    = name;
        n.inputs  = std::move(ins);
        n.outputs = {out};
        g.nodes.push_back(n);
    }

    const Node *findNode(const Graph &g, const std::string &name) {
        for (const Node &n: g.nodes)
        {
            if (n.name == name)
            {
                return &n;
            }
        }
        return nullptr;
    }

    int nodeIndex(const Graph &g, const std::string &name) {
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            if (g.nodes[i].name == name)
            {
                return (int) i;
            }
        }
        return -1;
    }

} // namespace

// --- rewireTensor redirects EVERY reference to a tensor: node inputs, graph outputs, and both
// fused edges. A fusedBias left behind would dangle at the retired tensor after a fusion pass
// folds its producer away. ---
TEST(FusedEdges, RewireTensorFollowsBothFusedEdges) {
    Graph    g;
    TensorId x  = addInputT(g, "x", {1, 4, 2, 2});
    TensorId r  = addAct(g, "r", {1, 4, 2, 2});
    TensorId r2 = addAct(g, "r2", {1, 4, 2, 2});
    TensorId b  = addAct(g, "b", {4});
    TensorId b2 = addAct(g, "b2", {4});
    TensorId y  = addAct(g, "y", {1, 4, 2, 2});
    Node     nd;
    nd.type          = OpType::Conv;
    nd.name          = "conv";
    nd.inputs        = {x, addFloatInit(g, "w", {4, 4, 1, 1})};
    nd.outputs       = {y};
    nd.fusedResidual = r;
    nd.fusedBias     = b;
    g.nodes.push_back(nd);
    g.outputs = {y};

    rewireTensor(g, r, r2);
    EXPECT_EQ(g.nodes[0].fusedResidual, r2) << "fusedResidual must follow the rewire";
    rewireTensor(g, b, b2);
    EXPECT_EQ(g.nodes[0].fusedBias, b2) << "fusedBias must follow the rewire";
}

// --- topoSort must order a node after its fused-edge producers even when the edge is not
// mirrored into node.inputs: the visible input edges alone leave the two nodes independent, and
// scheduling the reader first would execute it before the residual is written. ---
TEST(FusedEdges, TopoSortOrdersFusedResidualProducerFirst) {
    Graph    g;
    TensorId x = addInputT(g, "x", {1, 4, 2, 2});
    TensorId z = addInputT(g, "z", {1, 4, 2, 2});
    TensorId r = addAct(g, "r", {1, 4, 2, 2});
    TensorId y = addAct(g, "y", {1, 4, 2, 2});
    g.desc(y).isOutput = true;
    // Deliberately listed reader-first: only the fused edge relates the two nodes.
    Node conv;
    conv.type          = OpType::Conv;
    conv.name          = "conv";
    conv.inputs        = {x, addFloatInit(g, "w", {4, 4, 1, 1})};
    conv.outputs       = {y};
    conv.fusedResidual = r; // not in inputs: the legacy residual encoding
    g.nodes.push_back(conv);
    addNode(g, OpType::Relu, "mk_r", {z}, r);
    g.outputs = {y};

    g.topoSort();
    ASSERT_EQ(g.nodes.size(), 2u);
    EXPECT_EQ(g.nodes[0].name, "mk_r") << "the fused-residual producer must be scheduled before its reader";
    EXPECT_EQ(g.nodes[1].name, "conv");
}

// --- eliminateDeadNodes seeds liveness from graph outputs and follows fused edges: a producer
// whose only consumer is a fusedResidual/fusedBias reference must survive, while a genuinely
// unreferenced node is still removed. ---
TEST(FusedEdges, EliminateDeadNodesKeepsFusedEdgeProducers) {
    Graph    g;
    TensorId x  = addInputT(g, "x", {1, 4, 2, 2});
    TensorId q  = addInputT(g, "q", {1, 4, 4});
    TensorId k  = addInputT(g, "k", {1, 4, 4});
    TensorId r  = addAct(g, "r", {1, 4, 2, 2});
    TensorId b  = addAct(g, "b", {4});
    TensorId d  = addAct(g, "d", {1, 4, 2, 2});
    TensorId y  = addAct(g, "y", {1, 4, 2, 2});
    TensorId y2 = addAct(g, "y2", {1, 4, 4});
    addNode(g, OpType::Relu, "mk_r", {x}, r);
    addNode(g, OpType::Relu, "mk_b", {x}, b);
    addNode(g, OpType::Relu, "mk_dead", {x}, d); // nothing references d
    Node conv;
    conv.type          = OpType::Conv;
    conv.name          = "conv";
    conv.inputs        = {x, addFloatInit(g, "w", {4, 4, 1, 1})};
    conv.outputs       = {y};
    conv.fusedResidual = r; // not in inputs
    g.nodes.push_back(conv);
    Node mm;
    mm.type      = OpType::MatMul;
    mm.name      = "mm";
    mm.inputs    = {q, k};
    mm.outputs   = {y2};
    mm.fusedBias = b; // not in inputs
    g.nodes.push_back(mm);
    g.outputs = {y, y2};

    eliminateDeadNodes(g);
    EXPECT_NE(findNode(g, "mk_r"), nullptr) << "fusedResidual reference must keep its producer alive";
    EXPECT_NE(findNode(g, "mk_b"), nullptr) << "fusedBias reference must keep its producer alive";
    EXPECT_EQ(findNode(g, "mk_dead"), nullptr) << "an unreferenced node is still dead";
}

// --- pruneDeadInitializers keeps payloads referenced only through a fused edge (the edge is not
// always mirrored into node.inputs) and still drops genuinely orphaned payloads. ---
TEST(FusedEdges, PruneDeadInitializersKeepsFusedEdgeTargets) {
    Graph    g;
    TensorId x     = addInputT(g, "x", {1, 4, 2, 2});
    TensorId q     = addInputT(g, "q", {1, 4, 4});
    TensorId k     = addInputT(g, "k", {1, 4, 4});
    TensorId rInit = addFloatInit(g, "r_init", {1, 4, 2, 2});
    TensorId bInit = addFloatInit(g, "b_init", {4});
    TensorId dead  = addFloatInit(g, "dead_init", {4});
    TensorId y     = addAct(g, "y", {1, 4, 2, 2});
    TensorId y2    = addAct(g, "y2", {1, 4, 4});
    Node     conv;
    conv.type          = OpType::Conv;
    conv.name          = "conv";
    conv.inputs        = {x, addFloatInit(g, "w", {4, 4, 1, 1})};
    conv.outputs       = {y};
    conv.fusedResidual = rInit; // not in inputs
    g.nodes.push_back(conv);
    Node mm;
    mm.type      = OpType::MatMul;
    mm.name      = "mm";
    mm.inputs    = {q, k};
    mm.outputs   = {y2};
    mm.fusedBias = bInit; // not in inputs
    g.nodes.push_back(mm);
    g.outputs = {y, y2};

    pruneDeadInitializers(g);
    EXPECT_TRUE(g.initializers.count(rInit)) << "fusedResidual initializer must survive the prune";
    EXPECT_TRUE(g.initializers.count(bInit)) << "fusedBias initializer must survive the prune";
    EXPECT_FALSE(g.initializers.count(dead)) << "an orphaned payload is still dropped";
    EXPECT_FALSE(g.desc(dead).isInitializer);
}

// --- foldMatMulViews rewires the MatMul operand to the chain source and DCEs the chain, but a
// chain intermediate referenced by another node's fused edge must keep its producers: the fold
// never deletes a tensor a fused edge still reads. ---
TEST(FusedEdges, FoldMatMulViewsKeepsChainTensorReadByFusedEdge) {
    // The decode QK idiom at toy size (mirrors test_matmul_view): k reshapes/expands/reshapes to
    // r2 [1,6,5,4], transposes to [1,6,4,5], and MatMul(q, .) -> [1,6,1,5].
    Graph    g;
    TensorId q  = addInputT(g, "q", {1, 6, 1, 4});
    TensorId k  = addInputT(g, "k", {1, 2, 5, 4});
    TensorId r1 = addAct(g, "r1", {});
    TensorId ex = addAct(g, "ex", {});
    TensorId r2 = addAct(g, "r2", {});
    TensorId kt = addAct(g, "kt", {});
    addNode(g, OpType::Reshape, "reshape1", {k, addI64Init(g, "s1", {1, 2, 1, 5, 4})}, r1);
    addNode(g, OpType::Expand, "expand", {r1, addI64Init(g, "s2", {1, 2, 3, 5, 4})}, ex);
    addNode(g, OpType::Reshape, "reshape2", {ex, addI64Init(g, "s3", {1, 6, 5, 4})}, r2);
    {
        Node tr;
        tr.type    = OpType::Transpose;
        tr.name    = "kT";
        tr.inputs  = {r2};
        tr.outputs = {kt};
        Attr perm;
        perm.kind           = Attr::Ints;
        perm.ints           = {0, 1, 3, 2};
        tr.attr.map["perm"] = perm;
        g.nodes.push_back(tr);
    }
    TensorId y = addAct(g, "y", {});
    g.desc(y).isOutput = true;
    addNode(g, OpType::MatMul, "mm", {q, kt}, y);
    g.outputs = {y};
    runStandardPasses(g, PassOptions {});

    // A side Conv reads the chain intermediate r2 through its fused edge ONLY (legacy encoding):
    // no node input names r2, so the chain's liveness hangs on the edge alone.
    TensorId cy = addAct(g, "cy", {1, 6, 1, 4});
    g.desc(cy).isOutput = true;
    Node conv;
    conv.type          = OpType::Conv;
    conv.name          = "side_conv";
    conv.inputs        = {q, addFloatInit(g, "cw", {6, 6, 1, 1})};
    conv.outputs       = {cy};
    conv.fusedResidual = g.find("r2"); // the chain intermediate, outside inputs
    g.nodes.push_back(conv);
    g.outputs.push_back(cy);

    foldMatMulViews(g);
    const Node *mm = findNode(g, "mm");
    ASSERT_NE(mm, nullptr);
    EXPECT_TRUE(mm->attr.has(kMmView)) << "the chain must still fold";
    EXPECT_EQ(g.desc(mm->inputs[1]).name, "k") << "operand rewired to the chain source";
    EXPECT_EQ(findNode(g, "kT"), nullptr) << "the truly dead transpose is removed";
    EXPECT_NE(findNode(g, "reshape2"), nullptr) << "the fused-edge target's producer must survive";
    EXPECT_NE(findNode(g, "expand"), nullptr) << "everything feeding the fused-edge target must survive";
}

// --- insertLayoutConverts must treat a fused residual as a read in the node's own layout world:
// when the edge is mirrored into inputs (conv's bias slot doubles as the mirror; the conv kernel
// tests inputs[2] != fusedResidual to tell bias from residual), converting the inputs entry
// without rewiring the edge splits the two and turns the residual into a phantom bias. ---
TEST(FusedEdges, InsertLayoutConvertsBridgesMirroredFusedResidual) {
    Graph    g;
    TensorId x = addInputT(g, "x", {1, 4, 2, 2});
    TensorId z = addInputT(g, "z", {1, 4, 2, 2});
    TensorId r = addAct(g, "r", {1, 4, 2, 2});
    TensorId y = addAct(g, "y", {1, 4, 2, 2});
    g.desc(y).isOutput = true;
    {
        Node tr;
        tr.type    = OpType::Transpose; // a FLAT-class producer: r lands in the flat world
        tr.name    = "mk_r";
        tr.inputs  = {z};
        tr.outputs = {r};
        Attr perm;
        perm.kind           = Attr::Ints;
        perm.ints           = {0, 1, 3, 2};
        tr.attr.map["perm"] = perm;
        g.nodes.push_back(tr);
    }
    Node conv;
    conv.type          = OpType::Conv; // NC4HW4-class reader
    conv.name          = "conv";
    conv.inputs        = {x, addFloatInit(g, "w", {4, 4, 1, 1}), r};
    conv.outputs       = {y};
    conv.fusedResidual = r; // mirrored at inputs[2]
    g.nodes.push_back(conv);
    g.outputs = {y};

    insertLayoutConverts(g);
    const Node *c = findNode(g, "conv");
    ASSERT_NE(c, nullptr);
    ASSERT_GE(c->inputs.size(), 3u);
    EXPECT_EQ(c->fusedResidual, c->inputs[2]) << "the mirrored inputs entry and the fused edge must stay one tensor";
    EXPECT_NE(c->fusedResidual, r) << "the flat residual must be read through a layout convert";
    EXPECT_FALSE(g.desc(c->fusedResidual).gpuFlat) << "the residual must arrive in the conv's NC4HW4 world";
}

// --- The unmirrored (legacy .vxm) form: the fused edge is the ONLY reference, so the pass has to
// splice the convert for the edge itself and topoSort has to schedule that convert before the
// conv it feeds. ---
TEST(FusedEdges, InsertLayoutConvertsBridgesUnmirroredFusedResidual) {
    Graph    g;
    TensorId x = addInputT(g, "x", {1, 4, 2, 2});
    TensorId z = addInputT(g, "z", {1, 4, 2, 2});
    TensorId r = addAct(g, "r", {1, 4, 2, 2});
    TensorId y = addAct(g, "y", {1, 4, 2, 2});
    g.desc(y).isOutput = true;
    {
        Node tr;
        tr.type    = OpType::Transpose;
        tr.name    = "mk_r";
        tr.inputs  = {z};
        tr.outputs = {r};
        Attr perm;
        perm.kind           = Attr::Ints;
        perm.ints           = {0, 1, 3, 2};
        tr.attr.map["perm"] = perm;
        g.nodes.push_back(tr);
    }
    Node conv;
    conv.type          = OpType::Conv;
    conv.name          = "conv";
    conv.inputs        = {x, addFloatInit(g, "w", {4, 4, 1, 1})};
    conv.outputs       = {y};
    conv.fusedResidual = r; // not in inputs
    g.nodes.push_back(conv);
    g.outputs = {y};

    insertLayoutConverts(g);
    const Node *c = findNode(g, "conv");
    ASSERT_NE(c, nullptr);
    EXPECT_NE(c->fusedResidual, r) << "the edge itself must be rewired to a converted copy";
    EXPECT_FALSE(g.desc(c->fusedResidual).gpuFlat) << "the residual must arrive in the conv's NC4HW4 world";
    // The splice + topoSort must schedule the convert before the conv that reads it (the edge is
    // the only dependency relating them).
    int convertIdx = -1;
    for (size_t i = 0; i < g.nodes.size(); ++i)
    {
        for (TensorId o: g.nodes[i].outputs)
        {
            if (o == c->fusedResidual)
            {
                convertIdx = (int) i;
            }
        }
    }
    ASSERT_GE(convertIdx, 0) << "a ConvertLayout must produce the rewired residual";
    EXPECT_LT(convertIdx, nodeIndex(g, "conv")) << "the convert must be scheduled before its fused-edge reader";
}

// --- markFp32's frontier walk must bridge a storage-precision mismatch across a fused edge the
// same way it bridges node inputs, keeping a mirrored inputs entry and the edge identical. ---
TEST(FusedEdges, MarkFp32BridgesMirroredFusedResidualPrecision) {
    Graph    g;
    TensorId x = addInputT(g, "x", {1, 4, 2, 2});
    TensorId z = addInputT(g, "z", {1, 4, 2, 2}, true);
    TensorId r = addAct(g, "r", {1, 4, 2, 2}, true);
    TensorId y = addAct(g, "y", {1, 4, 2, 2});
    g.desc(y).isOutput  = true;
    g.desc(r).storeFp32 = true; // pinned by an earlier pass; the fp16 conv must not read it raw
    addNode(g, OpType::Relu, "mk_r", {z}, r);
    Node conv;
    conv.type          = OpType::Conv;
    conv.name          = "conv";
    conv.inputs        = {x, addFloatInit(g, "w", {4, 4, 1, 1}), r};
    conv.outputs       = {y};
    conv.fusedResidual = r; // mirrored at inputs[2]
    g.nodes.push_back(conv);
    g.outputs = {y};

    markFp32(g, "");
    const Node *c = findNode(g, "conv");
    ASSERT_NE(c, nullptr);
    ASSERT_GE(c->inputs.size(), 3u);
    EXPECT_EQ(c->fusedResidual, c->inputs[2]) << "the mirrored inputs entry and the fused edge must stay one tensor";
    EXPECT_NE(c->fusedResidual, r) << "the pinned fp32 residual must be read through a ConvertDtype";
    EXPECT_FALSE(g.desc(c->fusedResidual).storeFp32) << "the bridge lands in the conv's fp16 world";
}

// --- The unmirrored form of the same precision bridge: the edge is the only reference and must
// be rewired to the ConvertDtype output. ---
TEST(FusedEdges, MarkFp32BridgesUnmirroredFusedResidualPrecision) {
    Graph    g;
    TensorId x = addInputT(g, "x", {1, 4, 2, 2});
    TensorId z = addInputT(g, "z", {1, 4, 2, 2}, true);
    TensorId r = addAct(g, "r", {1, 4, 2, 2}, true);
    TensorId y = addAct(g, "y", {1, 4, 2, 2});
    g.desc(y).isOutput  = true;
    g.desc(r).storeFp32 = true;
    addNode(g, OpType::Relu, "mk_r", {z}, r);
    Node conv;
    conv.type          = OpType::Conv;
    conv.name          = "conv";
    conv.inputs        = {x, addFloatInit(g, "w", {4, 4, 1, 1})};
    conv.outputs       = {y};
    conv.fusedResidual = r; // not in inputs
    g.nodes.push_back(conv);
    g.outputs = {y};

    markFp32(g, "");
    const Node *c = findNode(g, "conv");
    ASSERT_NE(c, nullptr);
    EXPECT_NE(c->fusedResidual, r) << "the edge itself must be rewired to the ConvertDtype output";
    EXPECT_FALSE(g.desc(c->fusedResidual).storeFp32) << "the bridge lands in the conv's fp16 world";
}
