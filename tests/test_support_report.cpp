// Locks the support-report oracle to the engine: vkSupportSurvey (the code path behind
// `vknn_compile --support-report` and VulkanBackend::supportsNode) must assign a known graph's
// nodes to the expected backends with the expected refusal reasons. A change in gate behavior or
// reason vocabulary fails here before it silently shifts the scratchpad oracle snapshots.
#include "core/vk_gates.h"
#include "vknn/graph.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    Attr intAttr(int64_t v) {
        Attr a;
        a.kind = Attr::Int;
        a.i    = v;
        return a;
    }
    Attr strAttr(std::string s) {
        Attr a;
        a.kind = Attr::String;
        a.str  = std::move(s);
        return a;
    }

    TensorId tensor(Graph &g, const std::string &name, Shape shape, DType dt = DType::Float32) {
        TensorDesc d;
        d.name  = name;
        d.shape = std::move(shape);
        d.dtype = dt;
        return g.addTensor(d);
    }

    TensorId initializer(Graph &g, const std::string &name, Shape shape) {
        TensorId   id = tensor(g, name, std::move(shape));
        HostBuffer hb;
        hb.resizeElems((size_t) numElements(g.desc(id).shape), DType::Float32);
        g.initializers[id] = hb;
        return id;
    }

    void addNode(Graph &g, OpType type, const std::string &name, std::vector<TensorId> in, std::vector<TensorId> out, Attributes attr = {}) {
        Node n;
        n.type    = type;
        n.name    = name;
        n.inputs  = std::move(in);
        n.outputs = std::move(out);
        n.attr    = std::move(attr);
        g.nodes.push_back(std::move(n));
    }

} // namespace

TEST(SupportReport, SurveyMatchesExpectedAssignment) {
    Graph g;

    // dense conv: GPU (group == 1 path)
    TensorId x  = tensor(g, "x", {1, 8, 8, 8});
    TensorId w0 = initializer(g, "w0", {8, 8, 3, 3});
    TensorId c0 = tensor(g, "conv_out", {1, 8, 8, 8});
    addNode(g, OpType::Conv, "conv_dense", {x, w0}, {c0});

    // grouped (non-depthwise) conv surveyed WITHOUT the lowering pass (vkSupportSurvey gates the raw
    // graph): a surviving general grouped Conv means lowerGroupedConv could not fire, so the gate
    // routes it to the group-aware CPU op. In a real compile the constant-weight case here is lowered
    // to group-1 Convs + Concat (all GPU) before the survey.
    TensorId w1 = initializer(g, "w1", {8, 2, 3, 3});
    TensorId c1 = tensor(g, "gconv_out", {1, 8, 8, 8});
    Attributes gattr;
    gattr.map["group"] = intAttr(4);
    addNode(g, OpType::Conv, "conv_grouped", {c0, w1}, {c1}, gattr);

    // cubic GridSample: GPU (the mode list includes cubic/bicubic)
    TensorId grid = tensor(g, "grid", {1, 8, 8, 2});
    TensorId gs   = tensor(g, "gs_out", {1, 8, 8, 8});
    Attributes gsattr;
    gsattr.map["mode"] = strAttr("cubic");
    addNode(g, OpType::GridSample, "gridsample_cubic", {c1, grid}, {gs}, gsattr);

    // cast whose input is an int64 shape/index tensor to float: GPU (the int64 lanes decode to
    // compute-precision float at the pack boundary; `to`=1 is FLOAT).
    TensorId   idx = tensor(g, "idx", {4}, DType::Int64);
    TensorId   cast = tensor(g, "cast_out", {4});
    Attributes castattr;
    castattr.map["to"] = intAttr(1);
    addNode(g, OpType::Cast, "cast_i64", {idx}, {cast}, castattr);

    // TopK with a compile-time k attribute: GPU (per-slice selection on the flat path)
    TensorId   tv = tensor(g, "topk_vals", {1, 8, 8, 4});
    TensorId   ti = tensor(g, "topk_idx", {1, 8, 8, 4}, DType::Int64);
    Attributes tkattr;
    tkattr.map["k"] = intAttr(4);
    addNode(g, OpType::TopK, "topk", {gs}, {tv, ti}, tkattr);

    // Relu: GPU (registered, ungated)
    TensorId r = tensor(g, "relu_out", {1, 8, 8, 8});
    addNode(g, OpType::Relu, "relu", {gs}, {r});

    // Dropout: erased at import; a survivor has no kernel in either backend
    TensorId d = tensor(g, "drop_out", {1, 8, 8, 8});
    addNode(g, OpType::Dropout, "dropout_kept", {r}, {d});

    std::vector<NodeSupport> rows = vkSupportSurvey(g);
    ASSERT_EQ(rows.size(), 7u);

    EXPECT_EQ(rows[0].node, "conv_dense");
    EXPECT_EQ(rows[0].backend, "vulkan");
    EXPECT_TRUE(rows[0].reason.empty());

    EXPECT_EQ(rows[1].node, "conv_grouped");
    EXPECT_EQ(rows[1].backend, "cpu");
    EXPECT_EQ(rows[1].reason, "Conv: unlowered grouped conv (runtime weight or unresolved shapes)");

    EXPECT_EQ(rows[2].node, "gridsample_cubic");
    EXPECT_EQ(rows[2].backend, "vulkan");
    EXPECT_TRUE(rows[2].reason.empty());

    EXPECT_EQ(rows[3].node, "cast_i64");
    EXPECT_EQ(rows[3].backend, "vulkan");
    EXPECT_TRUE(rows[3].reason.empty());

    EXPECT_EQ(rows[4].node, "topk");
    EXPECT_EQ(rows[4].backend, "vulkan");
    EXPECT_TRUE(rows[4].reason.empty());

    EXPECT_EQ(rows[5].node, "relu");
    EXPECT_EQ(rows[5].backend, "vulkan");
    EXPECT_TRUE(rows[5].reason.empty());

    EXPECT_EQ(rows[6].node, "dropout_kept");
    EXPECT_EQ(rows[6].backend, "none");
    EXPECT_EQ(rows[6].reason, "no kernel in any backend");
}

TEST(SupportReport, CastFromInt64TargetGate) {
    // An int64 input Cast runs on the GPU for the shape-arithmetic targets (FLOAT/FLOAT16/DOUBLE,
    // INT32, INT64) and for INT8/UINT8: the int64 lanes decode to compute-precision float at the pack
    // boundary, so cast.comp reads them like any float operand, and it narrows INT8 (modulo-wrap) /
    // UINT8 (saturate) to match the CPU Cast op + readback narrowing bit-for-bit. INT16/UINT16/BOOL and
    // the 32/64-bit unsigned targets keep the CPU op. `to` is the ONNX TensorProto dtype.
    auto castNode = [&](Graph &g, const char *name, int64_t to, DType inDt) {
        TensorId   in  = tensor(g, std::string(name) + "_in", {4}, inDt);
        TensorId   out = tensor(g, std::string(name) + "_out", {4});
        Attributes a;
        a.map["to"] = intAttr(to);
        addNode(g, OpType::Cast, name, {in}, {out}, a);
    };
    Graph g;
    castNode(g, "cast_i64_to_float", 1, DType::Int64); // FLOAT  -> GPU
    castNode(g, "cast_i64_to_int32", 6, DType::Int64); // INT32  -> GPU
    castNode(g, "cast_i64_to_int64", 7, DType::Int64); // INT64  -> GPU
    castNode(g, "cast_i64_to_int8", 3, DType::Int64);  // INT8   -> GPU (modulo-wrap narrow)
    castNode(g, "cast_i64_to_uint8", 2, DType::Int64); // UINT8  -> GPU (saturate narrow)
    castNode(g, "cast_i64_to_int16", 5, DType::Int64); // INT16  -> CPU (fp32 output, no readback narrow)
    castNode(g, "cast_i64_to_bool", 9, DType::Int64);  // BOOL   -> CPU

    std::vector<NodeSupport> rows = vkSupportSurvey(g);
    ASSERT_EQ(rows.size(), 7u);
    EXPECT_EQ(rows[0].backend, "vulkan");
    EXPECT_TRUE(rows[0].reason.empty());
    EXPECT_EQ(rows[1].backend, "vulkan");
    EXPECT_EQ(rows[2].backend, "vulkan");
    EXPECT_EQ(rows[3].backend, "vulkan"); // INT8
    EXPECT_TRUE(rows[3].reason.empty());
    EXPECT_EQ(rows[4].backend, "vulkan"); // UINT8
    EXPECT_TRUE(rows[4].reason.empty());
    EXPECT_EQ(rows[5].backend, "cpu"); // INT16
    EXPECT_EQ(rows[5].reason, "Cast: int64 input to a narrow integer target");
    EXPECT_EQ(rows[6].backend, "cpu"); // BOOL
    EXPECT_EQ(rows[6].reason, "Cast: int64 input to a narrow integer target");
}

TEST(SupportReport, ConstantOfShapeIntegerFillRunsOnGpu) {
    // A resolved output shape runs on the GPU regardless of the fill dtype: an integer `value` fills as
    // the compute-precision float and the graph boundary repacks the declared int dtype on readback.
    // Only an unresolved (empty) output shape stays on the exact CPU op.
    auto cosNode = [&](Graph &g, const char *name, Shape outShape, bool intFill, DType outDt) {
        TensorId   shp = tensor(g, std::string(name) + "_shape", {(int64_t) outShape.size()}, DType::Int64);
        TensorId   out = tensor(g, std::string(name) + "_out", std::move(outShape), outDt);
        Attributes a;
        Attr       v;
        if (intFill)
        {
            v.kind = Attr::Ints;
            v.ints = {7};
        } else
        {
            v.kind   = Attr::Floats;
            v.floats = {1.5f};
        }
        a.map["value"] = v;
        addNode(g, OpType::ConstantOfShape, name, {shp}, {out}, a);
    };
    {
        Graph g;
        cosNode(g, "cos_int64", {6}, true, DType::Int64);   // integer fill, int64 output -> GPU
        cosNode(g, "cos_int32", {6}, true, DType::Int32);   // integer fill, int32 output -> GPU
        cosNode(g, "cos_float", {6}, false, DType::Float32); // float fill -> GPU (unchanged)
        std::vector<NodeSupport> rows = vkSupportSurvey(g);
        ASSERT_EQ(rows.size(), 3u);
        for (const NodeSupport &row: rows)
        {
            EXPECT_EQ(row.backend, "vulkan") << row.node;
            EXPECT_TRUE(row.reason.empty()) << row.node;
        }
    }
    {
        // unresolved (empty) output shape -> CPU
        Graph g;
        cosNode(g, "cos_unresolved", {}, true, DType::Int64);
        std::vector<NodeSupport> rows = vkSupportSurvey(g);
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].backend, "cpu");
        EXPECT_EQ(rows[0].reason, "ConstantOfShape: unresolved output shape");
    }
}

TEST(SupportReport, RangeIntegerRunsOnGpu) {
    // A Range with a resolved output size runs on the GPU regardless of the scalar dtype: integer
    // start/limit/delta generate the ramp in compute-precision float and the graph boundary repacks the
    // declared int dtype on readback. An unresolved output size stays on the exact CPU op.
    auto rangeNode = [&](Graph &g, const char *name, Shape outShape, DType scalarDt, DType outDt) {
        TensorId   s   = tensor(g, std::string(name) + "_s", {}, scalarDt);
        TensorId   l   = tensor(g, std::string(name) + "_l", {}, scalarDt);
        TensorId   d   = tensor(g, std::string(name) + "_d", {}, scalarDt);
        TensorId   out = tensor(g, std::string(name) + "_out", std::move(outShape), outDt);
        addNode(g, OpType::Range, name, {s, l, d}, {out});
    };
    {
        Graph g;
        rangeNode(g, "range_int64", {8}, DType::Int64, DType::Int64); // int64 scalars/output -> GPU
        rangeNode(g, "range_int32", {8}, DType::Int32, DType::Int32); // int32 scalars/output -> GPU
        rangeNode(g, "range_float", {8}, DType::Float32, DType::Float32); // float -> GPU (unchanged)
        std::vector<NodeSupport> rows = vkSupportSurvey(g);
        ASSERT_EQ(rows.size(), 3u);
        for (const NodeSupport &row: rows)
        {
            EXPECT_EQ(row.backend, "vulkan") << row.node;
            EXPECT_TRUE(row.reason.empty()) << row.node;
        }
    }
    {
        // unresolved output shape -> CPU
        Graph g;
        rangeNode(g, "range_unresolved", {}, DType::Int64, DType::Int64);
        std::vector<NodeSupport> rows = vkSupportSurvey(g);
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].backend, "cpu");
        EXPECT_EQ(rows[0].reason, "Range: fewer than 3 inputs or unresolved output shape");
    }
}

TEST(SupportReport, TopKGateOnCompileTimeK) {
    // TopK runs on the GPU when k is a compile-time value: the `k` attribute (opset < 10) or a constant
    // int64 input[1] (opset 10+). A runtime k input (a non-initializer) keeps the exact CPU op, since
    // the static plan fixes the output slot count.
    {
        Graph      g;
        TensorId   x  = tensor(g, "x", {1, 8, 8, 16});
        TensorId   v  = tensor(g, "v", {1, 8, 8, 4});
        TensorId   i  = tensor(g, "i", {1, 8, 8, 4}, DType::Int64);
        Attributes a;
        a.map["k"] = intAttr(4);
        addNode(g, OpType::TopK, "topk_attr_k", {x}, {v, i}, a);
        std::vector<NodeSupport> rows = vkSupportSurvey(g);
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].backend, "vulkan");
        EXPECT_TRUE(rows[0].reason.empty());
    }
    {
        Graph    g;
        TensorId x  = tensor(g, "x", {1, 8, 8, 16});
        TensorId k  = initializer(g, "k_const", {1}); // constant k input[1]
        TensorId v  = tensor(g, "v", {1, 8, 8, 4});
        TensorId i  = tensor(g, "i", {1, 8, 8, 4}, DType::Int64);
        addNode(g, OpType::TopK, "topk_const_k", {x, k}, {v, i});
        std::vector<NodeSupport> rows = vkSupportSurvey(g);
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].backend, "vulkan");
        EXPECT_TRUE(rows[0].reason.empty());
    }
    {
        Graph    g;
        TensorId x  = tensor(g, "x", {1, 8, 8, 16});
        TensorId k  = tensor(g, "k_runtime", {1}, DType::Int64); // runtime k, not an initializer
        TensorId v  = tensor(g, "v", {1, 8, 8, 4});
        TensorId i  = tensor(g, "i", {1, 8, 8, 4}, DType::Int64);
        addNode(g, OpType::TopK, "topk_runtime_k", {x, k}, {v, i});
        std::vector<NodeSupport> rows = vkSupportSurvey(g);
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].backend, "cpu");
        EXPECT_EQ(rows[0].reason, "TopK: runtime k input");
    }
}

TEST(SupportReport, BatchNormRuntimeParamsRunOnGpu) {
    // Constant params fold on the host; RUNTIME params (any of gamma/beta/mean/var a non-initializer)
    // bind as SSBOs and fold per channel in batchnorm_rt.comp. Both run on the GPU now.
    auto bnGate = [](bool allConst) {
        Graph    g;
        TensorId x = tensor(g, "x", {1, 8, 4, 4});
        auto     p = [&](const char *nm) {
            return allConst ? initializer(g, nm, {8}) : tensor(g, std::string(nm) + "_rt", {8});
        };
        TensorId ga = p("gamma"), be = p("beta"), me = p("mean"), va = p("var");
        TensorId y = tensor(g, "bn_out", {1, 8, 4, 4});
        addNode(g, OpType::BatchNorm, "bn", {x, ga, be, me, va}, {y});
        return vkSupportSurvey(g);
    };
    for (auto ac: {true, false})
    {
        std::vector<NodeSupport> rows = bnGate(ac);
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].backend, "vulkan") << "allConst=" << ac;
        EXPECT_TRUE(rows[0].reason.empty());
    }
    {
        // non-4D input still falls back to CPU
        Graph    g;
        TensorId x  = tensor(g, "x2", {8, 4});
        TensorId ga = tensor(g, "gamma2", {4}), be = tensor(g, "beta2", {4});
        TensorId me = tensor(g, "mean2", {4}), va = tensor(g, "var2", {4});
        TensorId y  = tensor(g, "bn_out2", {8, 4});
        addNode(g, OpType::BatchNorm, "bn2", {x, ga, be, me, va}, {y});
        std::vector<NodeSupport> rows = vkSupportSurvey(g);
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].backend, "cpu");
        EXPECT_EQ(rows[0].reason, "BatchNorm: fewer than 5 inputs or input not 4D");
    }
}

TEST(SupportReport, LayerNormRuntimeScaleBiasRunOnGpu) {
    // A constant scale/bias uploads flat; a RUNTIME scale/bias binds its activation buffer at dispatch.
    // Both run on the GPU flat path now (was gated to initializers).
    auto lnGate = [](bool constScale, bool constBias) {
        Graph      g;
        TensorId   x     = tensor(g, "x", {2, 8});
        TensorId   scale = constScale ? initializer(g, "scale", {8}) : tensor(g, "scale_rt", {8});
        TensorId   bias  = constBias ? initializer(g, "bias", {8}) : tensor(g, "bias_rt", {8});
        TensorId   out   = tensor(g, "ln_out", {2, 8});
        Attributes a;
        a.map["axis"] = intAttr(-1);
        addNode(g, OpType::LayerNorm, "ln", {x, scale, bias}, {out}, a);
        return vkSupportSurvey(g);
    };
    for (auto sc: {true, false})
    {
        for (auto bi: {true, false})
        {
            std::vector<NodeSupport> rows = lnGate(sc, bi);
            ASSERT_EQ(rows.size(), 1u);
            EXPECT_EQ(rows[0].backend, "vulkan") << "constScale=" << sc << " constBias=" << bi;
            EXPECT_TRUE(rows[0].reason.empty());
        }
    }
}

TEST(SupportReport, ConvTransposeRuntimeWeightBiasRunOnGpu) {
    // A constant weight/bias uploads flat; a RUNTIME weight/bias binds its activation buffer at
    // dispatch. Both run on the GPU flat path now (was gated to an initializer weight). A non-4D input
    // still falls back to CPU.
    auto ctGate = [](bool constW, bool constB) {
        Graph    g;
        TensorId x = tensor(g, "x", {1, 4, 8, 8});
        TensorId w = constW ? initializer(g, "w", {4, 4, 3, 3}) : tensor(g, "w_rt", {4, 4, 3, 3});
        TensorId b = constB ? initializer(g, "b", {4}) : tensor(g, "b_rt", {4});
        TensorId y = tensor(g, "y", {1, 4, 10, 10});
        addNode(g, OpType::ConvTranspose, "ct", {x, w, b}, {y});
        return vkSupportSurvey(g);
    };
    for (auto cw: {true, false})
    {
        for (auto cb: {true, false})
        {
            std::vector<NodeSupport> rows = ctGate(cw, cb);
            ASSERT_EQ(rows.size(), 1u);
            EXPECT_EQ(rows[0].backend, "vulkan") << "constW=" << cw << " constB=" << cb;
            EXPECT_TRUE(rows[0].reason.empty());
        }
    }
    {
        // non-4D input -> CPU
        Graph    g;
        TensorId x = tensor(g, "x3", {4, 8, 8});
        TensorId w = tensor(g, "w3_rt", {4, 4, 3});
        TensorId y = tensor(g, "y3", {4, 10, 10});
        addNode(g, OpType::ConvTranspose, "ct3", {x, w}, {y});
        std::vector<NodeSupport> rows = vkSupportSurvey(g);
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].backend, "cpu");
        EXPECT_EQ(rows[0].reason, "ConvTranspose: input not 4D");
    }
}

TEST(SupportReport, PadRuntimeValueRunsOnGpuButRuntimePadsStayCpu) {
    // A runtime pad VALUE (static geometry) runs on the GPU: flat_pad_rt reads the fill value from an
    // SSBO scalar. A runtime pads GEOMETRY keeps the CPU op, since the output shape is data-dependent.
    {
        // static pads (attr) + runtime pad value -> GPU
        Graph    g;
        TensorId x   = tensor(g, "x", {1, 4, 4, 4});
        TensorId val = tensor(g, "pad_val", {}); // runtime scalar, not an initializer
        TensorId out = tensor(g, "pad_out", {1, 4, 6, 6});
        Attributes a;
        Attr       pads;
        pads.kind = Attr::Ints;
        pads.ints = {0, 0, 1, 1, 0, 0, 1, 1};
        a.map["pads"] = pads;
        // input[1] (pads) absent via kNoTensor so the attr pads apply; input[2] is the runtime value.
        addNode(g, OpType::Pad, "pad_runtime_val", {x, kNoTensor, val}, {out}, a);
        std::vector<NodeSupport> rows = vkSupportSurvey(g);
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].backend, "vulkan");
        EXPECT_TRUE(rows[0].reason.empty());
    }
    {
        // runtime pads geometry -> CPU
        Graph    g;
        TensorId x    = tensor(g, "x", {1, 4, 4, 4});
        TensorId pads = tensor(g, "pads_runtime", {8}, DType::Int64); // runtime, not an initializer
        TensorId out  = tensor(g, "pad_out2", {1, 4, 6, 6});
        addNode(g, OpType::Pad, "pad_runtime_geom", {x, pads}, {out});
        std::vector<NodeSupport> rows = vkSupportSurvey(g);
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].backend, "cpu");
        EXPECT_EQ(rows[0].reason, "Pad: runtime pads input");
    }
}

TEST(SupportReport, QuantizeDequantizeRunOnGpuWithConstScale) {
    // Quantize/DequantizeLinear are the graph-boundary quant hops the import-time dequantize pass
    // keeps as nodes (a genuine int-graph-boundary dequant, e.g. a quantized embedding-table lookup).
    // With a constant scale/zero_point (uploaded flat in prepare) the flat GPU kernel runs them; the
    // survey assigns the Vulkan backend and no refusal reason.
    Graph    g;
    TensorId x  = tensor(g, "x", {1, 8, 8, 8});
    TensorId xs = initializer(g, "x_s", {1});
    TensorId zp = initializer(g, "x_zp", {1});
    TensorId q  = tensor(g, "q_out", {1, 8, 8, 8}, DType::UInt8);
    addNode(g, OpType::QuantizeLinear, "quant", {x, xs, zp}, {q});
    TensorId dq = tensor(g, "dq_out", {1, 8, 8, 8});
    addNode(g, OpType::DequantizeLinear, "dequant", {q, xs, zp}, {dq});

    std::vector<NodeSupport> rows = vkSupportSurvey(g);
    ASSERT_EQ(rows.size(), 2u);
    for (const NodeSupport &row: rows)
    {
        EXPECT_EQ(row.backend, "vulkan") << row.node;
        EXPECT_TRUE(row.reason.empty()) << row.node;
    }
}

TEST(SupportReport, DequantizeLinearRuntimeScaleFallsBackToCpu) {
    // A runtime (non-initializer) scale has no constant to bake in prepare, so the gate declines the
    // GPU and the exact CPU op runs the dequant. The refusal reason is stable and op-named.
    Graph    g;
    TensorId x  = tensor(g, "x", {1, 8, 8, 8}, DType::UInt8);
    TensorId xs = tensor(g, "x_s_runtime", {1}); // runtime tensor, not an initializer
    TensorId dq = tensor(g, "dq_out", {1, 8, 8, 8});
    addNode(g, OpType::DequantizeLinear, "dequant_runtime_scale", {x, xs}, {dq});

    std::vector<NodeSupport> rows = vkSupportSurvey(g);
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].backend, "cpu");
    EXPECT_EQ(rows[0].reason, "DequantizeLinear: runtime scale input");
}

TEST(SupportReport, FusedQLinearOpsReportNoKernel) {
    // The fused QLinear-family ops (QLinearConv/QLinearMatMul/QGemm/QLinearAdd/
    // QLinearGlobalAveragePool) are recognized at import but have no kernel in either backend: the
    // dequantize pass lowers them to plain float ops, so any survivor reports backend "none",
    // exactly like an Unknown op -- never a claimed Vulkan assignment.
    Graph    g;
    TensorId x = tensor(g, "x", {1, 8, 8, 8});
    TensorId w = initializer(g, "w", {8, 8, 1, 1});
    for (OpType t: {OpType::QLinearConv, OpType::QLinearMatMul, OpType::QGemm, OpType::QLinearAdd, OpType::QLinearGlobalAveragePool})
    {
        TensorId y = tensor(g, std::string("y_") + opTypeName(t), {1, 8, 8, 8});
        addNode(g, t, std::string("q_") + opTypeName(t), {x, w}, {y});
    }
    std::vector<NodeSupport> rows = vkSupportSurvey(g);
    ASSERT_EQ(rows.size(), 5u);
    for (const NodeSupport &row: rows)
    {
        EXPECT_EQ(row.backend, "none") << row.node;
        EXPECT_EQ(row.reason, "no kernel in any backend") << row.node;
    }
}

TEST(SupportReport, GateReasonIsNullSafeAndStable) {
    Graph g;
    // A Pad with runtime pads GEOMETRY refuses with a stable reason string; a null whyNot is legal.
    // (Runtime pads make the output shape data-dependent, so this gate stays CPU.)
    TensorId x    = tensor(g, "x", {1, 4, 4, 4});
    TensorId pads = tensor(g, "pads", {8}, DType::Int64); // runtime tensor, not an initializer
    TensorId out  = tensor(g, "pad_out", {1, 4, 6, 6});
    addNode(g, OpType::Pad, "pad_runtime_geom", {x, pads}, {out});

    EXPECT_FALSE(vkNodeGate(g, g.nodes[0], nullptr));
    std::string why;
    EXPECT_FALSE(vkNodeGate(g, g.nodes[0], &why));
    EXPECT_EQ(why, "Pad: runtime pads input");

    // An accepting gate leaves whyNot untouched.
    Node relu;
    relu.type    = OpType::Relu;
    relu.inputs  = {x};
    relu.outputs = {out};
    why          = "sentinel";
    EXPECT_TRUE(vkNodeGate(g, relu, &why));
    EXPECT_EQ(why, "sentinel");
}

TEST(SupportReport, ClipRuntimeBoundsRunOnGpu) {
    // A constant or absent min/max bakes into the push constant; a RUNTIME min/max binds as an SSBO
    // scalar read at dispatch (clip_rt.comp). Both cases run on the GPU flat path now.
    {
        Graph    g;
        TensorId x   = tensor(g, "x", {1, 4, 4, 4});
        TensorId lo  = initializer(g, "lo_const", {1}); // constant bound -> baked PC path
        TensorId out = tensor(g, "clip_const_out", {1, 4, 4, 4});
        addNode(g, OpType::Clip, "clip_const", {x, lo}, {out});
        std::vector<NodeSupport> rows = vkSupportSurvey(g);
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].backend, "vulkan");
        EXPECT_TRUE(rows[0].reason.empty());
    }
    {
        Graph    g;
        TensorId x   = tensor(g, "x", {1, 4, 4, 4});
        TensorId lo  = tensor(g, "lo_runtime", {1}); // runtime bound -> clip_rt SSBO path
        TensorId hi  = tensor(g, "hi_runtime", {1});
        TensorId out = tensor(g, "clip_rt_out", {1, 4, 4, 4});
        addNode(g, OpType::Clip, "clip_runtime", {x, lo, hi}, {out});
        std::vector<NodeSupport> rows = vkSupportSurvey(g);
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].backend, "vulkan");
        EXPECT_TRUE(rows[0].reason.empty());
    }
}
