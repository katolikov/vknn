// What a fused pointwise unit may carry when it rides a producer's store, rather than running as
// its own dispatch.
//
// shaders/pw_epilogue.glsl is #included by every epilogue-capable kernel -- conv, matmul, pooling,
// winograd, transpose and the rest -- so whatever the epilogue form declares is inlined into all of
// them, used or not. ONE thing is bounded for that reason: one extra dispatch for the rare unit,
// against slowing every producer in the graph.
//
// The OPERAND budget is deliberately NOT bounded, and the last test here pins that. Narrowing it
// changes answers rather than speed: a unit split for want of a slot rounds its intermediate
// through fp16 storage where the whole unit kept it in an fp32 register, and a production
// image-warp graph fell from matching the CPU oracle within one code to 15 dB when the epilogue
// was narrowed to 6 slots. It also bought nothing measurable -- see the note on kPwMaxOperands.
//
//   * broadcast classes, in the blocked world only. A geometric class (per-pixel, row/column,
//     packed) resolves its operand by first recovering the store's (n, channel-block, h, w) from
//     the output index; the direct classes address it from that index alone. The NC4HW4 appliers
//     carry an arm per class and the _epi builds compile PW_BCAST_MASK_DIRECT only. The flat
//     applier has no arms at all -- every class resolves through pwFlatIdx's one strided walk --
//     so a flat-world host keeps hosting every class, which is what the attention mask and the
//     RoPE tables depend on.
//
// A unit past either bound stays standalone, where the fused_pw_* kernels serve it with the full
// operand budget and every broadcast class compiled in. The gate is in fuse_pointwise_chains, and
// PwEpi::prepare refuses an attached blocked-world plan that carries a geometric class as a
// backstop -- the kernel has no arm for one, so it would otherwise fall through to a same-shape
// read and compute a wrong answer quietly.
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/op_type.h"
#include "vknn/session.h"
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    // Run extents. A single batch, because the per-pixel and *Splat classes are single-batch only
    // and this fixture uses the per-pixel one as its geometric case.
    constexpr int64_t kRunN = 1;
    constexpr int64_t kRunC = 8;
    constexpr int64_t kRunH = 5;
    constexpr int64_t kRunW = 4;
    // A broadcast (size-1) mask axis.
    constexpr int64_t kBcast = 1;

    // MSVC at C++17 rejects designated initializers; a tiny helper names the tensor.
    TensorDesc namedDesc(const std::string &name) {
        TensorDesc d;
        d.name = name;
        return d;
    }

    // x[N,C,H,W] -> Transpose(perm 0,1,3,2) -> one Binary per mask shape.
    //
    // Transpose is epilogue-capable and needs no weights, so the unit has a real producer to attach
    // to and the test reads the attach decision alone. It is also LayoutClass::Flat, which is the
    // world that keeps hosting every broadcast class; buildBlockedChainGraph below is its blocked
    // counterpart. Masks arrive as graph inputs through their own Abs so they reach the unit as
    // runtime activations; the ops alternate Mul/Add so no two steps merge. Mask shapes are given
    // in the TRANSPOSED run's extents.
    //
    // Every mask producer is emitted BEFORE the transpose. An epilogue runs inside the producer's
    // dispatch, so each operand it appends must already be computed by then -- mask producers
    // ordered after the transpose refuse the attach on availability, which would mask the bounds
    // this file is about.
    Graph buildHostedChainGraph(const std::vector<Shape> &maskShapes, bool blocked) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {kRunN, kRunC, kRunH, kRunW};
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs   = {x};

        std::vector<TensorId> maskVals;
        for (size_t k = 0; k < maskShapes.size(); ++k)
        {
            TensorDesc mi;
            mi.name    = "m" + std::to_string(k);
            mi.shape   = maskShapes[k];
            mi.isInput = true;
            TensorId m = g.addTensor(mi);
            g.inputs.push_back(m);

            TensorId mm = g.addTensor(namedDesc("mm" + std::to_string(k)));
            Node     mact;
            mact.type    = OpType::Unary;
            mact.name    = "mask_producer_" + std::to_string(k);
            mact.subOp   = (int) UnaryType::Abs;
            mact.inputs  = {m};
            mact.outputs = {mm};
            g.nodes.push_back(mact);
            maskVals.push_back(mm);
        }

        TensorId t = g.addTensor(namedDesc("t"));
        if (blocked)
        {
            // A 1x1 stride-1 MaxPool: shape-preserving and weight-free, and LayoutClass::Nc4, so
            // its epilogue is the blocked applier -- the one that carries a class arm each.
            Node mp;
            mp.type    = OpType::MaxPool;
            mp.name    = "producer";
            mp.inputs  = {x};
            mp.outputs = {t};
            Attr kernel, strides, pads;
            kernel.kind                 = Attr::Ints;
            kernel.ints                 = {1, 1};
            strides.kind                = Attr::Ints;
            strides.ints                = {1, 1};
            pads.kind                   = Attr::Ints;
            pads.ints                   = {0, 0, 0, 0};
            mp.attr.map["kernel_shape"] = kernel;
            mp.attr.map["strides"]      = strides;
            mp.attr.map["pads"]         = pads;
            g.nodes.push_back(mp);
        } else
        {
            Node tr;
            tr.type    = OpType::Transpose;
            tr.name    = "producer";
            tr.inputs  = {x};
            tr.outputs = {t};
            Attr perm;
            perm.kind           = Attr::Ints;
            perm.ints           = {0, 1, 3, 2};
            tr.attr.map["perm"] = perm;
            g.nodes.push_back(tr);
        }

        TensorId cur = t;
        for (size_t k = 0; k < maskVals.size(); ++k)
        {
            TensorId y = g.addTensor(namedDesc("y" + std::to_string(k)));
            Node     bin;
            bin.type    = OpType::Binary;
            bin.name    = "mask_step_" + std::to_string(k);
            bin.subOp   = (int) (k % 2 == 0 ? BinaryType::Mul : BinaryType::Add);
            bin.inputs  = {cur, maskVals[k]};
            bin.outputs = {y};
            g.nodes.push_back(bin);
            cur = y;
        }
        g.outputs = {cur};
        return g;
    }

    // Where the planned unit landed: on the producer's store, or on its own node.
    struct UnitPlacement {
        bool             attachedToProducer = false;
        bool             standalone         = false;
        std::vector<int> bcastClasses; // every step's class, in step order
    };

    UnitPlacement placeUnit(const std::vector<Shape> &maskShapes, bool blocked) {
        Graph g = buildHostedChainGraph(maskShapes, blocked);
        inferShapes(g, 1);
        fusePointwiseChains(g, /*strictFuse*/ false);

        UnitPlacement p;
        for (const Node &nd: g.nodes)
        {
            if (!nd.attr.has("pw_steps"))
            {
                continue;
            }
            if (nd.type == OpType::FusedPointwise)
            {
                p.standalone = true;
            } else
            {
                p.attachedToProducer = true;
            }
            const auto &steps = nd.attr.getints("pw_steps");
            for (size_t s = 0; s + kPwStepBcastField < steps.size(); s += kPwStepInts)
            {
                p.bcastClasses.push_back((int) steps[s + kPwStepBcastField]);
            }
        }
        return p;
    }

    bool anyGeometric(const std::vector<int> &classes) {
        for (int c: classes)
        {
            if (pwBcastClassIsGeometric(c))
            {
                return true;
            }
        }
        return false;
    }

    // n per-channel masks: the direct class, one operand each.
    std::vector<Shape> channelMasks(int n) {
        return std::vector<Shape>((size_t) n, Shape {kRunN, kRunC, kBcast, kBcast});
    }
} // namespace

// --- what the epilogue must keep carrying ------------------------------------------------------

// A geometric-class operand (per-pixel here) rides its producer's store like any other. Refusing it
// would SPLIT the unit, and a split rounds the intermediate through fp16 storage -- the same defect
// that narrowing the operand budget caused, measured at 70 dB -> 15 dB against the CPU oracle on a
// production image-warp graph. Only the STANDALONE kernels specialize by class (their _dc twin),
// because that choice is made per node at record time and leaves the graph alone.
TEST(PwEpilogueBudget, GeometricClassOperandStillRidesABlockedProducersStore) {
    UnitPlacement p = placeUnit({Shape {kBcast, kBcast, kRunH, kRunW}}, /*blocked*/ true);
    ASSERT_FALSE(p.bcastClasses.empty()) << "the chain must have been fused into a unit at all";
    EXPECT_TRUE(anyGeometric(p.bcastClasses)) << "a [1,1,H,W] operand must classify as the per-pixel class";
    EXPECT_TRUE(p.attachedToProducer) << "splitting a unit to avoid a class arm changes its answer";
    EXPECT_FALSE(p.standalone);
}

TEST(PwEpilogueBudget, GeometricClassOperandStillRidesAFlatProducersStore) {
    UnitPlacement p = placeUnit({Shape {kBcast, kBcast, kRunW, kRunH}}, /*blocked*/ false);
    ASSERT_FALSE(p.bcastClasses.empty());
    EXPECT_TRUE(anyGeometric(p.bcastClasses));
    EXPECT_TRUE(p.attachedToProducer) << "the flat applier resolves every class through one strided walk";
    EXPECT_FALSE(p.standalone);
}

TEST(PwEpilogueBudget, DirectClassOperandRidesTheProducersStore) {
    UnitPlacement p = placeUnit(channelMasks(1), /*blocked*/ true);
    EXPECT_TRUE(p.attachedToProducer);
    EXPECT_FALSE(p.standalone);
    EXPECT_FALSE(anyGeometric(p.bcastClasses));
}

// --- the operand budget is NOT bounded, on purpose --------------------------------------------

// A unit using every operand slot a standalone unit gets must still ride its producer's store.
// This is the test that fails if someone narrows the epilogue budget again for speed: it is the
// cheap, local signal standing in for the image-warp graph that caught it the expensive way.
TEST(PwEpilogueBudget, AUnitUsingEveryOperandSlotStillRidesTheProducersStore) {
    for (bool blocked: {false, true})
    {
        UnitPlacement p = placeUnit(channelMasks(kPwMaxOperands), blocked);
        EXPECT_TRUE(p.attachedToProducer) << "the epilogue must carry the full operand budget: a unit split for want "
                                             "of a slot rounds its intermediate through storage (blocked="
                                          << blocked << ")";
        EXPECT_FALSE(p.standalone) << "blocked=" << blocked;
    }
}

// --- the bound is a placement decision, not a value change --------------------------------------

// Whichever side of either bound a unit falls, the values are the chain's. The CPU backend
// evaluates it the same way in both placements, so a gate that changed an answer shows up here.
TEST(PwEpilogueBudget, PlacementDoesNotChangeTheComputedValues) {
    const std::vector<std::vector<Shape>> cases = {
        channelMasks(1),
        {Shape {kBcast, kBcast, kRunW, kRunH}},
        channelMasks(kPwMaxOperands),
    };
    for (size_t c = 0; c < cases.size(); ++c)
    {
        const std::vector<Shape> &masks = cases[c];
        Graph                     g     = buildHostedChainGraph(masks, /*blocked*/ false);
        Config                    cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        ASSERT_NE(sess, nullptr) << "case " << c;

        std::vector<IOTensor> in(1 + masks.size());
        const size_t          n = (size_t) (kRunN * kRunC * kRunH * kRunW);
        std::vector<float>    src(n);
        for (size_t i = 0; i < n; ++i)
        {
            src[i] = 0.5f + (float) (i % 11) * 0.125f;
        }
        in[0].name  = "x";
        in[0].shape = {kRunN, kRunC, kRunH, kRunW};
        in[0].dtype = DType::Float32;
        in[0].data.resize(n * sizeof(float));
        std::memcpy(in[0].data.data(), src.data(), n * sizeof(float));
        for (size_t k = 0; k < masks.size(); ++k)
        {
            std::vector<float> mv((size_t) numElements(masks[k]));
            for (size_t i = 0; i < mv.size(); ++i)
            {
                mv[i] = 0.25f + 0.125f * (float) (i % 13);
            }
            in[1 + k].name  = "m" + std::to_string(k);
            in[1 + k].shape = masks[k];
            in[1 + k].dtype = DType::Float32;
            in[1 + k].data.resize(mv.size() * sizeof(float));
            std::memcpy(in[1 + k].data.data(), mv.data(), mv.size() * sizeof(float));
        }

        std::vector<IOTensor> out;
        ASSERT_EQ(sess->run(in, out), Status::Ok) << "case " << c;
        ASSERT_EQ(out.size(), 1u) << "case " << c;
        const float *got = (const float *) out[0].data.data();
        // Reference: the transpose, then the chain, both by hand.
        for (int64_t ch = 0; ch < kRunC; ++ch)
        {
            for (int64_t w = 0; w < kRunW; ++w)
            {
                for (int64_t h = 0; h < kRunH; ++h)
                {
                    float v = src[(size_t) ((ch * kRunH + h) * kRunW + w)];
                    for (size_t k = 0; k < masks.size(); ++k)
                    {
                        const Shape &ms = masks[k];
                        // The mask indexes the TRANSPOSED run [N,C,W,H] with size-1 axes broadcast.
                        int64_t       mi = 0, stride = 1;
                        const int64_t coord[4] = {0, ch, w, h};
                        for (int a = 3; a >= 0; --a)
                        {
                            if (ms[(size_t) a] != 1)
                            {
                                mi += coord[a] * stride;
                            }
                            stride *= ms[(size_t) a];
                        }
                        float mv = 0.25f + 0.125f * (float) ((size_t) mi % 13);
                        v        = (k % 2 == 0) ? v * mv : v + mv;
                    }
                    const size_t idx = (size_t) ((ch * kRunW + w) * kRunH + h);
                    EXPECT_NEAR(got[idx], v, 1e-4f) << "case " << c << " at c=" << ch << " w=" << w << " h=" << h;
                }
            }
        }
    }
}
