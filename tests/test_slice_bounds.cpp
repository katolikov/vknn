// ONNX Slice bound resolution: the rule shared by shape inference, the CPU kernel, and the Vulkan
// flat gather. A reverse (negative-step) slice used to resolve to a ZERO-length axis at all three
// sites, which allocates nothing and makes every consumer downstream compute over an empty tensor —
// with no CPU fallback to announce it, because the node itself is fully supported.
#include "core/slice_bounds.h"
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {
    // Reference walk: the indices ONNX says a Slice visits on one axis, derived independently of
    // resolveSliceAxis so a shared off-by-one cannot hide.
    std::vector<int64_t> onnxWalk(int64_t dim, int64_t rawStart, int64_t rawEnd, int64_t step) {
        std::vector<int64_t> visited;
        if (dim <= 0 || step == 0)
        {
            return visited;
        }
        auto clampFwd = [&](int64_t v) {
            v = v < 0 ? v + dim : v;
            return std::max<int64_t>(0, std::min(v, dim));
        };
        auto clampRev = [&](int64_t v, int64_t lo) {
            v = v < 0 ? v + dim : v;
            return std::max<int64_t>(lo, std::min(v, dim - 1));
        };
        if (step > 0)
        {
            for (int64_t i = clampFwd(rawStart); i < clampFwd(rawEnd); i += step)
            {
                visited.push_back(i);
            }
        } else
        {
            for (int64_t i = clampRev(rawStart, 0), stop = clampRev(rawEnd, -1); i > stop; i += step)
            {
                visited.push_back(i);
            }
        }
        return visited;
    }
} // namespace

TEST(SliceBounds, ForwardMatchesTheOnnxWalk) {
    const int64_t dim = 8;
    for (int64_t start: {(int64_t) 0, (int64_t) 3, (int64_t) 8, (int64_t) 99, (int64_t) -2, (int64_t) -99})
    {
        for (int64_t end: {(int64_t) 0, (int64_t) 5, (int64_t) 8, (int64_t) 9223372036854775807LL, (int64_t) -1})
        {
            for (int64_t step: {(int64_t) 1, (int64_t) 2, (int64_t) 3})
            {
                SliceAxisBounds b   = resolveSliceAxis(dim, start, end, step);
                auto            ref = onnxWalk(dim, start, end, step);
                EXPECT_EQ(b.count, (int64_t) ref.size()) << "start=" << start << " end=" << end << " step=" << step;
                if (!ref.empty())
                {
                    EXPECT_EQ(b.start, ref.front()) << "start=" << start << " end=" << end << " step=" << step;
                }
            }
        }
    }
}

TEST(SliceBounds, ReverseMatchesTheOnnxWalk) {
    const int64_t dim = 8;
    for (int64_t start: {(int64_t) 7, (int64_t) 5, (int64_t) 99, (int64_t) -1, (int64_t) -3})
    {
        for (int64_t end: {(int64_t) -1, (int64_t) 0, (int64_t) 2, (int64_t) -9223372036854775807LL})
        {
            for (int64_t step: {(int64_t) -1, (int64_t) -2, (int64_t) -3})
            {
                SliceAxisBounds b   = resolveSliceAxis(dim, start, end, step);
                auto            ref = onnxWalk(dim, start, end, step);
                EXPECT_EQ(b.count, (int64_t) ref.size()) << "start=" << start << " end=" << end << " step=" << step;
                if (!ref.empty())
                {
                    EXPECT_EQ(b.start, ref.front()) << "start=" << start << " end=" << end << " step=" << step;
                }
            }
        }
    }
}

TEST(SliceBounds, ReverseOfATwoChannelAxisKeepsBothChannels) {
    // The channel-swap idiom: reverse a 2-wide axis with step -1. This used to resolve to 0.
    SliceAxisBounds b = resolveSliceAxis(/*dim*/ 2, /*start*/ -1, /*end*/ -9223372036854775807LL, /*step*/ -1);
    EXPECT_EQ(b.count, 2);
    EXPECT_EQ(b.start, 1);
}

TEST(SliceBounds, ZeroStepIsEmptyRatherThanDivisionByZero) {
    EXPECT_EQ(resolveSliceAxis(8, 0, 8, 0).count, 0);
}

// --- inferShapes: a reverse slice must keep the axis, not collapse it ---
TEST(Passes, ReverseSliceKeepsTheAxisExtent) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 2, 4, 4};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs   = {x};
    auto vec   = [&](const char *nm, std::vector<int64_t> v) {
        TensorDesc t;
        t.name          = nm;
        t.shape         = {(int64_t) v.size()};
        t.dtype         = DType::Int64;
        t.isInitializer = true;
        TensorId   id   = g.addTensor(t);
        HostBuffer hb;
        hb.resizeElems(v.size(), DType::Int64);
        for (size_t i = 0; i < v.size(); ++i)
        {
            hb.i64()[i] = v[i];
        }
        g.initializers[id] = hb;
        return id;
    };
    // Reverse the channel axis: starts=[-1], ends=[INT64_MIN+1], axes=[1], steps=[-1].
    TensorId st = vec("starts", {-1}), en = vec("ends", {-9223372036854775807LL});
    TensorId ax = vec("axes", {1}), sp = vec("steps", {-1});
    TensorId y = g.addTensor({.name = "y"});
    Node     sl;
    sl.type    = OpType::Slice;
    sl.name    = "reverse_channels";
    sl.inputs  = {x, st, en, ax, sp};
    sl.outputs = {y};
    g.nodes    = {sl};
    g.outputs  = {y};
    inferShapes(g, 1);
    ASSERT_EQ(g.desc(y).shape.size(), 4u);
    EXPECT_EQ(g.desc(y).shape[1], 2) << "a reverse slice of a 2-channel axis must keep both channels, not resolve to 0";
    EXPECT_EQ(g.desc(y).shape, (Shape {1, 2, 4, 4}));
}

// --- the CPU kernel (the byte oracle) must actually produce the reversed values ---
TEST(Ops, ReverseSliceOnCpuReversesTheAxis) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 2, 1, 3};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs   = {x};
    auto vec   = [&](const char *nm, std::vector<int64_t> v) {
        TensorDesc t;
        t.name          = nm;
        t.shape         = {(int64_t) v.size()};
        t.dtype         = DType::Int64;
        t.isInitializer = true;
        TensorId   id   = g.addTensor(t);
        HostBuffer hb;
        hb.resizeElems(v.size(), DType::Int64);
        for (size_t i = 0; i < v.size(); ++i)
        {
            hb.i64()[i] = v[i];
        }
        g.initializers[id] = hb;
        return id;
    };
    TensorId st = vec("starts", {-1}), en = vec("ends", {-9223372036854775807LL});
    TensorId ax = vec("axes", {1}), sp = vec("steps", {-1});
    TensorId y = g.addTensor({.name = "y"});
    Node     sl;
    sl.type    = OpType::Slice;
    sl.name    = "reverse_channels";
    sl.inputs  = {x, st, en, ax, sp};
    sl.outputs = {y};
    g.nodes    = {sl};
    g.outputs  = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_NE(sess, nullptr);
    std::vector<IOTensor> inputs(1), outputs;
    inputs[0].name  = "x";
    inputs[0].shape = {1, 2, 1, 3};
    inputs[0].dtype = DType::Float32;
    inputs[0].data.resize(6 * sizeof(float));
    const float src[6] = {1, 2, 3, 4, 5, 6}; // channel 0 = {1,2,3}, channel 1 = {4,5,6}
    std::memcpy(inputs[0].data.data(), src, sizeof(src));
    ASSERT_EQ(sess->run(inputs, outputs), Status::Ok);
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs[0].shape, (Shape {1, 2, 1, 3}));
    const float *got     = outputs[0].f32();
    const float  want[6] = {4, 5, 6, 1, 2, 3}; // channels swapped
    for (int i = 0; i < 6; ++i)
    {
        EXPECT_FLOAT_EQ(got[i], want[i]) << "element " << i;
    }
}
