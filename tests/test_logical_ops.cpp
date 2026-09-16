// Logical ops Or, Xor and Not.
//
// LogicalOps: the CPU oracle (src/backend/cpu/ops/{or,xor,not}.cpp) through a CPU Session -- a value is
// true iff nonzero (NaN true, both zeros false), int64 operands are read exactly, NumPy broadcasting
// covers rows, columns, 0-extent axes and rank-0 / 1-element operands, results are canonical fp32.
// LogicalOpsFold: all-constant logical nodes (int64 and 1-byte initializers) fold away with exact values.
// CpuThreading: the partitioned sweeps are byte-identical for every thread count.
// LogicalGpuRules: the flat GPU kernels cannot run on the host, so their host-side rules
// (backend/vulkan/ops/logical_geometry.h) run for real, and the shader functions run as C++
// transcriptions kept textually parallel to shaders/logical_truth.glsl, or.comp, xor.comp and not.comp;
// both are checked against the CPU oracle.
#include "backend/cpu/logical_ops.h"
#include "backend/cpu/parallel.h"
#include "backend/vulkan/ops/logical_geometry.h"
#include "import/passes.h"
#include "vknn/dtype.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    constexpr float   kNaN         = std::numeric_limits<float>::quiet_NaN();
    constexpr float   kInfinity    = std::numeric_limits<float>::infinity();
    constexpr float   kSubnormal   = std::numeric_limits<float>::denorm_min();
    constexpr int64_t kInt64Min    = std::numeric_limits<int64_t>::min();
    constexpr int64_t kLowWordZero = int64_t(1) << 32; // nonzero, but its low 32 bits read as fp32 0.0
    constexpr int64_t kAbove2Pow40 = (int64_t(1) << 40) + 3;
    // Node label the host-side geometry rules prefix their errors with (the kernels pass "<Op> '<name>'").
    const std::string kGeometryTestLabel = "Or 'geometry_probe'";

    // ---------------------------------------------------------------------------------------------
    // Single-node graphs on the CPU backend.

    // One operand of a single-node logical graph: a runtime graph input or a constant initializer, with
    // fp32 or int64 values.
    struct OperandSpec {
        Shape                shape;
        std::vector<float>   floats;
        std::vector<int64_t> ints;
        bool                 isInt64    = false;
        bool                 isConstant = false;
    };

    OperandSpec floatInput(Shape shape, std::vector<float> values) {
        OperandSpec spec;
        spec.shape  = std::move(shape);
        spec.floats = std::move(values);
        return spec;
    }
    OperandSpec floatConstant(Shape shape, std::vector<float> values) {
        OperandSpec spec = floatInput(std::move(shape), std::move(values));
        spec.isConstant  = true;
        return spec;
    }
    OperandSpec int64Input(Shape shape, std::vector<int64_t> values) {
        OperandSpec spec;
        spec.shape   = std::move(shape);
        spec.ints    = std::move(values);
        spec.isInt64 = true;
        return spec;
    }
    OperandSpec int64Constant(Shape shape, std::vector<int64_t> values) {
        OperandSpec spec = int64Input(std::move(shape), std::move(values));
        spec.isConstant  = true;
        return spec;
    }

    struct NodeResult {
        bool                 created = false;
        Status               status  = Status::RuntimeError;
        Shape                shape;
        DType                dtype = DType::Float32;
        std::vector<float>   values;
        std::vector<uint8_t> bytes;
    };

    HostBuffer hostBufferOf(const OperandSpec &spec) {
        HostBuffer hb;
        if (spec.isInt64)
        {
            hb.resizeElems((int64_t) spec.ints.size(), DType::Int64);
            if (!spec.ints.empty())
            {
                std::memcpy(hb.i64(), spec.ints.data(), spec.ints.size() * sizeof(int64_t));
            }
        } else
        {
            hb.resizeElems((int64_t) spec.floats.size(), DType::Float32);
            if (!spec.floats.empty())
            {
                std::memcpy(hb.f32(), spec.floats.data(), spec.floats.size() * sizeof(float));
            }
        }
        return hb;
    }

    // Build `type` over `operands` (named in0, in1, ...), writing output "y", and run it on the CPU
    // backend with `threads` workers.
    NodeResult runLogicalNode(OpType type, const std::vector<OperandSpec> &operands, int threads = 1) {
        Graph                 g;
        std::vector<TensorId> inputs;
        std::vector<IOTensor> feeds;
        for (size_t k = 0; k < operands.size(); ++k)
        {
            const OperandSpec &spec = operands[k];
            TensorDesc         desc;
            desc.name          = "in" + std::to_string(k);
            desc.shape         = spec.shape;
            desc.dtype         = spec.isInt64 ? DType::Int64 : DType::Float32;
            desc.isInput       = !spec.isConstant;
            desc.isInitializer = spec.isConstant;
            TensorId id        = g.addTensor(desc);
            inputs.push_back(id);
            if (spec.isConstant)
            {
                g.initializers[id] = hostBufferOf(spec);
                continue;
            }
            g.inputs.push_back(id);
            IOTensor feed;
            feed.name     = desc.name;
            feed.shape    = spec.shape;
            feed.dtype    = desc.dtype;
            HostBuffer hb = hostBufferOf(spec);
            feed.data.assign(hb.bytes.data(), hb.bytes.data() + hb.bytes.size());
            feeds.push_back(std::move(feed));
        }
        TensorDesc out;
        out.name     = "y";
        out.isOutput = true;
        TensorId y   = g.addTensor(out);
        Node     node;
        node.type    = type;
        node.name    = "logical";
        node.inputs  = inputs;
        node.outputs = {y};
        g.nodes.push_back(node);
        g.outputs = {y};

        NodeResult result;
        Config     cfg;
        cfg.backend    = BackendKind::Cpu;
        cfg.cpuThreads = threads;
        auto sess      = Session::create(std::move(g), cfg);
        result.created = (bool) sess;
        if (!sess)
        {
            return result;
        }
        std::vector<IOTensor> outs;
        result.status = sess->run(feeds, outs);
        if (result.status != Status::Ok || outs.empty())
        {
            return result;
        }
        result.shape = outs[0].shape;
        result.dtype = outs[0].dtype;
        result.bytes = outs[0].data;
        if (outs[0].dtype == DType::Float32)
        {
            result.values.assign(outs[0].data.size() / sizeof(float), 0.0f);
            if (!result.values.empty())
            {
                std::memcpy(result.values.data(), outs[0].data.data(), result.values.size() * sizeof(float));
            }
        }
        return result;
    }

    // Run and require success, returning the canonical fp32 result.
    NodeResult runOk(OpType type, const std::vector<OperandSpec> &operands, int threads = 1) {
        NodeResult result = runLogicalNode(type, operands, threads);
        EXPECT_TRUE(result.created);
        EXPECT_EQ(result.status, Status::Ok);
        EXPECT_EQ(result.dtype, DType::Float32) << "logical results are canonical fp32";
        return result;
    }

    // ---------------------------------------------------------------------------------------------
    // Reference semantics, written independently of the engine.

    bool referenceTruth(float value) {
        return !(value == 0.0f); // NaN compares unequal to 0, so it is true
    }

    // NumPy broadcast of two shapes: extents align from the right, 1 stretches, 0 wins.
    Shape referenceBroadcastShape(const Shape &a, const Shape &b) {
        const size_t rank = std::max(a.size(), b.size());
        Shape        out(rank, 1);
        for (size_t axis = 0; axis < rank; ++axis)
        {
            const int64_t aExtent = axis + a.size() < rank ? 1 : a[axis + a.size() - rank];
            const int64_t bExtent = axis + b.size() < rank ? 1 : b[axis + b.size() - rank];
            out[axis]             = (aExtent == 0 || bExtent == 0) ? 0 : std::max(aExtent, bExtent);
        }
        return out;
    }

    int64_t referenceCount(const Shape &shape) {
        int64_t count = 1;
        for (int64_t extent: shape)
        {
            count *= extent;
        }
        return count; // rank 0 -> 1
    }

    // Deterministic operand values: a mix of both zeros, NaN, infinities, subnormals and ordinary values,
    // so every element exercises a different truth path.
    std::vector<float> mixedValues(size_t count, uint32_t seed) {
        static const float kPalette[] = {0.0f, -0.0f, 1.0f, 2.0f, -3.0f, kNaN, kInfinity, -kInfinity, kSubnormal, -kSubnormal, 0.5f, 1e-10f, 70000.0f};
        std::mt19937       generator(seed);
        std::vector<float> values(count);
        for (float &value: values)
        {
            value = kPalette[generator() % (sizeof(kPalette) / sizeof(kPalette[0]))];
        }
        return values;
    }

    // Values whose truth an fp16 activation store keeps (the rounding of a magnitude below the smallest
    // fp16 subnormal to zero is a property of fp16 storage itself, outside the kernels).
    std::vector<float> fp16SafeValues(size_t count, uint32_t seed) {
        static const float kPalette[] = {0.0f, -0.0f, 1.0f, 2.0f, -3.0f, kNaN, kInfinity, -kInfinity, 0.5f, 70000.0f};
        std::mt19937       generator(seed);
        std::vector<float> values(count);
        for (float &value: values)
        {
            value = kPalette[generator() % (sizeof(kPalette) / sizeof(kPalette[0]))];
        }
        return values;
    }

    // ---------------------------------------------------------------------------------------------
    // C++ transcriptions of the logical shaders. Each mirrors the named GLSL function line for line;
    // GLSL `uint` is uint32_t, `int` is int32_t, `float` is float, floatBitsToUint is a bit copy.

    // Transcription of shaders/logical_truth.glsl.
    constexpr uint32_t kLogicalSignBitDrop = 1u;
    constexpr float    kLogicalTrue        = 1.0f;
    constexpr float    kLogicalFalse       = 0.0f;

    bool logicalIsTrue(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof bits); // floatBitsToUint(value)
        return (uint32_t) (bits << kLogicalSignBitDrop) != 0u;
    }

    // Push constants of or.comp / xor.comp and of not.comp.
    struct BinaryPushConstant {
        int rank, total;
    };
    struct NotPushConstant {
        int total;
    };

    // Transcription of the constants of or.comp / xor.comp.
    constexpr int kGeomAStrideArray = 1;
    constexpr int kGeomBStrideArray = 2;

    enum class LogicalCombine { Or, Xor };

    // Transcription of or.comp / xor.comp main() for the invocation whose recovered id is `gid`. `a`, `b`
    // are the bound operand buffers widened to fp32 (float(a[ai])), `g` the geometry SSBO, `d` the output.
    // .at() turns an out-of-range shader read or write into a test failure.
    void logicalBinaryMain(uint32_t gid, const BinaryPushConstant &pc, const std::vector<int32_t> &g, const std::vector<float> &a, const std::vector<float> &b, LogicalCombine combine, std::vector<float> &d) {
        if (gid >= (uint32_t) pc.total)
        {
            return;
        }
        int rem = (int) gid, ai = 0, bi = 0;
        for (int k = pc.rank - 1; k >= 0; --k)
        {
            int c = rem % g.at(k);
            rem /= g.at(k);
            ai += c * g.at(kGeomAStrideArray * pc.rank + k);
            bi += c * g.at(kGeomBStrideArray * pc.rank + k);
        }
        bool result = combine == LogicalCombine::Or ? (logicalIsTrue(a.at(ai)) || logicalIsTrue(b.at(bi))) : (logicalIsTrue(a.at(ai)) != logicalIsTrue(b.at(bi)));
        d.at(gid) = result ? kLogicalTrue : kLogicalFalse;
    }

    // Transcription of not.comp main().
    void logicalNotMain(uint32_t i, const NotPushConstant &pc, const std::vector<float> &s, std::vector<float> &d) {
        if (i >= (uint32_t) pc.total)
        {
            return;
        }
        d.at(i) = logicalIsTrue(s.at(i)) ? kLogicalFalse : kLogicalTrue;
    }

    // The flat kernels' local_size_x (flat::kFlatLocalSize).
    constexpr uint32_t kFlatLocalSize = 256;

    // Every recovered invocation id of a 1-D dispatch of `total` elements, reproducing
    // ComputePipeline::dispatch's spill of an x group count above `maxGroupsX` into y and the shaders'
    // two-term recovery `gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x *
    // gl_WorkGroupSize.x`.
    std::vector<uint32_t> dispatchInvocationIds(int64_t total, uint32_t maxGroupsX) {
        uint32_t groupsX = (uint32_t) ((total + kFlatLocalSize - 1) / kFlatLocalSize);
        uint32_t groupsY = 1;
        if (groupsX > maxGroupsX)
        {
            groupsY = (groupsX + maxGroupsX - 1) / maxGroupsX;
            groupsX = (groupsX + groupsY - 1) / groupsY;
        }
        std::vector<uint32_t> ids;
        for (uint32_t y = 0; y < groupsY; ++y)
        {
            for (uint32_t x = 0; x < groupsX * kFlatLocalSize; ++x)
            {
                ids.push_back(x + y * groupsX * kFlatLocalSize);
            }
        }
        return ids;
    }

    // flat::uploadFlatGeom's packing: arrays back to back, one placeholder int when all are empty.
    std::vector<int32_t> packGeometry(const logical::FlatBroadcastGeometry &geometry) {
        std::vector<int32_t> packed;
        for (const std::vector<int32_t> *array: {&geometry.outDim, &geometry.aStride, &geometry.bStride})
        {
            packed.insert(packed.end(), array->begin(), array->end());
        }
        if (packed.empty())
        {
            packed.push_back(0);
        }
        return packed;
    }

    // An fp32 buffer as the fp16 variant reads it: saturating fp16 store (the boundary pack and upload()),
    // then float(s[i]).
    std::vector<float> throughFp16Storage(const std::vector<float> &values) {
        std::vector<fp16_t> halves(values.size());
        floatToHalfSatBulk(values.data(), halves.data(), (int64_t) values.size());
        std::vector<float> widened(values.size());
        for (size_t k = 0; k < values.size(); ++k)
        {
            widened[k] = halfToFloat(halves[k]);
        }
        return widened;
    }

    // ---------------------------------------------------------------------------------------------
    // Thread-count byte invariance (the tests/test_cpu_threading.cpp protocol).

    const std::vector<int> kThreadCounts {2, 3, 5, 8};

    std::vector<uint8_t> runGraphBytes(Graph g, const Shape &xshape, const std::vector<float> &xdata, int threads) {
        Config cfg;
        cfg.backend    = BackendKind::Cpu;
        cfg.cpuThreads = threads;
        auto sess      = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        IOTensor in;
        in.name  = "x";
        in.shape = xshape;
        in.dtype = DType::Float32;
        in.data.resize(xdata.size() * sizeof(float));
        std::memcpy(in.data.data(), xdata.data(), in.data.size());
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        EXPECT_FALSE(outs.empty());
        return outs.empty() ? std::vector<uint8_t> {} : outs[0].data;
    }

    // Byte-identical output for every thread count, plus proof the partition engaged (a loop too small
    // for kMinChunkOps runs inline and would pass vacuously).
    void expectByteIdenticalAcrossThreads(const std::function<Graph()> &build, const Shape &xshape, const std::vector<float> &xdata) {
        const int64_t              dispatchesBefore = cpu::detail::poolDispatches();
        const std::vector<uint8_t> reference        = runGraphBytes(build(), xshape, xdata, 1);
        ASSERT_FALSE(reference.empty());
        for (int threads: kThreadCounts)
        {
            const std::vector<uint8_t> got = runGraphBytes(build(), xshape, xdata, threads);
            ASSERT_EQ(got.size(), reference.size()) << "threads=" << threads;
            EXPECT_EQ(0, std::memcmp(got.data(), reference.data(), reference.size())) << "threads=" << threads;
        }
        EXPECT_GT(cpu::detail::poolDispatches(), dispatchesBefore) << "shape too small to partition: the byte comparison is vacuous";
    }

    // A float input "x" with roughly half its elements exactly zero, so chunk boundaries land on both
    // truth values.
    std::vector<float> sparseValues(size_t count, uint32_t seed) {
        std::vector<float> values(count);
        uint32_t           state = seed;
        for (float &value: values)
        {
            state = state * 1664525u + 1013904223u;
            value = (state >> 31) ? (float) (int32_t) (state >> 8) : 0.0f;
        }
        return values;
    }

    Graph logicalGraphOverX(OpType type, const Shape &xshape, const Shape &maskShape, const std::vector<float> &mask) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = xshape;
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs.push_back(x);
        std::vector<TensorId> inputs {x};
        if (type != OpType::Not)
        {
            TensorDesc md;
            md.name           = "mask";
            md.shape          = maskShape;
            md.isInitializer  = true;
            TensorId    m     = g.addTensor(md);
            OperandSpec spec  = floatConstant(maskShape, mask);
            g.initializers[m] = hostBufferOf(spec);
            inputs.push_back(m);
        }
        TensorDesc yo;
        yo.name     = "y";
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     node;
        node.type    = type;
        node.name    = "logical";
        node.inputs  = inputs;
        node.outputs = {y};
        g.nodes.push_back(node);
        g.outputs = {y};
        return g;
    }

} // namespace

// =================================================================================================
// CPU oracle

TEST(LogicalOps, OrXorSameShape) {
    const std::vector<OperandSpec> operands {floatInput({2, 3}, {1, 0, 1, 0, 0, 2.5f}), floatConstant({2, 3}, {1, 1, 0, 0, -4, 0})};
    NodeResult                     orResult  = runOk(OpType::Or, operands);
    NodeResult                     xorResult = runOk(OpType::Xor, operands);
    EXPECT_EQ(orResult.shape, (Shape {2, 3}));
    EXPECT_EQ(orResult.values, (std::vector<float> {1, 1, 1, 0, 1, 1}));
    EXPECT_EQ(xorResult.shape, (Shape {2, 3}));
    EXPECT_EQ(xorResult.values, (std::vector<float> {0, 1, 1, 0, 1, 1}));
}

// A [2,3] mask against a [3] row, a [2,1] column, and two runtime operands broadcasting against each other.
TEST(LogicalOps, BroadcastRowAndColumn) {
    const OperandSpec x = floatInput({2, 3}, {1, 0, 1, 0, 1, 0});

    NodeResult orRow = runOk(OpType::Or, {x, floatConstant({3}, {0, 0, 1})});
    EXPECT_EQ(orRow.shape, (Shape {2, 3}));
    EXPECT_EQ(orRow.values, (std::vector<float> {1, 0, 1, 0, 1, 1}));

    NodeResult xorColumn = runOk(OpType::Xor, {x, floatConstant({2, 1}, {1, 0})});
    EXPECT_EQ(xorColumn.shape, (Shape {2, 3}));
    EXPECT_EQ(xorColumn.values, (std::vector<float> {0, 1, 0, 0, 1, 0}));

    NodeResult outer = runOk(OpType::Xor, {floatInput({2, 1}, {0, 7}), floatInput({1, 3}, {0, 1, -1})});
    EXPECT_EQ(outer.shape, (Shape {2, 3}));
    EXPECT_EQ(outer.values, (std::vector<float> {0, 1, 1, 1, 0, 0}));
}

// A 0 extent broadcasts to 0 (NumPy), never to 1: the result is empty.
TEST(LogicalOps, ZeroExtentBroadcastIsEmpty) {
    NodeResult orEmpty = runOk(OpType::Or, {floatInput({1, 3}, {1, 0, 1}), floatConstant({0, 1}, {})});
    EXPECT_EQ(orEmpty.shape, (Shape {0, 3}));
    EXPECT_TRUE(orEmpty.values.empty());

    NodeResult xorEmpty = runOk(OpType::Xor, {floatInput({2, 1}, {1, 0}), floatConstant({2, 0}, {})});
    EXPECT_EQ(xorEmpty.shape, (Shape {2, 0}));
    EXPECT_TRUE(xorEmpty.values.empty());
}

// A 1-element operand and a rank-0 operand broadcast over every output element.
TEST(LogicalOps, OneElementAndRankZeroOperands) {
    const OperandSpec x = floatInput({2, 3}, {1, 0, 2, 0, -0.0f, kNaN});

    NodeResult orNaN = runOk(OpType::Or, {x, floatConstant({1}, {kNaN})});
    EXPECT_EQ(orNaN.shape, (Shape {2, 3}));
    EXPECT_EQ(orNaN.values, (std::vector<float> {1, 1, 1, 1, 1, 1}));

    NodeResult xorScalarTrue = runOk(OpType::Xor, {x, floatConstant({}, {2.0f})});
    EXPECT_EQ(xorScalarTrue.shape, (Shape {2, 3}));
    EXPECT_EQ(xorScalarTrue.values, (std::vector<float> {0, 1, 0, 1, 1, 0}));

    NodeResult orScalarFalse = runOk(OpType::Or, {x, floatConstant({}, {-0.0f})});
    EXPECT_EQ(orScalarFalse.values, (std::vector<float> {1, 0, 1, 0, 0, 1}));

    NodeResult xorRankRaise = runOk(OpType::Xor, {x, floatConstant({1, 1, 1}, {1.0f})});
    EXPECT_EQ(xorRankRaise.shape, (Shape {1, 2, 3}));
    EXPECT_EQ(xorRankRaise.values, (std::vector<float> {0, 1, 0, 1, 1, 0}));
}

// Nonzero is true: 2.0, -3.0, NaN, infinities and subnormals are true; +0.0 and -0.0 are false.
TEST(LogicalOps, NonzeroIsTrue) {
    const std::vector<float> values {2.0f, -3.0f, kNaN, 0.0f, -0.0f, kInfinity, -kInfinity, kSubnormal};
    const std::vector<float> negated {0, 0, 0, 1, 1, 0, 0, 0};
    const size_t             count = values.size();

    NodeResult notResult = runOk(OpType::Not, {floatInput({(int64_t) count}, values)});
    EXPECT_EQ(notResult.shape, (Shape {(int64_t) count}));
    EXPECT_EQ(notResult.values, negated);

    const OperandSpec zeros   = floatConstant({(int64_t) count}, std::vector<float>(count, 0.0f));
    const OperandSpec ones    = floatConstant({(int64_t) count}, std::vector<float>(count, 1.0f));
    NodeResult        orZero  = runOk(OpType::Or, {floatInput({(int64_t) count}, values), zeros});
    NodeResult        xorOne  = runOk(OpType::Xor, {floatInput({(int64_t) count}, values), ones});
    NodeResult        xorZero = runOk(OpType::Xor, {zeros, floatInput({(int64_t) count}, values)});
    for (size_t k = 0; k < count; ++k)
    {
        const float truth = negated[k] == 0.0f ? 1.0f : 0.0f;
        EXPECT_EQ(orZero.values[k], truth) << "k=" << k;
        EXPECT_EQ(xorOne.values[k], negated[k]) << "k=" << k;
        EXPECT_EQ(xorZero.values[k], truth) << "k=" << k;
    }
}

// An int64 runtime input is tested exactly: 2^32 has an all-zero low word, which an fp32 reading of the
// int64 bytes would take for 0.0 (and the element offsets would be misaligned as well).
TEST(LogicalOps, Int64RuntimeInputsReadExactly) {
    const std::vector<int64_t> ints {0, 1, -1, kLowWordZero, kAbove2Pow40, kInt64Min};

    NodeResult notResult = runOk(OpType::Not, {int64Input({6}, ints)});
    EXPECT_EQ(notResult.shape, (Shape {6}));
    EXPECT_EQ(notResult.values, (std::vector<float> {1, 0, 0, 0, 0, 0}));

    NodeResult orMixed = runOk(OpType::Or, {int64Input({6}, ints), floatConstant({6}, {0, 0, 0, 0, 0, 0})});
    EXPECT_EQ(orMixed.values, (std::vector<float> {0, 1, 1, 1, 1, 1}));

    NodeResult orBroadcast = runOk(OpType::Or, {int64Input({2, 1}, {0, kLowWordZero}), int64Constant({3}, {0, 0, kAbove2Pow40})});
    EXPECT_EQ(orBroadcast.shape, (Shape {2, 3}));
    EXPECT_EQ(orBroadcast.values, (std::vector<float> {0, 0, 1, 1, 1, 1}));

    NodeResult xorBoth = runOk(OpType::Xor, {int64Input({4}, {0, kLowWordZero, 0, -1}), int64Input({4}, {0, 0, kAbove2Pow40, kInt64Min})});
    EXPECT_EQ(xorBoth.values, (std::vector<float> {0, 1, 1, 0}));
}

// A logical node missing an operand fails the run (a named InvalidArgument from the kernel), never a read
// past node.inputs.
TEST(LogicalOps, MissingOperandFailsRun) {
    NodeResult single = runLogicalNode(OpType::Or, {floatInput({3}, {1, 0, 1})});
    ASSERT_TRUE(single.created);
    EXPECT_NE(single.status, Status::Ok);
    NodeResult none = runLogicalNode(OpType::Not, {});
    ASSERT_TRUE(none.created);
    EXPECT_NE(none.status, Status::Ok);
}

// Operand shapes that do not broadcast ([2,3] against [2], [4] against [3]) fail the run with the kernel's
// named InvalidArgument instead of a stride walk past the shorter operand.
TEST(LogicalOps, IncompatibleBroadcastFailsRun) {
    NodeResult trailing = runLogicalNode(OpType::Or, {floatInput({2, 3}, {1, 0, 1, 0, 1, 0}), floatInput({2}, {1, 0})});
    ASSERT_TRUE(trailing.created);
    EXPECT_NE(trailing.status, Status::Ok);
    NodeResult sameRank = runLogicalNode(OpType::Xor, {floatInput({4}, {1, 0, 1, 0}), floatConstant({3}, {0, 1, 0})});
    ASSERT_TRUE(sameRank.created);
    EXPECT_NE(sameRank.status, Status::Ok);
}

// Less -> Not -> Or -> Where through a CPU Session: the logical results feed a select.
TEST(LogicalOps, LessNotOrWhereChain) {
    constexpr float          threshold = 0.5f;
    constexpr float          fill      = -7.0f;
    const Shape              xshape {2, 3};
    const std::vector<float> xd {0.25f, 0.75f, -1.0f, 3.0f, 0.5f, kNaN};
    const std::vector<float> rowMask {0, 1, 0};
    Graph                    g;
    auto                     addConstant = [&](const std::string &name, const Shape &shape, const std::vector<float> &values) {
        TensorDesc d;
        d.name             = name;
        d.shape            = shape;
        d.isInitializer    = true;
        TensorId id        = g.addTensor(d);
        g.initializers[id] = hostBufferOf(floatConstant(shape, values));
        return id;
    };
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = xshape;
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorId   limit    = addConstant("limit", {}, {threshold});
    TensorId   mask     = addConstant("row_mask", {3}, rowMask);
    TensorId   filler   = addConstant("fill", {1}, {fill});
    TensorId   below    = g.addTensor({"below"});
    TensorId   notBelow = g.addTensor({"not_below"});
    TensorId   keep     = g.addTensor({"keep"});
    TensorDesc yo;
    yo.name          = "y";
    yo.isOutput      = true;
    TensorId y       = g.addTensor(yo);
    auto     addNode = [&](OpType type, const std::string &name, std::vector<TensorId> inputs, TensorId output) {
        Node node;
        node.type    = type;
        node.name    = name;
        node.inputs  = std::move(inputs);
        node.outputs = {output};
        g.nodes.push_back(node);
    };
    addNode(OpType::Less, "less", {x, limit}, below);
    addNode(OpType::Not, "not", {below}, notBelow);
    addNode(OpType::Or, "or", {notBelow, mask}, keep);
    addNode(OpType::Where, "where", {keep, x, filler}, y);
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);
    IOTensor in;
    in.name  = "x";
    in.shape = xshape;
    in.data.resize(xd.size() * sizeof(float));
    std::memcpy(in.data.data(), xd.data(), in.data.size());
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_EQ(outs.size(), 1u);
    ASSERT_EQ(outs[0].shape, xshape);
    for (size_t k = 0; k < xd.size(); ++k)
    {
        const bool  isBelow = xd[k] < threshold; // NaN is not below
        const bool  keepIt  = !isBelow || rowMask[k % rowMask.size()] != 0.0f;
        const float expect  = keepIt ? xd[k] : fill;
        const float got     = outs[0].f32()[k];
        if (std::isnan(expect))
        {
            EXPECT_TRUE(std::isnan(got)) << "k=" << k;
        } else
        {
            EXPECT_EQ(got, expect) << "k=" << k;
        }
    }
}

// =================================================================================================
// constFold

namespace {
    TensorId addFoldInt64(Graph &g, const std::string &name, const Shape &shape, const std::vector<int64_t> &values) {
        TensorDesc d;
        d.name             = name;
        d.shape            = shape;
        d.dtype            = DType::Int64;
        d.isInitializer    = true;
        TensorId id        = g.addTensor(d);
        g.initializers[id] = hostBufferOf(int64Constant(shape, values));
        return id;
    }
    TensorId addFoldNode(Graph &g, OpType type, const std::string &name, std::vector<TensorId> inputs) {
        TensorId output = g.addTensor({name + "_out"});
        Node     node;
        node.type    = type;
        node.name    = name;
        node.inputs  = std::move(inputs);
        node.outputs = {output};
        g.nodes.push_back(node);
        g.outputs.push_back(output);
        return output;
    }
    void expectFoldedFloats(const Graph &g, TensorId id, const Shape &shape, const std::vector<float> &values) {
        ASSERT_TRUE(g.isInitializer(id)) << g.tensors[id].name;
        EXPECT_EQ(g.desc(id).dtype, DType::Float32) << g.tensors[id].name;
        EXPECT_EQ(g.desc(id).shape, shape) << g.tensors[id].name;
        const HostBuffer &hb = g.initializers.at(id);
        ASSERT_EQ(hb.bytes.size(), values.size() * sizeof(float)) << g.tensors[id].name;
        for (size_t k = 0; k < values.size(); ++k)
        {
            EXPECT_EQ(hb.f32()[k], values[k]) << g.tensors[id].name << " k=" << k;
        }
    }
} // namespace

// All-constant Or/Xor/Not/And over int64 initializers fold away with exact truth: 2^32 (zero low word)
// and values above 2^24 are true, and the broadcast [4] x [2,1] -> [2,4] is kept.
TEST(LogicalOpsFold, AllConstantInt64LogicalNodesFoldAway) {
    Graph    g;
    TensorId a      = addFoldInt64(g, "a", {4}, {0, kLowWordZero, -1, kAbove2Pow40});
    TensorId b      = addFoldInt64(g, "b", {2, 1}, {0, kLowWordZero + 1});
    TensorId orOut  = addFoldNode(g, OpType::Or, "or", {a, b});
    TensorId xorOut = addFoldNode(g, OpType::Xor, "xor", {a, b});
    TensorId andOut = addFoldNode(g, OpType::And, "and", {a, b});
    TensorId notOut = addFoldNode(g, OpType::Not, "not", {a});

    inferShapes(g, 1);
    constFold(g);
    EXPECT_TRUE(g.nodes.empty()) << "all-constant logical nodes must fold away";
    expectFoldedFloats(g, orOut, {2, 4}, {0, 1, 1, 1, 1, 1, 1, 1});
    expectFoldedFloats(g, xorOut, {2, 4}, {0, 1, 1, 1, 1, 0, 0, 0});
    expectFoldedFloats(g, andOut, {2, 4}, {0, 0, 0, 0, 0, 1, 1, 1});
    expectFoldedFloats(g, notOut, {4}, {1, 0, 0, 0});
}

// A rank-0 int64 constant folds to a rank-0 fp32 result.
TEST(LogicalOpsFold, RankZeroInt64NotStaysRankZero) {
    Graph    g;
    TensorId scalar = addFoldInt64(g, "scalar", {}, {kLowWordZero});
    TensorId zero   = addFoldInt64(g, "zero", {}, {0});
    TensorId notOut = addFoldNode(g, OpType::Not, "not", {scalar});
    TensorId xorOut = addFoldNode(g, OpType::Xor, "xor", {scalar, zero});
    inferShapes(g, 1);
    constFold(g);
    EXPECT_TRUE(g.nodes.empty());
    expectFoldedFloats(g, notOut, {}, {0});
    expectFoldedFloats(g, xorOut, {}, {1});
}

// 1-byte UINT8/INT8 initializers reach the fold kernels widened to their integer values.
TEST(LogicalOpsFold, ByteInitializersFoldByIntegerValue) {
    Graph      g;
    TensorDesc u8;
    u8.name                          = "u8";
    u8.shape                         = {3};
    u8.dtype                         = DType::UInt8;
    u8.isInitializer                 = true;
    TensorId unsignedId              = g.addTensor(u8);
    g.initializers[unsignedId].bytes = std::vector<uint8_t> {0, 1, 255};
    TensorDesc i8;
    i8.name                        = "i8";
    i8.shape                       = {3};
    i8.dtype                       = DType::Int8;
    i8.isInitializer               = true;
    TensorId signedId              = g.addTensor(i8);
    g.initializers[signedId].bytes = std::vector<uint8_t> {0xFF, 0, 0x80};
    TensorId notOut                = addFoldNode(g, OpType::Not, "not", {unsignedId});
    TensorId xorOut                = addFoldNode(g, OpType::Xor, "xor", {unsignedId, signedId});
    inferShapes(g, 1);
    constFold(g);
    EXPECT_TRUE(g.nodes.empty());
    expectFoldedFloats(g, notOut, {3}, {1, 0, 0});
    expectFoldedFloats(g, xorOut, {3}, {1, 1, 0});
}

// =================================================================================================
// Thread-count byte invariance (shapes past cpu::kMinChunkOps; the extent divides by none of 2/3/5/8)

TEST(CpuThreading, LogicalOrBitExact) {
    const Shape xs {7, 13, 1447};
    const auto  xd    = sparseValues(7 * 13 * 1447, 21);
    const auto  build = [&] {
        return logicalGraphOverX(OpType::Or, xs, {13, 1}, {0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0});
    };
    expectByteIdenticalAcrossThreads(build, xs, xd);
}

TEST(CpuThreading, LogicalXorBitExact) {
    const Shape xs {7, 13, 1447};
    const auto  xd    = sparseValues(7 * 13 * 1447, 22);
    const auto  build = [&] {
        return logicalGraphOverX(OpType::Xor, xs, {1447}, sparseValues(1447, 23));
    };
    expectByteIdenticalAcrossThreads(build, xs, xd);
}

TEST(CpuThreading, LogicalNotBitExact) {
    const Shape xs {7, 13, 1447};
    const auto  xd    = sparseValues(7 * 13 * 1447, 24);
    const auto  build = [&] {
        return logicalGraphOverX(OpType::Not, xs, {}, {});
    };
    expectByteIdenticalAcrossThreads(build, xs, xd);
}

// =================================================================================================
// GPU kernel rules and shader transcriptions against the CPU oracle

// The bit test of logical_truth.glsl agrees with the oracle's reading on fp32 bit patterns: the special
// values, every pattern around both zeros, and a pseudo-random sweep of the whole 32-bit space.
TEST(LogicalGpuRules, TruthBitTestMatchesOracleOnFp32Patterns) {
    auto oracleTruth = [](float value) {
        RtTensor tensor;
        tensor.shape = {1};
        tensor.dtype = DType::Float32;
        tensor.host.resizeElems(1, DType::Float32);
        tensor.host.f32()[0] = value;
        return cpu::LogicalOperand(tensor).isTrue(0);
    };
    auto check = [&](uint32_t bits) {
        float value;
        std::memcpy(&value, &bits, sizeof value);
        ASSERT_EQ(logicalIsTrue(value), oracleTruth(value)) << std::hex << "bits=0x" << bits;
        ASSERT_EQ(logicalIsTrue(value), referenceTruth(value)) << std::hex << "bits=0x" << bits;
    };
    constexpr uint32_t kSignBit        = 0x80000000u;
    constexpr uint32_t kNeighborhood   = 4096;
    constexpr uint32_t kRandomPatterns = 1u << 20;
    for (uint32_t offset = 0; offset < kNeighborhood; ++offset)
    {
        check(offset);
        check(kSignBit | offset);
        check(kSignBit - offset);
        check(0u - offset);
    }
    std::mt19937 generator(0x10a1);
    for (uint32_t k = 0; k < kRandomPatterns; ++k)
    {
        check((uint32_t) generator());
    }
    EXPECT_FALSE(logicalIsTrue(0.0f));
    EXPECT_FALSE(logicalIsTrue(-0.0f));
    EXPECT_TRUE(logicalIsTrue(kNaN));
    EXPECT_TRUE(logicalIsTrue(-kNaN));
    EXPECT_TRUE(logicalIsTrue(kSubnormal));
    EXPECT_TRUE(logicalIsTrue(-kInfinity));
}

// Every fp16 bit pattern, widened as the _fp16 variants read it (float(s[i])): exactly the two zeros are
// false, so NaN and subnormal halves stay true.
TEST(LogicalGpuRules, TruthBitTestMatchesOracleOnEveryFp16Pattern) {
    constexpr uint32_t kHalfPatterns      = 1u << 16;
    constexpr uint16_t kHalfMagnitudeMask = 0x7FFF;
    for (uint32_t pattern = 0; pattern < kHalfPatterns; ++pattern)
    {
        const fp16_t half    = (fp16_t) pattern;
        const float  widened = halfToFloat(half);
        const bool   isZero  = (half & kHalfMagnitudeMask) == 0;
        ASSERT_EQ(logicalIsTrue(widened), !isZero) << std::hex << "half=0x" << pattern;
        ASSERT_EQ(logicalIsTrue(widened), referenceTruth(widened)) << std::hex << "half=0x" << pattern;
    }
}

namespace {
    struct BroadcastCase {
        Shape a, b;
    };

    // Shape pairs covering same shape, rows, columns, two-sided broadcast, rank 0, 1-element operands,
    // rank raise, 0 extents and rank 5.
    const std::vector<BroadcastCase> kBroadcastCases {
        {{2, 3}, {2, 3}},       {{2, 3}, {3}},    {{2, 3}, {2, 1}},       {{2, 1, 4}, {3, 1}}, {{4}, {}},
        {{}, {2, 2}},           {{1}, {3, 1, 2}}, {{1, 3}, {0, 1}},       {{2, 0}, {2, 1}},    {{2, 1, 3, 1, 2}, {4, 1, 5, 1}},
        {{3, 1, 1}, {1, 1, 7}}, {{1, 1}, {1}},    {{5, 4, 3}, {5, 4, 3}},
    };

    // Run the transcribed binary kernel over its whole dispatch (with a forced 2-D spill) and return the
    // output buffer, requiring every element to be written exactly once.
    std::vector<float> runBinaryTranscription(const Shape &out, const Shape &aShape, const Shape &bShape, const std::vector<float> &aBuffer, const std::vector<float> &bBuffer, LogicalCombine combine) {
        static constexpr uint32_t            kForcedMaxGroupsX = 2; // spills every dispatch past 512 elements into y
        const logical::FlatBroadcastGeometry geometry          = logical::flatBroadcastGeometry(out, aShape, bShape, kGeometryTestLabel);
        BinaryPushConstant                   pc {};
        pc.rank                      = geometry.rank;
        pc.total                     = (int) logical::flatElementCount(out);
        const std::vector<int32_t> g = packGeometry(geometry);
        std::vector<float>         d((size_t) pc.total, -1.0f);
        std::vector<int>           writes((size_t) pc.total, 0);
        for (uint32_t gid: dispatchInvocationIds(pc.total, kForcedMaxGroupsX))
        {
            logicalBinaryMain(gid, pc, g, aBuffer, bBuffer, combine, d);
            if (gid < (uint32_t) pc.total)
            {
                ++writes[gid];
            }
        }
        for (size_t k = 0; k < writes.size(); ++k)
        {
            EXPECT_EQ(writes[k], 1) << "element " << k << " must be visited exactly once";
        }
        return d;
    }

    void expectMatchesOracle(const std::vector<float> &got, const NodeResult &oracle, const std::string &label) {
        ASSERT_EQ(got.size(), oracle.values.size()) << label;
        for (size_t k = 0; k < got.size(); ++k)
        {
            ASSERT_EQ(got[k], oracle.values[k]) << label << " k=" << k;
        }
    }
} // namespace

// or.comp / xor.comp against the oracle for every shape pair: the real geometry builder, the transcribed
// decode, runtime operands at fp32 storage and (for values whose truth fp16 storage keeps) at fp16, and a
// constant operand through the canonical upload at both precisions.
TEST(LogicalGpuRules, BinaryDecodeMatchesOracle) {
    uint32_t seed = 1;
    for (const BroadcastCase &shapes: kBroadcastCases)
    {
        const Shape  out   = referenceBroadcastShape(shapes.a, shapes.b);
        const size_t aSize = (size_t) referenceCount(shapes.a);
        const size_t bSize = (size_t) referenceCount(shapes.b);
        for (LogicalCombine combine: {LogicalCombine::Or, LogicalCombine::Xor})
        {
            const OpType      type  = combine == LogicalCombine::Or ? OpType::Or : OpType::Xor;
            const std::string label = std::string(combine == LogicalCombine::Or ? "Or" : "Xor") + " case seed " + std::to_string(seed);

            // Runtime operands, fp32 storage, the full value palette.
            const std::vector<float> aValues = mixedValues(aSize, seed++);
            const std::vector<float> bValues = mixedValues(bSize, seed++);
            NodeResult               oracle  = runOk(type, {floatInput(shapes.a, aValues), floatInput(shapes.b, bValues)});
            ASSERT_EQ(oracle.shape, out) << label;
            expectMatchesOracle(runBinaryTranscription(out, shapes.a, shapes.b, aValues, bValues, combine), oracle, label + " fp32");

            // Runtime operands, fp16 storage.
            const std::vector<float> aSafe      = fp16SafeValues(aSize, seed++);
            const std::vector<float> bSafe      = fp16SafeValues(bSize, seed++);
            NodeResult               safeOracle = runOk(type, {floatInput(shapes.a, aSafe), floatInput(shapes.b, bSafe)});
            expectMatchesOracle(runBinaryTranscription(out, shapes.a, shapes.b, throughFp16Storage(aSafe), throughFp16Storage(bSafe), combine), safeOracle, label + " fp16");

            // Constant operand B through canonicalConstantOperand, at both precisions.
            NodeResult               constOracle = runOk(type, {floatInput(shapes.a, aSafe), floatConstant(shapes.b, bValues)});
            const std::vector<float> canonical   = logical::canonicalConstantOperand(bValues, logical::flatElementCount(shapes.b));
            expectMatchesOracle(runBinaryTranscription(out, shapes.a, shapes.b, aSafe, canonical, combine), constOracle, label + " const fp32");
            expectMatchesOracle(runBinaryTranscription(out, shapes.a, shapes.b, throughFp16Storage(aSafe), throughFp16Storage(canonical), combine), constOracle, label + " const fp16");
        }
    }
}

// not.comp against the oracle over the dispatch of several sizes (including a 2-D spill, rank 0 and an
// empty tensor), at fp32 storage, fp16 storage and through the canonical constant upload.
TEST(LogicalGpuRules, NotDecodeMatchesOracle) {
    static constexpr uint32_t kForcedMaxGroupsX = 2; // spills every dispatch past 512 elements into y
    const std::vector<Shape>  shapes {{}, {1}, {0, 4}, {2, 3}, {3, 257}, {1, 1031}};
    uint32_t                  seed = 100;
    for (const Shape &shape: shapes)
    {
        const size_t size  = (size_t) referenceCount(shape);
        const auto   check = [&](const std::vector<float> &buffer, const NodeResult &oracle, const std::string &label) {
            NotPushConstant pc {};
            pc.total = (int) logical::flatElementCount(shape);
            std::vector<float> d((size_t) pc.total, -1.0f);
            for (uint32_t gid: dispatchInvocationIds(pc.total, kForcedMaxGroupsX))
            {
                logicalNotMain(gid, pc, buffer, d);
            }
            expectMatchesOracle(d, oracle, label);
        };
        const std::vector<float> values = mixedValues(size, seed++);
        const std::vector<float> safe   = fp16SafeValues(size, seed++);
        NodeResult               oracle = runOk(OpType::Not, {floatInput(shape, values)});
        ASSERT_EQ(oracle.shape, shape);
        check(values, oracle, "fp32");
        NodeResult safeOracle = runOk(OpType::Not, {floatInput(shape, safe)});
        check(throughFp16Storage(safe), safeOracle, "fp16");
        // A constant Not operand (one past constFold's element bound) uploads canonically; the oracle
        // reads the same values as a runtime input.
        const std::vector<float> canonical = logical::canonicalConstantOperand(values, logical::flatElementCount(shape));
        check(canonical, oracle, "const fp32");
        check(throughFp16Storage(canonical), oracle, "const fp16");
    }
}

// A constant operand keeps every element's truth at both storage precisions when decoded by initFloats
// from its stored dtype (fp32 including magnitudes an fp16 store rounds to zero, int64 including 2^32
// and INT64_MIN, 1-byte UINT8, rank 0) and canonicalized.
TEST(LogicalGpuRules, ConstantOperandKeepsTruthAtBothPrecisions) {
    Graph g;
    auto  addInitializer = [&](const std::string &name, const Shape &shape, DType dtype, std::vector<uint8_t> bytes) {
        TensorDesc d;
        d.name                   = name;
        d.shape                  = shape;
        d.dtype                  = dtype;
        d.isInitializer          = true;
        TensorId id              = g.addTensor(d);
        g.initializers[id].bytes = std::move(bytes);
        return id;
    };
    auto bytesOf = [](const HostBuffer &hb) {
        return std::vector<uint8_t>(hb.bytes.data(), hb.bytes.data() + hb.bytes.size());
    };
    const std::vector<float>   floats {0.0f, -0.0f, 1e-10f, -1e-30f, kSubnormal, kNaN, kInfinity, 1e6f, 2.0f, -3.0f};
    const std::vector<int64_t> ints {0, 1, -1, kLowWordZero, kAbove2Pow40, kInt64Min};
    TensorId floatId = addInitializer("floats", {(int64_t) floats.size()}, DType::Float32, bytesOf(hostBufferOf(floatConstant({(int64_t) floats.size()}, floats))));
    TensorId intId    = addInitializer("ints", {(int64_t) ints.size()}, DType::Int64, bytesOf(hostBufferOf(int64Constant({(int64_t) ints.size()}, ints))));
    TensorId byteId   = addInitializer("bytes", {4}, DType::UInt8, {0, 1, 128, 255});
    TensorId scalarId = addInitializer("scalar", {}, DType::Int64, bytesOf(hostBufferOf(int64Constant({}, {kLowWordZero}))));

    auto expectTruthKept = [&](TensorId id, const std::vector<bool> &truth) {
        const std::vector<float> canonical = logical::canonicalConstantOperand(initFloats(g, id), logical::flatElementCount(g.desc(id).shape));
        ASSERT_EQ(canonical.size(), truth.size()) << g.tensors[id].name;
        const std::vector<float> atFp16 = throughFp16Storage(canonical);
        for (size_t k = 0; k < truth.size(); ++k)
        {
            EXPECT_EQ(logicalIsTrue(canonical[k]), (bool) truth[k]) << g.tensors[id].name << " fp32 k=" << k;
            EXPECT_EQ(logicalIsTrue(atFp16[k]), (bool) truth[k]) << g.tensors[id].name << " fp16 k=" << k;
            EXPECT_TRUE(canonical[k] == logical::kLogicalTrue || canonical[k] == logical::kLogicalFalse);
        }
    };
    expectTruthKept(floatId, {false, false, true, true, true, true, true, true, true, true});
    expectTruthKept(intId, {false, true, true, true, true, true});
    expectTruthKept(byteId, {false, true, true, true});
    expectTruthKept(scalarId, {true});

    // The canonical encoding is what keeps 1e-10 true: a raw fp16 store rounds it to zero.
    EXPECT_FALSE(logicalIsTrue(throughFp16Storage({1e-10f})[0]));
}

// The dispatched element count treats rank 0 as one element and a 0 extent as none; an operand with more
// axes than the output, an operand extent that neither is 1 nor matches the output, and an output past the
// shader index range are rejected.
TEST(LogicalGpuRules, GeometryBoundsAndElementCount) {
    EXPECT_EQ(logical::flatElementCount({}), 1);
    EXPECT_EQ(logical::flatElementCount({0, 3}), 0);
    EXPECT_EQ(logical::flatElementCount({2, 3}), 6);
    EXPECT_THROW(logical::flatBroadcastGeometry({3}, {1, 3}, {3}, kGeometryTestLabel), Error);
    EXPECT_THROW(logical::flatBroadcastGeometry({2, 3}, {2, 3}, {2}, kGeometryTestLabel), Error);
    EXPECT_THROW(logical::flatBroadcastGeometry({0, 3}, {3, 3}, {1}, kGeometryTestLabel), Error);
    constexpr int64_t kPastIndexRange = logical::kMaxShaderElements + 1;
    EXPECT_THROW(logical::flatBroadcastGeometry({kPastIndexRange}, {1}, {1}, kGeometryTestLabel), Error);
    try
    {
        logical::flatBroadcastGeometry({2, 3}, {2, 3}, {2}, kGeometryTestLabel);
        ADD_FAILURE() << "an operand that does not broadcast must throw";
    } catch (const Error &error)
    {
        EXPECT_EQ(error.status(), Status::InvalidArgument);
        EXPECT_NE(std::string(error.what()).find(kGeometryTestLabel), std::string::npos) << error.what();
    }
    const logical::FlatBroadcastGeometry geometry = logical::flatBroadcastGeometry({2, 3, 4}, {3, 1}, {2, 1, 4}, kGeometryTestLabel);
    EXPECT_EQ(geometry.rank, 3);
    EXPECT_EQ(geometry.outDim, (std::vector<int32_t> {2, 3, 4}));
    EXPECT_EQ(geometry.aStride, (std::vector<int32_t> {0, 1, 0}));
    EXPECT_EQ(geometry.bStride, (std::vector<int32_t> {4, 0, 1}));
}
