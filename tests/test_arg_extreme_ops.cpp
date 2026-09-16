// ArgMax / ArgMin (ONNX opset 13) on the CPU oracle, and the host-checkable half of the GPU kernel.
//
// The CPU kernel (backend/cpu/arg_extreme.h) is checked through Session::run on hand-built graphs:
// axis and keepdims shapes, the sequential selection rule's tie / signed-zero / NaN / infinity
// consequences, exact int64 comparison, wide axes, the error cases, an index consumer that must keep
// its Cast to float, a .vxm round-trip and byte invariance across CPU thread counts.
//
// GLSL never runs on the host, so the GPU kernel is checked in three halves. The plan
// (backend/vulkan/ops/arg_extreme_plan.h) is run on graphs after the session's own Vulkan load
// sequence (planFlatLayoutAndStorage), proving the variant selection and the fp32-indices
// precondition against what the passes actually produce, and its geometry limits are shown to agree
// with the Vulkan gate. The scan itself is a C++ transcription of shaders/arg_extreme.comp, driven by
// the plan's push and specialization constants and compared with the CPU oracle over randomized data
// rich in ties, NaNs, infinities and signed zeros, for the fp32 variant and for the fp16 variant
// reading half-precision storage. The shader source is then read back and compared with the
// transcription (through a fixed GLSL-to-C++ substitution table) and with the plan's and the op's
// interface: push-constant members, specialization constant ids, bindings and their element types,
// local size and variant naming.
#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/parallel.h"
#include "backend/vulkan/ops/arg_extreme_plan.h"
#include "core/vk_gates.h"
#include "import/passes.h"
#include "vknn/dtype.h"
#include "vknn/exec_context.h"
#include "vknn/graph.h"
#include "vknn/op_descriptor.h"
#include "vknn/session.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    constexpr float   kNaN          = std::numeric_limits<float>::quiet_NaN();
    constexpr float   kInf          = std::numeric_limits<float>::infinity();
    constexpr int64_t kOnnxFloat    = 1; // TensorProto.DataType FLOAT, the Cast target
    constexpr int     kSingleThread = 1;

    Attr intAttr(int64_t value) {
        Attr a;
        a.kind = Attr::Int;
        a.i    = value;
        return a;
    }

    /// The three ONNX attributes, at their ONNX defaults.
    struct ArgAttrs {
        int64_t axis       = 0;
        int64_t keepDims   = 1;
        int64_t selectLast = 0;
    };

    ArgAttrs attrs(int64_t axis, int64_t keepDims, int64_t selectLast) {
        ArgAttrs a;
        a.axis       = axis;
        a.keepDims   = keepDims;
        a.selectLast = selectLast;
        return a;
    }

    Node argNode(OpType type, TensorId data, TensorId indices, const ArgAttrs &a) {
        Node n;
        n.type                          = type;
        n.name                          = type == OpType::ArgMax ? "argmax" : "argmin";
        n.inputs                        = {data};
        n.outputs                       = {indices};
        n.attr.map["axis"]              = intAttr(a.axis);
        n.attr.map["keepdims"]          = intAttr(a.keepDims);
        n.attr.map["select_last_index"] = intAttr(a.selectLast);
        return n;
    }

    TensorId addInput(Graph &g, const std::string &name, const Shape &shape, DType dtype) {
        TensorDesc d;
        d.name      = name;
        d.shape     = shape;
        d.dtype     = dtype;
        d.isInput   = true;
        TensorId id = g.addTensor(d);
        g.inputs.push_back(id);
        return id;
    }

    // The declared int64 indices output (the ONNX graph-output value_info), so readback is int64.
    TensorId addIndicesOutput(Graph &g, const std::string &name) {
        TensorDesc d;
        d.name      = name;
        d.dtype     = DType::Int64;
        d.isOutput  = true;
        TensorId id = g.addTensor(d);
        g.outputs.push_back(id);
        return id;
    }

    // x (runtime input of `dtype`) -> ArgMax/ArgMin -> int64 "indices".
    Graph argGraph(OpType type, const Shape &shape, DType dtype, const ArgAttrs &a) {
        Graph    g;
        TensorId x       = addInput(g, "x", shape, dtype);
        TensorId indices = addIndicesOutput(g, "indices");
        g.nodes.push_back(argNode(type, x, indices, a));
        return g;
    }

    IOTensor floatInput(const Shape &shape, const std::vector<float> &values) {
        IOTensor io;
        io.name  = "x";
        io.shape = shape;
        io.dtype = DType::Float32;
        io.data.resize(values.size() * sizeof(float));
        if (!values.empty())
        {
            std::memcpy(io.data.data(), values.data(), io.data.size());
        }
        return io;
    }

    IOTensor int64Input(const Shape &shape, const std::vector<int64_t> &values) {
        IOTensor io;
        io.name  = "x";
        io.shape = shape;
        io.dtype = DType::Int64;
        io.data.resize(values.size() * sizeof(int64_t));
        if (!values.empty())
        {
            std::memcpy(io.data.data(), values.data(), io.data.size());
        }
        return io;
    }

    struct ArgResult {
        Status               status = Status::RuntimeError;
        Shape                shape;
        DType                dtype = DType::Float32;
        std::vector<int64_t> indices;
        std::vector<uint8_t> bytes;
    };

    Config cpuConfig(int threads) {
        Config cfg;
        cfg.backend    = BackendKind::Cpu;
        cfg.cpuThreads = threads;
        return cfg;
    }

    ArgResult runSession(Session &sess, const std::vector<IOTensor> &inputs) {
        ArgResult             r;
        std::vector<IOTensor> outs;
        r.status = sess.run(inputs, outs);
        if (r.status != Status::Ok || outs.empty())
        {
            return r;
        }
        r.shape = outs[0].shape;
        r.dtype = outs[0].dtype;
        r.bytes = outs[0].data;
        if (r.dtype == DType::Int64)
        {
            r.indices.resize(outs[0].data.size() / sizeof(int64_t));
            if (!r.indices.empty())
            {
                std::memcpy(r.indices.data(), outs[0].data.data(), r.indices.size() * sizeof(int64_t));
            }
        }
        return r;
    }

    ArgResult runGraph(Graph g, const std::vector<IOTensor> &inputs, int threads = kSingleThread) {
        auto sess = Session::create(std::move(g), cpuConfig(threads));
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return {};
        }
        return runSession(*sess, inputs);
    }

    ArgResult runFloat(OpType type, const Shape &shape, const std::vector<float> &values, const ArgAttrs &a, int threads = kSingleThread) {
        return runGraph(argGraph(type, shape, DType::Float32, a), {floatInput(shape, values)}, threads);
    }

    ArgResult runInt64(OpType type, const Shape &shape, const std::vector<int64_t> &values, const ArgAttrs &a, int threads = kSingleThread) {
        return runGraph(argGraph(type, shape, DType::Int64, a), {int64Input(shape, values)}, threads);
    }

    // Runs `type` with `selectLast` over one rank-1 slice (axis 0, keepdims 0) and returns the selected
    // index, or -1 when the run fails.
    int64_t sliceIndex(OpType type, const std::vector<float> &slice, int64_t selectLast) {
        const Shape     shape = {(int64_t) slice.size()};
        const ArgResult r     = runFloat(type, shape, slice, attrs(0, 0, selectLast));
        EXPECT_EQ(r.status, Status::Ok);
        EXPECT_EQ(r.shape, (Shape {1}));
        return r.indices.size() == 1 ? r.indices[0] : -1;
    }

    // ---- CPU-independent restatement of the selection rule, for the randomized sweeps ----

    template <class T>
    std::vector<int64_t> referenceArgExtreme(const std::vector<T> &data, const Shape &shape, int64_t axis, bool selectLargest, bool selectLast) {
        const int64_t rank = (int64_t) shape.size();
        axis               = axis < 0 ? axis + rank : axis;
        int64_t outer = 1, inner = 1;
        for (int64_t d = 0; d < axis; ++d)
        {
            outer *= shape[(size_t) d];
        }
        for (int64_t d = axis + 1; d < rank; ++d)
        {
            inner *= shape[(size_t) d];
        }
        const int64_t        extent = shape[(size_t) axis];
        std::vector<int64_t> out((size_t) (outer * inner));
        for (int64_t o = 0; o < outer; ++o)
        {
            for (int64_t i = 0; i < inner; ++i)
            {
                const int64_t base   = o * extent * inner + i;
                T             best   = data[(size_t) base];
                int64_t       bestAt = 0;
                for (int64_t j = 1; j < extent; ++j)
                {
                    const T    x    = data[(size_t) (base + j * inner)];
                    const bool take = selectLargest ? (selectLast ? x >= best : x > best) : (selectLast ? x <= best : x < best);
                    if (take)
                    {
                        best   = x;
                        bestAt = j;
                    }
                }
                out[(size_t) (o * inner + i)] = bestAt;
            }
        }
        return out;
    }

    // ---- Transcription of shaders/arg_extreme.comp ----
    // Textually parallel to the shader: ArgExtremeShader.SourceMatchesTranscriptionAndInterface
    // compares the two after the substitutions in kGlslToTranscription, so any other edit to either
    // side fails it. The specialization constants SELECT_LARGEST / SELECT_LAST become parameters,
    // `float(data[i])` becomes readData(i) (an fp32 lane, or halfToFloat of an fp16 lane for the
    // arg_extreme_fp16 variant), `indices` is the fp32 buffer the shader declares, and the
    // push-constant block is the plan's ArgExtremePushConstants.

    using ShaderData = std::function<float(int)>;

    /// shaders/arg_extreme.comp local_size_x (the source test reads it back).
    constexpr uint32_t kShaderLocalSize = 256;

    // transcribes: bool takesCandidate(float candidate, float best)
    bool takesCandidate(float candidate, float best, int SELECT_LARGEST, int SELECT_LAST) {
        if (SELECT_LARGEST != 0)
        {
            return (SELECT_LAST != 0) ? (candidate >= best) : (candidate > best);
        }
        return (SELECT_LAST != 0) ? (candidate <= best) : (candidate < best);
    }

    // transcribes: void main()
    void shaderMain(uint32_t gid, const ShaderData &readData, std::vector<float> &indices, const ArgExtremePushConstants &pc, int SELECT_LARGEST, int SELECT_LAST) {
        if (gid >= uint32_t(pc.total))
        {
            return;
        }
        int outputIndex = int(gid);
        int block       = outputIndex / pc.inner;
        int innerPos    = outputIndex % pc.inner;
        int base        = (block * pc.extent) * pc.inner + innerPos;

        float best   = readData(base);
        int   bestAt = 0;
        if (!std::isnan(best))
        {
            for (int j = 1; j < pc.extent; ++j)
            {
                float candidate = readData(base + j * pc.inner);
                if (!std::isnan(candidate) && takesCandidate(candidate, best, SELECT_LARGEST, SELECT_LAST))
                {
                    best   = candidate;
                    bestAt = j;
                }
            }
        }
        indices[gid] = float(bestAt);
    }
    // end of the arg_extreme.comp transcription

    // Every invocation of the op's dispatch: groups(total, kShaderLocalSize) workgroups of
    // kShaderLocalSize, so the tail invocations past `total` exercise the bounds check.
    std::vector<float> dispatchShader(const ShaderData &readData, const ArgExtremePushConstants &pc, const std::vector<uint32_t> &specConstants) {
        std::vector<float> indices((size_t) pc.total, -1.0f);
        const uint32_t     groupCount = (uint32_t) ((pc.total + kShaderLocalSize - 1) / kShaderLocalSize);
        for (uint32_t gid = 0; gid < groupCount * kShaderLocalSize; ++gid)
        {
            shaderMain(gid, readData, indices, pc, (int) specConstants[0], (int) specConstants[1]);
        }
        return indices;
    }

    // ---- Reading the kernel's sources back ----

    /// Raw-text markers around the transcription above (the first occurrence of each is the comment).
    constexpr const char *kTranscriptionBegin = "// transcribes: bool takesCandidate(float candidate, float best)";
    constexpr const char *kTranscriptionEnd   = "// end of the arg_extreme.comp transcription";

    /// A GLSL fragment and the C++ fragment the transcription writes in its place, each in the
    /// normalized (comment- and whitespace-free) form, with the exact number of GLSL occurrences.
    struct SourceSubstitution {
        const char *glsl;
        const char *transcription;
        size_t      occurrences;
    };

    /// Every difference between shaders/arg_extreme.comp (from takesCandidate to the end) and the
    /// transcription. Any other edit to either side makes the translated shader differ from the
    /// transcription.
    const SourceSubstitution kGlslToTranscription[] = {
        {"booltakesCandidate(floatcandidate,floatbest){", "booltakesCandidate(floatcandidate,floatbest,intSELECT_LARGEST,intSELECT_LAST){", 1},
        {"voidmain(){uintgid=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*gl_NumWorkGroups.x*gl_WorkGroupSize.x;", "voidshaderMain(uint32_tgid,constShaderData&readData,std::vector<float>&indices,constArgExtremePushConstants&pc,intSELECT_LARGEST,intSELECT_LAST){", 1},
        {"if(gid>=uint(pc.total))return;", "if(gid>=uint32_t(pc.total)){return;}", 1},
        {"float(data[base])", "readData(base)", 1},
        {"float(data[base+j*pc.inner])", "readData(base+j*pc.inner)", 1},
        {"isnan(", "std::isnan(", 2},
        {"takesCandidate(candidate,best))", "takesCandidate(candidate,best,SELECT_LARGEST,SELECT_LAST))", 1},
    };

    /// The file at `path`, or an empty string when it cannot be read.
    std::string readSource(const std::string &path) {
        std::ifstream file(path, std::ios::binary);
        return file ? std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()) : std::string();
    }

    /// The repository root, from this file's compile-time path (<root>/tests/<file>).
    std::string repositoryRoot() {
        const std::string file  = __FILE__;
        const size_t      slash = file.find_last_of("/\\");
        return slash == std::string::npos ? std::string("..") : file.substr(0, slash) + "/..";
    }

    /// `text` with every `//` comment and every whitespace character removed. The sources compared
    /// here carry no `/* */` comment and no string literal containing `//`.
    std::string normalizeSource(const std::string &text) {
        std::string out;
        out.reserve(text.size());
        for (size_t i = 0; i < text.size(); ++i)
        {
            if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '/')
            {
                while (i < text.size() && text[i] != '\n')
                {
                    ++i;
                }
                continue;
            }
            if (!std::isspace((unsigned char) text[i]))
            {
                out += text[i];
            }
        }
        return out;
    }

    size_t countOccurrences(const std::string &text, const std::string &fragment) {
        size_t count = 0;
        for (size_t at = text.find(fragment); at != std::string::npos; at = text.find(fragment, at + fragment.size()))
        {
            ++count;
        }
        return count;
    }

    /// The text strictly between the first `open` and the next `close` after it, or "" when absent.
    std::string between(const std::string &text, const std::string &open, const std::string &close) {
        const size_t begin = text.find(open);
        if (begin == std::string::npos)
        {
            return {};
        }
        const size_t end = text.find(close, begin + open.size());
        return end == std::string::npos ? std::string() : text.substr(begin + open.size(), end - begin - open.size());
    }

    // Deterministic generator over a value alphabet heavy in ties, NaNs, infinities and signed zeros,
    // mixed with plain random values.
    std::vector<float> trickyValues(size_t n, uint32_t seed) {
        static constexpr float    kAlphabet[]   = {-2.0f, -1.0f,  -0.0f,    0.0f,     1.0f,      2.0f,    kNaN,   kInf,
                                                   -kInf, 1e-30f, 65504.0f, 70000.0f, -70000.0f, 2049.0f, 2048.0f};
        static constexpr uint32_t kAlphabetSize = sizeof(kAlphabet) / sizeof(kAlphabet[0]);
        static constexpr uint32_t kLcgMul = 1664525u, kLcgAdd = 1013904223u;
        // The LCG's low bits are weak; each decision reads a different window of its high bits.
        static constexpr uint32_t kKindShift = 16, kPlainShift = 8, kAlphabetShift = 12;
        static constexpr uint32_t kPlainRandomOneIn = 3;
        static constexpr int32_t  kPlainRandomSpan = 2001, kPlainRandomBias = 1000;
        static constexpr float    kPlainRandomStep = 0.25f;
        std::vector<float>        v(n);
        uint32_t                  s = seed;
        for (size_t i = 0; i < n; ++i)
        {
            s = s * kLcgMul + kLcgAdd;
            if ((s >> kKindShift) % kPlainRandomOneIn == 0)
            {
                v[i] = (float) ((int32_t) ((s >> kPlainShift) % kPlainRandomSpan) - kPlainRandomBias) * kPlainRandomStep;
            } else
            {
                v[i] = kAlphabet[(s >> kAlphabetShift) % kAlphabetSize];
            }
        }
        return v;
    }

    // Deterministic int64 values heavy in ties and in magnitudes above 2^53 that differ by one (a
    // comparison through double would tie them), including both int64 extremes.
    std::vector<int64_t> trickyInt64Values(size_t n, uint32_t seed) {
        static constexpr int64_t  kBig          = int64_t(1) << 60;
        static constexpr int64_t  kAlphabet[]   = {INT64_MIN, -kBig - 1, -kBig, -1, 0, 1, kBig, kBig + 1, 3 * kBig, INT64_MAX};
        static constexpr uint32_t kAlphabetSize = sizeof(kAlphabet) / sizeof(kAlphabet[0]);
        static constexpr uint32_t kLcgMul = 1664525u, kLcgAdd = 1013904223u;
        static constexpr uint32_t kAlphabetShift = 12, kOffsetShift = 20;
        static constexpr uint32_t kOffsetSpan = 3; // offsets -1, 0, +1 around 2^60 values
        std::vector<int64_t>      v(n);
        uint32_t                  s = seed;
        for (size_t i = 0; i < n; ++i)
        {
            s                     = s * kLcgMul + kLcgAdd;
            const int64_t base    = kAlphabet[(s >> kAlphabetShift) % kAlphabetSize];
            const bool    wrapped = base == INT64_MIN || base == INT64_MAX;
            v[i]                  = wrapped ? base : base + (int64_t) ((s >> kOffsetShift) % kOffsetSpan) - 1;
        }
        return v;
    }

    const Node *findNode(const Graph &g, OpType type) {
        for (const Node &n: g.nodes)
        {
            if (n.type == type)
            {
                return &n;
            }
        }
        return nullptr;
    }

    // Byte identity of the indices across thread counts, plus proof the partition engaged (a shape
    // too small to split would pass vacuously).
    void expectBytesInvariantAcrossThreads(const std::function<Graph()> &build, const IOTensor &input) {
        static const std::vector<int> kThreadCounts {2, 3, 5, 8};
        const int64_t                 dispatchesBefore = cpu::detail::poolDispatches();
        const ArgResult               reference        = runGraph(build(), {input}, kSingleThread);
        ASSERT_EQ(reference.status, Status::Ok);
        ASSERT_FALSE(reference.bytes.empty());
        for (int threads: kThreadCounts)
        {
            const ArgResult got = runGraph(build(), {input}, threads);
            ASSERT_EQ(got.status, Status::Ok) << "threads=" << threads;
            ASSERT_EQ(got.bytes.size(), reference.bytes.size()) << "threads=" << threads;
            EXPECT_EQ(0, std::memcmp(got.bytes.data(), reference.bytes.data(), reference.bytes.size())) << "threads=" << threads;
        }
        EXPECT_GT(cpu::detail::poolDispatches(), dispatchesBefore) << "shape too small to partition: the byte comparison is vacuous";
    }

    /// Layout of the hand-built plan graphs: flat everywhere (what the session's layout pass produces
    /// for an ArgMax/ArgMin node), the data left in the NC4HW4 layout, or the indices left in it.
    enum class PlanLayout { Flat, DataNc4, IndicesNc4 };

    // An ArgMax over `dataShape` along `axis` (keepdims 0) with hand-set layout and storage descriptors.
    Graph planGraph(const Shape &dataShape, const Shape &indicesShape, int64_t axis, bool indicesFp32, PlanLayout layout = PlanLayout::Flat) {
        Graph    g;
        TensorId x                = addInput(g, "x", dataShape, DType::Float32);
        TensorId indices          = addIndicesOutput(g, "indices");
        g.desc(x).gpuFlat         = layout != PlanLayout::DataNc4;
        g.desc(indices).shape     = indicesShape;
        g.desc(indices).gpuFlat   = layout != PlanLayout::IndicesNc4;
        g.desc(indices).storeFp32 = indicesFp32;
        g.nodes.push_back(argNode(OpType::ArgMax, x, indices, attrs(axis, 0, 0)));
        return g;
    }

    // The Status planArgExtreme ends with for an ArgMax over `dataShape` along `axis` (keepdims 0).
    Status planStatus(const Shape &dataShape, const Shape &indicesShape, int64_t axis, bool indicesFp32, bool baseFp16, PlanLayout layout = PlanLayout::Flat) {
        const Graph g      = planGraph(dataShape, indicesShape, axis, indicesFp32, layout);
        Status      status = Status::Ok;
        try
        {
            planArgExtreme(g, g.nodes[0], baseFp16); // stays Ok when every precondition holds
        } catch (const Error &e)
        {
            status = e.status(); // the named refusal
        }
        return status;
    }

} // namespace

// --- shapes and axes -------------------------------------------------------------------------------

TEST(ArgExtremeOps, AxisAndKeepDimsShapesOnRank3) {
    // x[i] = (i * kStep) % kCount with kStep coprime to kCount: distinct values in a scrambled order,
    // so every slice along every axis has a unique extreme.
    static constexpr int kCount = 24, kStep = 7;
    const Shape          shape = {2, 3, 4};
    std::vector<float>   x(kCount);
    for (int i = 0; i < kCount; ++i)
    {
        x[(size_t) i] = (float) ((i * kStep) % kCount);
    }
    struct Case {
        int64_t axis;
        int64_t keepDims;
        Shape   expectedShape;
    };
    const Case cases[] = {
        {0, 1, {1, 3, 4}}, {0, 0, {3, 4}},     {1, 1, {2, 1, 4}}, {1, 0, {2, 4}},  {2, 1, {2, 3, 1}},
        {2, 0, {2, 3}},    {-1, 1, {2, 3, 1}}, {-1, 0, {2, 3}},   {-3, 0, {3, 4}},
    };
    for (OpType type: {OpType::ArgMax, OpType::ArgMin})
    {
        for (const Case &c: cases)
        {
            const ArgResult r = runFloat(type, shape, x, attrs(c.axis, c.keepDims, 0));
            ASSERT_EQ(r.status, Status::Ok) << opTypeName(type) << " axis " << c.axis;
            EXPECT_EQ(r.dtype, DType::Int64);
            EXPECT_EQ(r.shape, c.expectedShape) << opTypeName(type) << " axis " << c.axis << " keepdims " << c.keepDims;
            EXPECT_EQ(r.indices, referenceArgExtreme(x, shape, c.axis, type == OpType::ArgMax, false)) << opTypeName(type) << " axis " << c.axis;
        }
    }
    // Hand-checked slices: along axis 1 at (0, *, 0) the values are x[0]=0, x[4]=4, x[8]=8.
    const ArgResult axis1 = runFloat(OpType::ArgMax, shape, x, attrs(1, 0, 0));
    ASSERT_EQ(axis1.indices.size(), 8u);
    EXPECT_EQ(axis1.indices[0], 2);
    const ArgResult axis2 = runFloat(OpType::ArgMin, shape, x, attrs(2, 0, 0));
    ASSERT_EQ(axis2.indices.size(), 6u);
    EXPECT_EQ(axis2.indices[0], 0); // row x[0..3] = {0, 7, 14, 21}
}

TEST(ArgExtremeOps, DefaultAttributesAreAxisZeroKeepDimsFirstIndex) {
    // Rows {9, 1} and {9, 2}: column 0 ties, so each default is observable. Axis 0, keepdims 1 and
    // select_last_index 0 give ArgMax {0, 1} and ArgMin {0, 0} of shape {1, 2}; select_last_index 1
    // would give {1, 1} / {1, 0}, and axis 1 would give {0, 0} / {1, 1}.
    for (OpType type: {OpType::ArgMax, OpType::ArgMin})
    {
        Graph    g;
        TensorId x       = addInput(g, "x", {2, 2}, DType::Float32);
        TensorId indices = addIndicesOutput(g, "indices");
        Node     n;
        n.type    = type;
        n.name    = "defaults";
        n.inputs  = {x};
        n.outputs = {indices};
        g.nodes.push_back(n);
        const ArgResult r = runGraph(std::move(g), {floatInput({2, 2}, {9, 1, 9, 2})});
        ASSERT_EQ(r.status, Status::Ok) << opTypeName(type);
        EXPECT_EQ(r.shape, (Shape {1, 2})) << opTypeName(type);
        EXPECT_EQ(r.indices, type == OpType::ArgMax ? (std::vector<int64_t> {0, 1}) : (std::vector<int64_t> {0, 0})) << opTypeName(type);
    }
}

TEST(ArgExtremeOps, RankOneWithoutKeepDimsIsOneElement) {
    for (OpType type: {OpType::ArgMax, OpType::ArgMin})
    {
        const ArgResult r = runFloat(type, {5}, {3, 8, -1, 8, -1}, attrs(0, 0, 0));
        ASSERT_EQ(r.status, Status::Ok);
        EXPECT_EQ(r.shape, (Shape {1})); // the IR has no rank-0 activations
        EXPECT_EQ(r.indices, (std::vector<int64_t> {type == OpType::ArgMax ? 1 : 2}));
    }
}

// --- the selection rule's consequences ---------------------------------------------------------------

TEST(ArgExtremeOps, TiesKeepFirstIndexOrLastWithSelectLastIndex) {
    const std::vector<float> slice = {4, 9, 1, 9, 1, 9, 4};
    EXPECT_EQ(sliceIndex(OpType::ArgMax, slice, 0), 1);
    EXPECT_EQ(sliceIndex(OpType::ArgMax, slice, 1), 5);
    EXPECT_EQ(sliceIndex(OpType::ArgMin, slice, 0), 2);
    EXPECT_EQ(sliceIndex(OpType::ArgMin, slice, 1), 4);
    // A constant slice: every element ties.
    EXPECT_EQ(sliceIndex(OpType::ArgMax, {7, 7, 7}, 0), 0);
    EXPECT_EQ(sliceIndex(OpType::ArgMax, {7, 7, 7}, 1), 2);
    EXPECT_EQ(sliceIndex(OpType::ArgMin, {7, 7, 7}, 1), 2);
}

TEST(ArgExtremeOps, SignedZerosTie) {
    for (const std::vector<float> &slice: {std::vector<float> {-0.0f, 0.0f}, std::vector<float> {0.0f, -0.0f}})
    {
        for (OpType type: {OpType::ArgMax, OpType::ArgMin})
        {
            EXPECT_EQ(sliceIndex(type, slice, 0), 0) << opTypeName(type);
            EXPECT_EQ(sliceIndex(type, slice, 1), 1) << opTypeName(type);
        }
    }
}

TEST(ArgExtremeOps, NaNAtIndexZeroWinsInEveryMode) {
    for (OpType type: {OpType::ArgMax, OpType::ArgMin})
    {
        for (int64_t selectLast: {0, 1})
        {
            EXPECT_EQ(sliceIndex(type, {kNaN, 5, -kInf, kInf, kNaN}, selectLast), 0) << opTypeName(type) << " select_last_index " << selectLast;
        }
    }
}

TEST(ArgExtremeOps, NaNElsewhereIsNeverSelected) {
    const std::vector<float> slice = {1, kNaN, 3, kNaN, -2, kNaN};
    for (int64_t selectLast: {0, 1})
    {
        EXPECT_EQ(sliceIndex(OpType::ArgMax, slice, selectLast), 2) << "select_last_index " << selectLast;
        EXPECT_EQ(sliceIndex(OpType::ArgMin, slice, selectLast), 4) << "select_last_index " << selectLast;
    }
    // A NaN between two equal extremes does not break the tie.
    EXPECT_EQ(sliceIndex(OpType::ArgMax, {5, kNaN, 5}, 1), 2);
    EXPECT_EQ(sliceIndex(OpType::ArgMin, {5, kNaN, 5}, 0), 0);
}

TEST(ArgExtremeOps, AllNaNSliceSelectsIndexZero) {
    for (OpType type: {OpType::ArgMax, OpType::ArgMin})
    {
        for (int64_t selectLast: {0, 1})
        {
            EXPECT_EQ(sliceIndex(type, {kNaN, kNaN, kNaN, kNaN}, selectLast), 0) << opTypeName(type);
        }
    }
}

TEST(ArgExtremeOps, InfinitiesAreOrdinaryExtremes) {
    const std::vector<float> slice = {-kInf, 3, kInf, kInf, -kInf};
    EXPECT_EQ(sliceIndex(OpType::ArgMax, slice, 0), 2);
    EXPECT_EQ(sliceIndex(OpType::ArgMax, slice, 1), 3);
    EXPECT_EQ(sliceIndex(OpType::ArgMin, slice, 0), 0);
    EXPECT_EQ(sliceIndex(OpType::ArgMin, slice, 1), 4);
    EXPECT_EQ(sliceIndex(OpType::ArgMax, {-kInf, -kInf}, 0), 0);
    EXPECT_EQ(sliceIndex(OpType::ArgMin, {kInf, 65504.0f}, 0), 1);
}

TEST(ArgExtremeOps, Int64DataComparesExactlyAboveTwoToThe53) {
    // 2^60 and 2^60 + 1 are one double apart only in int64: a comparison through double would tie them.
    static constexpr int64_t   kBig = int64_t(1) << 60;
    const std::vector<int64_t> row  = {kBig, kBig + 1, kBig, -kBig - 1, -kBig};
    const ArgResult            max  = runInt64(OpType::ArgMax, {5}, row, attrs(0, 1, 0));
    ASSERT_EQ(max.status, Status::Ok);
    EXPECT_EQ(max.dtype, DType::Int64);
    EXPECT_EQ(max.indices, (std::vector<int64_t> {1}));
    EXPECT_EQ(runInt64(OpType::ArgMin, {5}, row, attrs(0, 1, 0)).indices, (std::vector<int64_t> {3}));
    // Ties of equal huge values follow select_last_index.
    const std::vector<int64_t> ties = {kBig, kBig - 1, kBig};
    EXPECT_EQ(runInt64(OpType::ArgMax, {3}, ties, attrs(0, 0, 0)).indices, (std::vector<int64_t> {0}));
    EXPECT_EQ(runInt64(OpType::ArgMax, {3}, ties, attrs(0, 0, 1)).indices, (std::vector<int64_t> {2}));
    // A [2, 3] int64 tensor along the last axis.
    const std::vector<int64_t> matrix = {INT64_MIN, INT64_MAX, INT64_MAX - 1, 5, 5, 4};
    EXPECT_EQ(runInt64(OpType::ArgMax, {2, 3}, matrix, attrs(-1, 0, 0)).indices, (std::vector<int64_t> {1, 0}));
    EXPECT_EQ(runInt64(OpType::ArgMin, {2, 3}, matrix, attrs(-1, 0, 1)).indices, (std::vector<int64_t> {0, 2}));
}

TEST(ArgExtremeOps, WideAxisIndicesAbove2048OnTheCpu) {
    // Indices past 2048, the last integer fp16 storage holds consecutively. The CPU kernel stores int64
    // indices; the GPU kernel's exactness past 2048 rests on its fp32 indices buffer, which
    // ArgExtremeShader.SourceMatchesTranscriptionAndInterface pins in the shader source.
    static constexpr int64_t kExtent = 3000;
    std::vector<float>       x((size_t) (2 * kExtent));
    for (int64_t j = 0; j < kExtent; ++j)
    {
        x[(size_t) j]             = (float) (j % 97);
        x[(size_t) (kExtent + j)] = (float) -(j % 89);
    }
    x[2500]                      = 1000.0f; // row 0 maximum
    x[(size_t) (kExtent + 2999)] = -500.0f; // row 1 minimum
    const ArgResult max          = runFloat(OpType::ArgMax, {2, kExtent}, x, attrs(1, 0, 0));
    ASSERT_EQ(max.status, Status::Ok);
    ASSERT_EQ(max.indices.size(), 2u);
    EXPECT_EQ(max.indices[0], 2500);
    const ArgResult min = runFloat(OpType::ArgMin, {2, kExtent}, x, attrs(1, 0, 0));
    ASSERT_EQ(min.indices.size(), 2u);
    EXPECT_EQ(min.indices[1], 2999);
}

TEST(ArgExtremeOps, ZeroElementOuterDimensionYieldsEmptyIndices) {
    // Only the selected axis must be non-empty; an empty outer dimension is a valid empty result.
    const ArgResult r = runFloat(OpType::ArgMax, {0, 3}, {}, attrs(1, 1, 0));
    ASSERT_EQ(r.status, Status::Ok);
    EXPECT_EQ(r.shape, (Shape {0, 1}));
    EXPECT_TRUE(r.indices.empty());
}

// --- errors ------------------------------------------------------------------------------------------

TEST(ArgExtremeOps, InvalidAxisOrEmptyAxisFailsTheRun) {
    for (OpType type: {OpType::ArgMax, OpType::ArgMin})
    {
        EXPECT_NE(runFloat(type, {2, 3}, {1, 2, 3, 4, 5, 6}, attrs(2, 1, 0)).status, Status::Ok) << opTypeName(type) << " axis 2 on rank 2";
        EXPECT_NE(runFloat(type, {2, 3}, {1, 2, 3, 4, 5, 6}, attrs(-3, 1, 0)).status, Status::Ok) << opTypeName(type) << " axis -3 on rank 2";
        EXPECT_NE(runFloat(type, {2, 0, 3}, {}, attrs(1, 0, 0)).status, Status::Ok) << opTypeName(type) << " empty axis";
    }
}

TEST(ArgExtremeOps, RankZeroInputFailsTheRun) {
    for (OpType type: {OpType::ArgMax, OpType::ArgMin})
    {
        EXPECT_NE(runFloat(type, {}, {4.0f}, attrs(0, 1, 0)).status, Status::Ok) << opTypeName(type);
    }
}

TEST(ArgExtremeOps, KernelErrorsAreNamedInvalidArgument) {
    // Drive the registered CPU kernel directly so the Status and message are observable.
    struct Case {
        Shape       shape;
        int64_t     axis;
        const char *reason;
    };
    const Case cases[] = {{{}, 0, "rank-0"}, {{2, 3}, 2, "out of range"}, {{2, 3}, -3, "out of range"}, {{4, 0}, -1, "extent 0"}};
    for (OpType type: {OpType::ArgMax, OpType::ArgMin})
    {
        for (const Case &c: cases)
        {
            Graph    g;
            TensorId x       = addInput(g, "x", c.shape, DType::Float32);
            TensorId indices = addIndicesOutput(g, "indices");
            Node     n       = argNode(type, x, indices, attrs(c.axis, 1, 0));
            n.name           = "selector";

            std::vector<RtTensor> pool(g.tensors.size());
            pool[(size_t) x].shape = c.shape;
            pool[(size_t) x].dtype = DType::Float32;
            pool[(size_t) x].host.resizeElems(c.shape.empty() ? 1 : numElements(c.shape), DType::Float32);
            pool[(size_t) x].hostValid = true;
            ExecContext ctx;
            ctx.pool  = &pool;
            ctx.graph = &g;

            auto kernel = CpuOpRegistry::instance().create(type);
            ASSERT_TRUE(kernel);
            try
            {
                kernel->run(n, ctx);
                ADD_FAILURE() << opTypeName(type) << " accepted " << c.reason;
            } catch (const Error &e)
            {
                EXPECT_EQ(e.status(), Status::InvalidArgument) << e.what();
                const std::string message = e.what();
                EXPECT_NE(message.find(std::string(opTypeName(type)) + " 'selector': "), std::string::npos) << message;
                EXPECT_NE(message.find(c.reason), std::string::npos) << message;
            }
        }
    }
}

// --- graphs around the op ----------------------------------------------------------------------------

TEST(ArgExtremeOps, IndicesThroughCastToFloatKeepFractionalAdd) {
    // ArgMax/ArgMin -> Cast(to FLOAT) -> Add(0.5). The int64 indices make the Cast meaningful: were it
    // dropped, the Add would take its int64 path and truncate the 0.5 away.
    for (OpType type: {OpType::ArgMax, OpType::ArgMin})
    {
        auto build = [type]() {
            Graph      g;
            TensorId   x = addInput(g, "x", {2, 3}, DType::Float32);
            TensorDesc idxDesc;
            idxDesc.name       = "indices";
            TensorId   indices = g.addTensor(idxDesc);
            TensorDesc castDesc;
            castDesc.name      = "indices_float";
            TensorId   asFloat = g.addTensor(castDesc);
            TensorDesc halfDesc;
            halfDesc.name          = "half";
            halfDesc.shape         = {1};
            halfDesc.isInitializer = true;
            TensorId   half        = g.addTensor(halfDesc);
            HostBuffer hb;
            hb.resizeElems(1, DType::Float32);
            hb.f32()[0]          = 0.5f;
            g.initializers[half] = hb;
            TensorDesc yDesc;
            yDesc.name     = "y";
            yDesc.isOutput = true;
            TensorId y     = g.addTensor(yDesc);
            g.outputs      = {y};

            g.nodes.push_back(argNode(type, x, indices, attrs(1, 0, 0)));
            Node cast;
            cast.type           = OpType::Cast;
            cast.name           = "to_float";
            cast.inputs         = {indices};
            cast.outputs        = {asFloat};
            cast.attr.map["to"] = intAttr(kOnnxFloat);
            g.nodes.push_back(cast);
            Node add;
            add.type    = OpType::Add;
            add.name    = "offset";
            add.inputs  = {asFloat, half};
            add.outputs = {y};
            g.nodes.push_back(add);
            return g;
        };

        Graph passed = build();
        runStandardPasses(passed);
        EXPECT_NE(findNode(passed, OpType::Cast), nullptr) << opTypeName(type) << ": the Cast of int64 indices must survive the passes";

        auto sess = Session::create(build(), cpuConfig(kSingleThread));
        ASSERT_TRUE(sess);
        std::vector<IOTensor> outs;
        ASSERT_EQ(sess->run({floatInput({2, 3}, {1, 5, 3, 9, 2, 9})}, outs), Status::Ok);
        ASSERT_EQ(outs.size(), 1u);
        ASSERT_EQ(outs[0].dtype, DType::Float32);
        ASSERT_EQ(outs[0].data.size(), 2 * sizeof(float));
        std::vector<float> y(2);
        std::memcpy(y.data(), outs[0].data.data(), sizeof(float) * 2);
        const std::vector<float> expected = type == OpType::ArgMax ? std::vector<float> {1.5f, 0.5f} : std::vector<float> {0.5f, 1.5f};
        EXPECT_EQ(y, expected) << opTypeName(type);
    }
}

TEST(ArgExtremeOps, VxmRoundTripKeepsAttributesAndInt64Indices) {
    const Shape                shape    = {2, 3, 4};
    std::vector<float>         x        = trickyValues(24, 7u);
    const std::vector<int64_t> expected = referenceArgExtreme(x, shape, -1, false, true);

    Graph g = argGraph(OpType::ArgMin, shape, DType::Float32, attrs(-1, 0, 1));
    runStandardPasses(g);
    const std::string path = testing::TempDir() + "vknn_arg_extreme_roundtrip.vxm";
    ASSERT_TRUE(saveGraphBin(g, path));

    Graph loaded;
    ASSERT_TRUE(loadGraphBin(loaded, path));
    const Node *node = findNode(loaded, OpType::ArgMin);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->attr.geti("axis", 0), -1);
    EXPECT_EQ(node->attr.geti("keepdims", 1), 0);
    EXPECT_EQ(node->attr.geti("select_last_index", 0), 1);
    ASSERT_EQ(node->outputs.size(), 1u);
    EXPECT_EQ(loaded.desc(node->outputs[0]).dtype, DType::Int64);
    EXPECT_EQ(loaded.desc(node->outputs[0]).shape, (Shape {2, 3}));

    ArgResult r;
    {
        // The session may map the file, so it is removed only after the session is gone.
        auto sess = Session::createFromVxm(path, cpuConfig(kSingleThread));
        EXPECT_TRUE(sess);
        if (sess)
        {
            r = runSession(*sess, {floatInput(shape, x)});
        }
    }
    std::remove(path.c_str());
    ASSERT_EQ(r.status, Status::Ok);
    EXPECT_EQ(r.dtype, DType::Int64);
    EXPECT_EQ(r.shape, (Shape {2, 3}));
    EXPECT_EQ(r.indices, expected);
}

TEST(CpuThreading, ArgExtremeBitExact) {
    // Partitioned over the output elements: {7, 13, 1447} along axis 1 has 7 * 1447 outputs of a
    // 13-element scan; along axis 0 one outer block of 18811 outputs is split mid-block.
    const Shape        shape = {7, 13, 1447};
    std::vector<float> x     = trickyValues((size_t) numElements(shape), 11u);
    for (OpType type: {OpType::ArgMax, OpType::ArgMin})
    {
        for (int64_t axis: {1, 0})
        {
            for (int64_t selectLast: {0, 1})
            {
                const ArgAttrs a = attrs(axis, 1, selectLast);
                expectBytesInvariantAcrossThreads(
                    [&]() {
                        return argGraph(type, shape, DType::Float32, a);
                    },
                    floatInput(shape, x));
            }
        }
        // The threaded result is the rule itself.
        const ArgResult threaded = runFloat(type, shape, x, attrs(0, 0, 1), 5);
        ASSERT_EQ(threaded.status, Status::Ok);
        EXPECT_EQ(threaded.indices, referenceArgExtreme(x, shape, 0, type == OpType::ArgMax, true)) << opTypeName(type);
    }
}

TEST(CpuThreading, ArgExtremeInt64BitExact) {
    // The int64 arm of the partitioned scan, over the same geometry as the float test: values with
    // ties and neighbours above 2^53 that only an int64 comparison separates.
    const Shape                shape = {7, 13, 1447};
    const std::vector<int64_t> x     = trickyInt64Values((size_t) numElements(shape), 23u);
    for (OpType type: {OpType::ArgMax, OpType::ArgMin})
    {
        for (int64_t axis: {1, 0})
        {
            for (int64_t selectLast: {0, 1})
            {
                const ArgAttrs a = attrs(axis, 1, selectLast);
                expectBytesInvariantAcrossThreads(
                    [&]() {
                        return argGraph(type, shape, DType::Int64, a);
                    },
                    int64Input(shape, x));
            }
        }
        for (int64_t axis: {0, 1, 2})
        {
            const ArgResult threaded = runInt64(type, shape, x, attrs(axis, 0, 1), 5);
            ASSERT_EQ(threaded.status, Status::Ok);
            EXPECT_EQ(threaded.indices, referenceArgExtreme(x, shape, axis, type == OpType::ArgMax, true)) << opTypeName(type) << " axis " << axis;
        }
    }
}

// --- the GPU kernel's host-checkable half ------------------------------------------------------------

TEST(ArgExtremePlan, LoadSequenceLeavesDataUnbridgedAndIndicesFp32) {
    // After the Vulkan load sequence the plan takes the data at its own storage precision (fp16
    // under an fp16 base) and finds the fp32 indices it requires, for data reached through a layout
    // convert (an NC4HW4 producer) as well as for flat data.
    for (OpType type: {OpType::ArgMax, OpType::ArgMin})
    {
        Graph      g;
        TensorId   x = addInput(g, "x", {1, 21, 8, 8}, DType::Float32);
        TensorDesc actDesc;
        actDesc.name   = "act";
        actDesc.shape  = {1, 21, 8, 8};
        TensorId   act = g.addTensor(actDesc);
        TensorDesc idxDesc;
        idxDesc.name     = "indices";
        idxDesc.shape    = {1, 1, 8, 8};
        idxDesc.dtype    = DType::Int64;
        idxDesc.isOutput = true;
        TensorId indices = g.addTensor(idxDesc);
        g.outputs        = {indices};
        Node relu;
        relu.type    = OpType::Relu;
        relu.name    = "relu";
        relu.inputs  = {x};
        relu.outputs = {act};
        g.nodes.push_back(relu);
        g.nodes.push_back(argNode(type, act, indices, attrs(1, 1, 1)));

        planFlatLayoutAndStorage(g, "", nullptr);
        const Node *node = findNode(g, type);
        ASSERT_NE(node, nullptr);
        EXPECT_EQ(findNode(g, OpType::ConvertDtype), nullptr) << opTypeName(type) << ": the data must not be bridged";
        const TensorId data = node->inputs[0];
        EXPECT_TRUE(g.desc(data).gpuFlat);
        EXPECT_FALSE(g.desc(data).storeFp32);
        EXPECT_TRUE(g.desc(node->outputs[0]).storeFp32);

        const ArgExtremePlan fp16Base = planArgExtreme(g, *node, true);
        EXPECT_TRUE(fp16Base.dataFp16) << opTypeName(type);
        EXPECT_FALSE(fp16Base.constantData);
        EXPECT_TRUE(fp16Base.selectLast);
        EXPECT_EQ(fp16Base.push.outer, 1);
        EXPECT_EQ(fp16Base.push.extent, 21);
        EXPECT_EQ(fp16Base.push.inner, 64);
        EXPECT_EQ(fp16Base.push.total, 64);
        EXPECT_FALSE(planArgExtreme(g, *node, false).dataFp16) << opTypeName(type);
    }
}

TEST(ArgExtremePlan, IntegerAndConstantDataReadThroughTheFp32Variant) {
    // Int64-typed data is pinned fp32 by the load sequence; constant data is uploaded fp32 by the op.
    Graph    g;
    TensorId ids          = addInput(g, "ids", {1, 64}, DType::Int64);
    TensorId indices      = addIndicesOutput(g, "indices");
    g.desc(indices).shape = {1, 1};
    g.nodes.push_back(argNode(OpType::ArgMax, ids, indices, attrs(1, 1, 0)));
    planFlatLayoutAndStorage(g, "", nullptr);
    ASSERT_NE(findNode(g, OpType::ArgMax), nullptr);
    EXPECT_FALSE(planArgExtreme(g, *findNode(g, OpType::ArgMax), true).dataFp16);

    Graph      constant;
    TensorDesc tableDesc;
    tableDesc.name          = "table";
    tableDesc.shape         = {2, 3};
    tableDesc.isInitializer = true;
    TensorId   table        = constant.addTensor(tableDesc);
    HostBuffer hb;
    hb.resizeElems(6, DType::Float32);
    constant.initializers[table]          = hb;
    TensorId constIndices                 = addIndicesOutput(constant, "indices");
    constant.desc(constIndices).shape     = {2};
    constant.desc(constIndices).gpuFlat   = true;
    constant.desc(constIndices).storeFp32 = true;
    constant.nodes.push_back(argNode(OpType::ArgMin, table, constIndices, attrs(-1, 0, 0)));
    const ArgExtremePlan plan = planArgExtreme(constant, constant.nodes[0], true);
    EXPECT_TRUE(plan.constantData);
    EXPECT_FALSE(plan.dataFp16);
}

TEST(ArgExtremePlan, ReleasedConstantPayloadIsANamedError) {
    // An earlier weight upload of the same tensor releases its host bytes; the plan refuses to decode
    // the emptied payload (which would scan zeros) instead of planning it.
    Graph      g;
    TensorDesc tableDesc;
    tableDesc.name          = "table";
    tableDesc.shape         = {2, 3};
    tableDesc.dtype         = DType::Float16;
    tableDesc.isInitializer = true;
    TensorId   table        = g.addTensor(tableDesc);
    HostBuffer hb;
    hb.resizeElems(6, DType::Float16);
    g.initializers[table]     = hb;
    TensorId indices          = addIndicesOutput(g, "indices");
    g.desc(indices).shape     = {2};
    g.desc(indices).gpuFlat   = true;
    g.desc(indices).storeFp32 = true;
    g.nodes.push_back(argNode(OpType::ArgMax, table, indices, attrs(1, 0, 0)));
    EXPECT_TRUE(planArgExtreme(g, g.nodes[0], true).constantData);

    g.initializers[table].bytes.clear();
    try
    {
        planArgExtreme(g, g.nodes[0], true);
        ADD_FAILURE() << "a released constant payload was planned";
    } catch (const Error &e)
    {
        EXPECT_EQ(e.status(), Status::RuntimeError) << e.what();
        EXPECT_NE(std::string(e.what()).find("ArgMax 'argmax': the constant data payload holds 0 bytes"), std::string::npos) << e.what();
    }
}

TEST(ArgExtremePlan, AbsentAttributesPlanTheOnnxDefaults) {
    // No axis and no select_last_index: the plan reads axis 0 and first-index ties.
    for (OpType type: {OpType::ArgMax, OpType::ArgMin})
    {
        Graph    g;
        TensorId x                = addInput(g, "x", {2, 3}, DType::Float32);
        TensorId indices          = addIndicesOutput(g, "indices");
        g.desc(x).gpuFlat         = true;
        g.desc(indices).shape     = {1, 3};
        g.desc(indices).gpuFlat   = true;
        g.desc(indices).storeFp32 = true;
        Node n;
        n.type    = type;
        n.name    = "defaults";
        n.inputs  = {x};
        n.outputs = {indices};
        g.nodes.push_back(n);
        const ArgExtremePlan plan = planArgExtreme(g, g.nodes[0], true);
        EXPECT_FALSE(plan.selectLast) << opTypeName(type);
        EXPECT_EQ(plan.push.outer, 1);
        EXPECT_EQ(plan.push.extent, 2);
        EXPECT_EQ(plan.push.inner, 3);
        EXPECT_EQ(plan.push.total, 3);
    }
}

TEST(ArgExtremePlan, PushAndSpecializationConstantsMatchTheShader) {
    EXPECT_EQ(sizeof(ArgExtremePushConstants), 4 * sizeof(int32_t)); // {outer, extent, inner, total}
    EXPECT_EQ(argExtremeSpecConstants(true, false), (std::vector<uint32_t> {1, 0}));
    EXPECT_EQ(argExtremeSpecConstants(false, true), (std::vector<uint32_t> {0, 1}));
    EXPECT_STREQ(kArgExtremeShaderStem, "arg_extreme");
    EXPECT_EQ(kArgExtremeBufferCount, 2u);
}

TEST(ArgExtremePlan, RefusesWhatTheKernelCannotComputeExactly) {
    EXPECT_EQ(planStatus({2, 3}, {2}, 1, true, true), Status::Ok);
    EXPECT_EQ(planStatus({2, 3}, {2}, 1, false, false), Status::Ok) << "an fp32 base stores every tensor fp32";
    EXPECT_EQ(planStatus({2, 3}, {2}, 1, false, true), Status::RuntimeError) << "fp16 base with unpinned indices";
    EXPECT_EQ(planStatus({2, 3}, {2}, 2, true, true), Status::InvalidArgument);
    EXPECT_EQ(planStatus({2, 0}, {2}, 1, true, true), Status::InvalidArgument);
    EXPECT_EQ(planStatus({}, {1}, 0, true, true), Status::InvalidArgument);
    EXPECT_EQ(planStatus({2, 3}, {3}, 1, true, true), Status::InvalidArgument) << "indices descriptor disagrees with the reduction";
    // Indices 0..2^24 are exact in fp32; one more axis element would need index 2^24 + 1.
    EXPECT_EQ(planStatus({1, kArgExtremeMaxExactFp32Index + 1}, {1}, 1, true, true), Status::Ok);
    EXPECT_EQ(planStatus({1, kArgExtremeMaxExactFp32Index + 2}, {1}, 1, true, true), Status::Unsupported);
    EXPECT_EQ(planStatus({65536, 65536}, {65536}, 1, true, true), Status::Unsupported) << "beyond int32 addressing";
    // The int32 addressing bound at its edge, with extent 1 so no other limit applies.
    EXPECT_EQ(planStatus({kArgExtremeMaxShaderElements, 1}, {kArgExtremeMaxShaderElements}, 1, true, true), Status::Ok);
    EXPECT_EQ(planStatus({kArgExtremeMaxShaderElements + 1, 1}, {kArgExtremeMaxShaderElements + 1}, 1, true, true), Status::Unsupported);
    EXPECT_EQ(planStatus({2, 3}, {2}, 1, true, true, PlanLayout::DataNc4), Status::RuntimeError) << "data outside the flat layout";
    EXPECT_EQ(planStatus({2, 3}, {2}, 1, true, true, PlanLayout::IndicesNc4), Status::RuntimeError) << "indices outside the flat layout";
}

TEST(ArgExtremePlan, GateRefusesExactlyTheGeometriesThePlanCannotRun) {
    // A node the Vulkan gate admits must never make the plan throw at prepare (nothing on the load
    // path catches it, so the whole session would fail); a geometry past a kernel limit is refused by
    // the gate with a named reason and runs on the CPU op instead.
    struct Case {
        Shape       dataShape;
        int64_t     axis;
        const char *refusal; // nullptr: admitted
    };
    static constexpr const char *kAddressing = "ArgMax: data elements exceed the kernel's int32 addressing";
    static constexpr const char *kExactIndex = "ArgMax: axis extent has indices beyond the exact fp32 integer range";
    const Case                   cases[]     = {
        {{2, 3}, 1, nullptr},
        {{kArgExtremeMaxShaderElements, 1}, 1, nullptr},
        {{kArgExtremeMaxShaderElements + 1, 1}, 1, kAddressing},
        {{1, 65536, 32769}, 1, kAddressing},
        {{65536, 65536}, 1, kAddressing},
        {{1, kArgExtremeMaxExactFp32Index + 1}, 1, nullptr},
        {{1, kArgExtremeMaxExactFp32Index + 2}, 1, kExactIndex},
        {{kArgExtremeMaxExactFp32Index + 2, 1}, 0, kExactIndex},
        {{kArgExtremeMaxExactFp32Index + 2, 1}, 1, nullptr},
        {{0, 3, 4}, 1, nullptr},
        {{0, 4, 65536, 65536}, 1, kAddressing}, // empty, but inner does not fit the int32 push constant
    };
    for (const Case &c: cases)
    {
        const ArgExtremeGeometry geometry = argExtremeGeometry(c.dataShape, c.axis);
        const Shape              indicesShape {geometry.outer * geometry.inner};
        const Graph              g = planGraph(c.dataShape, indicesShape, c.axis, true);
        std::string              why;
        const bool               admitted = vkNodeGate(g, g.nodes[0], &why);
        const std::string        label    = shapeStr(c.dataShape) + " axis " + std::to_string(c.axis);
        EXPECT_EQ(admitted, c.refusal == nullptr) << label << ": " << why;
        if (c.refusal)
        {
            EXPECT_EQ(why, c.refusal) << label;
        }
        const Status planned = planStatus(c.dataShape, indicesShape, c.axis, true, true);
        EXPECT_EQ(planned == Status::Ok, admitted) << label;
    }
}

TEST(ArgExtremePlan, FlatLayoutClassKeepsTheSessionLayoutPassOn) {
    // The kernel reads and writes flat row-major only. The session keeps the flat-layout pass on for
    // any graph holding a LayoutClass::Flat op, even when Hint::FlatLayout asks to skip it, so an
    // ArgMax/ArgMin node never reaches the plan without flat tensors.
    EXPECT_EQ(opDescriptor(OpType::ArgMax).layout, LayoutClass::Flat);
    EXPECT_EQ(opDescriptor(OpType::ArgMin).layout, LayoutClass::Flat);
}

TEST(ArgExtremeShader, TranscriptionMatchesCpuOracleInBothStorageVariants) {
    // For every shape, axis and mode: plan the node (push + specialization constants), run the
    // transcribed shader over its full dispatch, and compare with the CPU kernel's indices. The fp32
    // variant reads the data as uploaded; the fp16 variant reads half-precision lanes, and its oracle
    // is the CPU kernel run on the same values widened back to fp32.
    struct ShapeCase {
        Shape                shape;
        std::vector<int64_t> axes;
    };
    const ShapeCase shapes[] = {{{5, 7, 3}, {0, 1, 2, -1}}, {{1, 300}, {1, 0}}, {{4, 1, 6, 2}, {1, 2, -1, 0}}, {{9}, {0}},
                                {{2, 3, 513}, {2}},         {{2, 2600}, {1}}};
    uint32_t        seed     = 1u;
    for (const ShapeCase &sc: shapes)
    {
        const int64_t            count    = numElements(sc.shape);
        const std::vector<float> fp32Data = trickyValues((size_t) count, seed++);
        std::vector<fp16_t>      halfLanes((size_t) count);
        std::vector<float>       widenedHalf((size_t) count);
        for (int64_t i = 0; i < count; ++i)
        {
            halfLanes[(size_t) i]   = floatToHalfSat(fp32Data[(size_t) i]);
            widenedHalf[(size_t) i] = halfToFloat(halfLanes[(size_t) i]);
        }
        for (int64_t axis: sc.axes)
        {
            for (OpType type: {OpType::ArgMax, OpType::ArgMin})
            {
                for (int64_t selectLast: {0, 1})
                {
                    for (bool fp16Variant: {false, true})
                    {
                        const ArgAttrs  a          = attrs(axis, fp16Variant ? 1 : 0, selectLast);
                        const auto     &oracleData = fp16Variant ? widenedHalf : fp32Data;
                        const ArgResult oracle     = runFloat(type, sc.shape, oracleData, a);
                        ASSERT_EQ(oracle.status, Status::Ok);

                        Graph g                        = argGraph(type, sc.shape, DType::Float32, a);
                        g.desc(g.inputs[0]).gpuFlat    = true;
                        g.desc(g.outputs[0]).gpuFlat   = true;
                        g.desc(g.outputs[0]).shape     = oracle.shape;
                        g.desc(g.outputs[0]).storeFp32 = true;
                        const ArgExtremePlan plan      = planArgExtreme(g, g.nodes[0], fp16Variant);
                        ASSERT_EQ(plan.dataFp16, fp16Variant);
                        const std::vector<uint32_t> spec = argExtremeSpecConstants(type == OpType::ArgMax, plan.selectLast);

                        ShaderData readData;
                        if (plan.dataFp16)
                        {
                            readData = [&halfLanes](int i) {
                                return halfToFloat(halfLanes[(size_t) i]);
                            };
                        } else
                        {
                            readData = [&fp32Data](int i) {
                                return fp32Data[(size_t) i];
                            };
                        }
                        const std::vector<float> shaderIndices = dispatchShader(readData, plan.push, spec);
                        ASSERT_EQ(shaderIndices.size(), oracle.indices.size());
                        for (size_t k = 0; k < shaderIndices.size(); ++k)
                        {
                            ASSERT_EQ((int64_t) shaderIndices[k], oracle.indices[k]) << opTypeName(type) << " axis " << axis << " select_last_index " << selectLast << " fp16 " << fp16Variant << " output " << k;
                        }
                    }
                }
            }
        }
    }
}

TEST(ArgExtremeShader, SourceMatchesTranscriptionAndInterface) {
    const std::string root       = repositoryRoot();
    const std::string testSource = readSource(__FILE__);
    if (testSource.empty())
    {
        GTEST_SKIP() << "the source tree is not present at " << __FILE__;
    }
    const std::string shaderPath = root + "/shaders/arg_extreme.comp";
    const std::string planPath   = root + "/src/backend/vulkan/ops/arg_extreme_plan.h";
    const std::string opPath     = root + "/src/backend/vulkan/ops/arg_extreme_vk.h";
    const std::string flatPath   = root + "/src/backend/vulkan/ops/flat_ops.h";
    const std::string shader     = normalizeSource(readSource(shaderPath));
    const std::string plan       = normalizeSource(readSource(planPath));
    const std::string op         = normalizeSource(readSource(opPath));
    const std::string flat       = normalizeSource(readSource(flatPath));
    ASSERT_FALSE(shader.empty()) << shaderPath;
    ASSERT_FALSE(plan.empty()) << planPath;
    ASSERT_FALSE(op.empty()) << opPath;
    ASSERT_FALSE(flat.empty()) << flatPath;

    // The kernel body: the shader from takesCandidate to its end, translated fragment by fragment,
    // is exactly the transcription.
    const size_t bodyStart = shader.find("booltakesCandidate(");
    ASSERT_NE(bodyStart, std::string::npos);
    std::string translated = shader.substr(bodyStart);
    for (const SourceSubstitution &sub: kGlslToTranscription)
    {
        ASSERT_EQ(countOccurrences(translated, sub.glsl), sub.occurrences) << sub.glsl;
        for (size_t at = translated.find(sub.glsl); at != std::string::npos; at = translated.find(sub.glsl, at + std::strlen(sub.transcription)))
        {
            translated.replace(at, std::strlen(sub.glsl), sub.transcription);
        }
    }
    const size_t transcriptionBegin = testSource.find(kTranscriptionBegin);
    ASSERT_NE(transcriptionBegin, std::string::npos);
    const size_t transcriptionEnd = testSource.find(kTranscriptionEnd, transcriptionBegin);
    ASSERT_NE(transcriptionEnd, std::string::npos);
    const std::string transcription = normalizeSource(testSource.substr(transcriptionBegin, transcriptionEnd - transcriptionBegin));
    EXPECT_EQ(translated, transcription) << "shaders/arg_extreme.comp and its transcription diverge";

    // Precision: the variant reads its data at storage precision without the RTE store helpers, and
    // the indices buffer is fp32 in both variants (exact past fp16's 2048).
    const size_t noRte = shader.find("#defineVKNN_NO_RTE1");
    ASSERT_NE(noRte, std::string::npos);
    EXPECT_LT(noRte, shader.find("#include\"precision.glsl\""));
    EXPECT_EQ(countOccurrences(shader, "binding="), (size_t) kArgExtremeBufferCount);
    EXPECT_EQ(countOccurrences(shader, "layout(std430,binding=0)readonlybufferData{STOREdata[];};"), 1u);
    EXPECT_EQ(countOccurrences(shader, "layout(std430,binding=1)writeonlybufferIndices{floatindices[];};"), 1u);
    EXPECT_EQ(countOccurrences(op, "pipe_->dispatch(cmd,{data->handle(),env.devBuf(node.outputs[0])->handle()},"), 1u)
        << "binding 0 is the data, binding 1 the indices";
    EXPECT_EQ(countOccurrences(op, "groups(plan_.push.total,flat::kFlatLocalSize)"), 1u);
    EXPECT_EQ(countOccurrences(op, "shader(kArgExtremeShaderStem,plan_.dataFp16)"), 1u);

    // Push constants: the shader block's members, in order, are the plan struct's.
    const std::string shaderPush = between(shader, "layout(push_constant)uniformPC{", "}pc;");
    std::string       planPush   = between(plan, "structArgExtremePushConstants{", "};");
    for (size_t at = planPush.find("int32_t"); at != std::string::npos; at = planPush.find("int32_t", at))
    {
        planPush.replace(at, std::strlen("int32_t"), "int");
    }
    EXPECT_EQ(shaderPush, "intouter;intextent;intinner;inttotal;");
    EXPECT_EQ(planPush, shaderPush);

    // Specialization constants: ids 0 and 1 are the order argExtremeSpecConstants returns.
    EXPECT_EQ(countOccurrences(shader, "layout(constant_id=0)constintSELECT_LARGEST="), 1u);
    EXPECT_EQ(countOccurrences(shader, "layout(constant_id=1)constintSELECT_LAST="), 1u);
    EXPECT_EQ(countOccurrences(shader, "constant_id="), 2u);
    EXPECT_EQ(countOccurrences(op, "argExtremeSpecConstants(selectLargest_,plan_.selectLast)"), 1u);

    // Local size: the shader's, the transcription's dispatch and the op's flat::kFlatLocalSize agree.
    const std::string localSize = std::to_string(kShaderLocalSize);
    EXPECT_EQ(countOccurrences(shader, "layout(local_size_x=" + localSize + ")in;"), 1u);
    EXPECT_EQ(countOccurrences(flat, "kFlatLocalSize=" + localSize + ";"), 1u);
}
