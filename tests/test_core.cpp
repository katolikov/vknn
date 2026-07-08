// vknn unit tests (host): dtype/fp16, config JSON, graph passes, layout packing math, and the
// ergonomic Session API. Operator correctness lives in test_ops.cpp; Vulkan correctness is
// validated on-device (see scripts).
#include "core/matmul_tile.h"
#include "vknn/config.h"
#include "vknn/dtype.h"
#include "vknn/graph.h"
#include "vknn/host_buffer.h"
#include "vknn/model.h"
#include "vknn/op_type.h"
#include "vknn/reduce_type.h"
#include "vknn/session.h"
#include "vknn/tensor_format.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

using namespace vknn;

TEST(DType, HalfRoundTrip) {
    for (float v: {0.f, 1.f, -1.f, 0.5f, 6.f, 3.14159f, -2.71828f, 65504.f})
    {
        float r = halfToFloat(floatToHalf(v));
        EXPECT_NEAR(r, v, std::fabs(v) * 0.01f + 1e-3f) << "v=" << v;
    }
    EXPECT_EQ(dtypeSize(DType::Float32), 4u);
    EXPECT_EQ(dtypeSize(DType::Float16), 2u);
}

namespace {
    float bitsToFloat(uint32_t b) {
        float f;
        std::memcpy(&f, &b, 4);
        return f;
    }
    // floatToHalfBulk() equals floatToHalf() bit-for-bit at every length, so both the vector body and the
    // scalar tail are exercised.
    void expectBulkMatchesScalar(const std::vector<float> &v) {
        std::vector<fp16_t> bulk(v.size(), 0xDEAD);
        for (size_t len = 0; len <= v.size(); ++len)
        {
            floatToHalfBulk(v.data(), bulk.data(), (int64_t) len);
            for (size_t i = 0; i < len; ++i)
            {
                uint32_t b;
                std::memcpy(&b, &v[i], 4);
                ASSERT_EQ(bulk[i], floatToHalf(v[i])) << "len=" << len << " i=" << i << " bits=0x" << std::hex << b;
            }
        }
    }
} // namespace

// Every fp16 bit pattern, widened and narrowed back, returns to itself -- except a NaN, whose payload and
// signaling bit floatToHalf collapses onto the quiet pattern by contract.
TEST(DType, HalfRoundTripAllPatterns) {
    for (uint32_t h = 0; h < 0x10000u; ++h)
    {
        fp16_t   half = (fp16_t) h;
        fp16_t   back = floatToHalf(halfToFloat(half));
        uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
        if (exp == 0x1F && mant != 0)
        {
            EXPECT_EQ(back, (fp16_t) ((h & 0x8000u) | 0x7E00u)) << "nan h=0x" << std::hex << h;
        } else
        {
            EXPECT_EQ(back, half) << "h=0x" << std::hex << h;
        }
    }
}

// The vectorized narrowing reproduces the scalar one exactly on the values that separate the rounding
// rules: the tie cases, the subnormal and flush-to-zero boundaries, the overflow boundary, and NaN.
TEST(DType, FloatToHalfBulkMatchesScalarOnBoundaries) {
    std::vector<float> v = {
            0.f, -0.f, 1.f, -1.f, 0.5f, 6.f, 3.14159f, -2.71828f,
            65504.f,                                            // largest finite half
            65519.996f, 65520.f,                                // the last value below the tie, and the tie that saturates
            65535.f, 65536.f, 65537.f,                          // at and past 2^16, where the exponent branch trips
            -65504.f, -65520.f, -65536.f,                       //
            6.10352e-5f, 6.09756e-5f,                           // smallest normal half, and the largest subnormal below it
            5.96046e-8f,                                        // smallest positive subnormal half (2^-24)
            2.98023e-8f, -2.98023e-8f,                          // 2^-25: the tie between zero and the smallest subnormal
            1.49012e-8f,                                        // 2^-26: below every tie, flushes to zero
            bitsToFloat(0x7F800000u), bitsToFloat(0xFF800000u), // +/- inf
            bitsToFloat(0x7FC00000u),                           // quiet nan
            bitsToFloat(0x7F800001u), bitsToFloat(0xFF800001u), // signaling nan, both signs
            bitsToFloat(0x7FFFFFFFu),                           // nan, every payload bit set
    };
    // Exact ties on the dropped mantissa bits (low 13 bits == 0x1000) over even and odd retained
    // mantissas: floatToHalf rounds both away from zero rather than to even.
    for (uint32_t retained: {0u, 1u, 2u, 3u, 0x3FEu, 0x3FFu})
    {
        for (uint32_t e: {113u, 127u, 142u})
        {
            uint32_t bits = (e << 23) | (retained << 13) | 0x1000u;
            v.push_back(bitsToFloat(bits));
            v.push_back(bitsToFloat(bits | 0x80000000u));
            v.push_back(bitsToFloat(bits + 1u)); // just above the tie
            v.push_back(bitsToFloat(bits - 1u)); // just below the tie
        }
    }
    // Exponents around the subnormal-half window, where the significand takes the variable-shift path.
    for (uint32_t e = 100; e <= 113; ++e)
    {
        for (uint32_t m: {0u, 0x1000u, 0x400000u, 0x7FFFFFu})
        {
            v.push_back(bitsToFloat((e << 23) | m));
            v.push_back(bitsToFloat(0x80000000u | (e << 23) | m));
        }
    }
    expectBulkMatchesScalar(v);
}

// The saturating boundary conversion (data entering fp16 COMPUTE storage): finite and infinite
// values beyond +/-65504 narrow to the largest finite half instead of +/-inf — mirroring GPU
// activation stores (vxSatF16) and imported constants (clampToFp16Range) — NaN passes through, and
// in-range values are bit-identical to the plain conversion. The bulk form matches the scalar one
// at every length (vector body + chunk boundary + scalar tail).
TEST(DType, FloatToHalfSatSaturatesInsteadOfInf) {
    const fp16_t maxFinite = floatToHalf(65504.f); // 0x7BFF
    EXPECT_EQ(floatToHalfSat(65505.f), maxFinite);
    EXPECT_EQ(floatToHalfSat(1e9f), maxFinite);
    EXPECT_EQ(floatToHalfSat(bitsToFloat(0x7F800000u)), maxFinite); // +inf
    EXPECT_EQ(floatToHalfSat(-65505.f), (fp16_t) (0x8000u | maxFinite));
    EXPECT_EQ(floatToHalfSat(bitsToFloat(0xFF800000u)), (fp16_t) (0x8000u | maxFinite)); // -inf
    EXPECT_EQ(floatToHalfSat(bitsToFloat(0x7FC00000u)), floatToHalf(bitsToFloat(0x7FC00000u))); // NaN passes
    for (float in: {0.f, -1.f, 3.14159f, 65504.f, -65504.f, 6.1e-5f})
    {
        EXPECT_EQ(floatToHalfSat(in), floatToHalf(in)) << "in-range must be bit-identical, v=" << in;
    }
    // Bulk == scalar at lengths spanning the 256-element chunk boundary and a non-multiple-of-4 tail.
    std::vector<float> v(517);
    uint32_t           s = 0xC0FFEEu;
    for (size_t i = 0; i < v.size(); ++i)
    {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        v[i] = (i % 7 == 0) ? 1e8f * ((i & 1) ? -1.f : 1.f) : bitsToFloat(s);
    }
    std::vector<fp16_t> bulk(v.size(), 0xDEAD);
    for (size_t len: {0u, 1u, 3u, 255u, 256u, 257u, 517u})
    {
        floatToHalfSatBulk(v.data(), bulk.data(), (int64_t) len);
        for (size_t i = 0; i < len; ++i)
        {
            ASSERT_EQ(bulk[i], floatToHalfSat(v[i])) << "len=" << len << " i=" << i;
        }
    }
}

// A pseudorandom sweep of the fp32 bit space, covering the exponent/mantissa combinations the boundary
// list does not name. The length is not a multiple of four, so the scalar tail runs.
TEST(DType, FloatToHalfBulkMatchesScalarOnRandomBits) {
    std::vector<float> v(4093);
    uint32_t           s = 0x12345678u;
    for (size_t i = 0; i < v.size(); ++i)
    {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        v[i] = bitsToFloat(s);
    }
    std::vector<fp16_t> bulk(v.size(), 0);
    floatToHalfBulk(v.data(), bulk.data(), (int64_t) v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        ASSERT_EQ(bulk[i], floatToHalf(v[i])) << "i=" << i;
    }
}

// resizeElems() zero-fills whenever the byte size moves, and leaves both the bytes and the allocation
// alone when it does not.
TEST(DType, HostBufferResizeElemsSemantics) {
    HostBuffer hb;
    hb.resizeElems(4, DType::Float32);
    EXPECT_EQ(hb.bytes.size(), 16u);
    for (uint8_t b: hb.bytes)
    {
        EXPECT_EQ(b, 0u); // a grow from empty zero-fills
    }
    for (int i = 0; i < 4; ++i)
    {
        hb.f32()[i] = (float) (i + 1);
    }
    const void *before = hb.bytes.data();
    hb.resizeElems(4, DType::Float32); // the same byte size: contents and allocation survive
    EXPECT_EQ((const void *) hb.bytes.data(), before);
    EXPECT_EQ(hb.f32()[3], 4.f);
    hb.resizeElems(8, DType::Float16); // the same byte size through a different dtype: still no wipe
    EXPECT_EQ(hb.bytes.size(), 16u);
    EXPECT_EQ(hb.f32()[3], 4.f);
    hb.resizeElems(8, DType::Float32); // grow: zero-filled
    EXPECT_EQ(hb.bytes.size(), 32u);
    for (uint8_t b: hb.bytes)
    {
        EXPECT_EQ(b, 0u);
    }
    hb.f32()[0] = 7.f;
    hb.resizeElems(2, DType::Float32); // shrink: zero-filled
    EXPECT_EQ(hb.bytes.size(), 8u);
    EXPECT_EQ(hb.f32()[0], 0.f);
    hb.resizeElems(0, DType::Float32); // a rank-0 shape reports 0 elements; the buffer empties
    EXPECT_TRUE(hb.bytes.empty());
    hb.resizeElems(-1, DType::Float32); // a negative count empties rather than overflowing the byte size
    EXPECT_TRUE(hb.bytes.empty());
}

TEST(Config, JsonRoundTrip) {
    Config c;
    c.backend        = BackendKind::Vulkan;
    c.precision      = Precision::Low;
    c.maxSubmitNodes = 250;
    c.profile        = true;
    c.tuning         = Tuning::Heavy;
    c.setHint(Hint::Winograd, (int) Mode::Off);
    c.setHint(Hint::FlatLayout, (int) Mode::Off); // exercise a non-default flat/fold hint
    std::string js = c.toJson();
    Config      d  = Config::fromJsonString(js);
    EXPECT_EQ(d.backend, BackendKind::Vulkan);
    EXPECT_EQ(d.precision, Precision::Low);
    EXPECT_EQ(d.maxSubmitNodes, 250);
    EXPECT_TRUE(d.profile);
    EXPECT_EQ(d.tuning, Tuning::Heavy);
    EXPECT_FALSE(d.flatLayout());   // --no-flat round-trips through the hint
    EXPECT_TRUE(d.gpuIslandFold()); // default stays on
    EXPECT_EQ(precisionFromStr("normal"), Precision::Normal);
    EXPECT_EQ(precisionFromStr("high"), Precision::High);
    EXPECT_EQ(precisionFromStr("low"), Precision::Low);
    EXPECT_EQ(d.hint(Hint::Winograd, 0), (int) Mode::Off);
}

TEST(Config, ParseExplicit) {
    Config c = Config::fromJsonString(R"({"backend":"CPU","precision":"fp32","fallback":["VULKAN","CPU"],"cacheDir":"/tmp/x"})");
    EXPECT_EQ(c.backend, BackendKind::Cpu);
    EXPECT_EQ(c.precision, Precision::High);
    EXPECT_EQ(c.cacheDir, "/tmp/x");
    ASSERT_EQ(c.fallback.size(), 2u);
    EXPECT_EQ(c.fallback[0], BackendKind::Vulkan);
}

TEST(Config, ListContainsWholeNameMatch) {
    // disableVkOps entries match whole op-type names: "Conv" leaves ConvTranspose/ConvGemm/
    // ConvertLayout enabled even though Conv is a prefix of all three.
    EXPECT_TRUE(Config::listContains("Conv", opTypeName(OpType::Conv)));
    EXPECT_FALSE(Config::listContains("Conv", opTypeName(OpType::ConvTranspose)));
    EXPECT_FALSE(Config::listContains("Conv", opTypeName(OpType::ConvGemm)));
    EXPECT_FALSE(Config::listContains("Conv", opTypeName(OpType::ConvertLayout)));

    // Multi-entry list disables exactly the named ops; entries are trimmed of whitespace.
    const std::string list = " Conv , MatMul ";
    EXPECT_TRUE(Config::listContains(list, opTypeName(OpType::Conv)));
    EXPECT_TRUE(Config::listContains(list, opTypeName(OpType::MatMul)));
    EXPECT_FALSE(Config::listContains(list, opTypeName(OpType::ConvTranspose)));
    EXPECT_FALSE(Config::listContains(list, opTypeName(OpType::Gemm)));
    EXPECT_FALSE(Config::listContains(list, opTypeName(OpType::Add)));

    // Reverse containment: an entry never disables an op whose name is its substring.
    EXPECT_TRUE(Config::listContains("PRelu", opTypeName(OpType::PRelu)));
    EXPECT_FALSE(Config::listContains("PRelu", opTypeName(OpType::Relu)));

    // Empty list / empty entries match nothing.
    EXPECT_FALSE(Config::listContains("", "Conv"));
    EXPECT_FALSE(Config::listContains(",,", "Conv"));
}

TEST(OpTypes, ReduceNamesSingleSource) {
    // reduceFromOnnx is the single source of reduce-family name recognition: opTypeFromOnnx
    // classifies a name as OpType::Reduce exactly when reduceFromOnnx recognizes it. Candidates
    // cover every current member plus unsupported Reduce* spellings and non-reduce names.
    for (const char *name: {"ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL2", "ReduceL1", "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare", "Reduce", "Conv"})
    {
        const bool isReduce = reduceFromOnnx(name) != ReduceType::Invalid;
        EXPECT_EQ(opTypeFromOnnx(name) == OpType::Reduce, isReduce) << "name=" << name;
    }
}

TEST(Layout, PackMath) {
    EXPECT_EQ(cBlocks(3), 1);
    EXPECT_EQ(cBlocks(4), 1);
    EXPECT_EQ(cBlocks(5), 2);
    EXPECT_EQ(cBlocks(1280), 320);
    NCHW s = NCHW::from({1, 32, 7, 7});
    EXPECT_EQ(s.n, 1);
    EXPECT_EQ(s.c, 32);
    EXPECT_EQ(s.h, 7);
}

// matmul_tiled tile-candidate table invariants. The tune table persists a winning INDEX into
// kMatMulTiles, and the shader sizes its shared/register arrays for the widest variant, so every
// entry must respect the shader's TM_MAX/TN_MAX/TK_MAX bounds and the cooperative-load loop
// divisibility (matmul_tiled.comp assigns one panel element per thread per iteration).
TEST(MatMulTile, CandidateTable) {
    // Index 0 is the Tuning::None default and must stay the pre-race fixed geometry: it is what
    // --tuning none dispatches and what a race-less build must byte-match.
    EXPECT_EQ(kMatMulTiles[0].tm, 128);
    EXPECT_EQ(kMatMulTiles[0].tn, 128);
    EXPECT_EQ(kMatMulTiles[0].tk, 16);
    EXPECT_GE(kMatMulTileCount, 1);
    for (int i = 0; i < kMatMulTileCount; ++i)
    {
        const MatMulTile &t = kMatMulTiles[i];
        EXPECT_LE(t.tm, 128) << "i=" << i; // TM_MAX
        EXPECT_LE(t.tn, 128) << "i=" << i; // TN_MAX
        EXPECT_LE(t.tk, 16) << "i=" << i;  // TK_MAX
        EXPECT_GE(t.tm, 16) << "i=" << i;  // RM >= 1 at the fixed 16x16 workgroup
        EXPECT_GE(t.tn, 16) << "i=" << i;
        EXPECT_GE(t.tk, 1) << "i=" << i;
        EXPECT_EQ(t.tm % 16, 0) << "i=" << i;
        EXPECT_EQ(t.tn % 16, 0) << "i=" << i;
        EXPECT_EQ((t.tm * t.tk) % 256, 0) << "i=" << i; // A-panel load loop
        EXPECT_EQ((t.tk * t.tn) % 256, 0) << "i=" << i; // B-panel load loop
    }
}

// The Vulkan MatMul op routes the default tile to the compile-time matmul_tiled_fast kernel and
// every other candidate to the spec-constant matmul_tiled kernel. isDefaultMatMulTile is that
// routing predicate: it must match index 0 (the {128,128,16} default) and reject every other tile.
TEST(MatMulTile, DefaultTileRouting) {
    EXPECT_TRUE(isDefaultMatMulTile(kMatMulTiles[0]));
    for (int i = 1; i < kMatMulTileCount; ++i)
    {
        EXPECT_FALSE(isDefaultMatMulTile(kMatMulTiles[i])) << "i=" << i;
    }
    // The predicate keys on the exact {128,128,16} geometry, not just any 128x128 tile.
    EXPECT_TRUE(isDefaultMatMulTile({128, 128, 16}));
    EXPECT_FALSE(isDefaultMatMulTile({128, 128, 8}));
    EXPECT_FALSE(isDefaultMatMulTile({64, 128, 16}));
    EXPECT_FALSE(isDefaultMatMulTile({128, 64, 16}));
}

// Ergonomic Tensor API: construct, shape/size accessors, argmax.
TEST(Api, TensorHelpers) {
    Tensor t({1.f, 5.f, 2.f, 9.f, 3.f, 0.f}, {1, 6});
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.size(), 6);
    EXPECT_EQ(t.dim(1), 6);
    EXPECT_EQ(t.shapeString(), "1x6");
    EXPECT_EQ(t.argmax(), 3);
    EXPECT_NEAR(t.max(), 9.f, 1e-6);
    Tensor flat(std::vector<float> {1.f, 2.f, 3.f});
    EXPECT_EQ(flat.rank(), 1);
    EXPECT_EQ(flat.shapeString(), "3");
}

// Session::run validates a caller-provided IOTensor shape against the planned buffers: buffer
// sizes, push constants, and dispatch geometry are frozen at plan() time, so a shape whose packed
// footprint differs from the plan is rejected instead of overrunning the boundary buffer at pack
// time. An empty caller shape adopts the planned shape; the planned shape itself runs.
TEST(Api, RunRejectsMismatchedInputShape) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 2, 2, 2};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorDesc yo;
    yo.name     = "y";
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     n;
    n.type    = OpType::Relu;
    n.name    = "relu";
    n.inputs  = {x};
    n.outputs = {y};
    g.nodes.push_back(n);
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);

    auto makeIn = [](const Shape &shape) {
        IOTensor in;
        in.name  = "x";
        in.shape = shape;
        in.dtype = DType::Float32;
        in.data.assign((size_t) numElements(shape) * 4, 0);
        return in;
    };
    std::vector<IOTensor> outs;
    // Larger spatial footprint than the planned [1,2,2,2] buffer: rejected, not adopted.
    EXPECT_EQ(sess->run({makeIn({1, 2, 4, 4})}, outs), Status::InvalidArgument);
    // A leading-dim (batch) mismatch is rejected the same way.
    EXPECT_EQ(sess->run({makeIn({2, 2, 2, 2})}, outs), Status::InvalidArgument);
    // An N/C/spatial-product-preserving reshape fits (the CPU dynamic-reshape contract).
    EXPECT_EQ(sess->run({makeIn({1, 2, 4, 1})}, outs), Status::Ok);
    // The planned shape runs.
    EXPECT_EQ(sess->run({makeIn({1, 2, 2, 2})}, outs), Status::Ok);
    // An empty caller shape adopts the planned shape.
    IOTensor noShape = makeIn({1, 2, 2, 2});
    noShape.shape.clear();
    EXPECT_EQ(sess->run({noShape}, outs), Status::Ok);
}

// boundShapeCompatible: the per-consumer input-rebinding contracts behind validateInputShape.
// The frozen-plan (GPU-consumed) branch demands byte-identical packing; the CPU branch only a
// footprint fit. The frozen NC4 case is the wrong-values class: equal PADDED footprints with a
// different channel split pack differently, and frozen kernels then misread every channel.
TEST(Api, BoundShapeCompatibleContracts) {
    // Frozen plan, NC4HW4 store: N, C and the spatial product must match.
    EXPECT_TRUE(boundShapeCompatible({1, 2, 4, 1}, {1, 2, 2, 2}, false, true));  // spatial reshape
    EXPECT_TRUE(boundShapeCompatible({1, 2, 1, 4}, {1, 2, 2, 2}, false, true));
    // [1,1,2,4] and [1,2,2,4] have the SAME padded NC4 footprint (both one 4-lane channel block x
    // 8 pixels = 32) but a different channel count: must be rejected under a frozen plan.
    EXPECT_TRUE(formatElems(TensorFormat::NC4HW4, NCHW::from({1, 1, 2, 4})) == formatElems(TensorFormat::NC4HW4, NCHW::from({1, 2, 2, 4})));
    EXPECT_FALSE(boundShapeCompatible({1, 1, 2, 4}, {1, 2, 2, 4}, false, true));
    EXPECT_FALSE(boundShapeCompatible({2, 1, 2, 2}, {1, 2, 2, 2}, false, true)); // N<->C swap
    // Frozen plan, flat store: dense row-major, any equal element count reinterprets losslessly.
    EXPECT_TRUE(boundShapeCompatible({4, 2}, {2, 4}, true, true));
    EXPECT_FALSE(boundShapeCompatible({4, 2}, {2, 5}, true, true));
    // CPU-consumed: the loose footprint-fit contract (ops follow the runtime shape).
    EXPECT_TRUE(boundShapeCompatible({1, 1, 2, 4}, {1, 2, 2, 4}, false, false)); // fits the alloc
    EXPECT_TRUE(boundShapeCompatible({2, 8}, {2, 6}, false, false));             // both pad to 2x8
    EXPECT_FALSE(boundShapeCompatible({2, 9}, {2, 6}, false, false));            // 3 blocks > 2
}

// Ergonomic API: infer()/inputInfo() — caller passes only data, metadata comes from the model.
TEST(Api, AutoShapesFromModel) {
    // input[1,2,1,1] -> Conv 1x1 (weight 2*I, bias {-3,0}) -> y
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {1, 2, 1, 1};
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorDesc wi;
    wi.name          = "w";
    wi.shape         = {2, 2, 1, 1};
    wi.isInitializer = true;
    TensorId   w     = g.addTensor(wi);
    HostBuffer wb;
    wb.resizeElems(4, DType::Float32);
    wb.f32()[0]       = 2;
    wb.f32()[1]       = 0;
    wb.f32()[2]       = 0;
    wb.f32()[3]       = 2;
    g.initializers[w] = wb;
    TensorDesc bi;
    bi.name          = "b";
    bi.shape         = {2};
    bi.isInitializer = true;
    TensorId   b     = g.addTensor(bi);
    HostBuffer bb;
    bb.resizeElems(2, DType::Float32);
    bb.f32()[0]       = -3;
    bb.f32()[1]       = 0;
    g.initializers[b] = bb;
    TensorDesc yo;
    yo.name     = "y";
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     conv;
    conv.type    = OpType::Conv;
    conv.name    = "conv";
    conv.inputs  = {x, w, b};
    conv.outputs = {y};
    g.nodes.push_back(conv);
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    // query metadata instead of hand-specifying it
    auto in = sess->inputInfo();
    ASSERT_EQ(in.size(), 1u);
    EXPECT_EQ(in[0].name, "x");
    EXPECT_EQ(in[0].elems, 2);
    EXPECT_EQ(sess->outputInfo()[0].elems, 2);
    // infer() with just data
    std::vector<float> out = sess->infer({1.0f, 5.0f}); // -> {2*1-3, 2*5} = {-1, 10}
    ASSERT_EQ(out.size(), 2u);
    EXPECT_NEAR(out[0], -1.0f, 1e-5);
    EXPECT_NEAR(out[1], 10.0f, 1e-5);
}
