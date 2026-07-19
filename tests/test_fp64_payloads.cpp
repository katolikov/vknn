// float64 tensors on the CPU path. The Fp64Payloads cases cover storage / serialization / IO: a DOUBLE
// initializer keeps native 8-byte storage (round-trips byte-identically through a .vxm; initDoubles
// recovers it, initFloats narrows it), and a DOUBLE graph input/output round-trips as fp64 at the
// session boundary. The Fp64Compute cases cover the fp64 compute path (Det/Sign/Cast/Binary), each with
// a value where fp32 and fp64 diverge, so a value beyond fp32 precision distinguishes the two.
#include "import/passes.h"
#include "vknn/dtype.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    // A value that is NOT representable in fp32: its low mantissa bits are lost when narrowed to float,
    // so it exposes any accidental fp32 round-trip. Exact in fp64.
    constexpr double kBeyondFp32 = 1.0 + 1.0 / 1099511627776.0; // 1 + 2^-40

    // A 2x2 matrix whose determinant is exactly -1: 1000001*999999 - 1000000*1000000 = (1e12-1) - 1e12.
    // The entries are fp32-exact (all < 2^24), but the ~1e12 products overflow fp32's 24-bit mantissa, so
    // the fp32 determinant is rounding garbage while the fp64 determinant is -1.
    const double kDetMatrix[4] = {1000001.0, 1000000.0, 1000000.0, 999999.0};

    // Build "x"[2,2] (dtype dt) -> Det -> "y"[1] (dtype dt), run on the CPU backend feeding `m` (four
    // doubles, narrowed to the input dtype), and return the single determinant element as a double.
    double runDet(DType dt, const double m[4]) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {2, 2};
        xi.dtype   = dt;
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs.push_back(x);
        TensorDesc yo;
        yo.name     = "y";
        yo.shape    = {1};
        yo.dtype    = dt;
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     det;
        det.type    = OpType::Det;
        det.name    = "det";
        det.inputs  = {x};
        det.outputs = {y};
        g.nodes.push_back(det);
        g.outputs = {y};

        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        if (!sess)
        {
            return 0;
        }
        IOTensor in;
        in.name  = "x";
        in.shape = {2, 2};
        in.dtype = dt;
        if (dt == DType::Float64)
        {
            in.data.resize(4 * sizeof(double));
            std::memcpy(in.data.data(), m, 4 * sizeof(double));
        } else
        {
            in.data.resize(4 * sizeof(float));
            for (int i = 0; i < 4; ++i)
            {
                reinterpret_cast<float *>(in.data.data())[i] = (float) m[i];
            }
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        EXPECT_FALSE(outs.empty());
        if (outs.empty())
        {
            return 0;
        }
        EXPECT_EQ(outs[0].dtype, dt);
        if (dt == DType::Float64)
        {
            return reinterpret_cast<const double *>(outs[0].data.data())[0];
        }
        return (double) outs[0].f32()[0];
    }

    // Add a native-fp64 initializer holding `vals` (8 bytes/elem), mirroring what the importer's
    // fillHostDouble does.
    TensorId addF64Init(Graph &g, const char *name, const Shape &shape, const std::vector<double> &vals) {
        TensorDesc d;
        d.name          = name;
        d.shape         = shape;
        d.dtype         = DType::Float64;
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) vals.size(), DType::Float64);
        for (size_t i = 0; i < vals.size(); ++i)
        {
            hb.f64()[i] = vals[i];
        }
        g.initializers[id] = std::move(hb);
        return id;
    }

} // namespace

TEST(Fp64Payloads, DtypeSizeAndMnemonic) {
    EXPECT_EQ(dtypeSize(DType::Float64), 8u);
    EXPECT_STREQ(dtypeStr(DType::Float64), "f64");
}

// A fp64 initializer occupies 8 bytes/elem. initDoubles recovers every bit; initFloats narrows to
// fp32. The beyond-fp32 value confirms the storage is fp64, not fp32.
TEST(Fp64Payloads, NativeInitializerStorageAndDecode) {
    Graph    g;
    TensorId id = addF64Init(g, "c", {3}, {1.5, -2.25, kBeyondFp32});

    ASSERT_EQ(g.desc(id).dtype, DType::Float64);
    ASSERT_EQ(g.initializers[id].bytes.size(), 3u * sizeof(double));

    std::vector<double> d = initDoubles(g, id);
    ASSERT_EQ(d.size(), 3u);
    EXPECT_EQ(d[0], 1.5);
    EXPECT_EQ(d[1], -2.25);
    EXPECT_EQ(d[2], kBeyondFp32);          // full fp64 precision preserved in storage
    EXPECT_NE(d[2], (double) (float) kBeyondFp32); // and it is genuinely beyond fp32

    std::vector<float> f = initFloats(g, id);
    ASSERT_EQ(f.size(), 3u);
    EXPECT_EQ(f[0], 1.5f);
    EXPECT_EQ(f[1], -2.25f);
    EXPECT_EQ(f[2], (float) kBeyondFp32);  // fp32 compute path sees the narrowed value
}

// hostInitElem reads a single element at the stored dtype without reinterpreting fp64 bytes as fp32.
TEST(Fp64Payloads, HostInitElemHonorsFp64) {
    Graph    g;
    TensorId id = addF64Init(g, "c", {2}, {kBeyondFp32, 7.0});
    EXPECT_EQ(hostInitElem(g.initializers[id], DType::Float64, 0), kBeyondFp32);
    EXPECT_EQ(hostInitElem(g.initializers[id], DType::Float64, 1), 7.0);
}

// A DOUBLE initializer survives a .vxm save/load byte-identically: the serializer stores the raw
// 8-byte payload and the dtype, so the reload recovers the exact fp64 bits (no fp32 round-trip).
TEST(Fp64Payloads, VxmRoundTripLossless) {
    Graph    g;
    TensorId c = addF64Init(g, "c", {4}, {kBeyondFp32, -kBeyondFp32, 3.0, 0.5});
    // A .vxm needs a graph shell; a single Identity from a declared input to a declared output holds
    // the initializer as a live tensor so it serializes.
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {4};
    xi.isInput = true;
    xi.dtype   = DType::Float64;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorDesc yo;
    yo.name     = "y";
    yo.shape    = {4};
    yo.dtype    = DType::Float64;
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     id;
    id.type    = OpType::Identity;
    id.name    = "id";
    id.inputs  = {x};
    id.outputs = {y};
    g.nodes.push_back(id);
    g.outputs = {y};

    const std::vector<uint8_t> before = g.initializers[c].bytes.toVector();

    std::string path = testing::TempDir() + "vknn_fp64_payloads_roundtrip.vxm";
    ASSERT_TRUE(saveGraphBin(g, path));
    Graph g2;
    ASSERT_TRUE(loadGraphBin(g2, path));
    std::remove(path.c_str());

    // Find the reloaded initializer by name and compare payload + dtype.
    TensorId c2 = g2.tensorByName.at("c");
    ASSERT_EQ(g2.desc(c2).dtype, DType::Float64);
    const std::vector<uint8_t> after = g2.initializers[c2].bytes.toVector();
    ASSERT_EQ(after.size(), before.size());
    EXPECT_EQ(std::memcmp(after.data(), before.data(), before.size()), 0); // byte-identical fp64
}

// A DOUBLE graph input/output round-trips as fp64 at the session boundary. The value passes through an
// Identity (a byte-mover), so an fp32-exact value round-trips exactly and the readback emits 8-byte
// fp64 (outs[0].dtype == Float64).
TEST(Fp64Payloads, IoReadbackEmitsDouble) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {2, 2};
    xi.dtype   = DType::Float64;
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorDesc yo;
    yo.name     = "y";
    yo.shape    = {2, 2};
    yo.dtype    = DType::Float64;
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     id;
    id.type    = OpType::Identity;
    id.name    = "id";
    id.inputs  = {x};
    id.outputs = {y};
    g.nodes.push_back(id);
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);

    const double         xv[4] = {1.5, -2.25, 3.0, 0.5}; // all fp32-exact
    IOTensor             in;
    in.name  = "x";
    in.shape = {2, 2};
    in.dtype = DType::Float64;
    in.data.resize(sizeof(xv));
    std::memcpy(in.data.data(), xv, sizeof(xv));

    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_FALSE(outs.empty());
    EXPECT_EQ(outs[0].dtype, DType::Float64);           // emitted as real fp64
    ASSERT_EQ(outs[0].data.size(), sizeof(xv));         // 8 bytes/elem
    const double *o = reinterpret_cast<const double *>(outs[0].data.data());
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(o[i], xv[i]) << "i=" << i;             // exact round-trip for fp32-representable values
    }
}

// ---- Phase B: real fp64 COMPUTE (the SVD / camera-head path) ----

// Det of a fp64 input matrix computes in double: kDetMatrix's determinant is -1, which only fp64
// recovers; the fp32 path returns rounding garbage far from -1.
TEST(Fp64Compute, DetIsRealDoubleNotFp32) {
    EXPECT_EQ(runDet(DType::Float64, kDetMatrix), -1.0);
    EXPECT_GT(std::abs(runDet(DType::Float32, kDetMatrix) + 1.0), 100.0);
}

// Sign of the fp64 determinant. det(kDetMatrix) = -1 in fp64 -> Sign = -1; the fp32 determinant is
// positive garbage -> Sign = +1. Det and Sign stay two ops for fp64 (pointwise fusion excludes it),
// so the sign follows the fp64 determinant.
TEST(Fp64Compute, SignOfFp64Determinant) {
    auto run = [](DType dt) {
        Graph      g;
        TensorDesc xi;
        xi.name    = "x";
        xi.shape   = {2, 2};
        xi.dtype   = dt;
        xi.isInput = true;
        TensorId x = g.addTensor(xi);
        g.inputs.push_back(x);
        TensorDesc di;
        di.name    = "d";
        di.shape   = {1};
        di.dtype   = dt;
        TensorId d = g.addTensor(di);
        TensorDesc yo;
        yo.name     = "y";
        yo.shape    = {1};
        yo.dtype    = dt;
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     det;
        det.type    = OpType::Det;
        det.name    = "det";
        det.inputs  = {x};
        det.outputs = {d};
        g.nodes.push_back(det);
        Node sign;
        sign.type    = OpType::Unary;
        sign.name    = "sign";
        sign.subOp   = (int) UnaryType::Sign;
        sign.inputs  = {d};
        sign.outputs = {y};
        g.nodes.push_back(sign);
        g.outputs = {y};

        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        IOTensor in;
        in.name  = "x";
        in.shape = {2, 2};
        in.dtype = dt;
        in.data.resize(4 * sizeof(double));
        std::memcpy(in.data.data(), kDetMatrix, 4 * sizeof(double));
        if (dt == DType::Float32)
        {
            in.data.resize(4 * sizeof(float));
            for (int i = 0; i < 4; ++i)
            {
                reinterpret_cast<float *>(in.data.data())[i] = (float) kDetMatrix[i];
            }
        }
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({in}, outs), Status::Ok);
        EXPECT_FALSE(outs.empty());
        if (dt == DType::Float64)
        {
            return reinterpret_cast<const double *>(outs[0].data.data())[0];
        }
        return (double) outs[0].f32()[0];
    };
    EXPECT_EQ(run(DType::Float64), -1.0); // sign of the fp64 determinant (-1)
    EXPECT_EQ(run(DType::Float32), 1.0);  // fp32 determinant is positive garbage
}

// Cast is the fp32->fp64 bridge: fp32-exact entries cast up to fp64, then Det computes in double. A
// graph enters the fp64 island from a fp32 input via Cast.
TEST(Fp64Compute, CastFp32ToFp64ThenDet) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {2, 2};
    xi.dtype   = DType::Float32; // fp32-exact entries
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    TensorDesc ci;
    ci.name    = "xd";
    ci.shape   = {2, 2};
    ci.dtype   = DType::Float64;
    TensorId xd = g.addTensor(ci);
    TensorDesc yo;
    yo.name     = "y";
    yo.shape    = {1};
    yo.dtype    = DType::Float64;
    yo.isOutput = true;
    TensorId y  = g.addTensor(yo);
    Node     cast;
    cast.type           = OpType::Cast;
    cast.name           = "to_f64";
    cast.inputs         = {x};
    cast.outputs        = {xd};
    cast.attr.map["to"] = [] { Attr a; a.kind = Attr::Int; a.i = 11 /*DOUBLE*/; return a; }();
    g.nodes.push_back(cast);
    Node det;
    det.type    = OpType::Det;
    det.name    = "det";
    det.inputs  = {xd};
    det.outputs = {y};
    g.nodes.push_back(det);
    g.outputs = {y};

    Config cfg;
    cfg.backend = BackendKind::Cpu;
    auto sess   = Session::create(std::move(g), cfg);
    ASSERT_TRUE(sess);
    IOTensor in;
    in.name  = "x";
    in.shape = {2, 2};
    in.dtype = DType::Float32;
    in.data.resize(4 * sizeof(float));
    for (int i = 0; i < 4; ++i)
    {
        reinterpret_cast<float *>(in.data.data())[i] = (float) kDetMatrix[i];
    }
    std::vector<IOTensor> outs;
    ASSERT_EQ(sess->run({in}, outs), Status::Ok);
    ASSERT_FALSE(outs.empty());
    ASSERT_EQ(outs[0].dtype, DType::Float64);
    EXPECT_EQ(reinterpret_cast<const double *>(outs[0].data.data())[0], -1.0);
}

// legalizeFp64 inserts a narrowing Cast(fp64->fp32) before a non-fp64-capable consumer (Softmax) while
// a fp64-capable consumer (Det) keeps its fp64 input.
TEST(Fp64Compute, LegalizeInsertsNarrowingCast) {
    Graph      g;
    TensorDesc xi;
    xi.name    = "x";
    xi.shape   = {2, 2};
    xi.dtype   = DType::Float64;
    xi.isInput = true;
    TensorId x = g.addTensor(xi);
    g.inputs.push_back(x);
    // Det(x) -> d (fp64-capable, keeps fp64), and Softmax(x) -> s (NOT fp64-capable, must narrow).
    TensorDesc di;
    di.name    = "d";
    di.shape   = {1};
    TensorId d = g.addTensor(di);
    TensorDesc si;
    si.name    = "s";
    si.shape   = {2, 2};
    TensorId s = g.addTensor(si);
    Node     det;
    det.type    = OpType::Det;
    det.name    = "det";
    det.inputs  = {x};
    det.outputs = {d};
    g.nodes.push_back(det);
    Node sm;
    sm.type    = OpType::Softmax;
    sm.name    = "softmax";
    sm.inputs  = {x};
    sm.outputs = {s};
    g.nodes.push_back(sm);
    // Two graph outputs so neither op is pruned.
    g.desc(d).isOutput = true;
    g.desc(s).isOutput = true;
    g.outputs          = {d, s};

    PassOptions opt;
    runStandardPasses(g, opt);

    // The Softmax now reads a fp32 tensor (a narrowing Cast was inserted); Det still reads the fp64 x.
    int  smIdx = -1, detIdx = -1;
    for (size_t i = 0; i < g.nodes.size(); ++i)
    {
        if (g.nodes[i].type == OpType::Softmax)
        {
            smIdx = (int) i;
        }
        if (g.nodes[i].type == OpType::Det)
        {
            detIdx = (int) i;
        }
    }
    ASSERT_GE(smIdx, 0);
    ASSERT_GE(detIdx, 0);
    EXPECT_EQ(g.desc(g.nodes[smIdx].inputs[0]).dtype, DType::Float32);  // Softmax narrowed
    EXPECT_EQ(g.desc(g.nodes[detIdx].inputs[0]).dtype, DType::Float64); // Det kept fp64
}

// fp64 elementwise arithmetic: (a - b) in double recovers a difference fp32 cancels. With a = 1 + 2^-40
// and b = 1 the exact difference is 2^-40; fp32 rounds both operands to 1.0 (difference 0), the fp64
// Binary computes 2^-40.
TEST(Fp64Compute, BinarySubIsRealDouble) {
    auto runSub = [](DType dt) {
        Graph      g;
        auto       mk = [&](const char *nm, bool input) {
            TensorDesc d;
            d.name    = nm;
            d.shape   = {1};
            d.dtype   = dt;
            d.isInput = input;
            return g.addTensor(d);
        };
        TensorId a = mk("a", true);
        TensorId b = mk("b", true);
        g.inputs.push_back(a);
        g.inputs.push_back(b);
        TensorDesc yo;
        yo.name     = "y";
        yo.shape    = {1};
        yo.dtype    = dt;
        yo.isOutput = true;
        TensorId y  = g.addTensor(yo);
        Node     sub;
        sub.type    = OpType::Binary;
        sub.name    = "sub";
        sub.subOp   = (int) BinaryType::Sub;
        sub.inputs  = {a, b};
        sub.outputs = {y};
        g.nodes.push_back(sub);
        g.outputs = {y};

        Config cfg;
        cfg.backend = BackendKind::Cpu;
        auto sess   = Session::create(std::move(g), cfg);
        EXPECT_TRUE(sess);
        auto feed = [&](const char *nm, double v) {
            IOTensor t;
            t.name  = nm;
            t.shape = {1};
            t.dtype = dt;
            if (dt == DType::Float64)
            {
                t.data.resize(sizeof(double));
                std::memcpy(t.data.data(), &v, sizeof(double));
            } else
            {
                t.data.resize(sizeof(float));
                float f = (float) v;
                std::memcpy(t.data.data(), &f, sizeof(float));
            }
            return t;
        };
        std::vector<IOTensor> outs;
        EXPECT_EQ(sess->run({feed("a", kBeyondFp32), feed("b", 1.0)}, outs), Status::Ok);
        EXPECT_FALSE(outs.empty());
        if (dt == DType::Float64)
        {
            return reinterpret_cast<const double *>(outs[0].data.data())[0];
        }
        return (double) outs[0].f32()[0];
    };
    EXPECT_EQ(runSub(DType::Float64), kBeyondFp32 - 1.0); // real fp64: exactly 2^-40
    EXPECT_EQ(runSub(DType::Float32), 0.0);               // fp32: both operands round to 1.0
}
