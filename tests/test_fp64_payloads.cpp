// Real IEEE-754 float64 tensors on the CPU path (Phase A: lossless storage / serialization / IO,
// compute still fp32). A DOUBLE initializer keeps NATIVE 8-byte host storage -- never narrowed at
// import -- so it round-trips byte-identically through a .vxm and a genuine fp64 reader (initDoubles)
// recovers it at full precision, while the fp32 compute path decodes it through initFloats. A DOUBLE
// graph input/output round-trips as real fp64 bytes at the session boundary. Phase B adds the fp64
// COMPUTE path (see test_ops.cpp Det tests); here the value is only ever moved, so an fp32-exact
// value round-trips exactly and a value beyond fp32 precision proves the STORAGE stays fp64.
#include "vknn/dtype.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    // A value that is NOT representable in fp32: its low mantissa bits are lost when narrowed to float,
    // so it exposes any accidental fp32 round-trip. Exact in fp64.
    constexpr double kBeyondFp32 = 1.0 + 1.0 / 1099511627776.0; // 1 + 2^-40

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

// A native-fp64 initializer occupies 8 bytes/elem. initDoubles recovers every bit; initFloats
// narrows to fp32 for the compute path. The beyond-fp32 value proves the storage is genuinely fp64,
// not fp32 in disguise.
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

// A DOUBLE graph input/output round-trips as real fp64 bytes at the session boundary. The value is
// only carried through an Identity, and the compute storage is fp32 in Phase A, so an fp32-exact
// value round-trips exactly. The readback emits 8-byte fp64 (outs[0].dtype == Float64).
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
