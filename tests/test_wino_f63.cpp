// F(6,3) Winograd host tests (core/wino_f63.h + core/wino_construct.h):
//   - the point-set construction reproduces the engine's proven F(2,3)/F(4,3) matrices;
//   - F(6,3) Winograd convolution equals the direct convolution to fp64 tolerance;
//   - the baked header tables equal the construction from the baked points, entry for entry;
//   - the chosen point set beats-or-matches the classic set AND beats the condition-number-best
//     fractional set on the shared fp16 pipeline scoring (the basis of the selection);
//   - the CPU-backend conv oracle matches the host Winograd fp64 simulation;
//   - the automatic F-unit rule is unchanged (F(2,3)/F(4,3) only — F(6,3) never auto-selected) and
//     is deterministic.
#include "core/wino_construct.h"
#include "core/wino_f63.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    const std::vector<double> kClassicF63Points = {0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0};
    // The condition-number-best set from the tools/wino_f63_points.cpp sweep — rejected because it
    // loses the empirical fp16 scoring to the chosen set (see the header comment).
    const std::vector<double> kConditionBestF63Points = {0.0, 0.5, -0.5, 1.0, -1.0, 4.0 / 3.0, -4.0 / 3.0};

    double maxAbsDiff(const std::vector<double> &a, const std::vector<double> &b) {
        EXPECT_EQ(a.size(), b.size());
        double m = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            m = std::max(m, std::fabs(a[i] - b[i]));
        }
        return m;
    }

    std::vector<double> directConv3x3Pad1Fp64(const std::vector<double> &input, int cin, int height, int width, const std::vector<double> &weights, int cout) {
        std::vector<double> out((size_t) cout * height * width, 0.0);
        for (int oc = 0; oc < cout; ++oc)
        {
            for (int oy = 0; oy < height; ++oy)
            {
                for (int ox = 0; ox < width; ++ox)
                {
                    double acc = 0.0;
                    for (int ic = 0; ic < cin; ++ic)
                    {
                        for (int ky = 0; ky < 3; ++ky)
                        {
                            for (int kx = 0; kx < 3; ++kx)
                            {
                                const int iy = oy - 1 + ky, ix = ox - 1 + kx;
                                if (iy < 0 || iy >= height || ix < 0 || ix >= width)
                                {
                                    continue;
                                }
                                acc += input[((size_t) ic * height + iy) * width + ix] * weights[(((size_t) oc * cin + ic) * 3 + ky) * 3 + kx];
                            }
                        }
                    }
                    out[((size_t) oc * height + oy) * width + ox] = acc;
                }
            }
        }
        return out;
    }

} // namespace

// The construction reproduces the engine's in-tree F(2,3) matrices (conv.cpp G2,
// wino_input_fp16.comp Bt, wino_out_fp16.comp At) from points {0, 1, -1}: the finite rows and
// columns entry for entry, and the infinity row/column up to the joint paired sign — the in-tree
// F(2,3) historically bakes (-M(x) coefficients, A^T infinity entry -1) where the construction
// (like the in-tree F(4,3)) uses (+M(x), +1); the per-point product, and therefore the algorithm,
// is identical either way.
TEST(WinoConstruct, ReproducesEngineF23Matrices) {
    const WinoMatrices w         = buildWinoMatrices(2, {0.0, 1.0, -1.0});
    const double       g2[4][3]  = {{1, 0, 0}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, {0, 0, 1}};
    const double       bt2[4][4] = {{1, 0, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, 0, -1}};
    const double       at2[2][4] = {{1, 1, 1, 0}, {0, 1, -1, -1}};
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_DOUBLE_EQ(w.filterG[(size_t) i * 3 + j], g2[i][j]) << "G2[" << i << "][" << j << "]";
        }
    }
    for (int i = 0; i < 3; ++i) // finite B^T rows match directly
    {
        for (int j = 0; j < 4; ++j)
        {
            EXPECT_DOUBLE_EQ(w.inputBt[(size_t) i * 4 + j], bt2[i][j]) << "Bt2[" << i << "][" << j << "]";
        }
    }
    for (int j = 0; j < 4; ++j) // infinity B^T row: the joint paired sign
    {
        EXPECT_DOUBLE_EQ(w.inputBt[(size_t) 3 * 4 + j], -bt2[3][j]) << "Bt2[3][" << j << "]";
    }
    for (int t = 0; t < 2; ++t)
    {
        for (int j = 0; j < 3; ++j) // finite A^T columns match directly
        {
            EXPECT_DOUBLE_EQ(w.outputAt[(size_t) t * 4 + j], at2[t][j]) << "At2[" << t << "][" << j << "]";
        }
    }
    EXPECT_DOUBLE_EQ(w.outputAt[(size_t) 1 * 4 + 3], -at2[1][3]); // infinity A^T entry: paired sign
}

// The construction reproduces the engine's in-tree F(4,3) matrices (conv.cpp G4,
// wino_input4_fp16.comp Bt, wino_out4_fp16.comp At) from points {0, 1, -1, 2, -2} exactly.
TEST(WinoConstruct, ReproducesEngineF43Matrices) {
    const WinoMatrices w        = buildWinoMatrices(4, {0.0, 1.0, -1.0, 2.0, -2.0});
    const float        g4[6][3] = {
        {0.25f, 0, 0}, {-1.f / 6, -1.f / 6, -1.f / 6}, {-1.f / 6, 1.f / 6, -1.f / 6}, {1.f / 24, 1.f / 12, 1.f / 6}, {1.f / 24, -1.f / 12, 1.f / 6}, {0, 0, 1}};
    const double bt4[6][6] = {{4, 0, -5, 0, 1, 0}, {0, -4, -4, 1, 1, 0}, {0, 4, -4, -1, 1, 0}, {0, -2, -1, 2, 1, 0}, {0, 2, -1, -2, 1, 0}, {0, 4, 0, -5, 0, 1}};
    const double at4[4][6] = {{1, 1, 1, 1, 1, 0}, {0, 1, -1, 2, -2, 0}, {0, 1, 1, 4, 4, 0}, {0, 1, -1, 8, -8, 1}};
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_FLOAT_EQ((float) w.filterG[(size_t) i * 3 + j], g4[i][j]) << "G4[" << i << "][" << j << "]";
        }
        for (int j = 0; j < 6; ++j)
        {
            EXPECT_DOUBLE_EQ(w.inputBt[(size_t) i * 6 + j], bt4[i][j]) << "Bt4[" << i << "][" << j << "]";
        }
    }
    for (int t = 0; t < 4; ++t)
    {
        for (int j = 0; j < 6; ++j)
        {
            EXPECT_DOUBLE_EQ(w.outputAt[(size_t) t * 6 + j], at4[t][j]) << "At4[" << t << "][" << j << "]";
        }
    }
}

// Transform correctness at fp64: F(6,3) Winograd convolution equals the direct convolution for
// random kernels and inputs, for the chosen points and for a fractional candidate set (the
// construction is point-set-generic).
TEST(WinoConstruct, F63WinogradEqualsDirectFp64) {
    for (const std::vector<double> *points: {&kClassicF63Points, &kConditionBestF63Points})
    {
        const WinoMatrices w = buildWinoMatrices(kWinoF63OutTile, *points);
        ASSERT_EQ(w.outTile, kWinoF63OutTile);
        ASSERT_EQ(w.alpha, kWinoF63Alpha);
        WinoNormalStream    noise(101);
        const int           cin = 3, cout = 4, height = 13, width = 11; // non-multiples of 6 exercise edge tiles
        std::vector<double> input((size_t) cin * height * width);
        std::vector<double> weights((size_t) cout * cin * 9);
        for (double &v: input)
        {
            v = noise.next();
        }
        for (double &v: weights)
        {
            v = noise.next();
        }
        const std::vector<double> wino   = winoConv2dFp64(w, input, cin, height, width, weights, cout);
        const std::vector<double> direct = directConv3x3Pad1Fp64(input, cin, height, width, weights, cout);
        EXPECT_LT(maxAbsDiff(wino, direct), 1e-10);
    }
    // F(2,3) and F(4,3) satisfy the same identity through the same construction.
    for (int unit: {2, 4})
    {
        const WinoMatrices  w = buildWinoMatrices(unit, unit == 2 ? std::vector<double> {0.0, 1.0, -1.0} : std::vector<double> {0.0, 1.0, -1.0, 2.0, -2.0});
        WinoNormalStream    noise(202);
        const int           cin = 2, cout = 2, height = 9, width = 7;
        std::vector<double> input((size_t) cin * height * width);
        std::vector<double> weights((size_t) cout * cin * 9);
        for (double &v: input)
        {
            v = noise.next();
        }
        for (double &v: weights)
        {
            v = noise.next();
        }
        EXPECT_LT(maxAbsDiff(winoConv2dFp64(w, input, cin, height, width, weights, cout), directConv3x3Pad1Fp64(input, cin, height, width, weights, cout)), 1e-10);
    }
}

// The baked header tables are exactly the float cast of the construction from the baked points —
// core/wino_f63.h cannot drift from core/wino_construct.h (tools/wino_f63_points.cpp re-checks the
// same identity offline).
TEST(WinoF63, HeaderTablesMatchConstruction) {
    const std::vector<double> points(kWinoF63Points, kWinoF63Points + (kWinoF63Alpha - 1));
    const WinoMatrices        w = buildWinoMatrices(kWinoF63OutTile, points);
    for (int i = 0; i < kWinoF63Alpha; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_EQ(kWinoF63G[i][j], (float) w.filterG[(size_t) i * 3 + j]) << "G[" << i << "][" << j << "]";
        }
        for (int j = 0; j < kWinoF63Alpha; ++j)
        {
            EXPECT_EQ(kWinoF63Bt[i][j], (float) w.inputBt[(size_t) i * kWinoF63Alpha + j]) << "Bt[" << i << "][" << j << "]";
        }
    }
    for (int t = 0; t < kWinoF63OutTile; ++t)
    {
        for (int j = 0; j < kWinoF63Alpha; ++j)
        {
            EXPECT_EQ(kWinoF63At[t][j], (float) w.outputAt[(size_t) t * kWinoF63Alpha + j]) << "At[" << t << "][" << j << "]";
        }
    }
}

// The selection contract on the shared scoring configs: the chosen (baked) points beat or match
// the classic set on worst-case SNR AND max-abs, and beat the condition-number-best fractional
// set — the empirical evidence that condition number alone would have picked a WORSE set for this
// fp16 pipeline. Deterministic: fixed seeds, portable noise stream.
TEST(WinoF63, ChosenPointsBeatOrMatchAlternatives) {
    const std::vector<double> chosenPoints(kWinoF63Points, kWinoF63Points + (kWinoF63Alpha - 1));
    const WinoMatrices        chosen        = buildWinoMatrices(kWinoF63OutTile, chosenPoints);
    const WinoMatrices        classic       = buildWinoMatrices(kWinoF63OutTile, kClassicF63Points);
    const WinoMatrices        conditionBest = buildWinoMatrices(kWinoF63OutTile, kConditionBestF63Points);
    auto                      worstCase     = [](const WinoMatrices &w) {
        WinoFp16Error worst;
        worst.snrDb = std::numeric_limits<double>::infinity();
        for (const WinoLayerConfig &cfg: winoScoreConfigs())
        {
            const WinoFp16Error e = simulateWinoFp16Pipeline(w, cfg);
            worst.snrDb           = std::min(worst.snrDb, e.snrDb);
            worst.maxAbs          = std::max(worst.maxAbs, e.maxAbs);
        }
        return worst;
    };
    const WinoFp16Error chosenErr = worstCase(chosen), classicErr = worstCase(classic), condErr = worstCase(conditionBest);
    EXPECT_GE(chosenErr.snrDb, classicErr.snrDb);
    EXPECT_LE(chosenErr.maxAbs, classicErr.maxAbs);
    EXPECT_GT(chosenErr.snrDb, condErr.snrDb);
    // The pipeline stays fp16-safe outright: finite error, comfortably above a usable-SNR floor.
    EXPECT_TRUE(std::isfinite(chosenErr.maxAbs));
    EXPECT_GT(chosenErr.snrDb, 35.0);
}

// CPU-oracle equivalence: a small 3x3 pad-1 conv with bias through the CPU backend equals the
// host F(6,3) Winograd fp64 simulation on the same data (correctness of the transform math
// against the engine's conv oracle; fp32-backend tolerance).
TEST(WinoF63, CpuOracleMatchesWinoSimulation) {
    const int          cin = 6, cout = 5, height = 12, width = 12;
    WinoNormalStream   noise(303);
    std::vector<float> input((size_t) cin * height * width);
    std::vector<float> weights((size_t) cout * cin * 9);
    std::vector<float> bias((size_t) cout);
    for (float &v: input)
    {
        v = (float) noise.next();
    }
    for (float &v: weights)
    {
        v = (float) noise.next() * 0.25f;
    }
    for (float &v: bias)
    {
        v = (float) noise.next();
    }

    Graph      g;
    TensorDesc xd;
    xd.name    = "x";
    xd.shape   = {1, cin, height, width};
    xd.isInput = true;
    TensorId x = g.addTensor(xd);
    g.inputs.push_back(x);
    TensorDesc wd;
    wd.name          = "w";
    wd.shape         = {cout, cin, 3, 3};
    wd.isInitializer = true;
    TensorId   w     = g.addTensor(wd);
    HostBuffer wb;
    wb.resizeElems(weights.size(), DType::Float32);
    std::memcpy(wb.f32(), weights.data(), weights.size() * sizeof(float));
    g.initializers[w] = wb;
    TensorDesc bd;
    bd.name          = "b";
    bd.shape         = {cout};
    bd.isInitializer = true;
    TensorId   b     = g.addTensor(bd);
    HostBuffer bb;
    bb.resizeElems(bias.size(), DType::Float32);
    std::memcpy(bb.f32(), bias.data(), bias.size() * sizeof(float));
    g.initializers[b] = bb;
    TensorDesc yd;
    yd.name     = "y";
    yd.isOutput = true;
    TensorId y  = g.addTensor(yd);
    Node     node;
    node.type = OpType::Conv;
    node.name = "conv";
    Attr pads;
    pads.kind = Attr::Ints;
    pads.ints = {1, 1, 1, 1};
    Attr strides;
    strides.kind             = Attr::Ints;
    strides.ints             = {1, 1};
    node.attr.map["pads"]    = pads;
    node.attr.map["strides"] = strides;
    node.inputs              = {x, w, b};
    node.outputs             = {y};
    g.nodes.push_back(node);
    g.outputs = {y};

    Config cfg;
    cfg.backend  = BackendKind::Cpu;
    auto session = Session::create(std::move(g), cfg);
    ASSERT_TRUE(session);
    IOTensor in;
    in.name  = "x";
    in.shape = {1, cin, height, width};
    in.dtype = DType::Float32;
    in.data.resize(input.size() * sizeof(float));
    std::memcpy(in.data.data(), input.data(), input.size() * sizeof(float));
    std::vector<IOTensor> outs;
    ASSERT_EQ(session->run({in}, outs), Status::Ok);
    ASSERT_FALSE(outs.empty());
    ASSERT_EQ(outs[0].shape, (std::vector<int64_t> {1, cout, height, width}));

    const std::vector<double> points(kWinoF63Points, kWinoF63Points + (kWinoF63Alpha - 1));
    const WinoMatrices        wm = buildWinoMatrices(kWinoF63OutTile, points);
    const std::vector<double> inputF64(input.begin(), input.end());
    const std::vector<double> weightsF64(weights.begin(), weights.end());
    const std::vector<double> wino    = winoConv2dFp64(wm, inputF64, cin, height, width, weightsF64, cout);
    const float              *cpu     = outs[0].f32();
    double                    maxDiff = 0.0;
    for (int oc = 0; oc < cout; ++oc)
    {
        for (int i = 0; i < height * width; ++i)
        {
            const double expected = wino[(size_t) oc * height * width + i] + bias[(size_t) oc];
            maxDiff               = std::max(maxDiff, std::fabs(expected - (double) cpu[(size_t) oc * height * width + i]));
        }
    }
    EXPECT_LT(maxDiff, 1e-4); // fp32 CPU backend vs fp64 winograd math
}

// GATING: the automatic F-unit rule (core/wino_f63.h winoAutoUnit, consumed by tuneWino) is
// BIT-IDENTICAL to the pre-F(6,3) rule over a shape sweep — F(6,3) must never be auto-selected;
// it is reachable only through the explicit WinogradUnit hint, and the device measurement that
// would have promoted it refuted it instead (accuracy regresses on every model that would use it,
// and the speed win is not a function of the shape — evidence at winoAutoUnit). The frozen replica
// below is the exact expression tuneWino carried before the rule moved into the shared header.
TEST(WinoF63, AutoUnitRuleUnchanged) {
    auto frozenRule = [](int64_t Cin, int64_t Cout) {
        auto winoCostPerOut = [&](int n) {
            double i = (double) Cin, o = (double) Cout, e = n + 2; // n + k - 1, k = 3
            return (2.0 * i * e + i * o * e + (double) n * e * (2.0 * n + 2.0)) / (double) (n * n);
        };
        return (winoCostPerOut(4) < winoCostPerOut(2)) ? 4 : 2;
    };
    const int64_t channels[] = {1, 3, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048};
    for (int64_t cin: channels)
    {
        for (int64_t cout: channels)
        {
            const int unit = winoAutoUnit(cin, cout);
            EXPECT_EQ(unit, frozenRule(cin, cout)) << "Cin=" << cin << " Cout=" << cout;
            EXPECT_TRUE(unit == 2 || unit == 4) << "F(6,3) leaked into the automatic rule at Cin=" << cin << " Cout=" << cout;
        }
    }
}

// Determinism: the rule is a pure function of the shape, so repeated evaluation yields the same
// unit — a timing-raced F-unit would break the byte-exactness the engine guarantees across runs
// and tuning levels (ADR-0009). The shapes where isolated single-shape probes put F(6,3) ahead are
// pinned here too: in-model those shapes reversed, so they must still resolve to F(4,3), and a
// future F(6,3) promotion cannot slip in without this gate turning red.
TEST(WinoF63, AutoUnitRuleIsDeterministicAndF63Free) {
    const int64_t channels[] = {32, 64, 96, 128, 256, 512};
    for (int64_t cin: channels)
    {
        for (int64_t cout: channels)
        {
            const int first = winoAutoUnit(cin, cout);
            for (int repeat = 0; repeat < 4; ++repeat)
            {
                EXPECT_EQ(winoAutoUnit(cin, cout), first) << "Cin=" << cin << " Cout=" << cout;
            }
        }
    }
    EXPECT_EQ(winoAutoUnit(64, 64), 4);   // probed ahead on F(6,3) at 28x28 / 40x40 / 56x56 / 80x80
    EXPECT_EQ(winoAutoUnit(96, 96), 4);   // probed ahead on F(6,3) at 35x35
    EXPECT_EQ(winoAutoUnit(32, 64), 4);   // probed ahead on F(6,3) at 147x147
    EXPECT_EQ(winoAutoUnit(128, 128), 4); // probed ahead on F(6,3) at 40x40 / 80x80
}
