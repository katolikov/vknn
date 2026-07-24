// Offline F(6,3) Winograd interpolation-point selection (NOVA-style, arXiv 2512.18453: pick the
// points that minimize the transform condition numbers; fractional points allowed).
//
// Enumerates every symmetric 7-point set {0, +-a, +-b, +-c} with a < b < c drawn from the reduced
// rationals p/q, q in {1,2,3,4}, 0 < p/q <= 4, builds the Cook-Toom matrices for each
// (core/wino_construct.h), and scores them two ways:
//   1. transform condition numbers kappa2(B^T), kappa2(A^T), kappa2(G);
//   2. empirical fp16 pipeline error (max-abs and SNR vs the fp64 direct conv) through the exact
//      GPU discipline: fp32 transforms, fp16 U/V/M storage, fp32 GEMM accumulation, saturating
//      RTE output store — over unit-normal and post-BatchNorm-range conv layers.
// The empirical stage runs on the condition-number top set plus the classic points
// {0, 1, -1, 2, -2, 1/2, -1/2}; the winner is the best worst-case SNR (tie-break: max-abs).
//
// The winning set and its derived float tables are baked as constexpr in src/core/wino_f63.h; this
// tool re-derives them on every run and exits non-zero when the baked tables disagree with the
// construction, so the header cannot drift from the search.
//
// Build: cmake target vknn_wino_f63_points (host). Run with no arguments; prints the score table.
#include "core/wino_construct.h"
#include "core/wino_f63.h"
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    // Candidate magnitudes: reduced rationals p/q with q in {1..4}, 0 < p/q <= 4.
    constexpr int    kMaxDenominator   = 4;
    constexpr double kMaxPointValue    = 4.0;
    // Empirical stage size: the condition-number top set that gets the (much slower) fp16 pipeline
    // simulation.
    constexpr size_t kEmpiricalTopK    = 16;

    struct Candidate {
        double              a = 0.0, b = 0.0, c = 0.0;
        std::string         label;
        std::vector<double> points; // {0, a, -a, b, -b, c, -c}
        WinoMatrices        mats;
        double              condBt = 0.0, condAt = 0.0, condG = 0.0;
        double              worstSnrDb = 0.0, worstMaxAbs = 0.0;
        bool                simulated = false;
    };

    int gcdInt(int x, int y) {
        while (y != 0)
        {
            const int r = x % y;
            x = y;
            y = r;
        }
        return x;
    }

    std::string rationalLabel(int num, int den) {
        return den == 1 ? std::to_string(num) : std::to_string(num) + "/" + std::to_string(den);
    }

    Candidate makeCandidate(double a, double b, double c, const std::string &label) {
        Candidate cand;
        cand.a      = a;
        cand.b      = b;
        cand.c      = c;
        cand.label  = label;
        cand.points = {0.0, a, -a, b, -b, c, -c};
        cand.mats   = buildWinoMatrices(6, cand.points);
        cand.condBt = winoCondition2(cand.mats.inputBt, cand.mats.alpha, cand.mats.alpha);
        cand.condAt = winoCondition2(cand.mats.outputAt, cand.mats.outTile, cand.mats.alpha);
        cand.condG  = winoCondition2(cand.mats.filterG, cand.mats.alpha, kWinoKernelTaps);
        return cand;
    }

    void simulate(Candidate &cand) {
        double worstSnr = std::numeric_limits<double>::infinity(), worstAbs = 0.0;
        for (const WinoLayerConfig &cfg: winoScoreConfigs())
        {
            const WinoFp16Error e = simulateWinoFp16Pipeline(cand.mats, cfg);
            worstSnr = std::min(worstSnr, e.snrDb);
            worstAbs = std::max(worstAbs, e.maxAbs);
        }
        cand.worstSnrDb  = worstSnr;
        cand.worstMaxAbs = worstAbs;
        cand.simulated   = true;
    }

    // %.9g with a guaranteed decimal point, so the emitted token is a valid C++ float literal
    // ("1" would otherwise print as the ill-formed "1f").
    std::string floatLiteral(float v) {
        char buf[40];
        std::snprintf(buf, sizeof(buf), "%.9g", (double) v);
        std::string s = buf;
        if (s.find('.') == std::string::npos && s.find('e') == std::string::npos)
        {
            s += ".0";
        }
        return s + "f";
    }

    void printFloatTable(const char *name, const std::vector<double> &m, int rows, int cols) {
        std::printf("constexpr float %s[%d][%d] = {\n", name, rows, cols);
        for (int i = 0; i < rows; ++i)
        {
            std::printf("    {");
            for (int j = 0; j < cols; ++j)
            {
                std::printf("%s%s", floatLiteral((float) m[(size_t) i * cols + j]).c_str(), j + 1 < cols ? ", " : "");
            }
            std::printf("},\n");
        }
        std::printf("};\n");
    }

    // The baked header tables must equal the float cast of the double construction, entry for entry.
    bool bakedTablesMatch(const WinoMatrices &m) {
        bool ok = true;
        for (int i = 0; i < m.alpha; ++i)
        {
            for (int j = 0; j < kWinoKernelTaps; ++j)
            {
                ok = ok && kWinoF63G[i][j] == (float) m.filterG[(size_t) i * kWinoKernelTaps + j];
            }
            for (int j = 0; j < m.alpha; ++j)
            {
                ok = ok && kWinoF63Bt[i][j] == (float) m.inputBt[(size_t) i * m.alpha + j];
            }
        }
        for (int i = 0; i < m.outTile; ++i)
        {
            for (int j = 0; j < m.alpha; ++j)
            {
                ok = ok && kWinoF63At[i][j] == (float) m.outputAt[(size_t) i * m.alpha + j];
            }
        }
        return ok;
    }

} // namespace

int main() {
    // Candidate magnitude values.
    struct Rational {
        int    num, den;
        double value;
        std::string label;
    };
    std::vector<Rational> values;
    for (int den = 1; den <= kMaxDenominator; ++den)
    {
        for (int num = 1; (double) num / den <= kMaxPointValue; ++num)
        {
            if (gcdInt(num, den) != 1)
            {
                continue;
            }
            values.push_back({num, den, (double) num / den, rationalLabel(num, den)});
        }
    }
    std::sort(values.begin(), values.end(), [](const Rational &x, const Rational &y) { return x.value < y.value; });

    std::vector<Candidate> cands;
    for (size_t i = 0; i < values.size(); ++i)
    {
        for (size_t j = i + 1; j < values.size(); ++j)
        {
            for (size_t k = j + 1; k < values.size(); ++k)
            {
                const std::string label = "{0, +-" + values[i].label + ", +-" + values[j].label + ", +-" + values[k].label + "}";
                cands.push_back(makeCandidate(values[i].value, values[j].value, values[k].value, label));
            }
        }
    }
    std::printf("wino_f63_points: %zu symmetric candidate sets (q <= %d, |p/q| <= %g)\n", cands.size(), kMaxDenominator, kMaxPointValue);

    // Rank by the input/output transform conditioning product (the NOVA objective; G is applied
    // once on the host in fp32, so it is reported but not ranked on).
    std::sort(cands.begin(), cands.end(), [](const Candidate &x, const Candidate &y) { return x.condBt * x.condAt < y.condBt * y.condAt; });

    const size_t classicIdx = [&] {
        for (size_t i = 0; i < cands.size(); ++i)
        {
            if (cands[i].a == 0.5 && cands[i].b == 1.0 && cands[i].c == 2.0)
            {
                return i;
            }
        }
        return cands.size();
    }();
    std::printf("classic {0, +-1/2, +-1, +-2} condition rank: %zu of %zu\n\n", classicIdx + 1, cands.size());

    // Empirical fp16 stage: the condition top-K plus the classic set.
    std::vector<size_t> simulateIdx;
    for (size_t i = 0; i < std::min(kEmpiricalTopK, cands.size()); ++i)
    {
        simulateIdx.push_back(i);
    }
    if (classicIdx < cands.size() && classicIdx >= kEmpiricalTopK)
    {
        simulateIdx.push_back(classicIdx);
    }
    for (size_t idx: simulateIdx)
    {
        simulate(cands[idx]);
    }

    std::printf("%-34s %10s %10s %10s %12s %12s\n", "points", "cond(Bt)", "cond(At)", "cond(G)", "worst SNR dB", "worst maxAbs");
    for (size_t idx: simulateIdx)
    {
        const Candidate &c = cands[idx];
        std::printf("%-34s %10.3f %10.3f %10.3f %12.2f %12.3e%s\n", c.label.c_str(), c.condBt, c.condAt, c.condG, c.worstSnrDb, c.worstMaxAbs,
                    idx == classicIdx ? "   <- classic" : "");
    }

    // Winner: best worst-case SNR among the simulated sets; max-abs breaks ties.
    size_t winner = simulateIdx[0];
    for (size_t idx: simulateIdx)
    {
        const Candidate &c = cands[idx], &best = cands[winner];
        if (c.worstSnrDb > best.worstSnrDb || (c.worstSnrDb == best.worstSnrDb && c.worstMaxAbs < best.worstMaxAbs))
        {
            winner = idx;
        }
    }
    const Candidate &win = cands[winner];
    std::printf("\nwinner: %s  (cond Bt %.3f, At %.3f, G %.3f; worst SNR %.2f dB, worst maxAbs %.3e)\n", win.label.c_str(), win.condBt, win.condAt, win.condG,
                win.worstSnrDb, win.worstMaxAbs);
    if (classicIdx < cands.size() && cands[classicIdx].simulated)
    {
        const Candidate &cl = cands[classicIdx];
        std::printf("classic: %s  (cond Bt %.3f, At %.3f, G %.3f; worst SNR %.2f dB, worst maxAbs %.3e)\n", cl.label.c_str(), cl.condBt, cl.condAt, cl.condG,
                    cl.worstSnrDb, cl.worstMaxAbs);
    }

    std::printf("\nbaked tables for src/core/wino_f63.h:\n");
    std::printf("// points: %s\n", win.label.c_str());
    printFloatTable("kWinoF63G", win.mats.filterG, win.mats.alpha, kWinoKernelTaps);
    printFloatTable("kWinoF63Bt", win.mats.inputBt, win.mats.alpha, win.mats.alpha);
    printFloatTable("kWinoF63At", win.mats.outputAt, win.mats.outTile, win.mats.alpha);

    // Consistency gate: the header must carry exactly this winner's tables.
    bool pointsMatch = true;
    for (int i = 0; i < (int) win.points.size(); ++i)
    {
        pointsMatch = pointsMatch && kWinoF63Points[i] == win.points[(size_t) i];
    }
    if (!pointsMatch || !bakedTablesMatch(win.mats))
    {
        std::printf("\nFAIL: src/core/wino_f63.h tables do not match this search's winner — re-bake them from the output above.\n");
        return 1;
    }
    std::printf("\nPASS: src/core/wino_f63.h matches the search winner.\n");
    return 0;
}
