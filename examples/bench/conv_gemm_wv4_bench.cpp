// vknn_conv_gemm_wv4_bench — A/B microbench + byte gate for the implicit-GEMM convolution's weight
// load width: times the scalar-weight kernel (shaders/conv_gemm.comp, one 16-bit load per output
// channel) against its STORE4-load twin (shaders/conv_gemm_wv4.comp, four channels per load) on the
// shapes the implicit-GEMM path actually serves, and byte-compares their outputs.
//
// The twin keeps conv_gemm's implicit im2col, LDS panel layout, chunked ascending-k fp32 reduction
// and store, changing only the width of the [K][Coutp] weight-panel load, so the outputs must be
// identical — the bytes_equal column is the gate, not a tolerance check. Both layouts are exercised:
//   packed  (Cout % 4 == 0): both kernels read the same [K][Cout] panel; Coutp == Cout.
//   padded  (Cout % 4 != 0): the scalar kernel reads the packed panel, the twin reads the
//                            padConvGemmWeightVec4 zero-padded copy at Coutp = roundUpConvGemmCout(Cout)
//                            (the "#gemmwp" repack layout).
//
// Timing mirrors the op's own candidate race: dedicated scratch operands, kRepsPerSubmit dispatches
// per command buffer with a serializing compute barrier after each, wall-clocked by
// CommandRunner::submitAndWait, min over kTimingSubmits submits, with the rounds INTERLEAVED across
// the two kernels so a clock ramp moves both kernels' samples together.
//
// Usage: vknn_conv_gemm_wv4_bench [N,Cin,H,W,Cout,KH,KW,SH,SW,PT,PL ...]
// Without arguments it runs the built-in shape set.
#include "backend/vulkan/vk_buffer.h"
#include "backend/vulkan/vk_command.h"
#include "backend/vulkan/vk_context.h"
#include "backend/vulkan/vk_pipeline.h"
#include "core/conv_gemm_route.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vknn;
using namespace vknn::vk;

namespace {

    constexpr int kRepsPerSubmit = 8;
    constexpr int kTimingSubmits = 5;

    // conv_gemm tile geometry: TN is fixed in the shader, the M tile is specialization constant 0
    // (16/32/64) and the op's heuristic picks it from M.
    constexpr int kTileN = 64;

    int convGemmTileM(int64_t m) {
        return m < 24 ? 16 : (m < 48 ? 32 : 64);
    }

    // Mirrors ConvGemmPC in src/backend/vulkan/ops/vk_op_common.h. Coutp (the weight panel's
    // physical row stride) is the trailing field, read only by the vec4-weight twin.
    struct ConvGemmPC {
        int32_t C, H, W, Cout, OH, OW, KH, KW, SH, SW, PT, PL, DH, DW;
        int32_t act, hasBias;
        float   actLo, actHi;
        int32_t Coutp;
    };

    struct ShapeSpec {
        int N, Cin, H, W, Cout, KH, KW, SH, SW, PT, PL;
    };

    // Shapes the implicit-GEMM path serves: the Conv rule takes it at Cout >= 512 and OH*OW >= 128,
    // which in practice means the wide patch-embed convs (a ViT/SigLIP stem) and the deep
    // large-map convs a lowered graph hands to the ConvGemm op. The last entry has Cout % 4 != 0 and
    // exercises the zero-padded repack.
    constexpr ShapeSpec kDefaultShapes[] = {
        {1, 3, 512, 512, 1152, 16, 16, 16, 16, 0, 0}, // SigLIP-class patch embed: M = 1024, K = 768
        {1, 3, 224, 224, 768, 16, 16, 16, 16, 0, 0},  // ViT-B/16 patch embed: M = 196, K = 768
        {1, 256, 28, 28, 512, 3, 3, 1, 1, 1, 1},      // deep large-map 3x3: M = 784, K = 2304
        {1, 512, 14, 14, 1024, 3, 3, 1, 1, 1, 1},     // deeper, smaller map: M = 196, K = 4608
        {1, 128, 56, 56, 512, 3, 3, 2, 2, 1, 1},      // strided 3x3 (Winograd refuses): M = 784
        {1, 256, 28, 28, 514, 3, 3, 1, 1, 1, 1},      // Cout % 4 != 0: the padded-panel route
        {1, 256, 28, 28, 516, 3, 3, 1, 1, 1, 1},      // the same shape 4-aligned, as its control
    };

    inline uint16_t f2h(float x) noexcept {
        __fp16   h = (__fp16) x;
        uint16_t o;
        std::memcpy(&o, &h, 2);
        return o;
    }

    // Deterministic bounded pseudo-random in [-0.5,0.5] from an index (reproducible across devices).
    inline float synth(uint32_t i) noexcept {
        uint32_t h = i * 2654435761u + 1013904223u;
        h ^= h >> 15;
        return (float) (h & 0xffff) / 65535.0f - 0.5f;
    }

    inline int64_t cBlocks(int64_t c) {
        return (c + 3) / 4;
    }

    bool parseShape(const char *s, ShapeSpec &out) {
        return sscanf(s, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", &out.N, &out.Cin, &out.H, &out.W, &out.Cout, &out.KH, &out.KW, &out.SH, &out.SW, &out.PT, &out.PL) == 11;
    }

} // namespace

int main(int argc, char **argv) {
    std::vector<ShapeSpec> shapes;
    for (int i = 1; i < argc; ++i)
    {
        ShapeSpec s {};
        if (!parseShape(argv[i], s))
        {
            fprintf(stderr, "bad shape '%s' (expect N,Cin,H,W,Cout,KH,KW,SH,SW,PT,PL)\n", argv[i]);
            return 2;
        }
        shapes.push_back(s);
    }
    if (shapes.empty())
    {
        shapes.assign(std::begin(kDefaultShapes), std::end(kDefaultShapes));
    }

    VulkanContext ctx;
    if (!ctx.initialized())
    {
        fprintf(stderr, "Vulkan init failed\n");
        return 1;
    }
    CommandRunner runner(ctx);

    printf("N,Cin,H,W,Cout,KHxKW,SHxSW,M,K,layout,tm,bytes_equal,scalar_ms,wv4_ms,delta_pct\n");
    for (const ShapeSpec &s: shapes)
    {
        const int64_t OH   = (s.H + 2 * s.PT - s.KH) / s.SH + 1;
        const int64_t OW   = (s.W + 2 * s.PL - s.KW) / s.SW + 1;
        const int64_t M    = OH * OW;
        const int64_t K    = (int64_t) s.Cin * s.KH * s.KW;
        const int64_t Cinb = cBlocks(s.Cin), Coutb = cBlocks(s.Cout);
        if (OH <= 0 || OW <= 0)
        {
            fprintf(stderr, "skip %dx%dx%d: empty output\n", s.Cin, s.H, s.W);
            continue;
        }

        const ConvGemmWVec4Route route = convGemmWVec4Route(s.Cout, /*weightRepackable=*/true);
        const int64_t            coutP = route.eligible ? route.coutP : s.Cout;
        const int                tm    = convGemmTileM(M);

        // NC4HW4 source, [K][Cout] weight panel (+ its zero-padded [K][coutP] copy when Cout is not
        // 4-aligned), rank-1 [Cout] bias — the layouts the op uploads.
        const size_t          srcHalfs = (size_t) s.N * Cinb * s.H * s.W * 4;
        const size_t          dstHalfs = (size_t) s.N * Coutb * OH * OW * 4;
        std::vector<uint16_t> hSrc(srcHalfs);
        for (size_t i = 0; i < srcHalfs; ++i)
        {
            hSrc[i] = f2h(synth((uint32_t) (i * 3 + 7)));
        }
        std::vector<float> wFloats((size_t) (K * s.Cout));
        for (size_t i = 0; i < wFloats.size(); ++i)
        {
            wFloats[i] = synth((uint32_t) (i * 5 + 1));
        }
        std::vector<uint16_t> hW(wFloats.size());
        for (size_t i = 0; i < wFloats.size(); ++i)
        {
            hW[i] = f2h(wFloats[i]);
        }
        std::vector<uint16_t> hBias((size_t) s.Cout);
        for (size_t i = 0; i < hBias.size(); ++i)
        {
            hBias[i] = f2h(synth((uint32_t) (i * 11 + 3)));
        }

        Buffer bufSrc(ctx, srcHalfs * 2, MemPref::kAuto);
        Buffer bufW(ctx, hW.size() * 2, MemPref::kAuto);
        Buffer bufBias(ctx, hBias.size() * 2, MemPref::kAuto);
        bufSrc.upload(hSrc.data(), srcHalfs * 2);
        bufW.upload(hW.data(), hW.size() * 2);
        bufBias.upload(hBias.data(), hBias.size() * 2);
        std::vector<uint16_t>().swap(hSrc);
        std::vector<uint16_t>().swap(hW);

        // The twin's panel: the same buffer when Cout is 4-aligned, else the zero-padded repack the
        // op uploads under the "#gemmwp" key.
        Buffer bufWv4(ctx, (size_t) (K * coutP) * 2, MemPref::kAuto);
        {
            const std::vector<float> padded = route.padW ? padConvGemmWeightVec4(wFloats, K, s.Cout, coutP) : wFloats;
            std::vector<uint16_t>    halves(padded.size());
            for (size_t i = 0; i < padded.size(); ++i)
            {
                halves[i] = f2h(padded[i]);
            }
            bufWv4.upload(halves.data(), halves.size() * 2);
        }
        std::vector<float>().swap(wFloats);

        ConvGemmPC pc {};
        pc.C       = s.Cin;
        pc.H       = s.H;
        pc.W       = s.W;
        pc.Cout    = s.Cout;
        pc.OH      = (int) OH;
        pc.OW      = (int) OW;
        pc.KH      = s.KH;
        pc.KW      = s.KW;
        pc.SH      = s.SH;
        pc.SW      = s.SW;
        pc.PT      = s.PT;
        pc.PL      = s.PL;
        pc.DH      = 1;
        pc.DW      = 1;
        pc.act     = 0;
        pc.hasBias = 1;
        pc.actLo   = 0.0f;
        pc.actHi   = 0.0f;
        pc.Coutp   = (int) coutP;

        ComputePipeline scalarPipe(ctx, "conv_gemm_fp16", 4, sizeof(ConvGemmPC), {(uint32_t) tm});
        ComputePipeline wv4Pipe(ctx, "conv_gemm_wv4_fp16", 4, sizeof(ConvGemmPC), {(uint32_t) tm});
        const uint32_t  gx = (uint32_t) ((s.Cout + kTileN - 1) / kTileN);
        const uint32_t  gy = (uint32_t) ((M + tm - 1) / tm);
        const uint32_t  gz = (uint32_t) s.N;

        // Correctness: one dispatch per kernel into host-readable outputs, then a byte compare.
        auto runInto = [&](ComputePipeline &pipe, VkBuffer weights, Buffer &dst) {
            runner.oneShot([&](VkCommandBuffer cmd) {
                pipe.dispatch(cmd, {bufSrc.handle(), weights, bufBias.handle(), dst.handle()}, &pc, sizeof(pc), gx, gy, gz);
            });
        };
        std::vector<uint16_t> outScalar(dstHalfs), outWv4(dstHalfs);
        {
            Buffer dScalar(ctx, dstHalfs * 2, MemPref::kReadback, 0, /*zeroInit=*/true);
            Buffer dWv4(ctx, dstHalfs * 2, MemPref::kReadback, 0, /*zeroInit=*/true);
            runInto(scalarPipe, bufW.handle(), dScalar);
            runInto(wv4Pipe, bufWv4.handle(), dWv4);
            dScalar.download(outScalar.data(), dstHalfs * 2);
            dWv4.download(outWv4.data(), dstHalfs * 2);
        }
        const bool equal = std::memcmp(outScalar.data(), outWv4.data(), dstHalfs * 2) == 0;
        std::vector<uint16_t>().swap(outScalar);
        std::vector<uint16_t>().swap(outWv4);

        // Timing: device-only output scratch, interleaved rounds, min over kTimingSubmits.
        Buffer dTime(ctx, dstHalfs * 2, MemPref::kDeviceOnly);
        auto   oneSubmit = [&](ComputePipeline &pipe, VkBuffer weights) {
            VkCommandBuffer cmd = runner.allocate();
            runner.begin(cmd);
            for (int r = 0; r < kRepsPerSubmit; ++r)
            {
                pipe.dispatch(cmd, {bufSrc.handle(), weights, bufBias.handle(), dTime.handle()}, &pc, sizeof(pc), gx, gy, gz);
                computeBarrier(ctx, cmd);
            }
            runner.end(cmd);
            const double ms = runner.submitAndWait(cmd);
            vkFreeCommandBuffers(ctx.device(), runner.pool(), 1, &cmd);
            return ms / kRepsPerSubmit;
        };
        double scalarMs = 1e30, wv4Ms = 1e30;
        for (int sub = 0; sub < kTimingSubmits; ++sub)
        {
            scalarMs = std::min(scalarMs, oneSubmit(scalarPipe, bufW.handle()));
            wv4Ms    = std::min(wv4Ms, oneSubmit(wv4Pipe, bufWv4.handle()));
        }

        printf("%d,%d,%d,%d,%d,%dx%d,%dx%d,%lld,%lld,%s,%d,%d,%.4f,%.4f,%+.1f\n", s.N, s.Cin, s.H, s.W, s.Cout, s.KH, s.KW, s.SH, s.SW, (long long) M, (long long) K, route.padW ? "padded" : "packed", tm, equal ? 1 : 0, scalarMs, wv4Ms, (wv4Ms - scalarMs) / scalarMs * 100.0);
        fflush(stdout);
    }
    return 0;
}
