// vknn_matmul_vec4_bench — A/B microbench for the vec4-load GEMM kernel family: times the
// scalar-load matmul_tiled_fast kernels against their f16vec4-load twins (the pair matmul.cpp's
// matmulVec4Route routes between) on flat row-major fp16 operands, covering the whole routing
// contract per shape:
//   packed  (N % 4 == 0): both kernels read the same packed B; bNp == N.
//   padded  (N % 4 != 0): the scalar kernel reads packed B, the v4 kernel reads the
//                         padMatMulRowsVec4 zero-padded copy at bNp = roundUpVec4(N) — the "#wv4"
//                         weight-repack layout (the padded geometry carries the K*bNp batch stride).
//   bias    (flagged shapes): the same comparison through the _bias twins with a random fp16 [N]
//                         bias added in the fp32 accumulator.
//
// Timing mirrors matmul.cpp pickTile's candidate race: dedicated scratch operands (never live
// activation buffers), kRepsPerSubmit dispatches per command buffer with a serializing compute
// barrier after each, wall-clocked by CommandRunner::submitAndWait, min over kTimingSubmits
// submissions. Per case it first runs each kernel once into host-readable buffers and byte-compares
// the outputs — the v4 kernels change only the global load width (pad zeros match the scalar
// kernel's column-guard value), so the outputs must be identical.
//
// Usage: vknn_matmul_vec4_bench [MxNxKxB ...]
// Without arguments it runs the built-in shape set (bias pairs run on its flagged shapes; a
// command-line shape runs the no-bias pair for its layout). Every shape must satisfy K % 4 == 0.
#include "backend/vulkan/vk_buffer.h"
#include "backend/vulkan/vk_command.h"
#include "backend/vulkan/vk_context.h"
#include "backend/vulkan/vk_pipeline.h"
#include "core/matmul_tile.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace vknn;
using namespace vknn::vk;

namespace {

    // pickTile's timing discipline: 8 dispatches per submit, min over 5 submits.
    constexpr int kRepsPerSubmit = 8;
    constexpr int kTimingSubmits = 5;

    // matmul_tiled_fast / _v4 tile geometry ({128,128,16}; TM/TN in the shaders).
    constexpr int kTileM = 128;
    constexpr int kTileN = 128;

    // Mirrors MatMulPC in src/backend/vulkan/ops/matmul.cpp. bNp is B's physical row stride: N for
    // packed B, roundUpVec4(N) for the padded copy (only the v4 kernels read it).
    struct MatMulPC {
        int32_t rank, total, M, N, K, aK, bK, bNp;
    };

    struct ShapeSpec {
        int  M, N, K, batch;
        bool bias; // also drive the _bias kernel pair for this shape
    };

    // Bench shapes: VLM vision-encoder class, LLM prefill tile / decode-adjacent, batched
    // CNN/encoder class, small-M class; the two N % 4 != 0 entries exercise the padded-B layout.
    // The bias pairs run on one aligned and one padded shape.
    constexpr ShapeSpec kDefaultShapes[] = {
        {1024, 1152, 1152, 1, true},
        {4300, 1152, 1152, 1, false},
        {256, 4096, 4096, 1, false},
        {1, 4096, 4096, 1, false},
        {784, 512, 512, 8, false},
        {49, 2048, 512, 1, false},
        {1024, 1150, 1152, 1, true},
        {49, 2047, 512, 1, false},
    };

    // float -> IEEE fp16 bit pattern (via the native __fp16 rounding).
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

    inline std::vector<uint16_t> toHalves(const std::vector<float> &src) {
        std::vector<uint16_t> halves(src.size());
        for (size_t i = 0; i < src.size(); ++i)
        {
            halves[i] = f2h(src[i]);
        }
        return halves;
    }

    bool parseShape(const char *s, ShapeSpec &out) {
        out.bias = false;
        return sscanf(s, "%dx%dx%dx%d", &out.M, &out.N, &out.K, &out.batch) == 4;
    }

    // One scalar-vs-v4 comparison + timing, printed as one CSV row. `scalarBufs`/`v4Bufs` differ
    // only in the B and geometry bindings (packed vs padded); slot 2 (D) is a placeholder swapped
    // for the readback/timing destination. `pc.bNp` carries the v4 kernel's row stride and is
    // ignored by the scalar kernel.
    void runCase(VulkanContext &ctx, CommandRunner &runner, const ShapeSpec &s, const char *layout, bool bias, ComputePipeline &scalarPipe, ComputePipeline &v4Pipe, const std::vector<VkBuffer> &scalarBufs,
                 const std::vector<VkBuffer> &v4Bufs, const MatMulPC &pc) {
        const size_t   dHalfs = (size_t) pc.total;
        const uint32_t gx     = (uint32_t) ((pc.N + kTileN - 1) / kTileN);
        const uint32_t gy     = (uint32_t) ((pc.M + kTileM - 1) / kTileM);
        const uint32_t gz     = (uint32_t) s.batch;

        auto bufsWithDst = [&](const std::vector<VkBuffer> &bufs, Buffer &dst) {
            std::vector<VkBuffer> out = bufs;
            out[2]                    = dst.handle();
            return out;
        };

        // Correctness: one dispatch per kernel into host-readable outputs, then a byte compare.
        Buffer dScalar(ctx, dHalfs * 2, MemPref::kReadback, 0, /*zeroInit=*/true);
        Buffer dV4(ctx, dHalfs * 2, MemPref::kReadback, 0, /*zeroInit=*/true);
        runner.oneShot([&](VkCommandBuffer cmd) {
            scalarPipe.dispatch(cmd, bufsWithDst(scalarBufs, dScalar), &pc, sizeof(pc), gx, gy, gz);
        });
        runner.oneShot([&](VkCommandBuffer cmd) {
            v4Pipe.dispatch(cmd, bufsWithDst(v4Bufs, dV4), &pc, sizeof(pc), gx, gy, gz);
        });
        std::vector<uint16_t> outScalar(dHalfs), outV4(dHalfs);
        dScalar.download(outScalar.data(), dHalfs * 2);
        dV4.download(outV4.data(), dHalfs * 2);
        const bool equal = std::memcmp(outScalar.data(), outV4.data(), dHalfs * 2) == 0;

        // Timing: device-only output scratch (pickTile's discipline), kRepsPerSubmit dispatches per
        // submit with a serializing barrier after each, min over kTimingSubmits submits.
        Buffer dTime(ctx, dHalfs * 2, MemPref::kDeviceOnly);
        auto   timeKernel = [&](ComputePipeline &pipe, const std::vector<VkBuffer> &bufs) {
            const std::vector<VkBuffer> timed = bufsWithDst(bufs, dTime);
            double                      best  = 1e30;
            for (int sub = 0; sub < kTimingSubmits; ++sub)
            {
                VkCommandBuffer cmd = runner.allocate();
                runner.begin(cmd);
                for (int r = 0; r < kRepsPerSubmit; ++r)
                {
                    pipe.dispatch(cmd, timed, &pc, sizeof(pc), gx, gy, gz);
                    computeBarrier(ctx, cmd);
                }
                runner.end(cmd);
                best = std::min(best, runner.submitAndWait(cmd));
                vkFreeCommandBuffers(ctx.device(), runner.pool(), 1, &cmd);
            }
            return best / kRepsPerSubmit;
        };
        const double scalarMs = timeKernel(scalarPipe, scalarBufs);
        const double v4Ms     = timeKernel(v4Pipe, v4Bufs);

        printf("%d,%d,%d,%d,%s,%d,%d,%.4f,%.4f,%+.1f\n", s.M, s.N, s.K, s.batch, layout, bias ? 1 : 0, equal ? 1 : 0, scalarMs, v4Ms, (v4Ms - scalarMs) / scalarMs * 100.0);
        fflush(stdout);
    }

} // namespace

int main(int argc, char **argv) {
    std::vector<ShapeSpec> shapes;
    for (int i = 1; i < argc; ++i)
    {
        ShapeSpec s {};
        if (!parseShape(argv[i], s))
        {
            fprintf(stderr, "bad shape '%s' (expect MxNxKxB)\n", argv[i]);
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
    fprintf(stderr, "subgroup=%u maxWGCount=%u/%u/%u\n", ctx.caps().subgroupSize, ctx.caps().maxWorkGroupCount[0], ctx.caps().maxWorkGroupCount[1], ctx.caps().maxWorkGroupCount[2]);

    ComputePipeline fastPipe(ctx, "matmul_tiled_fast_fp16", 4, sizeof(MatMulPC));
    ComputePipeline v4Pipe(ctx, "matmul_tiled_fast_v4_fp16", 4, sizeof(MatMulPC));
    ComputePipeline fastBiasPipe(ctx, "matmul_tiled_fast_bias_fp16", 5, sizeof(MatMulPC));
    ComputePipeline v4BiasPipe(ctx, "matmul_tiled_fast_v4_bias_fp16", 5, sizeof(MatMulPC));

    printf("M,N,K,batch,layout,bias,bytes_equal,scalar_ms,v4_ms,delta_pct\n");
    for (const ShapeSpec &s: shapes)
    {
        if (s.K % (int) kVec4Align != 0)
        {
            fprintf(stderr, "skip %dx%dx%dx%d: the v4 contract needs K %% 4 == 0\n", s.M, s.N, s.K, s.batch);
            continue;
        }
        const bool    padded = s.N % (int) kVec4Align != 0;
        const int     bNp    = (int) roundUpVec4(s.N);
        const size_t  aHalfs = (size_t) s.batch * s.M * s.K;
        const size_t  dHalfs = (size_t) s.batch * s.M * s.N;
        const int64_t bRows  = (int64_t) s.batch * s.K;

        MatMulPC pc {};
        pc.rank  = s.batch > 1 ? 3 : 2;
        pc.total = (int) dHalfs;
        pc.M     = s.M;
        pc.N     = s.N;
        pc.K     = s.K;
        pc.aK    = 1;
        pc.bK    = s.N;
        pc.bNp   = padded ? bNp : s.N;

        // Geometry SSBO layout as flat::uploadFlatGeom: outDim | aStride | bStride, rank each. The
        // tiled kernels read only the leading batch axes (indices 0 .. rank-3); the padded-B
        // geometry carries the padded per-batch stride K*bNp there.
        std::vector<int32_t> geomData, geomPaddedData;
        if (pc.rank == 3)
        {
            geomData       = {s.batch, s.M, s.N, s.M * s.K, s.K, 0, s.K * s.N, 0, 1};
            geomPaddedData = {s.batch, s.M, s.N, s.M * s.K, s.K, 0, s.K * bNp, 0, 1};
        } else
        {
            geomData       = {s.M, s.N, s.K, 0, 0, 1};
            geomPaddedData = geomData;
        }

        std::vector<uint16_t> ha(aHalfs);
        for (size_t i = 0; i < aHalfs; ++i)
        {
            ha[i] = f2h(synth((uint32_t) (i * 3 + 7)));
        }
        std::vector<float> bFloats((size_t) bRows * s.N);
        for (size_t i = 0; i < bFloats.size(); ++i)
        {
            bFloats[i] = synth((uint32_t) (i * 5 + 1));
        }
        const std::vector<uint16_t> hb = toHalves(bFloats);
        std::vector<float>          biasFloats((size_t) s.N);
        for (size_t i = 0; i < biasFloats.size(); ++i)
        {
            biasFloats[i] = synth((uint32_t) (i * 11 + 3));
        }
        const std::vector<uint16_t> hbias = toHalves(biasFloats);

        Buffer bufA(ctx, aHalfs * 2, MemPref::kAuto);
        Buffer bufB(ctx, hb.size() * 2, MemPref::kAuto);
        Buffer bufBias(ctx, hbias.size() * 2, MemPref::kAuto);
        Buffer geom(ctx, geomData.size() * 4, MemPref::kAuto);
        bufA.upload(ha.data(), aHalfs * 2);
        bufB.upload(hb.data(), hb.size() * 2);
        bufBias.upload(hbias.data(), hbias.size() * 2);
        geom.upload(geomData.data(), geomData.size() * 4);

        // Padded-B copy through the production repack helper: the scalar kernel keeps reading the
        // packed bufB, the v4 kernel reads bufBPadded at row stride bNp.
        std::shared_ptr<Buffer> bufBPadded, geomPadded;
        if (padded)
        {
            const std::vector<uint16_t> hbPadded = toHalves(padMatMulRowsVec4(bFloats, bRows, s.N, bNp));
            bufBPadded                           = std::make_shared<Buffer>(ctx, hbPadded.size() * 2, MemPref::kAuto);
            bufBPadded->upload(hbPadded.data(), hbPadded.size() * 2);
            geomPadded = std::make_shared<Buffer>(ctx, geomPaddedData.size() * 4, MemPref::kAuto);
            geomPadded->upload(geomPaddedData.data(), geomPaddedData.size() * 4);
        }
        VkBuffer bV4    = padded ? bufBPadded->handle() : bufB.handle();
        VkBuffer geomV4 = padded ? geomPadded->handle() : geom.handle();

        // Binding order per the shaders: A, B, D, [bias,] geometry; slot 2 (D) is the placeholder
        // runCase swaps for its destinations.
        const char                 *layout = padded ? "padded" : "packed";
        const std::vector<VkBuffer> scalarBufs {bufA.handle(), bufB.handle(), VK_NULL_HANDLE, geom.handle()};
        const std::vector<VkBuffer> v4Bufs {bufA.handle(), bV4, VK_NULL_HANDLE, geomV4};
        runCase(ctx, runner, s, layout, /*bias=*/false, fastPipe, v4Pipe, scalarBufs, v4Bufs, pc);
        if (s.bias)
        {
            const std::vector<VkBuffer> scalarBiasBufs {bufA.handle(), bufB.handle(), VK_NULL_HANDLE, bufBias.handle(), geom.handle()};
            const std::vector<VkBuffer> v4BiasBufs {bufA.handle(), bV4, VK_NULL_HANDLE, bufBias.handle(), geomV4};
            runCase(ctx, runner, s, layout, /*bias=*/true, fastBiasPipe, v4BiasPipe, scalarBiasBufs, v4BiasBufs, pc);
        }
    }
    return 0;
}
