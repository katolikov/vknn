// vknn_op_check - smoke test that dispatches the GPU elementwise `add` kernel over 1M floats,
// compares the result against a CPU reference, and exercises the persistent pipeline cache.
//
// Usage: vknn_op_check [cache-dir]
//   cache-dir  Directory holding pipeline.bin (the serialized VkPipelineCache); created if absent.
//              Defaults to /data/local/tmp/vxrt/cache. The cache is loaded before building the
//              pipeline and written back afterwards, so a second run reuses the driver's compiled
//              pipeline blob.
//
// Prints one line with the max absolute error and the on-disk cache size, and exits 0 (PASS) when
// the error is below 1e-4. Exit codes: 0 pass, 1 no Vulkan device, 2 built without Vulkan,
// 3 numeric mismatch.
#include "vknn/logging.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/stat.h>
#include <vector>
#if defined(VKNN_ENABLE_VULKAN)
#include "backend/vulkan/vk_buffer.h"
#include "backend/vulkan/vk_command.h"
#include "backend/vulkan/vk_pipeline.h"
#endif

int main(int argc, char **argv) {
#if defined(VKNN_ENABLE_VULKAN)
    using namespace vknn;
    const std::string cacheDir = (argc > 1) ? argv[1] : "/data/local/tmp/vxrt/cache";
    ::mkdir(cacheDir.c_str(), 0755);

    vk::VulkanContext ctx;
    if (!ctx.initialized())
    {
        fprintf(stderr, "no Vulkan\n");
        return 1;
    }

    // Seed the PipelineCache from the previous run's blob (empty on the first run).
    std::string       cachePath = cacheDir + "/pipeline.bin";
    std::vector<char> cacheInit;
    if (std::ifstream cf {cachePath, std::ios::binary})
    {
        cacheInit.assign(std::istreambuf_iterator<char>(cf), std::istreambuf_iterator<char>());
    }
    vk::PipelineCache   cache(ctx, cacheInit);
    vk::CommandRunner   runner(ctx);
    vk::ComputePipeline add(ctx, "add", /*numBuffers=*/3, /*pushBytes=*/sizeof(uint32_t), {}, cache.handle());

    const uint32_t     N = 1u << 20; // 1M elements
    vk::Buffer         ba(ctx, N * 4), bb(ctx, N * 4), bc(ctx, N * 4, vk::MemPref::kReadback);
    std::vector<float> a(N), b(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        a[i] = (float) i * 0.5f;
        b[i] = (float) (N - i) * 0.25f;
    }
    ba.upload(a.data(), N * 4);
    bb.upload(b.data(), N * 4);

    // add.comp declares local_size_x = 256; one workgroup covers 256 elements.
    constexpr uint32_t kWorkgroupSize = 256;
    uint32_t           count          = N;
    uint32_t           groups         = (N + kWorkgroupSize - 1) / kWorkgroupSize;
    runner.oneShot([&](VkCommandBuffer cmd) {
        add.dispatch(cmd, {ba.handle(), bb.handle(), bc.handle()}, &count, sizeof(count), groups);
    });

    std::vector<float> c(N);
    bc.download(c.data(), N * 4);
    double maxErr = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        maxErr = std::max(maxErr, (double) std::fabs(c[i] - (a[i] + b[i])));
    }

    // Persist the (now-populated) pipeline cache so the next run skips shader compilation.
    if (std::ofstream of {cachePath, std::ios::binary | std::ios::trunc})
    {
        auto data = cache.getData();
        of.write(data.data(), (std::streamsize) data.size());
    }
    printf("add %u elems: maxAbsErr=%.3e  pipelineCache=%zu bytes  => %s\n", N, maxErr, cache.diskBytes(), maxErr < 1e-4 ? "PASS" : "FAIL");
    return maxErr < 1e-4 ? 0 : 3;
#else
    fprintf(stderr, "built without Vulkan\n");
    return 2;
#endif
}
