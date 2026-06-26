// vknn_op_check - GPU elementwise add vs CPU reference, exercising the pipeline cache.
#include "vknn/logging.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

    vk::PipelineCache   cache(ctx, cacheDir + "/pipeline.bin");
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

    uint32_t count  = N;
    uint32_t groups = (N + 255) / 256;
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

    cache.save();
    printf("add %u elems: maxAbsErr=%.3e  pipelineCache=%zu bytes  => %s\n", N, maxErr, cache.diskBytes(), maxErr < 1e-4 ? "PASS" : "FAIL");
    return maxErr < 1e-4 ? 0 : 3;
#else
    fprintf(stderr, "built without Vulkan\n");
    return 2;
#endif
}
