// vknn_op_check - a first look at running a single GPU kernel with VKNN's low-level backend.
//
// This example is a smoke test: it fills two arrays of 1M floats, adds them together on the GPU
// with VKNN's built-in elementwise `add` compute kernel, checks the result against the same sum
// computed on the CPU, and reports the largest disagreement. Along the way it shows the persistent
// pipeline cache: a small on-disk blob the driver reuses so a second run skips shader compilation.
//
// Unlike most examples, this one talks to VKNN's Vulkan BACKEND directly (vk::* types) instead of the
// high-level Model API, so you can see the raw pieces a kernel dispatch is made of.
//
// The VKNN API used here:
//   vk::VulkanContext ctx;                     // picks a Vulkan device; ctx.initialized() says if one exists
//   vk::PipelineCache cache(ctx, bytes);       // seed the driver's compiled-pipeline cache from prior bytes
//   vk::CommandRunner runner(ctx);             // records + submits command buffers and waits for the GPU
//   vk::ComputePipeline add(ctx, "add", ...);  // load the embedded "add" SPIR-V kernel as a runnable pipeline
//   vk::Buffer buf(ctx, bytes[, MemPref]);     // a GPU buffer; upload()/download() copy host <-> device
//   runner.oneShot([&](VkCommandBuffer cmd){ add.dispatch(cmd, {buffers}, push, n, groups); });  // run it
//   cache.getData() / cache.diskBytes();       // serialize the now-warm pipeline cache back out
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

    // ----- Not VKNN - just plumbing so the demo runs -----
    // Read the cache directory from argv (default under /data/local/tmp) and make sure it exists.
    const std::string cacheDir = (argc > 1) ? argv[1] : "/data/local/tmp/vxrt/cache";
    ::mkdir(cacheDir.c_str(), 0755);

    // Step 1 - open a Vulkan device. VulkanContext discovers and initializes a GPU; without one there
    // is nothing to run the kernel on, so bail out early.
    vk::VulkanContext context;
    if (!context.initialized())
    {
        fprintf(stderr, "no Vulkan\n");
        return 1;
    }

    // Step 2 - seed the pipeline cache from the previous run's blob (empty on the very first run). This
    // is a driver-owned cache of compiled pipelines; handing it back the old bytes lets pipeline
    // creation reuse already-compiled shaders instead of recompiling them.
    // ----- Not VKNN - just plumbing: slurp the file into a byte vector -----
    const std::string cachePath = cacheDir + "/pipeline.bin";
    std::vector<char> cacheInitBytes;
    if (std::ifstream cacheFile {cachePath, std::ios::binary})
    {
        cacheInitBytes.assign(std::istreambuf_iterator<char>(cacheFile), std::istreambuf_iterator<char>());
    }
    // ----- Back to VKNN -----
    vk::PipelineCache pipelineCache(context, cacheInitBytes);

    // Step 3 - build the runnable pieces. CommandRunner records and submits work to the compute queue;
    // ComputePipeline loads the embedded "add" SPIR-V kernel, wired to 3 storage buffers (two inputs +
    // one output) and a 4-byte push constant (the element count), reusing the pipeline cache.
    vk::CommandRunner   commandRunner(context);
    vk::ComputePipeline addPipeline(context, "add", /*numBuffers=*/3, /*pushBytes=*/sizeof(uint32_t), {}, pipelineCache.handle());

    // Step 4 - allocate GPU buffers and fill the two inputs. The output buffer asks for kReadback
    // memory so the CPU can read the GPU's result back efficiently.
    const uint32_t     elementCount = 1u << 20; // 1M elements
    vk::Buffer         inputBufferA(context, elementCount * 4);
    vk::Buffer         inputBufferB(context, elementCount * 4);
    vk::Buffer         outputBuffer(context, elementCount * 4, vk::MemPref::kReadback);
    std::vector<float> inputA(elementCount), inputB(elementCount);
    for (uint32_t i = 0; i < elementCount; ++i)
    {
        // ----- Not VKNN - just arbitrary test data so the sum is easy to check -----
        inputA[i] = (float) i * 0.5f;
        inputB[i] = (float) (elementCount - i) * 0.25f;
    }
    // upload() memcpys host data into the (host-visible) GPU buffers - no staging copy on this hardware.
    inputBufferA.upload(inputA.data(), elementCount * 4);
    inputBufferB.upload(inputB.data(), elementCount * 4);

    // Step 5 - dispatch the kernel. add.comp declares local_size_x = 256, so each workgroup handles 256
    // elements; launch enough workgroups to cover every element. oneShot records the dispatch, submits
    // it, and waits for the GPU to finish. The element count is passed as the push constant.
    constexpr uint32_t kWorkgroupSize = 256;
    uint32_t           count          = elementCount;
    uint32_t           groups         = (elementCount + kWorkgroupSize - 1) / kWorkgroupSize;
    commandRunner.oneShot([&](VkCommandBuffer cmd) {
        addPipeline.dispatch(cmd, {inputBufferA.handle(), inputBufferB.handle(), outputBuffer.handle()}, &count, sizeof(count), groups);
    });

    // Step 6 - read the result back and compare against the CPU sum, tracking the worst element.
    std::vector<float> gpuResult(elementCount);
    outputBuffer.download(gpuResult.data(), elementCount * 4);
    double maxErr = 0;
    for (uint32_t i = 0; i < elementCount; ++i)
    {
        maxErr = std::max(maxErr, (double) std::fabs(gpuResult[i] - (inputA[i] + inputB[i])));
    }

    // Step 7 - persist the now-warm pipeline cache so the next run skips shader compilation. getData()
    // serializes the current cache; diskBytes() reports how large it is on disk.
    // ----- Not VKNN - just plumbing: write the bytes back to the file -----
    if (std::ofstream cacheFile {cachePath, std::ios::binary | std::ios::trunc})
    {
        std::vector<char> cacheData = pipelineCache.getData();
        cacheFile.write(cacheData.data(), (std::streamsize) cacheData.size());
    }

    // Step 8 - report. PASS when the GPU and CPU agree to within 1e-4.
    printf("add %u elems: maxAbsErr=%.3e  pipelineCache=%zu bytes  => %s\n", elementCount, maxErr, pipelineCache.diskBytes(), maxErr < 1e-4 ? "PASS" : "FAIL");
    return maxErr < 1e-4 ? 0 : 3;
#else
    fprintf(stderr, "built without Vulkan\n");
    return 2;
#endif
}
