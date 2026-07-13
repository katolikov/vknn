// vknn_zerocopy_simple - the smallest useful zero-copy + cache program.
//
// Loads a pre-compiled .vxm model, hands the engine one caller-owned DMA-BUF per input and per output,
// runs inference with NO host staging buffers, and reads the result straight out of the output DMA-BUF.
// A unified per-model cache file next to the model gives a fast warm start on the second run.
//
// The DMA-BUFs here are declared NCHW fp32 (the default): you fill them with ordinary row-major float
// data and the engine converts to/from its device-native boundary on the GPU, so there is still no host
// copy. If you already hold device-native bytes, declare TensorFormat::Auto instead and the fd is bound
// directly with zero conversion (see IOInfo::deviceFormat / deviceBytes).
//
// The VKNN API used here:
//   Config.cacheFile           - where to write/read the warm-start cache (defaults next to the model).
//   Model::load(path, cfg)     - read the compiled .vxm and hand back a ready-to-run model handle.
//   model.session()            - the live inference session behind the handle.
//   session->inputInfo()       - the inputs the model expects (name, shape, element count).
//   session->outputInfo()      - the outputs the model produces (name, shape, element count).
//   Tensor::fromDmaBuf(fd,...)  - bind a caller-owned DMA-BUF fd as a zero-copy INPUT.
//   Tensor::toDmaBuf(fd,...)    - bind a caller-owned DMA-BUF fd as a zero-copy OUTPUT.
//   model.run(inputs, outputs) - run on the GPU, reading input fds and writing output fds directly.
//   session->config()          - the effective config (used here to print the cache path).
//
//   vknn_zerocopy_simple model.vxm
#include "vknn/model.h"
#include "vknn/session.h"
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

using namespace vknn;

// ---------------------------------------------------------------------------------------------------
// Not VKNN - just plumbing so the demo runs.
//
// A caller-owned DMA-BUF from /dev/dma_heap/system, CPU-mapped (Android). In a real app these fds come
// from the camera / gralloc / ION stack; vknn never allocates them. This stand-in exists ONLY so the
// example is self-contained on an Android device.
// ---------------------------------------------------------------------------------------------------
struct DmaBuffer {
    int    fd         = -1;        // the DMA-BUF file descriptor
    void  *cpuMapping = nullptr;   // CPU-visible mmap of the buffer (fill inputs / read outputs here)
    size_t byteLength = 0;         // size of the buffer in bytes
};

static DmaBuffer allocDmaBuf(size_t bytes) noexcept
{
    DmaBuffer buffer;
    buffer.byteLength = bytes;
    struct {
        uint64_t len;
        uint32_t fd, fd_flags;
        uint64_t heap_flags;
    } request {bytes, 0, O_RDWR | O_CLOEXEC, 0};
    // DMA_HEAP_IOCTL_ALLOC == _IOWR('H', 0, request): dir READ|WRITE (3)<<30, size<<16, 'H'<<8, nr 0.
    constexpr unsigned long kIoctlAlloc = (3UL << 30) | (sizeof(request) << 16) | ('H' << 8) | 0x0;
    int                     heap        = ::open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
    if (heap < 0)
    {
        return buffer;
    }
    if (::ioctl(heap, kIoctlAlloc, &request) == 0)
    {
        buffer.fd = (int) request.fd;
        // mmap reports failure as MAP_FAILED ((void*)-1), not nullptr; normalize it so the caller's
        // `!cpuMapping` check detects a failed mapping instead of dereferencing (void*)-1.
        void *mapped      = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, buffer.fd, 0);
        buffer.cpuMapping = mapped == MAP_FAILED ? nullptr : mapped;
    }
    ::close(heap);
    return buffer;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("usage: %s model.vxm\n", argv[0]);
        return 1;
    }

    // Step 1 - load the model.
    // The cache file (here "<model>.vxm.cache") is written on teardown and reused next time for a fast
    // warm start. Passing an empty Config also works — it defaults to "<model>.cache" next to the model.
    Config cfg;
    cfg.cacheFile    = std::string(argv[1]) + ".cache";
    Model  model     = Model::load(argv[1], cfg);
    if (!model)
    {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    // Step 2 - see what the model expects and produces.
    // inputInfo()/outputInfo() report each boundary tensor's name, NCHW shape and element count, so we
    // can size a DMA-BUF for every one of them without hand-specifying anything.
    Session             *session      = model.session();
    std::vector<IOInfo>  modelInputs   = session->inputInfo();
    std::vector<IOInfo>  modelOutputs  = session->outputInfo();

    // Step 3 - give the engine one caller-owned DMA-BUF per input.
    // Each fd is bound as a zero-copy INPUT declared NCHW fp32 (the default): we write ordinary
    // row-major float data into it and the GPU converts to its device-native boundary on read.
    std::vector<DmaBuffer> inputBuffers, outputBuffers;
    std::vector<Tensor>    inputTensors, outputTensors;
    for (const IOInfo &inputInfo: modelInputs)
    {
        DmaBuffer buffer = allocDmaBuf((size_t) inputInfo.elems * sizeof(float));
        if (buffer.fd < 0 || !buffer.cpuMapping)
        {
            fprintf(stderr, "DMA-BUF alloc failed (need /dev/dma_heap/system; Android only)\n");
            return 2;
        }
        float *inputData = (float *) buffer.cpuMapping; // fill with your real input; a constant here
        for (int64_t elementIndex = 0; elementIndex < inputInfo.elems; ++elementIndex)
        {
            inputData[elementIndex] = 0.5f;
        }
        inputBuffers.push_back(buffer);
        inputTensors.push_back(Tensor::fromDmaBuf(buffer.fd, inputInfo.shape, inputInfo.name)); // NCHW fp32 (default)
    }

    // Step 4 - give the engine one caller-owned DMA-BUF per output.
    // Each fd is bound as a zero-copy OUTPUT: the GPU writes the result straight into it, no host buffer.
    for (const IOInfo &outputInfo: modelOutputs)
    {
        DmaBuffer buffer = allocDmaBuf((size_t) outputInfo.elems * sizeof(float));
        if (buffer.fd < 0 || !buffer.cpuMapping)
        {
            fprintf(stderr, "DMA-BUF alloc failed\n");
            return 2;
        }
        outputBuffers.push_back(buffer);
        outputTensors.push_back(Tensor::toDmaBuf(buffer.fd, outputInfo.shape, outputInfo.name));
    }

    // Step 5 - run inference. The GPU reads the input fds and writes the output fds directly — no host copies.
    model.run(inputTensors, outputTensors);

    // Step 6 - read the first output straight from its DMA-BUF (no readback into a host buffer).
    const float *outputData = (const float *) outputBuffers[0].cpuMapping;
    printf("output '%s' [%lld values]: %.4f %.4f %.4f %.4f ...\n", modelOutputs[0].name.c_str(), (long long) modelOutputs[0].elems, outputData[0], outputData[1], outputData[2], outputData[3]);

    // Step 7 - release the caller-owned buffers (unmap + close each fd).
    for (DmaBuffer &buffer: inputBuffers)
    {
        ::munmap(buffer.cpuMapping, buffer.byteLength);
        ::close(buffer.fd);
    }
    for (DmaBuffer &buffer: outputBuffers)
    {
        ::munmap(buffer.cpuMapping, buffer.byteLength);
        ::close(buffer.fd);
    }
    printf("done (cache: %s)\n", session->config().cacheFile.c_str());
    return 0;
}
