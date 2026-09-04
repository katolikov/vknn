// vknn_zerocopy_bench - the per-run wall cost of the two I/O paths, on one model, side by side.
//
// Host mode hands the engine IOTensor payloads: every run packs the fp32 input into the device
// boundary on the CPU, uploads it, downloads the output and unpacks it. Zero-copy mode binds one
// caller-owned DMA-BUF per input and per output (NCHW fp32, the default declaration), so the GPU
// converts straight from and into the caller's buffers and no host copy exists. This program times
// the loops with the same model and reports min / median wall per run, which is the number an
// application pipeline sees. The DMA-BUFs come from /dev/dma_heap/system (Android only).
//
//   vknn_zerocopy_bench model.vxm [--repeat N] [--precision low|normal|high] [--power normal|high]
#include "vknn/pinned_host_memory.h"
#include "vknn/model.h"
#include "vknn/session.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

using namespace vknn;

namespace {

    struct DmaBuffer {
        int    fd         = -1;
        void  *cpuMapping = nullptr;
        size_t byteLength = 0;
    };

    // A system dma-heap allocation, CPU-mapped; stands in for the buffer a camera or gralloc hands an
    // application. Not part of vknn.
    DmaBuffer allocDmaBuf(size_t bytes) noexcept {
        DmaBuffer buffer;
        buffer.byteLength = bytes;
        struct {
            uint64_t len;
            uint32_t fd, fd_flags;
            uint64_t heap_flags;
        } request {bytes, 0, O_RDWR | O_CLOEXEC, 0};
        constexpr unsigned long kIoctlAlloc = (3UL << 30) | (sizeof(request) << 16) | ('H' << 8) | 0x0;
        int                     heap        = ::open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
        if (heap < 0)
        {
            return buffer;
        }
        if (::ioctl(heap, kIoctlAlloc, &request) == 0)
        {
            buffer.fd         = (int) request.fd;
            void *mapped      = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, buffer.fd, 0);
            buffer.cpuMapping = mapped == MAP_FAILED ? nullptr : mapped;
        }
        ::close(heap);
        return buffer;
    }

    void releaseDmaBuf(DmaBuffer &buffer) noexcept {
        if (buffer.cpuMapping)
        {
            ::munmap(buffer.cpuMapping, buffer.byteLength);
        }
        if (buffer.fd >= 0)
        {
            ::close(buffer.fd);
        }
        buffer = DmaBuffer {};
    }

    const char *optValue(int argc, char **argv, const char *name, const char *fallback) {
        for (int i = 2; i + 1 < argc; ++i)
        {
            if (!strcmp(argv[i], name))
            {
                return argv[i + 1];
            }
        }
        return fallback;
    }

    Precision precisionFromStr(const char *s) {
        return !strcmp(s, "high") ? Precision::High : !strcmp(s, "normal") ? Precision::Normal : Precision::Low;
    }

    // Wall milliseconds of each call of `runOnce`, the first (cold) call discarded.
    struct WallStats {
        double minMs = 0, medianMs = 0;
        int    runs  = 0;
    };

    template <class F>
    WallStats timeRuns(int repeat, F runOnce) {
        std::vector<double> ms;
        for (int i = 0; i < repeat + 1; ++i)
        {
            auto t0 = std::chrono::steady_clock::now();
            if (!runOnce())
            {
                return WallStats {};
            }
            auto t1 = std::chrono::steady_clock::now();
            if (i > 0)
            {
                ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
        }
        std::sort(ms.begin(), ms.end());
        WallStats stats;
        stats.runs     = (int) ms.size();
        stats.minMs    = ms.empty() ? 0.0 : ms.front();
        stats.medianMs = ms.empty() ? 0.0 : ms[ms.size() / 2];
        return stats;
    }

} // namespace

int main(int argc, char **argv) {
    if (argc < 2)
    {
        printf("usage: %s model.vxm [--repeat N] [--precision low|normal|high] [--power normal|high] [--timing]\n", argv[0]);
        return 1;
    }
    const int repeat = std::max(1, atoi(optValue(argc, argv, "--repeat", "30")));
    Config    cfg;
    cfg.precision = precisionFromStr(optValue(argc, argv, "--precision", "low"));
    cfg.power     = !strcmp(optValue(argc, argv, "--power", "normal"), "high") ? Power::High : Power::Normal;
    cfg.cacheFile = std::string(argv[1]) + ".cache";
    for (int i = 2; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--timing"))
        {
            cfg.timing = true; // the engine prints pack / submit+gpu / unpack per run
        }
    }
    Model model   = Model::load(argv[1], cfg);
    if (!model)
    {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    Session                  *session      = model.session();
    const std::vector<IOInfo> modelInputs  = session->inputInfo();
    const std::vector<IOInfo> modelOutputs = session->outputInfo();

    // --- host mode: IOTensor payloads, the engine packs / uploads / downloads / unpacks per run ---
    std::vector<IOTensor> hostInputs, hostOutputs;
    for (const IOInfo &info: modelInputs)
    {
        IOTensor input;
        input.name  = info.name;
        input.shape = info.shape;
        input.dtype = info.dtype;
        input.data.assign((size_t) (numElements(info.shape) * (int64_t) dtypeSize(info.dtype)), 0);
        float *values = reinterpret_cast<float *>(input.data.data());
        for (int64_t i = 0; info.dtype == DType::Float32 && i < info.elems; ++i)
        {
            values[i] = 0.5f;
        }
        hostInputs.push_back(std::move(input));
    }
    const WallStats host = timeRuns(repeat, [&] {
        return session->run(hostInputs, hostOutputs) == Status::Ok;
    });
    // Pinned host blocks (PinnedHostMemory): the same IOTensor loop, the payloads living in blocks the
    // engine binds to the GPU directly (no copy in either direction on a device that imports host
    // memory; a copy elsewhere). The blocks are allocated once and reused every run, as an
    // application would.
    std::vector<IOTensor> pinnedInputs, pinnedOutputs;
    for (const IOInfo &info: modelInputs)
    {
        IOTensor input;
        input.name   = info.name;
        input.shape  = info.shape;
        input.dtype  = info.dtype;
        input.pinned = PinnedHostMemory::alloc((size_t) info.elems * dtypeSize(info.dtype));
        if (input.pinned && info.dtype == DType::Float32)
        {
            float *values = reinterpret_cast<float *>(input.pinned->data());
            for (int64_t i = 0; i < info.elems; ++i)
            {
                values[i] = 0.5f;
            }
        }
        pinnedInputs.push_back(std::move(input));
    }
    for (const IOInfo &info: modelOutputs)
    {
        IOTensor output;
        output.name   = info.name;
        output.shape  = info.shape;
        output.dtype  = info.dtype;
        output.pinned = PinnedHostMemory::alloc((size_t) info.elems * dtypeSize(info.dtype));
        pinnedOutputs.push_back(std::move(output));
    }
    const WallStats pinned = timeRuns(repeat, [&] {
        return session->run(pinnedInputs, pinnedOutputs) == Status::Ok;
    });
    // The pinned outputs must be the host outputs byte for byte: the same GPU convert produces both.
    bool   pinnedMatches = pinnedOutputs.size() == hostOutputs.size();
    size_t pinnedDiffs   = 0;
    for (size_t i = 0; pinnedMatches && i < pinnedOutputs.size(); ++i)
    {
        const IOTensor &p = pinnedOutputs[i], &h = hostOutputs[i];
        if (!p.pinned || p.pinned->bytes() < h.data.size())
        {
            pinnedMatches = false;
            break;
        }
        for (size_t b = 0; b < h.data.size(); ++b)
        {
            pinnedDiffs += p.pinned->data()[b] != h.data[b];
        }
    }

    // --- zero-copy mode: one caller-owned DMA-BUF per boundary, NCHW fp32, converted on the GPU ---
    std::vector<DmaBuffer> inputBuffers, outputBuffers;
    std::vector<Tensor>    inputTensors, outputTensors;
    bool                   dmaOk = true;
    for (const IOInfo &info: modelInputs)
    {
        DmaBuffer buffer = allocDmaBuf((size_t) info.elems * sizeof(float));
        dmaOk            = dmaOk && buffer.fd >= 0 && buffer.cpuMapping != nullptr;
        if (dmaOk)
        {
            float *values = (float *) buffer.cpuMapping;
            for (int64_t i = 0; i < info.elems; ++i)
            {
                values[i] = 0.5f;
            }
            inputTensors.push_back(Tensor::fromDmaBuf(buffer.fd, info.shape, info.name));
        }
        inputBuffers.push_back(buffer);
    }
    for (const IOInfo &info: modelOutputs)
    {
        DmaBuffer buffer = allocDmaBuf((size_t) info.elems * sizeof(float));
        dmaOk            = dmaOk && buffer.fd >= 0 && buffer.cpuMapping != nullptr;
        if (dmaOk)
        {
            outputTensors.push_back(Tensor::toDmaBuf(buffer.fd, info.shape, info.name));
        }
        outputBuffers.push_back(buffer);
    }
    WallStats zero;
    if (dmaOk)
    {
        zero = timeRuns(repeat, [&] {
            return !model.run(inputTensors, outputTensors).empty(); // the bound output tensors, filled in place
        });
    } else
    {
        fprintf(stderr, "DMA-BUF alloc failed (need /dev/dma_heap/system; Android only): zero-copy loop skipped\n");
    }
    for (DmaBuffer &buffer: inputBuffers)
    {
        releaseDmaBuf(buffer);
    }
    for (DmaBuffer &buffer: outputBuffers)
    {
        releaseDmaBuf(buffer);
    }
    printf("wall ms/run (n=%d, first run discarded)\n", host.runs);
    printf("  host mode (IOTensor payloads): min %.3f  median %.3f\n", host.minMs, host.medianMs);
    printf("  pinned host (PinnedHostMemory): min %.3f  median %.3f  (outputs %s the host mode's%s)\n", pinned.minMs, pinned.medianMs, pinnedMatches && pinnedDiffs == 0 ? "byte-identical to" : "DIFFER from", pinnedMatches ? "" : ": shape/count mismatch");
    if (dmaOk)
    {
        printf("  zero-copy (DMA-BUF in/out):    min %.3f  median %.3f\n", zero.minMs, zero.medianMs);
    }
    return 0;
}
