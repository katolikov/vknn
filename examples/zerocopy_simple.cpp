// vknn_zerocopy_simple - the smallest useful zero-copy + cache program.
//
// Loads a pre-compiled .vxm (a unified per-model cache file gives a fast warm start on the second run),
// hands the engine one caller-owned DMA-BUF per input and per output, runs with NO host staging buffers,
// and reads the result straight out of the output DMA-BUF.
//
// The DMA-BUFs here are declared NCHW fp32 (the default) — you fill them with ordinary row-major float
// data and the engine converts to/from its device-native boundary on the GPU, so there is still no host
// copy. If you already hold device-native bytes, declare TensorFormat::Auto instead and the fd is bound
// directly with zero conversion (see IOInfo::deviceFormat / deviceBytes).
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

// A caller-owned DMA-BUF from /dev/dma_heap/system, CPU-mapped (Android). In a real app these fds come
// from the camera / gralloc / ION stack; vknn never allocates them.
struct Buf {
    int    fd  = -1;
    void  *map = nullptr;
    size_t len = 0;
};
static Buf allocDmaBuf(size_t bytes) {
    Buf b;
    b.len = bytes;
    struct {
        uint64_t len;
        uint32_t fd, fd_flags;
        uint64_t heap_flags;
    } d {bytes, 0, O_RDWR | O_CLOEXEC, 0};
    constexpr unsigned long kAlloc = (3UL << 30) | (sizeof(d) << 16) | ('H' << 8) | 0x0;
    int                     heap   = ::open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
    if (heap < 0)
    {
        return b;
    }
    if (::ioctl(heap, kAlloc, &d) == 0)
    {
        b.fd  = (int) d.fd;
        b.map = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, b.fd, 0);
    }
    ::close(heap);
    return b;
}

int main(int argc, char **argv) {
    if (argc < 2)
    {
        printf("usage: %s model.vxm\n", argv[0]);
        return 1;
    }
    // The cache file (here "<model>.vxm.cache") is written on teardown and reused next time for a fast
    // warm start. Passing an empty Config also works — it defaults to "<model>.cache" next to the model.
    Config cfg;
    cfg.cacheFile = std::string(argv[1]) + ".cache";
    Model net     = Model::load(argv[1], cfg);
    if (!net)
    {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    Session *sess = net.session();
    auto     ins  = sess->inputInfo();
    auto     outs = sess->outputInfo();

    // one DMA-BUF per input (NCHW fp32) ...
    std::vector<Buf>    inBufs, outBufs;
    std::vector<Tensor> zin, zout;
    for (const auto &info: ins)
    {
        Buf b = allocDmaBuf((size_t) info.elems * sizeof(float));
        if (b.fd < 0 || !b.map)
        {
            fprintf(stderr, "DMA-BUF alloc failed (need /dev/dma_heap/system; Android only)\n");
            return 2;
        }
        float *px = (float *) b.map; // fill with your real input; a constant here
        for (int64_t k = 0; k < info.elems; ++k)
        {
            px[k] = 0.5f;
        }
        inBufs.push_back(b);
        zin.push_back(Tensor::fromDmaBuf(b.fd, info.shape, info.name)); // declared NCHW fp32 (default)
    }
    // ... and one per output.
    for (const auto &info: outs)
    {
        Buf b = allocDmaBuf((size_t) info.elems * sizeof(float));
        if (b.fd < 0 || !b.map)
        {
            fprintf(stderr, "DMA-BUF alloc failed\n");
            return 2;
        }
        outBufs.push_back(b);
        zout.push_back(Tensor::toDmaBuf(b.fd, info.shape, info.name));
    }

    net.run(zin, zout); // GPU reads the input fds and writes the output fds directly — no host copies

    // read the first output straight from its DMA-BUF
    const float *y = (const float *) outBufs[0].map;
    printf("output '%s' [%lld values]: %.4f %.4f %.4f %.4f ...\n", outs[0].name.c_str(), (long long) outs[0].elems, y[0], y[1], y[2], y[3]);

    for (auto &b: inBufs)
    {
        ::munmap(b.map, b.len);
        ::close(b.fd);
    }
    for (auto &b: outBufs)
    {
        ::munmap(b.map, b.len);
        ::close(b.fd);
    }
    printf("done (cache: %s)\n", sess->config().cacheFile.c_str());
    return 0;
}
