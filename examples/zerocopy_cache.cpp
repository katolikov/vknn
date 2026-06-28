// vknn_zerocopy_cache - run a .vxm with caller-owned DMA-BUF I/O and a unified cache file.
//
// The caller allocates one DMA-BUF per model input and per model output, fills the inputs, hands vknn
// the fds, runs, and reads the outputs straight from the output DMA-BUFs. vknn never allocates these
// I/O buffers — it reads each input directly from the caller's fd and writes each output directly into
// the caller's fd (Tensor::fromDmaBuf / Tensor::toDmaBuf). A unified per-model cache file makes the
// second load fast (skips shader compile + conv autotune + Winograd transform); it is created on the
// first run and updated on teardown.
//
//   vknn_zerocopy_cache model.vxm [cache.file]
//
// Shown for a 3-input / 3-output model, but the loops handle any input/output count.
#include "vknn/logging.h"
#include "vknn/model.h"
#include "vknn/session.h" // Session::config() for the resolved cache-file path
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

using namespace vknn;

// The caller's allocator (NOT part of vknn): a DMA-BUF from /dev/dma_heap/system, CPU-mapped. In a
// real app these fds come from the camera / gralloc / ION stack. Returns fd=-1 if unavailable.
struct UserBuf {
    int    fd    = -1;
    void  *map   = nullptr;
    size_t bytes = 0;
};
static UserBuf allocDmaBuf(size_t bytes) {
    UserBuf b;
    b.bytes = bytes;
    struct alloc_data {
        uint64_t len;
        uint32_t fd;
        uint32_t fd_flags;
        uint64_t heap_flags;
    } d {};
    constexpr unsigned long kIoctlAlloc = (3UL << 30) | (sizeof(alloc_data) << 16) | ('H' << 8) | 0x0;
    int                     h           = ::open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
    if (h < 0)
    {
        return b;
    }
    d.len      = bytes;
    d.fd_flags = O_RDWR | O_CLOEXEC;
    if (::ioctl(h, kIoctlAlloc, &d) == 0)
    {
        b.fd  = (int) d.fd;
        b.map = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, b.fd, 0);
        if (b.map == MAP_FAILED)
        {
            b.map = nullptr;
        }
    }
    ::close(h);
    return b;
}
static void freeDmaBuf(UserBuf &b) {
    if (b.map)
    {
        ::munmap(b.map, b.bytes);
    }
    if (b.fd >= 0)
    {
        ::close(b.fd);
    }
}

int main(int argc, char **argv) {
    if (argc < 2)
    {
        printf("usage: %s model.vxm [cache.file]\n", argv[0]);
        return 1;
    }
    std::string modelPath = argv[1];
    Config      cfg;
    if (argc > 2)
    {
        cfg.cacheFile = argv[2]; // else Model::load uses "<model>.cache"
    }

    Model net = Model::load(modelPath, cfg);
    if (!net)
    {
        fprintf(stderr, "failed to load %s\n", modelPath.c_str());
        return 1;
    }
    auto inInfo  = net.inputs();
    auto outInfo = net.outputs();
    printf("model %s: %zu inputs, %zu outputs\n", modelPath.c_str(), inInfo.size(), outInfo.size());

    // One caller DMA-BUF per input (filled here) and per output (written by vknn).
    std::vector<UserBuf> inBufs, outBufs;
    std::vector<Tensor>  inputs, outputs;
    for (const auto &ii: inInfo)
    {
        UserBuf b = allocDmaBuf((size_t) ii.count * sizeof(float));
        if (b.fd < 0 || !b.map)
        {
            fprintf(stderr, "DMA-BUF alloc failed (need /dev/dma_heap/system; Android only)\n");
            return 2;
        }
        float *f = (float *) b.map; // fill the input (here a ramp; in a real app: sensor/camera data)
        for (int64_t k = 0; k < ii.count; ++k)
        {
            f[k] = (float) (k % 255) / 255.f;
        }
        inBufs.push_back(b);
        inputs.push_back(Tensor::fromDmaBuf(b.fd, ii.shape, ii.name));
    }
    for (const auto &oi: outInfo)
    {
        UserBuf b = allocDmaBuf((size_t) oi.count * sizeof(float));
        if (b.fd < 0 || !b.map)
        {
            fprintf(stderr, "DMA-BUF alloc failed\n");
            return 2;
        }
        outBufs.push_back(b);
        outputs.push_back(Tensor::toDmaBuf(b.fd, oi.shape, oi.name));
    }

    // Inputs are read from their fds; outputs are written into their fds. No vknn-side I/O buffers.
    auto result = net.run(inputs, outputs);
    if (result.empty())
    {
        fprintf(stderr, "run failed\n");
        return 3;
    }
    for (size_t i = 0; i < outInfo.size(); ++i)
    {
        const float *f = (const float *) outBufs[i].map; // read straight from the caller's DMA-BUF
        printf("  output '%s' %s  fd=%d  [0]=%.5f\n", outInfo[i].name.c_str(), outInfo[i].shapeString().c_str(), outBufs[i].fd, f[0]);
    }

    for (auto &b: inBufs)
    {
        freeDmaBuf(b);
    }
    for (auto &b: outBufs)
    {
        freeDmaBuf(b);
    }
    // ~Model -> ~Session writes/updates the unified cache file (so the next load is warm). The session
    // resolved the actual path (the "<model>.cache" default when none was given).
    printf("done. cache file: %s\n", net.session()->config().cacheFile.c_str());
    return 0;
}
