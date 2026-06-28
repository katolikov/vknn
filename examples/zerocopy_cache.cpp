// vknn_zerocopy_cache - true zero-copy: the caller's DMA-BUF fd IS the GPU boundary buffer.
//
// vknn never allocates I/O buffers. The caller allocates one DMA-BUF per model input and per output,
// fills the inputs with DEVICE-NATIVE bytes (row-major NCHW at the model's compute precision — fp16 for
// an fp16 model; the exact size is IOInfo::deviceBytes), hands vknn the fds, runs, and reads the outputs
// straight from the output DMA-BUFs. The GPU reads each input fd and writes each output fd directly: no
// pack, no unpack, no copy. A unified per-model cache file makes the second load fast.
//
// It also SELF-VERIFIES: it runs once the normal way (host fp32 in -> outputs) and once zero-copy (the
// same inputs as fp16 in DMA-BUFs), and checks the two outputs match.
//
//   vknn_zerocopy_cache model.vxm [cache.file]
#include "vknn/dtype.h"
#include "vknn/logging.h"
#include "vknn/model.h"
#include "vknn/session.h"
#include "vknn/tensor_format.h" // NCHW / cBlocks, to format the device-native boundary buffer
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

using namespace vknn;

// The caller's allocator (NOT part of vknn): a DMA-BUF from /dev/dma_heap/system, CPU-mapped. In a real
// app these fds come from the camera / gralloc / ION stack. Returns fd=-1 if unavailable.
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
static std::string shpJoin(const Shape &s) {
    std::string r;
    for (size_t i = 0; i < s.size(); ++i)
    {
        r += (i ? "x" : "") + std::to_string(s[i]);
    }
    return r.empty() ? "scalar" : r;
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

// Format NCHW fp32 -> the device-native fp16 boundary bytes vknn reads/writes (flat NCHW or NC4HW4, per
// IOInfo::deviceFormat). A real app fills its dma-buf this way (or its camera/preprocessor produces it).
static void packDevice(const IOInfo &info, const float *nchw, void *dst) {
    fp16_t *d = (fp16_t *) dst;
    if (info.deviceFormat == TensorFormat::NCHW)
    {
        for (int64_t i = 0; i < info.elems; ++i)
        {
            d[i] = floatToHalf(nchw[i]);
        }
        return;
    }
    NCHW    x  = NCHW::from(info.shape);
    int64_t Cb = cBlocks(x.c);
    for (int64_t n = 0; n < x.n; ++n)
    {
        for (int64_t cb = 0; cb < Cb; ++cb)
        {
            for (int64_t h = 0; h < x.h; ++h)
            {
                for (int64_t w = 0; w < x.w; ++w)
                {
                    for (int l = 0; l < 4; ++l)
                    {
                        int64_t c                                        = cb * 4 + l;
                        float   v                                        = (c < x.c) ? nchw[((n * x.c + c) * x.h + h) * x.w + w] : 0.f;
                        d[(((n * Cb + cb) * x.h + h) * x.w + w) * 4 + l] = floatToHalf(v);
                    }
                }
            }
        }
    }
}
static void unpackDevice(const IOInfo &info, const void *src, float *nchw) {
    const fp16_t *s = (const fp16_t *) src;
    if (info.deviceFormat == TensorFormat::NCHW)
    {
        for (int64_t i = 0; i < info.elems; ++i)
        {
            nchw[i] = halfToFloat(s[i]);
        }
        return;
    }
    NCHW    x  = NCHW::from(info.shape);
    int64_t Cb = cBlocks(x.c);
    for (int64_t n = 0; n < x.n; ++n)
    {
        for (int64_t c = 0; c < x.c; ++c)
        {
            int64_t cb = c / 4, l = c % 4;
            for (int64_t h = 0; h < x.h; ++h)
            {
                for (int64_t w = 0; w < x.w; ++w)
                {
                    nchw[((n * x.c + c) * x.h + h) * x.w + w] = halfToFloat(s[(((n * Cb + cb) * x.h + h) * x.w + w) * 4 + l]);
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2)
    {
        printf("usage: %s model.vxm [cache.file]\n", argv[0]);
        return 1;
    }
    Config cfg;
    if (argc > 2)
    {
        cfg.cacheFile = argv[2]; // else Model::load uses "<model>.cache"
    }
    Model net = Model::load(argv[1], cfg);
    if (!net)
    {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    Session *sess    = net.session();
    auto     inInfo  = sess->inputInfo();
    auto     outInfo = sess->outputInfo();
    printf("model %s: %zu inputs, %zu outputs\n", argv[1], inInfo.size(), outInfo.size());

    // reference fp32 inputs (a ramp; a real app uses sensor/camera data)
    std::vector<std::vector<float>> refIn(inInfo.size());
    std::vector<Tensor>             hostInputs;
    for (size_t i = 0; i < inInfo.size(); ++i)
    {
        refIn[i].resize((size_t) inInfo[i].elems);
        for (int64_t k = 0; k < inInfo[i].elems; ++k)
        {
            refIn[i][(size_t) k] = (float) ((k * 37) % 255) / 255.f - 0.5f;
        }
        hostInputs.push_back(Tensor(refIn[i], inInfo[i].shape, inInfo[i].name));
    }

    // (1) host path — normal fp32 in, fp32 out (the reference)
    auto hostOut = net.run(hostInputs);

    // (2) zero-copy path — one caller DMA-BUF per input/output; inputs as device-native fp16
    std::vector<UserBuf> inBufs, outBufs;
    std::vector<Tensor>  zcInputs, zcOutputs;
    for (size_t i = 0; i < inInfo.size(); ++i)
    {
        UserBuf b = allocDmaBuf((size_t) inInfo[i].deviceBytes);
        if (b.fd < 0 || !b.map)
        {
            fprintf(stderr, "DMA-BUF alloc failed (need /dev/dma_heap/system; Android only)\n");
            return 2;
        }
        packDevice(inInfo[i], refIn[i].data(), b.map); // device-native input bytes into the caller's fd
        inBufs.push_back(b);
        zcInputs.push_back(Tensor::fromDmaBuf(b.fd, inInfo[i].shape, inInfo[i].name));
    }
    for (size_t i = 0; i < outInfo.size(); ++i)
    {
        UserBuf b = allocDmaBuf((size_t) outInfo[i].deviceBytes);
        if (b.fd < 0 || !b.map)
        {
            fprintf(stderr, "DMA-BUF alloc failed\n");
            return 2;
        }
        outBufs.push_back(b);
        zcOutputs.push_back(Tensor::toDmaBuf(b.fd, outInfo[i].shape, outInfo[i].name));
    }
    net.run(zcInputs, zcOutputs); // GPU reads input fds, writes output fds — no host copies

    // (3) verify: zero-copy output (fp16 in the DMA-BUF) == host output (fp32)
    double worst = 0;
    bool   ok    = true;
    for (size_t i = 0; i < outInfo.size(); ++i)
    {
        const Tensor *h = findTensor(hostOut, outInfo[i].name);
        if (!h)
        {
            ok = false;
            continue;
        }
        std::vector<float> zc((size_t) outInfo[i].elems);
        unpackDevice(outInfo[i], outBufs[i].map, zc.data()); // device-native fd bytes -> NCHW fp32
        double mx = 0;
        for (int64_t k = 0; k < outInfo[i].elems; ++k)
        {
            mx = std::max(mx, std::fabs((double) zc[(size_t) k] - (double) h->values()[(size_t) k]));
        }
        worst = std::max(worst, mx);
        printf("  output '%s' %s  fd=%d  zero-copy vs host maxAbsErr=%.3e\n", outInfo[i].name.c_str(), shpJoin(outInfo[i].shape).c_str(), outBufs[i].fd, mx);
    }
    ok = ok && worst < 1e-3;
    printf("zero-copy %s host  (maxAbsErr=%.3e)\n", ok ? "MATCHES" : "DIFFERS FROM", worst);

    for (auto &b: inBufs)
    {
        freeDmaBuf(b);
    }
    for (auto &b: outBufs)
    {
        freeDmaBuf(b);
    }
    printf("cache file: %s\n", sess->config().cacheFile.c_str());
    return ok ? 0 : 3;
}
