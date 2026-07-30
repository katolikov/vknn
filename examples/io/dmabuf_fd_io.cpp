// vknn_dmabuf_fd_io - zero-copy inference across CALLER-PROVIDED dma-buf fds, both in and out.
//
// This example takes fds it does NOT own or allocate: an input fd already holding the model's input
// bytes and an output fd sized to receive the result. In a real deployment those fds come from the
// camera / gralloc / another process / an ION or dma-heap allocator owned by the caller — vknn never
// allocates them and never closes them. Both boundaries are bound zero-copy: the GPU reads the input
// straight from its fd and writes the output straight into its fd, with no host staging buffer on
// either side.
//
// The zero-copy contract:
//   - Tensor::fromDmaBuf / toDmaBuf declare each fd's boundary LAYOUT + DTYPE. When they match the
//     model's device-native boundary (IOInfo::deviceFormat / deviceDtype) the fd is bound directly;
//     otherwise the GPU converts on read/write. TensorFormat::Auto means "already device-native".
//   - Ownership stays with the CALLER for the whole call. vknn imports each fd for the duration of
//     run() and releases its import on return; it never takes ownership, so this example does NOT
//     close the fds. Closing them is the caller's job, once it is done with the buffers.
//
//   vknn_dmabuf_fd_io model.vxm <input_fd> <output_fd>
//   vknn_dmabuf_fd_io model.vxm            # no fds given: acquireCallerFd() stubs them (see below)
#include "vknn/model.h"
#include "vknn/session.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>

using namespace vknn;

// Stand-in for however the caller obtains a dma-buf fd of at least `bytes`. In production this fd comes
// from the camera / gralloc / another process / an ION or dma-heap allocator owned by the caller, and
// is passed into this code (e.g. over a binder call or as a CLI argument) already populated for an
// input, or pre-sized for an output. The dma-heap allocation here exists ONLY so the example is
// self-contained on an Android device; it is not part of vknn and not what a real integration does.
static int acquireCallerFd(size_t bytes) {
    struct alloc_data {
        uint64_t len;
        uint32_t fd;
        uint32_t fd_flags;
        uint64_t heap_flags;
    } d {};
    // DMA_HEAP_IOCTL_ALLOC == _IOWR('H', 0, alloc_data): dir READ|WRITE (3)<<30, size<<16, 'H'<<8, nr 0.
    constexpr unsigned long kIoctlAlloc = (3UL << 30) | (sizeof(alloc_data) << 16) | ('H' << 8) | 0x0;
    int                     heap        = ::open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
    if (heap < 0)
    {
        return -1;
    }
    d.len      = bytes;
    d.fd_flags = O_RDWR | O_CLOEXEC;
    int fd     = (::ioctl(heap, kIoctlAlloc, &d) == 0) ? (int) d.fd : -1;
    ::close(heap);
    return fd;
}

int main(int argc, char **argv) {
    if (argc < 2)
    {
        printf("usage: %s model.vxm [<input_fd> <output_fd>]\n", argv[0]);
        return 1;
    }

    // A unified per-model cache file next to the model gives a fast warm start on the second run.
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
    if (ins.empty() || outs.empty())
    {
        fprintf(stderr, "model needs at least one input and one output\n");
        return 1;
    }

    // Declare each boundary as NCHW fp32: ordinary row-major float bytes in the caller's fd, which the
    // GPU converts to/from its device-native boundary on read/write (still no host copy). To bind an fd
    // with zero conversion, declare TensorFormat::Auto and hold device-native bytes (see IOInfo).
    const TensorFormat kLayout = TensorFormat::NCHW;
    const DType        kDtype  = DType::Float32;
    const IOInfo      &inInfo  = ins.front();
    const IOInfo      &outInfo = outs.front();

    // The input fd already holds the input bytes; the output fd is sized to receive the result. Both
    // come from the caller (a CLI arg here, or the acquireCallerFd() stand-in when none is given).
    bool ownFds  = false; // true only for the acquireCallerFd() stand-in, whose fds we then clean up
    int  inputFd = -1, outputFd = -1;
    if (argc >= 4)
    {
        inputFd  = ::atoi(argv[2]);
        outputFd = ::atoi(argv[3]);
    } else
    {
        inputFd  = acquireCallerFd((size_t) inInfo.elems * sizeof(float));
        outputFd = acquireCallerFd((size_t) outInfo.elems * sizeof(float));
        ownFds   = true;
        if (inputFd < 0 || outputFd < 0)
        {
            fprintf(stderr, "no fds given and acquireCallerFd() failed (needs /dev/dma_heap/system; Android only)\n");
            if (inputFd >= 0)
            {
                ::close(inputFd);
            }
            if (outputFd >= 0)
            {
                ::close(outputFd);
            }
            return 2;
        }
    }

    // Bind the input fd as a zero-copy INPUT and the output fd as a zero-copy OUTPUT. The names tie each
    // binding to its model boundary tensor; the shape is the logical NCHW shape the fd holds/receives.
    std::vector<Tensor> inputs  = {Tensor::fromDmaBuf(inputFd, inInfo.shape, inInfo.name, kLayout, kDtype)};
    std::vector<Tensor> outputs = {Tensor::toDmaBuf(outputFd, outInfo.shape, outInfo.name, kLayout, kDtype)};

    // Run. The GPU reads the input fd and writes the output fd directly — no host input or output
    // buffer. The returned tensor for a dma-buf-bound output carries no host data; the result lives in
    // the caller's output fd, which the caller reads (e.g. by mmap-ing it).
    net.run(inputs, outputs);
    printf("ran '%s': input fd %d -> output '%s' fd %d (%lld elems), zero-copy both ways\n", argv[1], inputFd, outInfo.name.c_str(), outputFd,
           (long long) outInfo.elems);

    // Ownership rule: the fds are CALLER-owned, so this example does NOT close caller-supplied fds — the
    // caller closes them when done. The only fds cleaned up here are the ones acquireCallerFd() minted
    // as a stand-in for a caller that did not pass any.
    if (ownFds)
    {
        ::close(inputFd);
        ::close(outputFd);
    }
    return 0;
}
