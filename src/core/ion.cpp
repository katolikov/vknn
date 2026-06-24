#include "vx/ion.h"
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "vx/logging.h"

namespace vx {

// DMA-BUF heap uAPI (linux/dma-heap.h). Defined locally to avoid header-availability
// issues across NDK API levels; matches the stable kernel ABI.
namespace {
struct dma_heap_allocation_data {
  uint64_t len;
  uint32_t fd;
  uint32_t fd_flags;
  uint64_t heap_flags;
};
// _IOWR('H', 0x0, struct dma_heap_allocation_data)
constexpr unsigned long kDmaHeapIoctlAlloc =
    (3UL << 30) | (sizeof(dma_heap_allocation_data) << 16) | ('H' << 8) | 0x0;
}  // namespace

std::unique_ptr<IonBuffer> IonBuffer::alloc(size_t bytes, const std::string& heap) {
  std::string path = "/dev/dma_heap/" + heap;
  int hfd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (hfd < 0) {
    VX_WARN << "ION: cannot open dma-heap " << path << " (errno " << errno << ")";
    return nullptr;
  }
  dma_heap_allocation_data data{};
  data.len = bytes;
  data.fd_flags = O_RDWR | O_CLOEXEC;
  int rc = ::ioctl(hfd, kDmaHeapIoctlAlloc, &data);
  ::close(hfd);
  if (rc < 0) {
    VX_WARN << "ION: DMA_HEAP_IOCTL_ALLOC failed (errno " << errno << ")";
    return nullptr;
  }
  auto b = std::unique_ptr<IonBuffer>(new IonBuffer());
  b->fd_ = (int)data.fd;
  b->size_ = bytes;
  b->owns_ = true;
  b->heap_ = heap;
  b->map_ = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, b->fd_, 0);
  if (b->map_ == MAP_FAILED) {
    VX_WARN << "ION: mmap failed (errno " << errno << ")";
    b->map_ = nullptr;
  }
  VX_INFO << "ION: allocated " << bytes << " bytes from /dev/dma_heap/" << heap << " -> fd "
          << b->fd_ << (b->map_ ? " (mapped)" : " (unmapped)");
  return b;
}

std::unique_ptr<IonBuffer> IonBuffer::wrapFd(int fd, size_t bytes, bool takeOwnership) {
  auto b = std::unique_ptr<IonBuffer>(new IonBuffer());
  b->fd_ = fd;
  b->size_ = bytes;
  b->owns_ = takeOwnership;
  b->heap_ = "<wrapped>";
  b->map_ = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (b->map_ == MAP_FAILED) {
    b->map_ = nullptr;
    VX_WARN << "ION: wrapFd mmap failed";
  }
  VX_INFO << "ION: wrapped fd " << fd << " (" << bytes
          << " bytes, ownership=" << (takeOwnership ? "engine" : "caller") << ")";
  return b;
}

IonBuffer::~IonBuffer() {
  if (map_) ::munmap(map_, size_);
  if (owns_ && fd_ >= 0) ::close(fd_);
}

}  // namespace vx
