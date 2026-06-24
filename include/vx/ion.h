// vxrt — Exynos ION / DMA-BUF heap allocation (zero-copy). See docs/adr/0005.
//
// On this device classic /dev/ion is gone; the "exynos_ion" mechanism is DMA-BUF heaps
// (/dev/dma_heap/system). An IonBuffer wraps a dma-buf fd + CPU mmap; the fd is imported
// into Vulkan via VK_EXT_external_memory_dma_buf for true zero-copy GPU access.
#pragma once
#include <cstddef>
#include <memory>
#include <string>

namespace vx {

class IonBuffer {
 public:
  ~IonBuffer();
  IonBuffer(const IonBuffer&) = delete;
  IonBuffer& operator=(const IonBuffer&) = delete;

  // Mode A — library-allocated: allocate `bytes` from a dma-heap (default "system"),
  // mmap for CPU access. Returns null if the heap/ioctl is unavailable.
  static std::unique_ptr<IonBuffer> alloc(size_t bytes, const std::string& heap = "system");

  // Mode B — user-supplied fd: wrap an existing dma-buf fd. By default does NOT take
  // ownership (caller keeps/closes the fd). mmap for CPU access.
  static std::unique_ptr<IonBuffer> wrapFd(int fd, size_t bytes, bool takeOwnership = false);

  int fd() const { return fd_; }
  size_t size() const { return size_; }
  void* data() const { return map_; }  // CPU-mapped pointer (may be null if mmap failed)
  const std::string& heap() const { return heap_; }

 private:
  IonBuffer() = default;
  int fd_ = -1;
  size_t size_ = 0;
  void* map_ = nullptr;
  bool owns_ = false;
  std::string heap_;
};

}  // namespace vx
