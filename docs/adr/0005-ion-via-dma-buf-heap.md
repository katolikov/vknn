# ADR-0005: Caller-owned DMA-BUF I/O + VkImportMemoryFdInfoKHR

## Status
Accepted (2026-06-24)

## Context
DMA-BUF I/O is required. On this device, `/dev/ion` is absent (classic ION is removed on
Android 12+), while `/dev/dma_heap/` is present and exposes a `system` heap. The Vulkan driver
exposes `VK_EXT_external_memory_dma_buf`, `VK_KHR_external_memory_fd`, and AHB.

## Decision
vknn binds I/O to **caller-provided dma-buf fds only** — it never allocates dma-bufs. The fd
comes from the caller's camera / gralloc / ION stack (on this device, a `/dev/dma_heap/system`
allocation via the kernel `DMA_HEAP_IOCTL_ALLOC` ioctl, which needs no vendor library link).

- `vknn::IonBuffer::wrapFd(int fd, size_t bytes, bool takeOwnership = false)` wraps a caller fd and
  `mmap`s it for CPU access.
- `Tensor::fromDmaBuf(fd, shape, name)` binds a model input to a caller fd; `Tensor::toDmaBuf(fd,
  shape, name)` binds a model output to one. `Model::run(inputs, outputs)` reads each fd-bound input
  straight from the fd and writes each fd-bound output straight into the fd, with no vknn-side host
  I/O buffer.
- The low-level primitive `vk::Buffer::importDmaBufFd` imports the same fd into Vulkan via
  `VkImportMemoryFdInfoKHR` (handle type `DMA_BUF_BIT_EXT`), querying allowed types with
  `vkGetMemoryFdPropertiesKHR`, for advanced / true-GPU use: the GPU reads the dma-buf directly,
  with no staging copy.

When dma-buf import fails on a build, the fallback chain is the AHB route, then a staged copy
with a logged `LIMITATIONS.md` entry.

## Consequences
- No dependency on Samsung's libion headers; uses stable kernel uAPI.
- vknn allocates no dma-bufs: buffer lifetime and the dma-heap allocation are the caller's job.
- The fd is `dup()`'d for import (the driver takes ownership of its dup), leaving the caller's fd
  valid; `wrapFd` does not close the caller's fd unless `takeOwnership` is set.
