# ADR-0005: ION zero-copy via DMA-BUF heaps + VkImportMemoryFdInfoKHR

## Status
Accepted (2026-06-24)

## Context
ION zero-copy is required. On this device, `/dev/ion` is absent (classic ION is removed on
Android 12+), while `/dev/dma_heap/` is present and exposes a `system` heap. The Vulkan driver
exposes `VK_EXT_external_memory_dma_buf`, `VK_KHR_external_memory_fd`, and AHB.

## Decision
The ION mechanism on this build is DMA-BUF heaps. `vknn::IonAllocator`:
- Allocates from `/dev/dma_heap/system` via the kernel `DMA_HEAP_IOCTL_ALLOC` ioctl
  (`linux/dma-heap.h`), requiring no vendor library link and remaining robust across builds.
- Returns a dma-buf fd and `mmap`s it for CPU access.
- Imports the same fd into Vulkan via `VkImportMemoryFdInfoKHR` (handle type
  `DMA_BUF_BIT_EXT`), querying allowed types with `vkGetMemoryFdPropertiesKHR`. This is true
  zero-copy: the GPU reads the ION buffer directly, with no staging copy.

Two API modes (Section 6.7): **A — library-allocated** (`Tensor::createIon`) and
**B — user-supplied fd** (`Tensor::wrapIonFd`, ownership configurable).

When dma-buf import fails on a build, the fallback chain is the AHB route, then a staged copy
with a logged `LIMITATIONS.md` entry.

## Consequences
- No dependency on Samsung's libion headers; uses stable kernel uAPI.
- The fd is `dup()`'d for import (the driver takes ownership of its dup), leaving the caller's fd
  valid; in mode B the caller's fd is not closed by default.
