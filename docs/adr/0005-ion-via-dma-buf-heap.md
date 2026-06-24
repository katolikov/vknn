# ADR-0005: ION zero-copy via DMA-BUF heaps + VkImportMemoryFdInfoKHR

## Status
Accepted (2026-06-24)

## Context
The task calls for Exynos ION zero-copy. On-device probing found `/dev/ion` **absent** (classic
ION removed on Android 12+), but `/dev/dma_heap/` present with a `system` heap, and
`libion.so`/`libion_exynos.so`/`libdmabufheap.so` present. The Vulkan driver exposes
`VK_EXT_external_memory_dma_buf` + `VK_KHR_external_memory_fd` + AHB.

## Decision
The "exynos_ion" mechanism on this build is **DMA-BUF heaps**. `vx::IonAllocator`:
- Allocates from `/dev/dma_heap/system` via the kernel `DMA_HEAP_IOCTL_ALLOC` ioctl
  (`linux/dma-heap.h` — no vendor lib link needed, robust across builds).
- Returns a dma-buf **fd**, `mmap`s it for CPU access.
- Imports the **same fd** into Vulkan via `VkImportMemoryFdInfoKHR` (handle type
  `DMA_BUF_BIT_EXT`), querying allowed types with `vkGetMemoryFdPropertiesKHR`. This is true
  zero-copy: GPU reads the ION buffer directly, no staging copy.

Two API modes (Section 6.7): **A — library-allocated** (`Tensor::createIon`) and
**B — user-supplied fd** (`Tensor::wrapIonFd`, ownership configurable).

Fallback chain if dma-buf import fails on some build: AHB route, then a staged copy with a
logged `LIMITATIONS.md` entry.

## Consequences
- No dependency on Samsung's libion headers; uses stable kernel uAPI.
- The fd is `dup()`'d for import (the driver takes ownership of its dup), leaving the caller's fd
  valid; in mode B we do not close the caller's fd by default.
