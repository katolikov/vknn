# ADR-0003: UMA memory strategy — direct-mapped device-local buffers

## Status
Accepted (2026-06-24)

## Context
The target GPU is a unified-memory (UMA) GPU. `vknn_probe` confirmed memory types that are
simultaneously `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` (type 0; type 1 adds HOST_CACHED).
On discrete GPUs you must stage host→device through a separate transfer buffer; on UMA you do
not.

## Decision
The `vk::Buffer` allocator prefers, by `MemPref`:
- **kAuto** (inputs/weights/intermediates): DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT,
  *avoiding* HOST_CACHED (write-combined is best for CPU writes). Persistently mapped →
  `upload()` is a plain `memcpy`, **no staging buffer, no transfer command**.
- **kReadback** (outputs): DEVICE_LOCAL + HOST_VISIBLE + HOST_CACHED for efficient CPU reads.
- **kDeviceOnly**: DEVICE_LOCAL only (fallback / external imports).

If no host-visible device-local type exists (non-UMA GPU), the allocator falls back to a plain
host-visible type, preserving correctness on other hardware.

## Consequences
- Eliminates the staging-copy class of overhead entirely on the target.
- Makes ION zero-copy (ADR-0005) a natural extension rather than a special case.
- Memory is persistently mapped for the buffer's lifetime (fewer map/unmap calls).
