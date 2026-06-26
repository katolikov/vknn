# ADR-0003: UMA memory strategy — direct-mapped device-local buffers

## Status
Accepted (2026-06-24)

## Context
The target GPU uses unified memory (UMA). `vknn_probe` reports memory types that are
simultaneously `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` (type 0; type 1 adds HOST_CACHED).
A discrete GPU requires staging host→device through a separate transfer buffer. UMA does not.

## Decision
The `vk::Buffer` allocator selects a memory type by `MemPref`:
- **kAuto** (inputs/weights/intermediates): DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT,
  avoiding HOST_CACHED (write-combined is best for CPU writes). The buffer is persistently
  mapped, so `upload()` is a plain `memcpy` with no staging buffer and no transfer command.
- **kReadback** (outputs): DEVICE_LOCAL + HOST_VISIBLE + HOST_CACHED for efficient CPU reads.
- **kDeviceOnly**: DEVICE_LOCAL only (fallback / external imports).

If no host-visible device-local type exists (non-UMA GPU), the allocator falls back to a plain
host-visible type, preserving correctness on other hardware.

## Consequences
- Eliminates the staging-copy class of overhead on the target.
- Makes ION zero-copy (ADR-0005) a natural extension rather than a special case.
- Memory is persistently mapped for the buffer's lifetime, reducing map/unmap calls.
