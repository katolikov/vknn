# ADR-0001: C++17, CMake + NDK, static library

## Status
Accepted (2026-06-24)

## Context
The engine is a fast native inference runtime for Android arm64-v8a, cross-compiled
with the NDK and host-buildable for unit tests. The Vulkan/NEON code is C/C++.

## Decision
- **C++17** (broad NDK r27 support, `if constexpr`, structured bindings; avoids C++20 module churn).
- **CMake + Ninja** with the NDK's `android.toolchain.cmake`. A single `CMakeLists.txt` builds
  both host (CPU backend + IR + import + tests, no Vulkan needed) and Android (full engine).
- **Static library** `libvknn.a` linked into examples/tests. Avoids `LD_LIBRARY_PATH` juggling on
  device and keeps each pushed binary self-contained (shaders embedded, see ADR-0002).
- `minSdk = 33` (Android 13). The target device runs API 33+; 33 keeps headroom and dma-heap (API 30+) works.

## Consequences
- Easy `adb push <binary>` deployment, no shared-lib dependency chasing.
- Host CI for everything except Vulkan; Vulkan validated on-device every milestone.
