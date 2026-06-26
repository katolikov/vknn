# ADR-0002: Direct NDK Vulkan linking + embedded SPIR-V

## Status
Accepted (2026-06-24)

## Context
Three ways to get the Vulkan loader: vendor `volk`, dlopen `libvulkan.so`, or link the NDK's
`libvulkan`. For shaders: ship `.spv` files next to the binaries, or embed them.

## Decision
- **Link the NDK `libvulkan`** directly and resolve the extension entrypoints in use
  (`vkCmdPushDescriptorSetKHR`, `vkGetMemoryFdKHR`, `vkGetMemoryFdPropertiesKHR`) via
  `vkGetDeviceProcAddr`. The device is Vulkan 1.3+, so every promoted core function (timeline
  semaphore, properties2, and the rest) is available without an extension loader. No third-party
  loader dependency.
- **Embed SPIR-V** into the library: `glslc` compiles `shaders/*.comp` → `.spv` at build time,
  `tools/embed_spirv.py` packs them into one generated `.cpp` that exposes
  `vknn::embeddedShaders()`. The binary stands alone, requires no per-shader file pushes, and
  versions the shader set with the code.

## Consequences
- Zero runtime file dependencies for shaders.
- Adding a shader: drop a `.comp` in `shaders/`; CMake recompiles and re-embeds automatically.
- A non-Android/no-glslc host build gets an empty shader registry (CPU backend still works).
