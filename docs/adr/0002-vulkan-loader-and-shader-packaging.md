# ADR-0002: Direct NDK Vulkan linking + embedded SPIR-V

## Status
Accepted (2026-06-24)

## Context
Options for the Vulkan loader: vendor `volk`, dlopen `libvulkan.so`, or link the NDK's
`libvulkan`. Options for shaders: ship `.spv` files alongside binaries, or embed them.

## Decision
- **Link the NDK `libvulkan`** directly and resolve the few extension entrypoints we use
  (`vkCmdPushDescriptorSetKHR`, `vkGetMemoryFdKHR`, `vkGetMemoryFdPropertiesKHR`) via
  `vkGetDeviceProcAddr`. The device is Vulkan 1.4 so all promoted core functions (timeline
  semaphore, properties2, etc.) are available without an extension loader. No third-party
  loader dependency.
- **Embed SPIR-V** into the library: `glslc` compiles `shaders/*.comp` → `.spv` at build time,
  `tools/embed_spirv.py` packs them into one generated `.cpp` exposing
  `vx::embeddedShaders()`. The binary is self-contained; no per-shader file pushes; the shader
  set is versioned with the code.

## Consequences
- Zero runtime file dependencies for shaders.
- Adding a shader = drop a `.comp` in `shaders/`; CMake recompiles + re-embeds automatically.
- A non-Android/no-glslc host build gets an empty shader registry (CPU backend still works).
