# shaders/experimental/

Scratch and benchmark kernels. Nothing in this directory is compiled, embedded into the
engine, or counted in `embeddedShadersHash()` (the kernel-version guard for device model
caches): the shader glob in `CMakeLists.txt` only picks up `.comp` files directly under
`shaders/` and explicitly excludes this directory. Files here are typically untracked.

To promote a kernel, move it up to `shaders/` (and, if it carries a pointwise epilogue,
add its stem to `_epi_stems` in `CMakeLists.txt`). Note that any `.comp` directly under
`shaders/` changes the embedded-shaders hash and invalidates existing device model caches.
