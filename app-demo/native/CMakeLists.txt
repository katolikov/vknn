# Builds libvknnchat.so — the JNI decoder bridge — against the repo's static `vknn` engine (Vulkan
# backend + embedded shaders). Configured by build_native.sh with the NDK arm64-v8a toolchain and a
# cmake >= 3.24 (for the WHOLE_ARCHIVE generator expression the op registries need). VKNN_ROOT is the
# repo root, passed on the command line.
cmake_minimum_required(VERSION 3.24)
project(vknnchat LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Only the engine library is needed; skip its tests and example executables.
set(VKNN_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(VKNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(${VKNN_ROOT} ${CMAKE_BINARY_DIR}/vknn_build)

# raster_core is the splatting rasterizer shared with the vknn_yonosplat example (one copy of the
# pipeline code; the example's device gates cover it).
add_library(vknnchat SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/vknn_jni.cpp
  ${VKNN_ROOT}/examples/splatting/raster_core.cpp)
target_include_directories(vknnchat PRIVATE ${VKNN_ROOT}/examples/splatting)
# WHOLE_ARCHIVE keeps the op-registry self-registration static initializers (dropped by a normal link,
# which would leave the engine with no kernels). vknn's PUBLIC includes + transitive Vulkan/android/log/
# msgpack/Threads deps propagate through the target.
target_link_libraries(vknnchat PRIVATE
  "$<LINK_LIBRARY:WHOLE_ARCHIVE,vknn>"
  log android vulkan)
# Align load segments to 16 KB so the .so loads on 16-KB-page-size devices (Android 15+). NDK r27 still
# defaults to 4 KB, which trips the "ELF alignment check failed" app-compatibility warning on newer devices.
target_link_options(vknnchat PRIVATE "-Wl,-z,max-page-size=16384" "-Wl,-z,common-page-size=16384")
