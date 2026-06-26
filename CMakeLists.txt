cmake_minimum_required(VERSION 3.22)
project(vxrt LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# vxrt — Vulkan-eXynos RunTime. On-device CNN inference engine.
#   Host build  : CPU backend + IR + ONNX import + tests (no Vulkan needed).
#   Android build: full engine incl. Vulkan backend (NDK toolchain, arm64-v8a).
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

option(VXRT_ENABLE_VULKAN "Build the Vulkan compute backend" ${ANDROID})
option(VXRT_ENABLE_NEON    "Build NEON-optimized CPU kernels" ${ANDROID})
option(VXRT_BUILD_TESTS    "Build unit tests" ON)
option(VXRT_BUILD_EXAMPLES "Build example executables" ON)

if(ANDROID)
  message(STATUS "vxrt: Android build, ABI=${ANDROID_ABI}, API=${ANDROID_PLATFORM}")
  add_compile_definitions(VXRT_ANDROID=1)
endif()

# --- find glslc (build-time SPIR-V compilation) ----------------------------
find_program(GLSLC glslc HINTS /opt/homebrew/bin /usr/local/bin)
if(NOT GLSLC)
  message(WARNING "glslc not found; shaders will not be (re)compiled")
endif()

# ---------------------------------------------------------------------------
# Shader compilation + embedding
# ---------------------------------------------------------------------------
set(VXRT_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(VXRT_SHADER_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${VXRT_SHADER_GEN_DIR})

set(EMBEDDED_SHADERS_CPP ${VXRT_SHADER_GEN_DIR}/vx_embedded_shaders.cpp)
set(_spv_files "")
if(VXRT_ENABLE_VULKAN AND GLSLC)
  file(GLOB VXRT_COMP_SHADERS CONFIGURE_DEPENDS ${VXRT_SHADER_DIR}/*.comp)
  foreach(_comp ${VXRT_COMP_SHADERS})
    get_filename_component(_name ${_comp} NAME_WE)
    set(_spv ${VXRT_SHADER_GEN_DIR}/${_name}.spv)
    add_custom_command(
      OUTPUT ${_spv}
      COMMAND ${GLSLC} -O --target-env=vulkan1.3 -fshader-stage=compute
              ${_comp} -o ${_spv}
      DEPENDS ${_comp} ${VXRT_SHADER_DIR}/common.glsl
      COMMENT "glslc ${_name}.comp -> ${_name}.spv"
      VERBATIM)
    list(APPEND _spv_files ${_spv})
  endforeach()

  add_custom_command(
    OUTPUT ${EMBEDDED_SHADERS_CPP}
    COMMAND ${CMAKE_COMMAND} -E env python3
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spirv.py
            ${EMBEDDED_SHADERS_CPP} ${_spv_files}
    DEPENDS ${_spv_files} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spirv.py
    COMMENT "Embedding SPIR-V blobs -> vx_embedded_shaders.cpp"
    VERBATIM)
else()
  # Stub embedded shader registry for host / no-glslc builds.
  file(WRITE ${EMBEDDED_SHADERS_CPP}
    "#include <cstdint>\n#include <cstddef>\n#include <map>\n#include <string>\n#include <vector>\n"
    "namespace vx { const std::map<std::string,std::vector<uint32_t>>& embeddedShaders(){ static std::map<std::string,std::vector<uint32_t>> m; return m; } }\n")
endif()

# ---------------------------------------------------------------------------
# Core library sources (host + android)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE VXRT_CORE_SRC CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/import/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/layout/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/enn/*.cpp)

set(VXRT_SRC ${VXRT_CORE_SRC} ${EMBEDDED_SHADERS_CPP})

if(VXRT_ENABLE_VULKAN)
  file(GLOB_RECURSE VXRT_VK_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/vulkan/*.cpp)
  list(APPEND VXRT_SRC ${VXRT_VK_SRC})
endif()

add_library(vxrt STATIC ${VXRT_SRC})
target_include_directories(vxrt PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_compile_features(vxrt PUBLIC cxx_std_17)

if(VXRT_ENABLE_VULKAN)
  target_compile_definitions(vxrt PUBLIC VXRT_ENABLE_VULKAN=1)
  if(ANDROID)
    target_link_libraries(vxrt PUBLIC vulkan android log)
  else()
    find_package(Vulkan)
    if(Vulkan_FOUND)
      target_link_libraries(vxrt PUBLIC Vulkan::Vulkan)
    endif()
  endif()
endif()
if(VXRT_ENABLE_NEON)
  target_compile_definitions(vxrt PUBLIC VXRT_ENABLE_NEON=1)
endif()
if(ANDROID)
  target_link_libraries(vxrt PUBLIC log)
endif()
find_package(Threads REQUIRED)
target_link_libraries(vxrt PUBLIC Threads::Threads)

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(VXRT_BUILD_EXAMPLES)
  foreach(ex probe classify profile ion_zerocopy backend_switch op_check image_bench predict run_io convert)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/${ex}.cpp)
      add_executable(vx_${ex} ${CMAKE_CURRENT_SOURCE_DIR}/examples/${ex}.cpp)
      # whole-archive so self-registering backends/operators (static initializers in the
      # static lib) are retained even though no symbol references them directly.
      target_link_libraries(vx_${ex} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,vxrt>")
    endif()
  endforeach()
endif()

# ---------------------------------------------------------------------------
# Tests (GoogleTest, host-focused but buildable on-device too)
# ---------------------------------------------------------------------------
if(VXRT_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt)
  add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
  file(GLOB VXRT_TEST_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
  add_executable(vx_tests ${VXRT_TEST_SRC})
  target_link_libraries(vx_tests PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,vxrt>" gtest gtest_main)
endif()
