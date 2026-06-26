cmake_minimum_required(VERSION 3.22)
project(vknn LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# vknn — Vulkan-eXynos RunTime. On-device CNN inference engine.
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

option(VKNN_ENABLE_VULKAN "Build the Vulkan compute backend" ${ANDROID})
option(VKNN_ENABLE_NEON    "Build NEON-optimized CPU kernels" ${ANDROID})
option(VKNN_BUILD_TESTS    "Build unit tests" ON)
option(VKNN_BUILD_EXAMPLES "Build example executables" ON)

if(ANDROID)
  message(STATUS "vknn: Android build, ABI=${ANDROID_ABI}, API=${ANDROID_PLATFORM}")
  add_compile_definitions(VKNN_ANDROID=1)
endif()

# --- find glslc (build-time SPIR-V compilation) ----------------------------
find_program(GLSLC glslc HINTS /opt/homebrew/bin /usr/local/bin)
if(NOT GLSLC)
  message(WARNING "glslc not found; shaders will not be (re)compiled")
endif()

# ---------------------------------------------------------------------------
# Shader compilation + embedding
# ---------------------------------------------------------------------------
set(VKNN_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(VKNN_SHADER_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${VKNN_SHADER_GEN_DIR})

set(EMBEDDED_SHADERS_CPP ${VKNN_SHADER_GEN_DIR}/vknn_embedded_shaders.cpp)
set(_spv_files "")
if(VKNN_ENABLE_VULKAN AND GLSLC)
  file(GLOB VKNN_COMP_SHADERS CONFIGURE_DEPENDS ${VKNN_SHADER_DIR}/*.comp)
  foreach(_comp ${VKNN_COMP_SHADERS})
    get_filename_component(_name ${_comp} NAME_WE)
    set(_spv ${VKNN_SHADER_GEN_DIR}/${_name}.spv)
    add_custom_command(
      OUTPUT ${_spv}
      COMMAND ${GLSLC} -O --target-env=vulkan1.3 -fshader-stage=compute
              ${_comp} -o ${_spv}
      DEPENDS ${_comp} ${VKNN_SHADER_DIR}/common.glsl
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
    COMMENT "Embedding SPIR-V blobs -> vknn_embedded_shaders.cpp"
    VERBATIM)
else()
  # Stub embedded shader registry for host / no-glslc builds.
  file(WRITE ${EMBEDDED_SHADERS_CPP}
    "#include <cstdint>\n#include <cstddef>\n#include <map>\n#include <string>\n#include <vector>\n"
    "namespace vknn { const std::map<std::string,std::vector<uint32_t>>& embeddedShaders(){ static std::map<std::string,std::vector<uint32_t>> m; return m; } }\n")
endif()

# ---------------------------------------------------------------------------
# Core library sources (host + android)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE VKNN_CORE_SRC CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/import/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/layout/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/enn/*.cpp)

set(VKNN_SRC ${VKNN_CORE_SRC} ${EMBEDDED_SHADERS_CPP})

if(VKNN_ENABLE_VULKAN)
  file(GLOB_RECURSE VKNN_VK_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/vulkan/*.cpp)
  list(APPEND VKNN_SRC ${VKNN_VK_SRC})
endif()

add_library(vknn STATIC ${VKNN_SRC})
target_include_directories(vknn PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_compile_features(vknn PUBLIC cxx_std_17)

if(VKNN_ENABLE_VULKAN)
  target_compile_definitions(vknn PUBLIC VKNN_ENABLE_VULKAN=1)
  if(ANDROID)
    target_link_libraries(vknn PUBLIC vulkan android log)
  else()
    find_package(Vulkan)
    if(Vulkan_FOUND)
      target_link_libraries(vknn PUBLIC Vulkan::Vulkan)
    endif()
  endif()
endif()
if(VKNN_ENABLE_NEON)
  target_compile_definitions(vknn PUBLIC VKNN_ENABLE_NEON=1)
endif()
if(ANDROID)
  target_link_libraries(vknn PUBLIC log)
endif()
find_package(Threads REQUIRED)
target_link_libraries(vknn PUBLIC Threads::Threads)

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(VKNN_BUILD_EXAMPLES)
  foreach(ex probe classify profile ion_zerocopy backend_switch op_check image_bench predict run_io convert yonosplat)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/${ex}.cpp)
      add_executable(vknn_${ex} ${CMAKE_CURRENT_SOURCE_DIR}/examples/${ex}.cpp)
      # whole-archive so self-registering backends/operators (static initializers in the
      # static lib) are retained even though no symbol references them directly.
      target_link_libraries(vknn_${ex} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,vknn>")
    endif()
  endforeach()
endif()

# ---------------------------------------------------------------------------
# Tests (GoogleTest, host-focused but buildable on-device too)
# ---------------------------------------------------------------------------
if(VKNN_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt)
  add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
  file(GLOB VKNN_TEST_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
  add_executable(vknn_tests ${VKNN_TEST_SRC})
  target_link_libraries(vknn_tests PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,vknn>" gtest gtest_main)
endif()
