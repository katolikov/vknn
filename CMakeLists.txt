cmake_minimum_required(VERSION 3.22)

# Version, from the repo-root VERSION file. The same file is read by app-demo/app/build.gradle.kts
# for the Android versionName, so the engine and the app report one value.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" VKNN_VERSION_RAW)
string(STRIP "${VKNN_VERSION_RAW}" VKNN_VERSION)
if(NOT VKNN_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR "VERSION must read major.minor.patch, got '${VKNN_VERSION}'")
endif()

project(vknn VERSION ${VKNN_VERSION} LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# vknn — Vulkan Neural Network. On-device inference engine (CNNs, transformers, 3D Gaussian Splatting).
#   Host build  : CPU backend + IR + ONNX import + tests (no Vulkan needed).
#   Android build: full engine incl. Vulkan backend (NDK toolchain, arm64-v8a).
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
  # cl.exe has no -O3; keep its Release optimizer defaults and just pin NDEBUG.
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

option(VKNN_ENABLE_VULKAN "Build the Vulkan compute backend" ${ANDROID})
option(VKNN_ENABLE_NEON    "Build NEON-optimized CPU kernels" ${ANDROID})
option(VKNN_BUILD_TESTS    "Build unit tests" ON)
option(VKNN_BUILD_EXAMPLES "Build example executables" ON)
option(VKNN_SANITIZE       "Build with AddressSanitizer + UndefinedBehaviorSanitizer for host memory-safety checking (see build.sh --leakcheck)" OFF)

if(ANDROID)
  message(STATUS "vknn: Android build, ABI=${ANDROID_ABI}, API=${ANDROID_PLATFORM}")
  add_compile_definitions(VKNN_ANDROID=1)
endif()

# Host memory-safety instrumentation for build.sh --leakcheck. Catches use-after-free, double-free,
# heap/stack overflow, and undefined behaviour; on Linux, ASan additionally reports leaks (LeakSanitizer).
# Applied to every target (library, tests, examples) so the whole codebase is instrumented. Host-only.
if(VKNN_SANITIZE)
  message(STATUS "vknn: sanitizers ON (address, undefined)")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

# --- shader compiler (build-time GLSL -> SPIR-V) ---------------------------
# Prefer the vendored glslang (third_party/glslang, built host-side by build.sh) so shader compilation
# is reproducible and independent of a system install; fall back to a system glslc. Both emit the same
# SPIR-V, including the RoundingModeRTE fp16-store rounding (store16.glsl). glslang needs the include
# directive enabled up front via -P (a preamble injected after #version); glslc enables it implicitly.
find_program(GLSLANG NAMES glslang glslangValidator
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glslang/build-host/StandAlone NO_DEFAULT_PATH)
find_program(GLSLC glslc HINTS /opt/homebrew/bin /usr/local/bin)
if(GLSLANG)
  set(SHADERCC ${GLSLANG})
  set(SHADERCC_PRE -V --target-env vulkan1.3 -S comp "-P#extension GL_GOOGLE_include_directive : require")
  message(STATUS "VKNN shader compiler: vendored glslang (${GLSLANG})")
elseif(GLSLC)
  set(SHADERCC ${GLSLC})
  set(SHADERCC_PRE -O --target-env=vulkan1.3 -fshader-stage=compute)
  message(STATUS "VKNN shader compiler: system glslc (${GLSLC})")
else()
  message(WARNING "no shader compiler (glslang or glslc) found; shaders will not be (re)compiled")
endif()

# ---------------------------------------------------------------------------
# Shader compilation + embedding
# ---------------------------------------------------------------------------
set(VKNN_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(VKNN_SHADER_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${VKNN_SHADER_GEN_DIR})

# Every .comp directly under shaders/ — tracked or not — is compiled, embedded, and counted in
# embeddedShadersHash(), so a stray scratch file invalidates every device model cache. Warn at
# configure time; scratch/bench kernels belong in shaders/experimental/ (see its README.md).
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR} ls-files --others --exclude-standard -- shaders
    OUTPUT_VARIABLE _untracked_shader_files
    RESULT_VARIABLE _git_ls_result
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_git_ls_result EQUAL 0 AND _untracked_shader_files)
    string(REPLACE "\n" ";" _untracked_shader_files "${_untracked_shader_files}")
    set(_stray_comps "")
    foreach(_f IN LISTS _untracked_shader_files)
      if(_f MATCHES "^shaders/[^/]+\\.comp$")
        list(APPEND _stray_comps ${_f})
      endif()
    endforeach()
    if(_stray_comps)
      list(JOIN _stray_comps ", " _stray_comps)
      message(WARNING
        "untracked .comp directly under shaders/: ${_stray_comps} — these are embedded into the "
        "engine and change embeddedShadersHash(), invalidating every device model cache. Move "
        "scratch kernels to shaders/experimental/.")
    endif()
  endif()
endif()

set(EMBEDDED_SHADERS_CPP ${VKNN_SHADER_GEN_DIR}/vknn_embedded_shaders.cpp)
set(_spv_files "")
if(VKNN_ENABLE_VULKAN AND SHADERCC)
  file(GLOB VKNN_COMP_SHADERS CONFIGURE_DEPENDS ${VKNN_SHADER_DIR}/*.comp)
  # shaders/experimental/ holds scratch and bench kernels: never compiled, embedded, or counted in
  # embeddedShadersHash(). The glob above is non-recursive; the filter keeps the exclusion explicit
  # should the glob ever become recursive.
  list(FILTER VKNN_COMP_SHADERS EXCLUDE REGEX "/experimental/")
  # every .spv depends on every include: a .glsl edit must recompile all kernels that may pull it in
  file(GLOB VKNN_GLSL_INCLUDES CONFIGURE_DEPENDS ${VKNN_SHADER_DIR}/*.glsl)

  # --- derive the epilogue-capable stem set from the shader sources -----------------------------
  # A kernel that #includes pw_epilogue.glsl self-declares that its store can carry a fused
  # pointwise-chain epilogue, so it needs _epi/_epi_rx SPIR-V variants (see the PW_EPI block below).
  # Deriving the set from the sources means adding an epilogue-capable op = writing the shader; there
  # is no hand-maintained stem list to drift from the shaders. tools/check_epi_sync.py enforces that
  # this derivation, the C++ pwEpilogueCapable() OpType set, and the op-file stem call sites agree.
  #   plain/templated <stem>.comp      -> _epi_stems   (base name is the stem)
  #   hand-written    <stem>_fp16.comp -> _epi_stems   (stem = name without the _fp16 suffix)
  #   the standalone pointwise VM      -> _rx_standalone (fused_pw_*; _rx twin, no _epi)
  set(_epi_stems "")
  set(_rx_standalone "")
  foreach(_comp ${VKNN_COMP_SHADERS})
    get_filename_component(_name ${_comp} NAME_WE)
    file(READ ${_comp} _src)
    # Match the include DIRECTIVE, not an incidental comment mention of the file.
    if(_src MATCHES "#include[ \t]+\"pw_epilogue\\.glsl\"")
      if(_name MATCHES "^fused_pw_")
        list(APPEND _rx_standalone ${_name})
      elseif(_name MATCHES "^(.+)_fp16$")
        list(APPEND _epi_stems ${CMAKE_MATCH_1})
      else()
        list(APPEND _epi_stems ${_name})
      endif()
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _epi_stems)
  list(REMOVE_DUPLICATES _rx_standalone)

  # Fail configuration — not a device load — when the epilogue surfaces disagree (the shader set
  # derived just above, the op-file kernel-name call sites, and the C++ pwEpilogueCapable OpType
  # set) or a shader breaks the store16 / VKNN_NO_RTE contracts. These run only when shaders are
  # actually being compiled (this VKNN_ENABLE_VULKAN branch); the host build has no shaders to check.
  find_program(PYTHON3 NAMES python3 python)
  if(PYTHON3)
    execute_process(
      COMMAND ${PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_epi_sync.py --repo ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE _epi_sync_rc OUTPUT_VARIABLE _epi_sync_out ERROR_VARIABLE _epi_sync_out)
    if(NOT _epi_sync_rc EQUAL 0)
      message(FATAL_ERROR "check_epi_sync.py: epilogue-surface drift\n${_epi_sync_out}")
    endif()
    execute_process(
      COMMAND ${PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_shader_contracts.py --shaders ${VKNN_SHADER_DIR}
      RESULT_VARIABLE _shader_rc OUTPUT_VARIABLE _shader_out ERROR_VARIABLE _shader_out)
    if(NOT _shader_rc EQUAL 0)
      message(FATAL_ERROR "check_shader_contracts.py: shader contract violation\n${_shader_out}")
    endif()
  else()
    message(WARNING "python3 not found; skipping check_epi_sync.py / check_shader_contracts.py "
      "configure-time sync checks")
  endif()

  foreach(_comp ${VKNN_COMP_SHADERS})
    get_filename_component(_name ${_comp} NAME_WE)
    # boundary_convert is built below as four cross-dtype variants, not the fp32/fp16 pair.
    if(_name STREQUAL "boundary_convert")
      continue()
    endif()
    set(_spv ${VKNN_SHADER_GEN_DIR}/${_name}.spv)
    add_custom_command(
      OUTPUT ${_spv}
      COMMAND ${SHADERCC} ${SHADERCC_PRE} -o ${_spv} ${_comp}
      DEPENDS ${_comp} ${VKNN_GLSL_INCLUDES}
      COMMENT "compile ${_name}.comp -> ${_name}.spv"
      VERBATIM)
    list(APPEND _spv_files ${_spv})
    # A precision-templated shader (one that #includes precision.glsl) also compiles to an fp16
    # variant <name>_fp16.spv with -DUSE_FP16 — one source of truth for both, no hand-written
    # _fp16.comp duplicate.
    file(READ ${_comp} _src)
    if(_src MATCHES "precision.glsl" AND NOT _name MATCHES "_fp16$")
      set(_spv16 ${VKNN_SHADER_GEN_DIR}/${_name}_fp16.spv)
      add_custom_command(
        OUTPUT ${_spv16}
        COMMAND ${SHADERCC} ${SHADERCC_PRE} -DUSE_FP16=1 -o ${_spv16} ${_comp}
        DEPENDS ${_comp} ${VKNN_GLSL_INCLUDES}
        COMMENT "compile ${_name}.comp -DUSE_FP16 -> ${_name}_fp16.spv"
        VERBATIM)
      list(APPEND _spv_files ${_spv16})
    endif()
    # Producer kernels that can carry a fused pointwise-chain epilogue (fusePointwiseChains
    # attaching a chain to its producer) also emit PW_EPI variants that pull in
    # shaders/pw_epilogue.glsl. The hosting stems are derived above from the sources, not curated.
    # Each PW_EPI build comes in two rounding disciplines selected at compile time (a runtime
    # branch would carry both bodies in every kernel): the strict per-step-rounded "_epi" and the
    # fp32-chained "_epi_rx" (-DPW_RELAX, for units carrying the pw_relax attr):
    #   plain/templated <stem>.comp      -> <stem>_epi[_rx].spv (+ <stem>_epi[_rx]_fp16.spv when templated)
    #   hand-written    <stem>_fp16.comp -> <stem>_epi[_rx]_fp16.spv
    list(FIND _epi_stems ${_name} _epiPlain)
    list(FIND _rx_standalone ${_name} _rxAlone)
    if(_epiPlain GREATER -1)
      foreach(_relax "" "_rx")
        set(_relaxFlag "")
        if(_relax STREQUAL "_rx")
          set(_relaxFlag "-DPW_RELAX=1")
        endif()
        set(_spvEpi ${VKNN_SHADER_GEN_DIR}/${_name}_epi${_relax}.spv)
        add_custom_command(
          OUTPUT ${_spvEpi}
          COMMAND ${SHADERCC} ${SHADERCC_PRE} -DPW_EPI=1 ${_relaxFlag} -o ${_spvEpi} ${_comp}
          DEPENDS ${_comp} ${VKNN_GLSL_INCLUDES}
          COMMENT "compile ${_name}.comp -DPW_EPI ${_relaxFlag} -> ${_name}_epi${_relax}.spv"
          VERBATIM)
        list(APPEND _spv_files ${_spvEpi})
        if(_src MATCHES "precision.glsl")
          set(_spvEpi16 ${VKNN_SHADER_GEN_DIR}/${_name}_epi${_relax}_fp16.spv)
          add_custom_command(
            OUTPUT ${_spvEpi16}
            COMMAND ${SHADERCC} ${SHADERCC_PRE} -DPW_EPI=1 ${_relaxFlag} -DUSE_FP16=1 -o ${_spvEpi16} ${_comp}
            DEPENDS ${_comp} ${VKNN_GLSL_INCLUDES}
            COMMENT "compile ${_name}.comp -DPW_EPI ${_relaxFlag} -DUSE_FP16 -> ${_name}_epi${_relax}_fp16.spv"
            VERBATIM)
          list(APPEND _spv_files ${_spvEpi16})
        endif()
      endforeach()
    elseif(_rxAlone GREATER -1)
      # fused_pw_* compile the strict discipline as the base name (rule above); add the _rx twin
      # and its fp16 form.
      set(_spvRx ${VKNN_SHADER_GEN_DIR}/${_name}_rx.spv)
      add_custom_command(
        OUTPUT ${_spvRx}
        COMMAND ${SHADERCC} ${SHADERCC_PRE} -DPW_RELAX=1 -o ${_spvRx} ${_comp}
        DEPENDS ${_comp} ${VKNN_GLSL_INCLUDES}
        COMMENT "compile ${_name}.comp -DPW_RELAX -> ${_name}_rx.spv"
        VERBATIM)
      list(APPEND _spv_files ${_spvRx})
      if(_src MATCHES "precision.glsl")
        set(_spvRx16 ${VKNN_SHADER_GEN_DIR}/${_name}_rx_fp16.spv)
        add_custom_command(
          OUTPUT ${_spvRx16}
          COMMAND ${SHADERCC} ${SHADERCC_PRE} -DPW_RELAX=1 -DUSE_FP16=1 -o ${_spvRx16} ${_comp}
          DEPENDS ${_comp} ${VKNN_GLSL_INCLUDES}
          COMMENT "compile ${_name}.comp -DPW_RELAX -DUSE_FP16 -> ${_name}_rx_fp16.spv"
          VERBATIM)
        list(APPEND _spv_files ${_spvRx16})
      endif()
    elseif(_name MATCHES "^(.+)_fp16$")
      list(FIND _epi_stems ${CMAKE_MATCH_1} _epiHand)
      if(_epiHand GREATER -1)
        foreach(_relax "" "_rx")
          set(_relaxFlag "")
          if(_relax STREQUAL "_rx")
            set(_relaxFlag "-DPW_RELAX=1")
          endif()
          set(_spvEpi16 ${VKNN_SHADER_GEN_DIR}/${CMAKE_MATCH_1}_epi${_relax}_fp16.spv)
          add_custom_command(
            OUTPUT ${_spvEpi16}
            COMMAND ${SHADERCC} ${SHADERCC_PRE} -DPW_EPI=1 ${_relaxFlag} -o ${_spvEpi16} ${_comp}
            DEPENDS ${_comp} ${VKNN_GLSL_INCLUDES}
            COMMENT "compile ${_name}.comp -DPW_EPI ${_relaxFlag} -> ${CMAKE_MATCH_1}_epi${_relax}_fp16.spv"
            VERBATIM)
          list(APPEND _spv_files ${_spvEpi16})
        endforeach()
      endif()
    endif()
  endforeach()

  # boundary_convert: one source -> cross-dtype SPIR-V (declared <-> device-native), selected at
  # runtime by (srcDtype, dstDtype). fp16 variants enable the 16-bit storage extensions; uint8 variants
  # enable 8-bit storage. Trailing args are extra -D flags for that variant.
  set(_bc ${VKNN_SHADER_DIR}/boundary_convert.comp)
  function(bc_variant suffix srct dstt)
    set(_o ${VKNN_SHADER_GEN_DIR}/boundary_convert_${suffix}.spv)
    add_custom_command(
      OUTPUT ${_o}
      COMMAND ${SHADERCC} ${SHADERCC_PRE} ${ARGN} -DSRC_T=${srct} -DDST_T=${dstt} -o ${_o} ${_bc}
      DEPENDS ${_bc} ${VKNN_SHADER_DIR}/store16.glsl
      COMMENT "compile boundary_convert ${suffix}"
      VERBATIM)
    set(_spv_files ${_spv_files} ${_o} PARENT_SCOPE)
  endfunction()
  bc_variant(f32_f32 float      float)
  bc_variant(f32_f16 float      float16_t  -DNEED_FP16=1)
  bc_variant(f16_f32 float16_t  float      -DNEED_FP16=1)
  bc_variant(f16_f16 float16_t  float16_t  -DNEED_FP16=1)
  # uint8 image I/O (8-bit storage). Source uint8 -> fp16/fp32 device (exact for 0..255); device
  # fp16/fp32/uint8 -> uint8 output truncates+clamps to [0,255] (INT_DST_U8).
  bc_variant(u8_f16  uint8_t    float16_t  -DNEED_FP16=1 -DNEED_INT8=1)
  bc_variant(u8_f32  uint8_t    float      -DNEED_INT8=1)
  bc_variant(f16_u8  float16_t  uint8_t    -DNEED_FP16=1 -DNEED_INT8=1 -DINT_DST_U8=1)
  bc_variant(f32_u8  float      uint8_t    -DNEED_INT8=1 -DINT_DST_U8=1)
  bc_variant(u8_u8   uint8_t    uint8_t    -DNEED_INT8=1 -DINT_DST_U8=1)

  add_custom_command(
    OUTPUT ${EMBEDDED_SHADERS_CPP}
    COMMAND ${CMAKE_COMMAND} -E env python3
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spirv.py
            ${EMBEDDED_SHADERS_CPP} ${_spv_files}
    DEPENDS ${_spv_files} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spirv.py
    COMMENT "Embedding SPIR-V blobs -> vknn_embedded_shaders.cpp"
    VERBATIM)
else()
  # Stub embedded shader registry for host / no-glslc builds. embeddedShadersHash() is the cache's
  # kernel-version guard; with no compiled kernels it is a fixed placeholder.
  file(WRITE ${EMBEDDED_SHADERS_CPP}
    "#include <cstdint>\n#include <cstddef>\n#include <map>\n#include <string>\n#include <vector>\n"
    "namespace vknn { const std::map<std::string,std::vector<uint32_t>>& embeddedShaders(){ static std::map<std::string,std::vector<uint32_t>> m; return m; }\n"
    "const char* embeddedShadersHash(){ return \"no-embedded-shaders\"; } }\n")
endif()

# ---------------------------------------------------------------------------
# MessagePack (msgpack-c, C API) — serializes the warm-start model cache.
# ---------------------------------------------------------------------------
set(MSGPACK_ENABLE_SHARED   OFF CACHE BOOL "" FORCE)
set(MSGPACK_ENABLE_STATIC   ON  CACHE BOOL "" FORCE)
set(MSGPACK_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(MSGPACK_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/msgpack-c EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Core library sources (host + android)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE VKNN_CORE_SRC CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/import/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/layout/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cpu/*.cpp)

set(VKNN_SRC ${VKNN_CORE_SRC} ${EMBEDDED_SHADERS_CPP})

if(VKNN_ENABLE_VULKAN)
  file(GLOB_RECURSE VKNN_VK_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/vulkan/*.cpp)
  list(APPEND VKNN_SRC ${VKNN_VK_SRC})
endif()

add_library(vknn STATIC ${VKNN_SRC})
# Compiled in, so vknnVersion() reports the linked engine rather than a caller's header.
target_compile_definitions(vknn PUBLIC
  VKNN_VERSION_STRING="${PROJECT_VERSION}"
  VKNN_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  VKNN_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  VKNN_VERSION_PATCH=${PROJECT_VERSION_PATCH})
target_include_directories(vknn PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_compile_features(vknn PUBLIC cxx_std_17)
target_link_libraries(vknn PRIVATE msgpack-c) # cache (de)serialization

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
  # Examples are grouped by theme under examples/<group>/; the target name is the file basename
  # (vknn_classify, vknn_run_io, ...), unchanged by the grouping so scripts and docs still reference it.
  set(_vknn_examples
      basics/probe basics/readme_quickstart basics/backend_switch basics/op_check
      vision/classify vision/predict vision/predict_cache vision/image_bench
      llm/chat llm/vlm
      splatting/yonosplat
      io/run_io
      bench/profile)
  if(NOT WIN32)
    # The dma-buf zero-copy demos drive Linux kernel interfaces (ioctl/mmap on dma-buf fds),
    # which do not exist on Windows.
    list(APPEND _vknn_examples io/dmabuf_fd_io io/zerocopy_cache io/zerocopy_simple)
  endif()
  if(VKNN_ENABLE_VULKAN)
    # microbench drives Vulkan kernels directly (includes vk_context.h), so it needs the backend.
    list(APPEND _vknn_examples bench/microbench)
    # A/B microbench for the vec4-load GEMM kernel pair (matmul_tiled_fast_v4_fp16.comp vs the
    # scalar matmul_tiled_fast_fp16); like microbench it drives Vulkan kernels directly.
    list(APPEND _vknn_examples bench/matmul_vec4_bench)
    # A/B microbench + byte gate for the implicit-GEMM conv's weight load width (conv_gemm_wv4.comp
    # vs conv_gemm.comp); drives the Vulkan kernels directly like the pair above.
    list(APPEND _vknn_examples bench/conv_gemm_wv4_bench)
  endif()
  foreach(ex ${_vknn_examples})
    get_filename_component(_name ${ex} NAME) # target name = basename, independent of the group dir
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/${ex}.cpp)
      add_executable(vknn_${_name} ${CMAKE_CURRENT_SOURCE_DIR}/examples/${ex}.cpp)
      # whole-archive so self-registering backend/operators (static initializers in the
      # static lib) are retained even though no symbol references them directly.
      target_link_libraries(vknn_${_name} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,vknn>")
    endif()
  endforeach()
  # The splatting rasterizer lives in raster_core.cpp so the app-demo JNI bridge compiles the
  # same pipeline; the example adds it as a second source.
  if(TARGET vknn_yonosplat)
    target_sources(vknn_yonosplat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples/splatting/raster_core.cpp)
  endif()
endif()

# ---------------------------------------------------------------------------
# Benchmark on-device tools (driven by benchmark/run.py)
# ---------------------------------------------------------------------------
if(VKNN_BUILD_EXAMPLES)
  foreach(bt benchmark)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/${bt}.cpp)
      add_executable(vknn_${bt} ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/${bt}.cpp)
      target_link_libraries(vknn_${bt} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,vknn>")
    endif()
  endforeach()
endif()

# ---------------------------------------------------------------------------
# Model compiler (ONNX -> optimized .vxm). Its own top-level dir, not an example.
# ---------------------------------------------------------------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/convert/compile.cpp)
  add_executable(vknn_compile ${CMAKE_CURRENT_SOURCE_DIR}/convert/compile.cpp)
  target_link_libraries(vknn_compile PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,vknn>")
endif()

# ---------------------------------------------------------------------------
# Offline host tools with a C++ build step
# ---------------------------------------------------------------------------
# Winograd F(6,3) interpolation-point selection: enumerates + scores candidate point sets and
# re-verifies the tables baked in src/core/wino_f63.h (header-only dependencies, no vknn link).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/wino_f63_points.cpp)
  add_executable(vknn_wino_f63_points ${CMAKE_CURRENT_SOURCE_DIR}/tools/wino_f63_points.cpp)
  target_include_directories(vknn_wino_f63_points PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()

# ---------------------------------------------------------------------------
# Tests (GoogleTest, host-focused but buildable on-device too)
# ---------------------------------------------------------------------------
if(VKNN_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt)
  if(MSVC)
    # Link gtest against the shared CRT like every other target, or MSVC links fail with
    # LNK2038 runtime-library mismatches.
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  endif()
  add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
  file(GLOB VKNN_TEST_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
  add_executable(vknn_tests ${VKNN_TEST_SRC})
  # tests/test_main.cpp provides main() (it mutes engine logging by default), so gtest_main is not linked.
  target_link_libraries(vknn_tests PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,vknn>" gtest)
endif()
