# Declared-format zero-copy + release-prep Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax. Verification in this repo is the
> host build + the 11 `vknn_tests`, then the on-device example with cosine/bit-exact checks — not a
> per-step unit-test harness. Each task ends in a build + the relevant gate + a commit.

**Goal:** Let a caller declare the layout/dtype of a zero-copy DMA-BUF; convert on the GPU when it
differs from the device-native boundary. Plus release prep: drop the `k` enum prefix, a naming-clarity
sweep, an on-device verification matrix, and a codebase-wide bug audit.

**Architecture:** Declared `(format, dtype)` rides next to `dmaBufFd` through `Tensor → IOTensor →
RtTensor`. `VulkanSegment::rebind` binds the fd directly when the declared format matches device-native,
else registers a boundary-convert; `record` emits a `boundary_convert` dispatch (imported↔pooled
boundary) in the existing command buffer, re-recording on change. The conversion is one kernel
(push-constant layout codes, 4 dtype SPIR-V variants) mirroring `packToBuffer`.

**Tech Stack:** C++17, Vulkan compute (GLSL → SPIR-V via glslc), Android NDK arm64, `build.sh`.

**Spec:** [docs/superpowers/specs/2026-06-28-declared-format-zerocopy-design.md](../specs/2026-06-28-declared-format-zerocopy-design.md)

---

## Phase 0 — Public enum k-prefix removal

### Task 0: Drop `k` from every `include/vknn/` enum

**Files:** `include/vknn/{config,logging,op,common,tensor_format,dtype}.h`; every `Enum::kX` use in
`src/`, `examples/`, `tests/`, `convert/`.

- [ ] **Step 1:** Per-enum qualified replacement `Enum::kX → Enum::X` for the enums in the spec table.
  Build a scripted, scope-aware pass: for each public enum, enumerate its members, then replace the
  qualified token `Enum::kMember` (and `LogLevel::kX` in the logging macros, and `Attr::kX` /
  unqualified `kX` inside `struct Attr`). Internal enums declared under `src/` keep their `k`.
- [ ] **Step 2:** Add no new value yet (`TensorFormat::Auto` lands in Task 2). Keep `Unknown`.
- [ ] **Step 3:** `./build.sh` → clean.
- [ ] **Step 4:** `./build/host/vknn_tests` → 11/11 pass.
- [ ] **Step 5:** `clang-format -i` touched headers/sources.
- [ ] **Step 6:** Commit `refactor: drop k-prefix from public vknn enums`.

Verification detail — confirm no enumerator name is stringized anywhere (config maps to string
literals, already checked); grep for `#k` and name-reflection to be safe.

---

## Phase 1 — Declared-format plumbing + API

### Task 1: `TensorFormat::Auto` + enum compare helper

**Files:** `include/vknn/tensor_format.h`

- [ ] Add `Auto` to `TensorFormat` (sentinel: bytes already device-native, never convert).
- [ ] Build + 11 tests. Commit.

### Task 2: Public API — declared layout/dtype on the DMA-BUF Tensor

**Files:** `include/vknn/model.h`, `src/core/model.cpp`

- [ ] `fromDmaBuf` / `toDmaBuf` gain `TensorFormat layout = TensorFormat::NCHW, DType dtype =
  DType::Float32`. `Tensor` gains `dmaBufFormat_` / `dmaBufDtype_` + accessors. `Model::run` copies
  them into the `IOTensor` it builds (the existing fd-marshalling loop).
- [ ] Build + 11 tests. Commit.

### Task 3: Carry declared fields through Session

**Files:** `include/vknn/session.h` (`IOTensor`), `include/vknn/tensor.h` (`RtTensor`),
`src/core/session.cpp`

- [ ] `IOTensor` + `RtTensor` gain `dmaBufFormat` / `dmaBufDtype` (default `NCHW` / `Float32`).
- [ ] `Session::run` input-bind loop (`:554`) and output pre-bind loop (`:584`) copy the declared
  fields into the pool entry; reset to defaults after the run alongside `dmaBufFd`.
- [ ] Build + 11 tests. Commit.

---

## Phase 2 — The GPU converter

### Task 4: `boundary_convert` shader (4 dtype variants)

**Files:** `shaders/boundary_convert.comp`; CMake shader list.

Shader skeleton (one source; `SRC_T`/`DST_T` macros select fp32/fp16; layout codes are push
constants `0=NCHW 1=NHWC 2=NC4HW4`):

```glsl
#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly  buffer Src { SRC_T src[]; };
layout(std430, binding = 1) writeonly buffer Dst { DST_T dst[]; };
layout(push_constant) uniform PC { int N, C, H, W, srcFmt, dstFmt, count; } pc;

int idxNCHW (int n,int c,int h,int w){ return ((n*pc.C + c)*pc.H + h)*pc.W + w; }
int idxNHWC (int n,int c,int h,int w){ return ((n*pc.H + h)*pc.W + w)*pc.C + c; }
int cBlk(int c){ return (c + 3) / 4; }
int idxNC4 (int n,int c,int h,int w){ int Cb=cBlk(pc.C),cb=c/4,l=c%4;
  return ((((n*Cb + cb)*pc.H + h)*pc.W + w)*4) + l; }
int encode(int fmt,int n,int c,int h,int w){
  return fmt==0?idxNCHW(n,c,h,w):fmt==1?idxNHWC(n,c,h,w):idxNC4(n,c,h,w); }

void main(){
  int i = int(gl_GlobalInvocationID.x);
  if (i >= pc.count) return;
  // decode destination linear index i -> (n,c,h,w) per dstFmt
  int n,c,h,w;
  if (pc.dstFmt == 2){ int Cb=cBlk(pc.C); int l=i%4,t=i/4;
    w=t%pc.W;t/=pc.W; h=t%pc.H;t/=pc.H; int cb=t%Cb;t/=Cb; n=t; c=cb*4+l; }
  else if (pc.dstFmt == 1){ c=i%pc.C; int t=i/pc.C; w=t%pc.W;t/=pc.W; h=t%pc.H;t/=pc.H; n=t; }
  else { w=i%pc.W; int t=i/pc.W; h=t%pc.H;t/=pc.H; c=t%pc.C;t/=pc.C; n=t; }
  if (c >= pc.C){ dst[i] = DST_T(0); return; }            // NC4HW4 padding lane
  dst[i] = DST_T(src[encode(pc.srcFmt,n,c,h,w)]);
}
```

- [ ] Add the source; teach CMake to emit 4 SPIR-V: `boundary_convert_f32_f32`, `_f32_f16`,
  `_f16_f32`, `_f16_f16` (e.g. `-DSRC_T=float -DDST_T=float16_t`, etc.), registered for `shader()`
  lookup by name.
- [ ] `./build.sh` compiles all 4. Commit.

### Task 5: `boundary_convert` op (pipelines + record helper)

**Files:** `src/backend/vulkan/ops/boundary_convert.{h,cpp}`

- [ ] A small class owned by the segment: holds the 4 pipelines (lazy), exposes
  `record(cmd, srcBuf, dstBuf, NCHW shape, TensorFormat srcFmt, DType srcDt, TensorFormat dstFmt,
  DType dstDt)` that picks the variant by `(srcDt,dstDt)`, sets push constants, dispatches
  `groups(dstCount, 256)`. `dstCount` = device-native element count (input) or declared count
  (output). NHWC of rank<4 is treated as NCHW (H,W==1).
- [ ] Build. Commit.

### Task 6: Wire direct-vs-convert into the segment

**Files:** `src/backend/vulkan/vk_backend.cpp`

- [ ] In `rebind()` (`:1127`): device-native `(fmt,dt) = (desc.gpuFlat?NCHW:NC4HW4, useFp16?Float16:
  Float32)`. If `decl==Auto || decl==device-native` → swap imported into `buffers_[tid]` (today).
  Else keep `origBoundary_[tid]` and push a boundary-convert binding for this tid (in/out).
- [ ] Track declared formats in the re-record signature so a changed declaration re-records.
- [ ] In `record()` (`:959`): emit input converts (imported→pooled) + `computeBarrier` before the op
  loop; emit output converts (pooled→imported) after the final barrier.
- [ ] Build. (Device verification in Phase 4.) Commit.

### Task 7: NHWC in the CPU reference / fallback

**Files:** `src/backend/vulkan/vk_backend.cpp` (`packToBuffer`/`unpackFromBuffer`, `:534-660`)

- [ ] Add the NHWC index case to both, matching the shader math; used when `importDmaBufFd` returns
  null (staged-copy fallback) and as the oracle.
- [ ] Build + 11 tests. Commit.

---

## Phase 3 — Naming-clarity sweep

### Task 8: Rename opaque short identifiers

**Files:** codebase-wide (`src/`, `examples/`, `convert/`, public headers where a param is opaque).

- [ ] Rename genuinely unclear names — a `Buffer b`, a param `a`/`c`, a `Model n` — to descriptive
  names. **Preserve** idiomatic loop counters and tensor-index math (`i/j/k`, `n/c/h/w`, `cb/l`,
  `x` for an `NCHW`), which are the clearest choice in numeric/GPU code.
- [ ] Do it file-group by file-group; build after each group. Keep it a pure rename (no behavior
  change).
- [ ] `./build.sh` clean + 11 tests. `clang-format -i`. Commit (may be a few commits by area).

---

## Phase 4 — On-device verification matrix

### Task 9: Extend the example to declared formats

**Files:** `examples/zerocopy_cache.cpp`

- [ ] Generalize `packDevice`/`unpackDevice` to `pack(shape, fmt, dtype, const float* nchw, void*)`
  / `unpack(...)` handling `{NCHW, NHWC, NC4HW4} × {Float32, Float16}`.
- [ ] For each input and output binding, loop the four declared formats
  `{NCHW f32, NCHW f16, NHWC f32, NC4HW4 f16}` (NHWC only for rank-4). Allocate the DMA-BUF at the
  declared byte size, fill (inputs) from the reference fp32-NCHW, run `Model::run`, unpack (outputs)
  back to fp32-NCHW, compare to the host fp32 path: bit-exact when no fp16 in the path, else
  cosine ≥ 0.999. Print a per-(binding,format) line.
- [ ] Build host. Commit.

### Task 10: Run on device

- [ ] `./build.sh --android`; `adb push` `vknn_*` + ensure `resnet50.vxm`, `encoder8_fp16.vxm`
  under `/data/local/tmp/vxrt/bench`.
- [ ] Run the example for both models; capture the full matrix output.
- [ ] Gate: every (binding, format) passes its bit-exact/cosine threshold on both models.
- [ ] If a format fails, debug (systematic-debugging) before proceeding.

---

## Phase 5 — Release bug audit

### Task 11: Codebase-wide correctness audit (multi-agent)

- [ ] Run a review workflow: dimensions (DMA-BUF/fd lifetime & ownership; Vulkan buffer/memory
  lifetime & barriers; the new convert path; cache file read/write; threading/`Session` teardown;
  shape/format edge cases; integer overflow in index math; error paths) → find → **adversarially
  verify** each finding (≥2 skeptics, default-refute) → keep only confirmed.
- [ ] Triage confirmed findings by severity; fix the release-blockers; re-build + re-run host tests
  and the device matrix after fixes.
- [ ] Record residual non-blockers in the final report.

---

## Self-review notes

- Spec coverage: Part 0 → Task 0; API → Tasks 1-2; plumbing → Task 3; converter shader/op/segment →
  Tasks 4-6; CPU NHWC/fallback → Task 7; verification matrix → Tasks 9-10; naming sweep (added by
  user) → Task 8; bug audit (added by user) → Task 11.
- No placeholders: shader is complete; signatures are concrete; index math is specified.
- Type consistency: `dmaBufFormat`/`dmaBufDtype` names are identical across `Tensor`/`IOTensor`/
  `RtTensor`; converter `record(...)` signature is fixed in Task 5 and used in Task 6.
