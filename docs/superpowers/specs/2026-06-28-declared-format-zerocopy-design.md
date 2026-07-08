# Declared-format zero-copy DMA-BUF I/O + public-enum k-prefix removal

## Summary

Let a caller of `Tensor::fromDmaBuf` / `toDmaBuf` **declare** the layout
(`NCHW` / `NHWC` / `NC4HW4`) and dtype (`Float32` / `Float16`) of their zero-copy DMA-BUF.
The engine compares the declared format against the model's device-native boundary format and
either (a) binds the fd directly as the boundary buffer when they match — the path already shipped —
or (b) records a GPU-side conversion that reads the imported DMA-BUF in the declared format and
writes the device-native boundary buffer (input), or the reverse (output). No host copy.

This work also drops the `k` prefix from every enumerator in the public `include/vknn/` headers
(`NCHW`, `Float32`, `Add`, …) as a self-contained prerequisite commit.

## Goals

- Caller declares `(layout, dtype)` per DMA-BUF binding; default `NCHW` + `Float32`.
- Declared format == device-native → direct bind (existing path), zero conversion.
- Declared format != device-native → GPU convert in the recorded command buffer, no host copy.
- A `TensorFormat::Auto` sentinel asserts "bytes are already device-native — never convert."
- Verify on device: each declared format for inputs and outputs, on resnet50 + yonosplat.
- Public enums read without the `k` prefix.

## Non-goals

- No unification of the in-graph `convert_layout` op into the new boundary converter; the in-graph
  node path stays as-is.
- No NHWC in the internal op set; NHWC is a boundary-only declared format.
- No per-graph recompile for declared formats; the conversion is a runtime, in-segment dispatch.
- No new dtypes beyond `Float32` / `Float16` at the boundary.

---

## Part 0 — Public enum k-prefix removal (prerequisite commit)

Drop the leading `k` from every enumerator of every enum declared in `include/vknn/`. The enums:

| Header | Enum | Members (after) |
|---|---|---|
| `config.h` | `BackendKind` | `Vulkan`, `Cpu` |
| `config.h` | `Precision` | `Fp32`, `Fp16`, `Auto` |
| `config.h` | `PowerHint` | `Normal`, `High`, `Low` |
| `config.h` | `TuningLevel` | `Off`, `Fast`, `Thorough` |
| `config.h` | `WinogradMode` | `Auto`, `On`, `Off` |
| `config.h` | `Hint` | `WinogradVariant`, `WinogradUnit`, `DirectConv3x3`, … |
| `logging.h` | `LogLevel` | `Debug`, `Info`, `Warn`, `Error`, `None` |
| `op.h` | `ActType` | `None`, `Relu`, `Relu6`, `Clip`, `HardSwish`, `SiLU` |
| `op.h` | `OpType` | `Conv`, `MatMul`, `Add`, … (~80) |
| `op.h` | `UnaryType` | `Sigmoid`, `Tanh`, … |
| `op.h` | `BinaryType` | `Invalid`, `Mul`, `Sub`, `Div`, `Max`, `Min`, `Pow`, `Add` |
| `op.h` | `ReduceType` | `Invalid`, `Mean`, `Sum`, `Max`, `Min`, `Prod`, `L2` |
| `op.h` | `Attr::Kind` | `None`, `Int`, `Float`, `Ints`, `Floats`, `String` (plain `enum`) |
| `common.h` | `Status` | `Ok`, `InvalidArgument`, `NotFound`, `Unsupported`, `IoError`, `RuntimeError`, `DeviceLost`, `NoTensor` |
| `tensor_format.h` | `TensorFormat` | `NCHW`, `NHWC`, `NC4HW4`, `Unknown`, `Auto` (new) |
| `dtype.h` | `DType` | `Float32`, `Float16`, `Int32`, `Int8`, `UInt8`, `Int64` |

Mechanics and safety:

- The project is C++17 and contains no `using enum`, so every `enum class` member is used fully
  qualified (`OpType::kConv`). The rename is a per-enum qualified replacement `Enum::kX` → `Enum::X`
  across `src/`, `examples/`, `tests/`, `convert/`. Qualification disambiguates from any internal
  enum in `src/` that happens to share a member spelling (e.g. an internal `kAdd`); internal enums
  declared outside `include/vknn/` keep their `k` prefix and are not touched.
- `Attr::Kind` is a plain `enum`; its members are referenced as `Attr::kInt` or unqualified inside
  `Attr`. Update those sites directly.
- The logging macros reference enumerators (`VKNN_LOG(LVL)` → `LogLevel::LVL`,
  `VKNN_DEBUG = VKNN_LOG(kDebug)`, `VKNN_WARN_THROTTLE` → `LogLevel::kWarn`). Update the macro bodies.
- `config.cpp` maps enums to/from **string literals** (`"vulkan"`, `"fp16"`, `"auto"`), not the C++
  identifier names, so JSON config and the binary cache file (`VKNNCAC1`) are unaffected by the rename.
- Gate: host build clean, 11 host tests pass — pure mechanical change, no behavior difference.

This lands as commit 1; Part 1 is written against the new names.

---

## Part 1 — Declared-format zero-copy

### API surface (`include/vknn/model.h`, `src/core/model.cpp`)

```cpp
static Tensor fromDmaBuf(int fd, std::vector<int64_t> shape, std::string name = "",
                         TensorFormat layout = TensorFormat::NCHW,
                         DType dtype = DType::Float32);
static Tensor toDmaBuf(int fd, std::vector<int64_t> shape, std::string name = "",
                       TensorFormat layout = TensorFormat::NCHW,
                       DType dtype = DType::Float32);
```

`Tensor` gains `dmaBufFormat_` / `dmaBufDtype_` and accessors `dmaBufFormat()` / `dmaBufDtype()`,
alongside the existing `fd_`. `TensorFormat::Auto` means "the fd already holds device-native bytes —
take the direct path, never convert" (the dtype argument is ignored). Adding defaulted parameters is
source-compatible; the 11 host tests compile unchanged.

### Plumbing the declaration (`session.h`, `tensor.h`, `session.cpp`, `model.cpp`)

The declared `(format, dtype)` rides next to the fd at every layer that already carries `dmaBufFd`:

- `IOTensor` (`session.h:21`) gains `dmaBufFormat` / `dmaBufDtype`.
- `RtTensor` (`tensor.h:54`) gains `dmaBufFormat` / `dmaBufDtype`.
- `Model::run` marshals `Tensor → IOTensor` (the loop at `model.cpp:130` that already copies the fd).
- `Session::run` binds `IOTensor → RtTensor` (the loops at `session.cpp:554` inputs / `:584` outputs),
  and resets the declared fields to the defaults after the run, mirroring `dmaBufFd`.

### Direct-vs-convert decision (`src/backend/vulkan/vk_backend.cpp`)

The segment knows the device-native boundary `(format, dtype)`:
`format = desc.gpuFlat ? NCHW : NC4HW4`, `dtype = useFp16_ ? Float16 : Float32`. In `rebind()`
(`vk_backend.cpp:1127`), after importing the fd:

- Declared is `Auto`, **or** declared `(format, dtype)` == device-native → **direct**: swap the
  imported buffer into `buffers_[tid]` (today's path).
- Otherwise → **convert**: leave `buffers_[tid]` = the pooled dedicated boundary buffer
  (`origBoundary_[tid]`); register a boundary-convert binding
  `{tid, importedBuf, NCHW shape, declFmt, declDtype, devFmt, devDtype, direction}`.

`record()` (`vk_backend.cpp:959`) then emits, inside the existing command buffer:

- **input convert**: `convert(importedBuf → pooledBoundary)` at the start, then a `computeBarrier`,
  before the op loop.
- **output convert**: after the op loop and final barrier, `convert(pooledBoundary → importedBuf)`.

The existing `reRecord` flag fires when the imported-buffer set **or** the declared formats change
versus the last recording, so a steady binding re-records once (same cost as a reused fd today). The
imported DMA-BUF is created with storage usage (the direct path binds it as an SSBO), so it binds as
the converter's source/destination directly.

### The convert kernel (`src/backend/vulkan/ops/boundary_convert.{h,cpp}`, `shaders/boundary_convert.comp`)

One op file (one-op-per-file; `convert_layout` is untouched). A single kernel handles every layout
pair in both directions:

- Each thread owns one **destination** element, decodes its `(n, c, h, w)` from the destination
  layout, and: if `c >= C` (an NC4HW4 padding lane) writes `0`; else encodes the source index from
  the source layout and copies with dtype conversion.
- Index math mirrors `packToBuffer` / `unpackFromBuffer` exactly:
  - NCHW: `((n*C + c)*H + h)*W + w`
  - NHWC: `((n*H + h)*W + w)*C + c`
  - NC4HW4: `((((n*Cb + cb)*H + h)*W + w)*4) + l`, with `Cb = (C+3)/4`, `cb = c/4`, `l = c%4`
- `srcLayout` / `dstLayout` are push constants — one SPIR-V covers all layout pairs including NHWC.
- Cross-precision cannot be a runtime branch on SSBO member types, so dtype is a compile variant:
  **4 SPIR-V** — `f32→f32`, `f32→f16`, `f16→f32`, `f16→f16` — selected by `(srcDtype, dstDtype)`.
  The shader source uses `SRC_T` / `DST_T` macros; CMake compiles the four variants.
- Dispatch count = number of destination elements; workgroup local size 256 (matches
  `convert_layout`).

`NCHW::from` normalizes ranks < 4 (`[N,C,L]→[N,C,L,1]`, `[N,C]→[N,C,1,1]`, `[C]→[1,C,1,1]`); for
those ranks `H`/`W` collapse so NHWC ≡ NCHW. Declared NHWC on a non-4D tensor is therefore accepted
as NCHW rather than an error.

### CPU reference + fallback (`src/backend/vulkan/vk_backend.cpp`)

`packToBuffer` / `unpackFromBuffer` (`vk_backend.cpp:534-660`) gain the NHWC index case. This is the
**fallback** path — if `importDmaBufFd` returns null (a device without DMA-BUF import), the binding
falls back to the staged copy, CPU-converting declared↔device-native with the same math — and it
keeps the CPU reference a faithful oracle for the GPU kernel.

---

## Data flow

```
Tensor::fromDmaBuf(fd, shape, name, layout, dtype)
  → Tensor{ fd_, dmaBufFormat_, dmaBufDtype_ }
  → Model::run  → IOTensor{ dmaBufFd, dmaBufFormat, dmaBufDtype }
  → Session::run → RtTensor{ dmaBufFd, dmaBufFormat, dmaBufDtype }
  → VulkanSegment::rebind:
       device-native = (gpuFlat?NCHW:NC4HW4, useFp16?Float16:Float32)
       Auto or declared==device-native → swap imported into buffers_[tid]   (direct)
       else → keep pooled boundary buffer + register boundary-convert        (convert)
  → VulkanSegment::record:
       input  convert: imported → pooledBoundary  (before op loop, + barrier)
       output convert: pooledBoundary → imported  (after op loop + final barrier)
```

## File-by-file change list

Part 0 (rename): `include/vknn/{config,logging,op,common,tensor_format,dtype}.h` plus every
`Enum::kX` use in `src/`, `examples/`, `tests/`, `convert/`.

Part 1 (feature):
- `include/vknn/tensor_format.h` — add `TensorFormat::Auto`.
- `include/vknn/model.h`, `src/core/model.cpp` — `fromDmaBuf`/`toDmaBuf` params; `Tensor` fields +
  accessors; `Model::run` marshalling.
- `include/vknn/session.h` — `IOTensor` declared fields.
- `include/vknn/tensor.h` — `RtTensor` declared fields.
- `src/core/session.cpp` — bind declared fields in/out, reset after run.
- `src/backend/vulkan/vk_backend.cpp` — `rebind()` direct-vs-convert; `record()` convert dispatches;
  NHWC in `packToBuffer`/`unpackFromBuffer`.
- `src/backend/vulkan/ops/boundary_convert.{h,cpp}` — new converter op (pipelines + record helper).
- `shaders/boundary_convert.comp` — new shader; CMake 4-variant compile.
- `examples/zerocopy_cache.cpp` — declared-format verification matrix.

## Testing & verification

- **Host**: `./build.sh`; `vknn_tests` 11/11 pass after Part 0 and after Part 1.
- **Device** (the target mobile GPU): `./build.sh --android`; push `vknn_*`; run the extended
  example for `resnet50.vxm` and `encoder8_fp16.vxm`.
- **Verification matrix**: for each model, for inputs and outputs, run each declared format in
  `{NCHW Float32, NCHW Float16, NHWC Float32, NC4HW4 Float16}`. The host fills each input DMA-BUF in
  the declared format (from the reference fp32-NCHW input) and unpacks each output DMA-BUF back to
  fp32-NCHW. Compare against the host fp32 path: **bit-exact** when no fp16 is involved,
  **cosine ≥ 0.999** when the path crosses fp16. NHWC variants run only for 4D tensors.
- `clang-format -i` on every touched file (`.clang-format`, Google base, 100 col).

The example generalizes `packDevice`/`unpackDevice` to take a declared `(TensorFormat, DType)`
instead of the hardcoded device-native fp16, and loops over the format list per binding.

## Risks

- A missed `Enum::kX` site fails the build (caught immediately); the rename carries no behavior change.
- Cross-precision fp16 SSBO reads require `shaderFloat16` + `16bit_storage` (present on both target
  GPUs).
- Host visibility of an output DMA-BUF written by the convert shader uses the same submit/fence
  guarantee the shipped direct-output path already relies on.

## Commit sequencing

1. Drop `k` prefix from public enums (build + 11 tests green).
2. Declared-format plumbing + segment direct-vs-convert + `boundary_convert` op/shader + CPU NHWC.
3. Extend the example; on-device verification matrix.

Branch `declared-format-zerocopy`; merge to main and push only when asked.
