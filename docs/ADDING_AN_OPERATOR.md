# Adding an operator to VKNN

A copy-pasteable walkthrough for adding a new ONNX operator to VKNN
(Vulkan Neural Network). The worked example is **LeakyRelu**
(`y = x` for `x >= 0`, `y = alpha * x` otherwise), an elementwise unary op with
one float attribute. The same pattern fits any pointwise op (`Mul`,
`Sigmoid`, `Tanh`, ...).

There are four pieces. You can stop after any of them and still have a working
build: under VKNN's capability/fallback model a CPU-only op runs on the
CPU backend, and the Vulkan path falls back to it automatically.

1. Declare the op: `OpType` enum value + name mappings.
2. CPU reference: subclass `vknn::CpuOp`, implement `run()`, register it.
3. Vulkan kernel: a GLSL `.comp` shader + subclass `vknn::VulkanOp` (`prepare()` /
   `record()`), register it.
4. (Build) Nothing to wire up — self-registration relies on whole-archive
   linking; see [the last section](#self-registration-and-whole-archive-linking).

Throughout, "register" means dropping a static `VKNN_REGISTER_*` line at file
scope. No edits to any core dispatch code are required.

---

## 1. Declare the op type

`OpType` is the backend-agnostic operator tag carried by every IR `Node`. Add a
value, then keep the two switch/map tables in `src/core/op.cpp` in sync.

### `include/vknn/op.h`

Add an enum value to `OpType` (placement is not significant — it is never
serialized as an integer):

```cpp
enum class OpType {
  kUnknown = 0,
  kConv,
  kClip,
  kRelu,
  kLeakyRelu,       // <-- new: y = x>=0 ? x : alpha*x
  kAdd,
  // ...
};
```

### `src/core/op.cpp`

Add the `OpType` → display-name mapping in `opTypeName()`:

```cpp
const char* opTypeName(OpType t) {
  switch (t) {
    case OpType::kConv: return "Conv";
    case OpType::kClip: return "Clip";
    case OpType::kRelu: return "Relu";
    case OpType::kLeakyRelu: return "LeakyRelu";   // <-- new
    // ...
  }
}
```

And the ONNX-op-name → `OpType` mapping in `opTypeFromOnnx()` (this string is the
ONNX node `op_type`, so it must match the ONNX spec exactly):

```cpp
OpType opTypeFromOnnx(const std::string& s) {
  static const std::unordered_map<std::string, OpType> m = {
      {"Conv", OpType::kConv},
      {"Clip", OpType::kClip},
      {"Relu", OpType::kRelu},
      {"LeakyRelu", OpType::kLeakyRelu},   // <-- new
      // ...
  };
  // ...
}
```

Anything missing from this map imports as `OpType::kUnknown`, and the ONNX
attributes still attach to the `Node` (`LeakyRelu` carries a float `alpha`,
default `0.01`), retrievable via `node.attr.getf("alpha", 0.01f)`.

That's the entire core-side change. The op now flows through the import →
IR → graph-pass → partition pipeline; all that's missing is a kernel on at
least one backend.

---

## 2. CPU reference kernel

The CPU backend is the scalar reference (plus NEON kernels for `Add`/`Gemm`). It's
also the fallback target for every op the primary backend declines, so writing
the CPU kernel first gives you a correct baseline to diff against.

A CPU op is a subclass of `vknn::CpuOp` (declared in
`src/backends/cpu/cpu_backend.h`):

```cpp
class CpuOp {
 public:
  virtual ~CpuOp() = default;
  virtual void run(const Node& node, ExecContext& ctx) = 0;
  // Which dtypes this op supports (capability/fallback). Default: fp32 + int64.
  virtual bool supportsDType(DType dt) const {
    return dt == DType::kFloat32 || dt == DType::kInt64;
  }
};
```

`run()` reads its inputs and writes its outputs through the `ExecContext`. The
context resolves a `TensorId` to its runtime tensor:

```cpp
struct ExecContext {
  std::vector<RtTensor>* pool = nullptr;   // indexed by TensorId
  const Graph* graph = nullptr;
  const Config* config = nullptr;
  Profiler* profiler = nullptr;
  RtTensor& t(TensorId id) { return (*pool)[id]; }
};
```

Host buffers are canonical **NCHW, fp32**. Use `RtTensor::host.f32()` to get the
data pointer and `RtTensor::elems()` for the element count. The
`cpu::allocOut(RtTensor&, const Shape&)` helper sizes the output's host buffer
and returns a `float*` (there is also `cpu::allocOutI64` for integer outputs, and
`cpu::applyAct` to apply a fused activation in place).

Add the implementation to `src/backends/cpu/ops_basic.cpp` (inside the existing
anonymous `namespace`, next to `ReluCpuOp`):

```cpp
struct LeakyReluCpuOp : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    float alpha = node.attr.getf("alpha", 0.01f);
    int64_t n = X.elems();
    float* y = cpu::allocOut(Y, X.shape);
    const float* x = X.host.f32();
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] >= 0 ? x[i] : alpha * x[i];
  }
};
```

Register it at namespace scope (bottom of the file, alongside the other
`VKNN_REGISTER_CPU_OP` lines):

```cpp
VKNN_REGISTER_CPU_OP(OpType::kLeakyRelu, LeakyReluCpuOp);
```

`VKNN_REGISTER_CPU_OP(OPTYPE, CLASS)` expands to a static `CpuOpRegistrar` whose
constructor calls `CpuOpRegistry::instance().reg(...)`. After this, the CPU
backend's `supports()` returns `true` for `kLeakyRelu` (fp32/int64/int32) and the
session can place the node on CPU.

That's a complete, correct (if slow) operator. Build the host target
and the op runs on the CPU backend, or as a Vulkan-segment fallback.

---

## 3. Vulkan compute kernel

To run the op on the GPU you need two things: a GLSL compute shader, and a
`vknn::VulkanOp` subclass that builds the pipeline and records the dispatch.

VKNN's internal device layout is **NC4HW4** — channels packed in `vec4` blocks.
For a pure elementwise op the packing is transparent: every packed element is
processed independently, like the existing `add.comp`. So `LeakyRelu` can
operate on the flat packed buffer and never has to reason about the layout.

### 3a. The shader: `shaders/leakyrelu.comp`

Shaders are GLSL compute, compiled at build time by `glslc`
(`--target-env=vulkan1.3 -O`) and embedded into the static lib by
`tools/embed_spirv.py` (exposed as `vknn::embeddedShaders()`). The shader's base
name (here `leakyrelu`) is the key you look up when creating the pipeline.

The push-constant block and binding count must match the C++ side exactly. Model
it on `shaders/add.comp`:

```glsl
#version 450
// Elementwise LeakyRelu over the NC4HW4-packed buffer: y = x>=0 ? x : alpha*x.
#include "common.glsl"

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly  buffer BufX { float x[]; };
layout(std430, binding = 1) writeonly buffer BufY { float y[]; };

layout(push_constant) uniform PC { uint count; float alpha; } pc;

void main() {
  uint i = gl_GlobalInvocationID.x;
  if (i >= pc.count) return;
  float v = x[i];
  y[i] = v >= 0.0 ? v : pc.alpha * v;
}
```

`common.glsl` holds the shared fused-activation helper `vx_act()` and the
`ACT_*` codes (kept in sync with `vknn::ActType`). LeakyRelu doesn't use them
here, but include the header for consistency.

**fp16 variant.** VKNN's fp16 device path uses fp16 *storage* with fp32
*accumulation*, and selects a `_fp16`-suffixed shader at runtime via the `sv()`
helper (`sv("conv", true)` → `"conv_fp16"`). If you want LeakyRelu to run in the
fp16 pipeline, add `shaders/leakyrelu_fp16.comp` with `float16_t` storage buffers
(enable `GL_EXT_shader_16bit_storage` / `GL_EXT_shader_explicit_arithmetic_types_float16`),
computing in `float`. If you skip the fp16 variant, only register the op for the
fp32 path (or guard pipeline creation on `env.useFp16`); a missing `_fp16` shader
would fail pipeline creation.

`glslc` is discovered by CMake (`find_program(GLSLC glslc ...)`). Any `*.comp`
under `shaders/` is picked up automatically by the `file(GLOB ...)` in
`CMakeLists.txt` — no build-file edit is needed to add a shader. (On a host build
without `glslc`, `embeddedShaders()` is a stub and the Vulkan path is compiled
out; the CPU kernel is what runs.)

### 3b. The op: subclass `vknn::VulkanOp`

A Vulkan op (declared in `src/backends/vulkan/vk_backend.h`) has two phases:

```cpp
class VulkanOp {
 public:
  virtual ~VulkanOp() = default;
  /// Create pipeline(s), prepack + upload weights, allocate op-private buffers.
  virtual void prepare(const Node& node, VkOpEnv& env) = 0;
  /// Record dispatch(es) into the command buffer.
  virtual void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) = 0;
};
```

`prepare()` runs once at session-creation time (build the pipeline, prepack and
upload any weights, read static shapes). `record()` runs once per static segment,
emitting the dispatch into the segment's pre-recorded command buffer. The
command buffer is recorded once and replayed every inference, so `record()` must
only reference buffers that stay stable across runs — activation buffers are
fetched fresh through `env.devBuf(id)`.

`prepare()`/`record()` receive a `VkOpEnv`:

```cpp
struct VkOpEnv {
  VulkanBackend* backend;
  vk::VulkanContext* ctx;
  vk::PipelineCache* cache;
  const Graph* graph;
  const Config* config;
  std::function<vk::Buffer*(TensorId)> devBuf;  // activation buffer for a tensor id
  bool useFp16;
  WeightCache* weights;                          // prepacked-weight + tuning cache (may be null)
  vk::CommandRunner* runner;                      // for on-device autotuning benchmarks
  TuningLevel tuning;
};
```

Key APIs used below:

- `env.graph->desc(id).shape` — logical NCHW shape of a tensor (use
  `packedElems(shape)` from `vk_backend.h` for the NC4HW4 element count).
- `env.devBuf(id)` — the `vk::Buffer*` holding the activation for tensor `id`.
- `vk::ComputePipeline(ctx, shaderName, numBuffers, pushConstBytes, specData,
  cacheHandle)` — builds the pipeline from an embedded shader.
- `pipe->dispatch(cmd, {bufHandles...}, &pc, sizeof(pc), groupsX)` — records bind
  + push-descriptors + push-constants + dispatch.

Add this to `src/backends/vulkan/vk_ops.cpp` (inside the anonymous `namespace`,
next to `AddVulkanOp`). The push-constant struct must byte-match the shader's
`PC` block:

```cpp
struct LeakyReluPC { uint32_t count; float alpha; };

struct LeakyReluVulkanOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  LeakyReluPC pc{};

  void prepare(const Node& node, VkOpEnv& env) override {
    pc.count = (uint32_t)packedElems(env.graph->desc(node.outputs[0]).shape);
    pc.alpha = node.attr.getf("alpha", 0.01f);
    // 2 buffers (x, y); sv() selects the _fp16 variant when env.useFp16.
    pipe = std::make_unique<vk::ComputePipeline>(
        *env.ctx, sv("leakyrelu", env.useFp16).c_str(), /*numBuffers=*/2,
        sizeof(LeakyReluPC), std::vector<uint32_t>{}, env.cache->handle());
  }

  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* x = env.devBuf(node.inputs[0]);
    vk::Buffer* y = env.devBuf(node.outputs[0]);
    // local_size_x = 256 in the shader -> ceil(count / 256) workgroups.
    uint32_t groups = (uint32_t)((pc.count + 255) / 256);
    pipe->dispatch(cmd, {x->handle(), y->handle()}, &pc, sizeof(pc), groups);
  }
};
```

Notes:

- The workgroup size in the dispatch (`/256`) must match `local_size_x` in the
  shader. The general `conv` shader instead uses a spec-constant `local_size_x`
  (the `specData` argument) so its workgroup size can be autotuned on-device via
  `env.runner` and cached; an elementwise op doesn't need that.
- Ops with weights (see `ConvVulkanOp` / `GemmVulkanOp`) prepack the initializer
  into NC4HW4 / transposed layout in `prepare()` and upload it via
  `uploadCached(env, key, computeFn)`, which consults the content-keyed weight
  cache so warm starts skip the repack. `LeakyRelu` has no weights, so this is
  not needed.

Register it at namespace scope (bottom of the file, with the other
`VKNN_REGISTER_VK_OP` lines):

```cpp
VKNN_REGISTER_VK_OP(OpType::kLeakyRelu, LeakyReluVulkanOp);
```

After this, the Vulkan backend's `supports()` returns `true` for `kLeakyRelu`
(because `VkOpRegistry::instance().has(kLeakyRelu)` is now true), and the session
will place LeakyRelu nodes in Vulkan segments.

---

## 4. Self-registration and whole-archive linking

There is no central table of operators to edit. Every `VKNN_REGISTER_CPU_OP`,
`VKNN_REGISTER_VK_OP`, and `VKNN_REGISTER_BACKEND` declares a file-scope static whose
constructor inserts the factory into the relevant registry
(`CpuOpRegistry` / `VkOpRegistry` / `BackendRegistry`) before `main()` runs.

The catch: a static-library object file that nothing references gets dropped by the
linker, taking its self-registration with it. VKNN avoids this by linking the
static lib **whole-archive** everywhere it's consumed
(`CMakeLists.txt`):

```cmake
target_link_libraries(vknn_${ex}  PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,vknn>")
target_link_libraries(vknn_tests  PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,vknn>" gtest gtest_main)
```

This pulls in every object file (and therefore every registrar) whether or not
the symbol is directly referenced. As long as your new op lives in a
source file already globbed into the `vknn` target — `src/backends/cpu/*.cpp` and
`src/backends/vulkan/*.cpp` both are — registration "just works" with no further
build wiring.

---

## The capability / fallback model

VKNN assigns each node to a backend at session-creation time, then partitions the
topo-ordered node list into maximal same-backend **segments**. Two capability
hooks drive this.

### `Backend::supports(OpType, DType)`

Each backend answers whether it can run a given op at a given compute dtype
(`include/vknn/backend.h`). The implementations are thin wrappers over the op
registries:

- **Vulkan** (`src/backends/vulkan/vk_backend.cpp`):

  ```cpp
  bool supports(OpType t, DType dt) const override {
    if (!available()) return false;
    // Debug hook: VKNN_DISABLE_VK_OPS="Add,Conv" forces those ops to fall back.
    if (const char* dis = std::getenv("VKNN_DISABLE_VK_OPS")) {
      std::string list = dis, name = opTypeName(t);
      if (list.find(name) != std::string::npos) return false;
    }
    return VkOpRegistry::instance().has(t);
  }
  ```

- **CPU** (`src/backends/cpu/cpu_backend.cpp`):

  ```cpp
  bool supports(OpType t, DType dt) const override {
    auto& r = CpuOpRegistry::instance();
    if (!r.has(t)) return false;
    return dt == DType::kFloat32 || dt == DType::kInt64 || dt == DType::kInt32;
  }
  ```

So registering your op (step 2 / step 3) is what makes
`supports()` return true — there is no separate capability list to maintain. The
per-op `CpuOp::supportsDType()` hook lets a CPU op narrow the dtypes it accepts
(e.g. the shape ops accept all dtypes by returning `true`); the default is
fp32 + int64.

### Backend selection and fallback

`Session` (`src/core/session.cpp`) builds a priority-ordered backend list — the
configured `cfg.backend` first, then the `fallback` list, with CPU appended last
when `allowCpuFallback` is set. For each node it picks the highest-priority
backend whose `supports()` returns true:

```cpp
for (size_t bi = 0; bi < backends_.size(); ++bi) {
  if (backends_[bi]->supports(nd.type, dt)) { chosen = (int)bi; break; }
}
if (chosen < 0) throw Error(Status::kUnsupported, "no backend supports op ...");
```

If the chosen backend isn't the configured primary (because the primary's
`supports()` said no), the session emits a throttled fallback warning. Contiguous
nodes assigned to the same backend are then merged into one segment; tensor
residency is reconciled at segment boundaries via `toHost()` / `toDevice()`, so a
CPU fallback segment in the middle of a Vulkan graph triggers an
unpack/repack around it. A CPU segment created because the primary backend couldn't
run its ops is tagged `Segment::isFallback = true` (drives the warning and
the profiler tag).

This is why each step is independently shippable:

- **Only the CPU kernel registered** → Vulkan's `supports()` returns false for
  the op, the node falls back to a CPU segment, output stays correct (just with a
  host round-trip at the segment boundary).
- **Both kernels registered** → the op runs in-place in the Vulkan segment with no
  extra sync.

You can exercise the fallback path without touching code by forcing the op back
to CPU at runtime:

```sh
VKNN_DISABLE_VK_OPS="LeakyRelu" ./vknn_classify ...
```

This is the same mechanism used to validate the NEON fallback path
(`VKNN_DISABLE_VK_OPS="Add,GlobalAveragePool"` re-partitions the graph into
Vulkan/CPU segments while keeping the output bit-comparable).

---

## Checklist

- [ ] `include/vknn/op.h`: add `OpType::kLeakyRelu`.
- [ ] `src/core/op.cpp`: add to `opTypeName()` and `opTypeFromOnnx()`.
- [ ] `src/backends/cpu/ops_basic.cpp`: `LeakyReluCpuOp` + `VKNN_REGISTER_CPU_OP`.
- [ ] `shaders/leakyrelu.comp` (+ optional `leakyrelu_fp16.comp`).
- [ ] `src/backends/vulkan/vk_ops.cpp`: `LeakyReluVulkanOp` + `VKNN_REGISTER_VK_OP`.
- [ ] Build and run; diff Vulkan output against the CPU reference (and against
      `scripts/get_golden.py` if you want an external check).
