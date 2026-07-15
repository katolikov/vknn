# Adding an operator to VKNN

A walkthrough for adding a new ONNX operator to VKNN
(Vulkan Neural Network). The worked example is **LeakyRelu**
(`y = x` for `x >= 0`, `y = alpha * x` otherwise), an elementwise unary op with
one float attribute. The walkthrough shows the full standalone-op recipe; note that in the
shipped engine the elementwise families are sub-codes — a unary (`Sigmoid`, `Tanh`,
`LeakyRelu`, ...) imports as `OpType::Unary` with a `UnaryType` in `Node::subOp`, a binary
(`Mul`, `Div`, ...) as `OpType::Binary` with a `BinaryType` — so a new pointwise op usually
extends a family enum + its CPU/GPU switch rather than adding a whole `OpType`.

There are four pieces. Each is independently shippable: under VKNN's
capability/fallback model a CPU-only op runs on the CPU backend, and the Vulkan
path falls back to it automatically.

1. Declare the op: `OpType` enum value + name mappings.
2. CPU reference: subclass `vknn::CpuOp`, implement `run()`, register it.
3. Vulkan kernel: a GLSL `.comp` shader + subclass `vknn::VulkanOp` (`prepare()` /
   `record()`), register it.
4. (Build) Self-registration relies on whole-archive linking; see
   [the last section](#self-registration-and-whole-archive-linking).

Throughout, "register" means a static `VKNN_REGISTER_*` line at file
scope. No edits to any core dispatch code are required.

---

## 1. Declare the op type

`OpType` is the backend-agnostic operator tag carried by every IR `Node`. Add a
value, then keep the two switch/map tables in `src/core/op.cpp` in sync.

### `include/vknn/op_type.h`

Append the enum value at the **end** of `OpType` (`op.h` re-exports it). The enum is
**append-only**: `model_io` serializes it as a raw integer into `.vxm` files, so a value
inserted mid-enum shifts every later op and silently corrupts existing models:

```cpp
enum class OpType {
  Unknown = 0,
  Conv,
  Clip,
  Relu,
  Add,
  // ...
  FusedAttention,  // <-- the current last value
  LeakyRelu,       // <-- new values go at the END: y = x>=0 ? x : alpha*x
};
```

### `src/core/op.cpp`

Add the `OpType` → display-name mapping in `opTypeName()`:

```cpp
const char* opTypeName(OpType t) {
  switch (t) {
    case OpType::Conv: return "Conv";
    case OpType::Clip: return "Clip";
    case OpType::Relu: return "Relu";
    case OpType::LeakyRelu: return "LeakyRelu";   // <-- new
    // ...
  }
}
```

And the ONNX-op-name → `OpType` mapping in `opTypeFromOnnx()` (this string is the
ONNX node `op_type`, so it must match the ONNX spec exactly):

```cpp
OpType opTypeFromOnnx(const std::string& s) {
  static const std::unordered_map<std::string, OpType> m = {
      {"Conv", OpType::Conv},
      {"Clip", OpType::Clip},
      {"Relu", OpType::Relu},
      {"LeakyRelu", OpType::LeakyRelu},   // <-- new
      // ...
  };
  // ...
}
```

Anything missing from this map imports as `OpType::Unknown`, and the ONNX
attributes still attach to the `Node` (`LeakyRelu` carries a float `alpha`,
default `0.01`), retrievable via `node.attr.getf("alpha", 0.01f)`.

This is the entire core-side change. The op flows through the import →
IR → graph-pass → partition pipeline; the remaining work is the kernels. The CPU
kernel is mandatory: `tools/check_support_consistency.py` (run by `scripts/ci_host.sh`)
fails when an OpType mapped in `opTypeFromOnnx()` has no `VKNN_REGISTER_CPU_OP` under
`src/backend/cpu/ops/` — a recognized op with no CPU kernel cannot run even as a
fallback. (Its `CPU_KERNEL_EXEMPT` list excuses only ops an import pass lowers away
before planning — the quantized family, `Dropout`, `InstanceNorm`, the ORT contrib
family.) The Vulkan kernel is the optional half.

---

## 2. CPU reference kernel

The CPU backend is the scalar reference (plus NEON kernels for `Add`/`Gemm`). It
is also the fallback target for every op the primary backend declines, so the
CPU kernel provides a correct baseline to diff against.

A CPU op is a subclass of `vknn::CpuOp` (declared in
`src/backend/cpu/cpu_backend.h`):

```cpp
class CpuOp {
 public:
  virtual ~CpuOp() = default;
  virtual void run(const Node& node, ExecContext& ctx) = 0;
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

Add the implementation as its own file `src/backend/cpu/ops/leakyrelu.cpp` — one operator
per file; model it on `src/backend/cpu/ops/relu.cpp`:

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

Register it at the bottom of the same file:

```cpp
VKNN_REGISTER_CPU_OP(OpType::LeakyRelu, LeakyReluCpuOp);
```

`VKNN_REGISTER_CPU_OP(OPTYPE, CLASS)` expands to a static `CpuOpRegistrar` whose
constructor calls `CpuOpRegistry::instance().reg(...)`. The CPU
backend's `supports()` then returns `true` for `LeakyRelu` (fp32/int64/int32) and the
session can place the node on CPU.

This is a complete, correct operator. The host target builds and the op runs on
the CPU backend, or as a Vulkan-segment fallback.

---

## 3. Vulkan compute kernel

Running the op on the GPU requires two things: a GLSL compute shader, and a
`vknn::VulkanOp` subclass that builds the pipeline and records the dispatch.

VKNN's internal device layout is **NC4HW4** — channels packed in `vec4` blocks.
For a pure elementwise op the packing is transparent: every packed element is
processed independently, like `add.comp`. `LeakyRelu` operates on the flat
packed buffer and never reasons about the layout.

### 3a. The shader: `shaders/leakyrelu.comp`

Shaders are GLSL compute, compiled at build time by the vendored glslang
(`third_party/glslang`; a system `glslc` is the fallback), targeting `vulkan1.3`, and
embedded into the static lib by `tools/embed_spirv.py` (exposed as
`vknn::embeddedShaders()`). The shader's base
name (here `leakyrelu`) is the key you look up when creating the pipeline.

The push-constant block and binding count must match the C++ side exactly. Model
it on `shaders/add.comp`:

```glsl
#version 450
#define VKNN_NO_RTE 1 // stores round via vknnRte16: must match the fused-unit per-step rounding exactly
#include "precision.glsl"
// Elementwise LeakyRelu over the NC4HW4-packed buffer: y = x>=0 ? x : alpha*x.
#include "common.glsl"

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly  buffer BufX { STORE x[]; };
layout(std430, binding = 1) writeonly buffer BufY { STORE y[]; };

layout(push_constant) uniform PC { uint count; float alpha; } pc;

void main() {
  // 2-term flat-id recovery: dispatch() spills a group count past the device's
  // x-limit into y, so a bare gl_GlobalInvocationID.x drops the tail.
  uint i = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x;
  if (i >= pc.count) return;
  float v = float(x[i]);
  y[i] = TO_STORE(v >= 0.0 ? v : pc.alpha * v);
}
```

`common.glsl` holds the shared fused-activation helper `vx_act()` and the
`ACT_*` codes (kept in sync with `vknn::ActType`). LeakyRelu doesn't use them
here, but include the header for consistency.

**fp16 variant.** VKNN's fp16 device path uses fp16 *storage* with fp32
*accumulation*, and selects a `_fp16`-suffixed shader at runtime via the `shader()`
helper in `vk_op_common.h` (`shader("conv", true)` → `"conv_fp16"`). Shaders are
precision-templated, not duplicated: `#include "precision.glsl"` and declare the storage
buffers with the `STORE` element type (read `float(x[i])`, write `y[i] = STORE(v)`;
arithmetic stays fp32). The build compiles each such shader twice, emitting both the
fp32 and `_fp16` variants from one source.

`glslc` is discovered by CMake (`find_program(GLSLC glslc ...)`). Any `*.comp`
under `shaders/` is picked up automatically by the `file(GLOB ...)` in
`CMakeLists.txt`; adding a shader needs no build-file edit. On a host build the Vulkan
backend is compiled out by default (`VKNN_ENABLE_VULKAN` defaults to on only for
Android) and `embeddedShaders()` is a stub; the CPU kernel runs. With Vulkan enabled
but no shader compiler found, the shader build is skipped and the same stub is used.

### 3b. The op: subclass `vknn::VulkanOp`

A Vulkan op (declared in `src/backend/vulkan/vk_backend.h`) has two phases:

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
command buffer is recorded once and replayed every inference, so `record()`
references only buffers that stay stable across runs — activation buffers are
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
  Tuning tuning;                                  // Config::tuning effort (None/Fast/Heavy)
  Mode winograd;                                  // Hint::Winograd value (Auto/On/Off)
  std::string modelTag;                           // per-model weight-cache namespace
  std::string gpuTag;                             // per-GPU autotune namespace (kernel choice is device-specific)
};
```

Key APIs used below:

- `env.graph->desc(id).shape` — logical NCHW shape of a tensor (use
  `packedElems(shape)` from `vk_backend.h` for the NC4HW4 element count).
- `env.devBuf(id)` — the `vk::Buffer*` holding the activation for tensor `id`.
- `env.pipeline(shaderName, numBuffers, pushConstBytes, specData)` — returns the
  session-shared `std::shared_ptr<vk::ComputePipeline>` for an embedded shader; nodes
  with the same kernel configuration share one pipeline.
- `pipe->dispatch(cmd, {bufHandles...}, &pc, sizeof(pc), groupsX)` — records bind
  + push-descriptors + push-constants + dispatch.

Add this as its own file `src/backend/vulkan/ops/leakyrelu.cpp` — one operator per
file; model it on `src/backend/vulkan/ops/relu.cpp`, starting with
`#include "vk_op_common.h"`. The push-constant struct must byte-match the shader's
`PC` block:

```cpp
struct LeakyReluPC { uint32_t count; float alpha; };

struct LeakyReluVulkanOp : VulkanOp {
  std::shared_ptr<vk::ComputePipeline> pipe;
  LeakyReluPC pc{};

  void prepare(const Node& node, VkOpEnv& env) override {
    pc.count = (uint32_t)packedElems(env.graph->desc(node.outputs[0]).shape);
    pc.alpha = node.attr.getf("alpha", 0.01f);
    // 2 buffers (x, y); shader() selects the _fp16 variant when env.useFp16.
    pipe = env.pipeline(shader("leakyrelu", env.useFp16), /*numBuffers=*/2,
                        sizeof(LeakyReluPC), std::vector<uint32_t>{});
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

Register it at the bottom of the same file:

```cpp
VKNN_REGISTER_VK_OP(OpType::LeakyRelu, LeakyReluVulkanOp);
```

### 3c. The capability gate: `src/core/vk_gates.cpp`

The Vulkan backend's `supportsNode()` is not free-form — it delegates to two pure
functions in `src/core/vk_gates.cpp` (declared in `vk_gates.h`). This file lives in
**core**, so it compiles into every build, including a host with no Vulkan backend;
that is what lets `vknn_compile --support-report` and the host tests evaluate the
exact gate the device engine runs, with no chance of the two drifting.

- **`vkKernelDeclared(OpType)`** — a `switch` that mirrors the `VKNN_REGISTER_VK_OP`
  set. It returns `true` by default, and lists (as `return false`) the ops that have
  *no* GPU kernel: the const-folded / import-lowered ops (`Unknown`, `Identity`,
  `Constant`, `Shape`, `EyeLike`, `Dropout`, `InstanceNorm`), the quantized family
  lowered by the dequantize pass, and the ORT contrib family expanded by
  `lowerOrtContribOps` (`SimplifiedLayerNorm` … `MatMulNBits`). A new op with a Vulkan kernel is already covered by the `default:
  return true` — leave it alone unless the op has no kernel.

- **`vkNodeGate(const Graph&, const Node&, std::string* whyNot)`** — the shape/attribute
  gate. It returns `true` when the GPU kernel accepts this node's shapes, attributes,
  and operand constness, and on refusal fills `*whyNot` with a short stable
  `"<Op>: <reason>"` string (the reason that shows up in the fallback warning and the
  support report). A **pure pointwise** op like `LeakyRelu` needs nothing here — it
  is not listed, so it falls through to the gate's `return true`. Add a case only when
  the kernel cannot handle every shape/attribute the op admits (a non-4D input, a
  runtime operand the kernel can't bind, an unresolved output shape); model it on the
  `Pad` / `ConstantOfShape` / `TopK` cases.

### 3d. The capability descriptor: `op_descriptor.cpp`

The OpType-keyed capability facts an op declares — its GPU **layout class** and its
**pointwise-fusion roles** — live in one row of the table in `src/core/op_descriptor.cpp`
(`opDescriptor(OpType)`, declared in `include/vknn/op_descriptor.h`). The layout classifier
(`gpuFlatNode`) and the pointwise-fusion pass read that table instead of each keeping a parallel
`OpType` switch, so these facts are edited in one place, not three. The `OpDescriptor` fields:

- **`layout`** (`LayoutClass`) — how the op's kernel reads its tensors. VKNN has two GPU layouts:
  the CNN-default `NC4HW4` (channels packed in `vec4` blocks) and a **flat row-major** path for
  generic N-D ops; the layout pass splices `ConvertLayout` nodes at the boundary. `LayoutClass::Nc4`
  (the default) is the pointwise / CPU-only case — a `LeakyRelu` that processes the packed buffer
  element-by-element stays here and needs no entry. Use `LayoutClass::Flat` for an op whose kernel
  reasons about logical N-D shape at *every* node (a gather/broadcast/reduce/matmul/compare-shaped
  op). Use `LayoutClass::ShapeDependent` only when the layout is a per-node function of shapes or
  attributes (e.g. `Concat` that isn't 4D channel-axis 4-aligned goes flat, otherwise NC4HW4) —
  then add the matching per-node arm to the `switch` in `gpuFlatNode`
  (`src/import/insert_layout_converts.cpp`), and keep it in sync with the `vkNodeGate` case so the
  two agree on which nodes are GPU-eligible. A pure pointwise op needs `Nc4` (the default) and no
  `gpuFlatNode` arm.
- **`pwMember`** — set `true` for a per-element op that can join a fused-pointwise unit (a new
  member type; the fusion pass still applies its own float-dtype/shape/bound checks per node).
- **`pwEpilogue`** — set `true` for an op whose GPU kernel family carries an `_epi` store variant
  that can host a fused pointwise unit (Conv/Gemm/MatMul/pools/Softmax/... do; a bare pointwise op
  does not).

`LeakyRelu` is a pure pointwise unary, so it takes the all-default row (`Nc4`, not a pw member, no
epilogue) — no descriptor edit is needed. `op_descriptor.cpp`'s comment header enumerates which
OpTypes deviate from the default. The `OpDescriptor.LayoutClassAgreesWithGpuFlatNode` test asserts
the descriptor and `gpuFlatNode` never disagree.

The Vulkan backend's `supports()` then returns `true` for `LeakyRelu`
(because `VkOpRegistry::instance().has(LeakyRelu)` is true and `vkKernelDeclared`
agrees), `supportsNode()` passes it through `vkNodeGate`, and the session places
LeakyRelu nodes in Vulkan segments.

### 3e. Hosting a pointwise-chain epilogue (producer ops only)

`fusePointwiseChains` folds a chain of pointwise ops (activation → bias → scale …)
into the store of the *producer* that feeds it, so the chain never becomes a
standalone `FusedPointwise` node. `LeakyRelu` is itself pointwise and is *consumed*
into a producer, not a host — skip this section for a pointwise op. A **producer**
op (a Conv/MatMul/pool/reduce/gather-shaped kernel that others read from) can opt in
to hosting the epilogue at its store. Three things wire it up, and a build check
keeps them in sync:

1. **Shader.** Under `#ifdef PW_EPI`, `#include "pw_epilogue.glsl"` and apply the
   unit to each stored value (`v = pw_apply(v, idx)` for the flat world,
   `pw_apply_nc4` for NC4HW4) before `STORE`. Including that header is what marks the
   kernel epilogue-capable: the build sniffs the `#include` and generates the
   `<stem>_epi` (strict per-step rounding) and `<stem>_epi_rx` (`-DPW_RELAX`,
   fp32-chained; see [ADR-0011](adr/0011-fp32-chained-fusion.md)) SPIR-V variants.
   There is no hand-maintained stem list — adding an epilogue-capable kernel is just
   writing the shader.
2. **Op.** Wire `PwEpi` into the kernel (`src/backend/vulkan/ops/pw_plan.h`):
   `epi.prepare(node, env, flat, out)`, request the variant with
   `shader((std::string("<stem>") + epi.suffix()).c_str(), …)` sized
   `nbuf + epi.extraBufs()`, and `epi.append(bufs, …)` after the kernel's own
   buffers. `epi.suffix()` resolves to `""` / `_epi` / `_epi_rx`.
3. **Capability.** Mark the `OpType` epilogue-capable so `fusePointwiseChains`
   attaches units to this producer — the flag `pwEpilogueCapable()` reads (in
   `src/import/fuse_pointwise_chains.cpp`, or the op's descriptor row where the
   OpType-keyed capability facts are tabled).

`tools/check_epi_sync.py` (run at configure time and in `scripts/ci_host.sh`)
FATAL-ERRORs if these disagree — a stem requested by an op with no `_epi` shader, a
`pw_epilogue.glsl` shader no op requests, or a `pwEpilogueCapable` OpType whose kernel
never requests an epilogue variant. That turns what used to be a device-load-time
`shader not found` throw into a build diagnostic. `tools/check_shader_contracts.py`
separately enforces the fp16 store contracts the epilogue relies on (`store16.glsl`
inclusion, `VKNN_NO_RTE` ordering).

---

## Shape inference (ops that change shape)

`LeakyRelu` is pointwise, so its output shape equals its input shape and the
default rule covers it. An op whose output shape differs from its input — `Conv`,
`ConvTranspose`, `Reshape`, `Slice`, `Gather`, a broadcasting binary — needs a
rule in `inferShapes()` (`src/import/infer_shapes.cpp`). The Vulkan path sizes its
buffers at plan time from these shapes. A missing rule leaves the output shape
empty; an empty shape on a produced tensor means *unresolved* — it is never treated
as a rank-0 scalar and never fabricated to `{1,1,1,1}` — and it propagates until a
downstream op cannot plan. A *wrong* rule is worse: a `Shape` node const-folds the
lie into the model's shape arithmetic.

A shape rule must reproduce ONNX's output size for **every** shape-affecting
attribute, not just the common ones. `Conv`/`ConvTranspose`/pooling read
`auto_pad` (`SAME_UPPER` / `SAME_LOWER` / `VALID`) and `ConvTranspose` also reads
`output_shape` — when present these override the explicit `pads`, and SAME pads
are clamped to `>= 0`. The geometry an op shares between its shape rule and its
kernels lives in one helper so the two cannot drift; `ConvTranspose` uses
`convTransposeGeom()` (`src/core/conv_geom.h`) from `inferShapes` and from both
the CPU and Vulkan kernels.

Cross-check the rule against the reference: run `onnx.shape_inference` on a model
with concrete input shapes and diff every live tensor's shape against VKNN's, and
confirm the CPU op output matches `onnxruntime` (`vknn_run_io --backend cpu`)
across the attribute matrix (strides, kernels, `auto_pad`, `output_shape`,
`output_padding`, `dilations`, `group`). A single-config op test passes even when
an attribute variant is unhandled.

---

## 4. Self-registration and whole-archive linking

There is no central table of operators to edit. Every `VKNN_REGISTER_CPU_OP`,
`VKNN_REGISTER_VK_OP`, and `VKNN_REGISTER_BACKEND` declares a file-scope static whose
constructor inserts the factory into the relevant registry
(`CpuOpRegistry` / `VkOpRegistry` / `BackendRegistry`) before `main()` runs.

A static-library object file that nothing references gets dropped by the
linker, taking its self-registration with it. VKNN avoids this by linking the
static lib **whole-archive** everywhere it is consumed
(`CMakeLists.txt`):

```cmake
target_link_libraries(vknn_${_name} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,vknn>")
# tests/test_main.cpp provides main(), so gtest_main is not linked.
target_link_libraries(vknn_tests    PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,vknn>" gtest)
```

This pulls in every object file (and therefore every registrar) whether or not
the symbol is directly referenced. As long as the new op lives in a
source file already globbed into the `vknn` target — the `file(GLOB_RECURSE ...)`
patterns cover everything under `src/backend/cpu/` and `src/backend/vulkan/`,
including the `ops/` subdirectories — registration requires no further
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

- **Vulkan** (`src/backend/vulkan/vk_backend.cpp`):

  ```cpp
  bool supports(OpType t, DType dt) const override {
    if (!available()) return false;
    // Debug/fallback hook: Config::disableVkOps="Add,Conv" forces those ops to fall back.
    // Entries match whole op-type names ("Conv" leaves ConvTranspose on the GPU).
    if (!disabledOps_.empty() && Config::listContains(disabledOps_, opTypeName(t)))
      return false;
    return VkOpRegistry::instance().has(t);
  }
  ```

- **CPU** (`src/backend/cpu/cpu_backend.cpp`):

  ```cpp
  bool supports(OpType t, DType dt) const override {
    auto& r = CpuOpRegistry::instance();
    if (!r.has(t)) return false;
    return dt == DType::Float32 || dt == DType::Int64 || dt == DType::Int32;
  }
  ```

Registering the op (step 2 / step 3) is what makes
`supports()` return true — there is no separate capability list to maintain. The
dtype gate is central, not per-op: `CpuBackend::supports()` accepts fp32, int64, and
int32 for any registered op (`src/backend/cpu/cpu_backend.cpp`); `CpuOp` has no
per-op dtype hook.

### Backend selection and fallback

`Session` (`src/core/session.cpp`) builds a priority-ordered backend list — the
configured `cfg.backend` first, then the `fallback` list, with CPU appended last
when `allowCpuFallback` is set. For each node it picks the highest-priority
backend whose `supports()` returns true:

```cpp
for (size_t bi = 0; bi < backends_.size(); ++bi) {
  if (backends_[bi]->supportsNode(graph_, nd, dt)) { chosen = (int)bi; break; }
}
if (chosen < 0) throw Error(Status::Unsupported, "no backend supports op ...");
```

`supportsNode()` defaults to `supports()`; a backend overrides it to gate specific
ops on shapes or attributes. The Vulkan backend's override is `vkNodeGate`
(`src/core/vk_gates.cpp`, §3c) behind the availability + registry pre-checks, so a
node is refused (and its reason reported) whenever the GPU kernel can't handle its
shape/attributes — an unresolved-shape node, a runtime `k` on `TopK`, an
int64→narrow-integer `Cast`, and so on.

If the chosen backend is not the configured primary (because the primary's
`supports()` said no), the session emits a throttled fallback warning. Contiguous
nodes assigned to the same backend are then merged into one segment; tensor
residency is reconciled at segment boundaries via `toHost()` / `toDevice()`, so a
CPU fallback segment in the middle of a Vulkan graph triggers an
unpack/repack around it. A CPU segment created because the primary backend cannot
run its ops is tagged `Segment::isFallback = true` (drives the warning and
the profiler tag).

Each step is independently shippable:

- **Only the CPU kernel registered** → Vulkan's `supports()` returns false for
  the op, the node falls back to a CPU segment, output stays correct (just with a
  host round-trip at the segment boundary).
- **Both kernels registered** → the op runs in-place in the Vulkan segment with no
  extra sync.

The fallback path can be exercised without touching code by forcing the op back
to CPU at runtime (`Config::disableVkOps`):

```cpp
cfg.disableVkOps = "LeakyRelu";
```

This is the same mechanism that validates the NEON fallback path
(`cfg.disableVkOps = "Add,GlobalAveragePool"` re-partitions the graph into
Vulkan/CPU segments while keeping the output bit-comparable).

---

## Checklist

- [ ] `include/vknn/op_type.h`: append `OpType::LeakyRelu` at the END of the enum
      (append-only — `.vxm` files store the raw integer).
- [ ] `src/core/op.cpp`: add to `opTypeName()` and `opTypeFromOnnx()`.
- [ ] `src/import/infer_shapes.cpp`: add an `inferShapes()` rule if the op changes shape
      (cross-check vs `onnx.shape_inference` over the attribute matrix).
- [ ] `src/backend/cpu/ops/leakyrelu.cpp`: `LeakyReluCpuOp` + `VKNN_REGISTER_CPU_OP` (one op per file).
- [ ] `shaders/leakyrelu.comp` (`precision.glsl` + `STORE` buffers; the build emits the `_fp16` variant).
- [ ] `src/backend/vulkan/ops/leakyrelu.cpp`: `LeakyReluVulkanOp` + `VKNN_REGISTER_VK_OP` (one op per file).
- [ ] `src/core/vk_gates.cpp`: a `vkNodeGate` case only if the kernel can't take every
      shape/attribute the op admits (pure pointwise needs none); leave `vkKernelDeclared`
      alone unless the op has no GPU kernel.
- [ ] `src/core/op_descriptor.cpp`: the op's capability row — layout class (`Flat` /
      `ShapeDependent` / default `Nc4`) and fusion roles (`pwMember` / `pwEpilogue`). This is the
      single place those OpType-keyed facts live; `gpuFlatNode` and the pointwise-fusion pass read
      it. A pure pointwise op keeps the all-default row (no edit). Only a `ShapeDependent` layout
      also needs a per-node arm in `gpuFlatNode` (`src/import/insert_layout_converts.cpp`), kept in
      sync with its `vkNodeGate` case.
- [ ] For a **producer** op that should host a fused pointwise-chain epilogue (§3e):
      `#include "pw_epilogue.glsl"` under `#ifdef PW_EPI` in the shader (auto-derives the
      `_epi` variants), `PwEpi` wiring in the op, and set the descriptor row's `pwEpilogue`.
      `tools/check_epi_sync.py` enforces the three agree — no CMake stem list to edit.
- [ ] Add a gtest under `tests/` and run it with `./build.sh --test` (builds + runs the host unit
      tests only); diff Vulkan output against the CPU reference (and against `scripts/get_golden.py`
      for an external check). Confirm the node lands on the GPU with
      `vknn_compile <model>.onnx out.vxm --support-report r.json`. Before opening a PR,
      `./build.sh --leakcheck` runs the suite under memory-leak detection (ASan+LeakSanitizer on
      Linux; the `leaks` tool on macOS).
