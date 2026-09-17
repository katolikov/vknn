# How to add an operator

Importing a new ONNX op requires a CPU reference implementation (the correctness oracle) and an
optional GPU kernel. The convention is **one operator per file**. Full writeup:
[../docs/adding-an-operator.md](../docs/adding-an-operator.md).

## The five touch points

1. **`include/vknn/op_type.h`** — add an `OpType::Foo` enumerator at the END of the enum (values
   serialize as raw integers into `.vxm`, so the enum is append-only; a mid-enum insert shifts every
   later op and corrupts existing compiled models).
2. **`src/core/op.cpp`** — map the ONNX op name to `OpType::Foo` in `opTypeFromOnnx` and add the
   `opTypeName` string (logs and the support tools parse it). The importer fills the node's attribute
   bag generically; ops read attributes with `node.attr.geti()` / `node.attr.getints()`
   (`include/vknn/attributes.h`).
3. **`src/import/infer_shapes.cpp`** — add a shape rule for `OpType::Foo` in `inferShapes` so the planner can size
   buffers. (`readI64Param` reads a param from an attribute *or* an initializer.) The Vulkan path
   requires concrete shapes at plan time. The rule must reproduce ONNX's output size for **every**
   shape-affecting attribute, not just the common ones — a conv-family op reads `auto_pad`,
   `output_shape`, `output_padding`, `dilations`, and `group`, not only `pads`. Keep geometry that the
   shape rule and the kernels both need in one helper so they cannot drift (ConvTranspose uses
   `src/core/conv_geom.h` from `inferShapes` and from both the CPU and Vulkan ops).
4. **`src/backend/cpu/ops/foo.cpp`** — the CPU oracle (always required; it is the correctness ground
   truth and the fallback).
5. **`src/backend/vulkan/ops/foo.cpp`** + **`shaders/foo.comp`** — optional GPU kernel, gated by
   `supportsNode`.

A GPU kernel also touches the shared tables: a `vkNodeGate` refusal in `src/core/vk_gates.cpp` for every
shape or attribute the kernel cannot take (refuse by name there — a `prepare()` throw fails the whole
session load instead of falling back); an OpDescriptor row in `src/core/op_descriptor.cpp` for a non-NC4HW4
layout, with `kMaxOp` raised to the new last enumerator (a row past it throws when the table is built)
and the `OpDescriptor.LayoutClassAgreesWithGpuFlatNode` loop bound in `tests/test_support_report.cpp`
moved with it; and, for an op whose GPU result holds integers, a seed in `pinIntegerResultsFp32`
(`src/import/mark_fp32.cpp`) so the integer region stores fp32 (exact within ±2^24; fp16 only within
±2^11). An op the importer lowers away has no kernel: list it in `CPU_KERNEL_EXEMPT`
(`tools/check_support_consistency.py`) and in `vkKernelDeclared`'s false list (`Mean`, `InstanceNorm`).

## CPU op pattern

`src/backend/cpu/ops/foo.cpp` — one struct, one registration:

```cpp
#include "backend/cpu/cpu_backend.h"

namespace vknn {
namespace {
struct FooCpu : CpuOp {
  void run(const Node& node, ExecContext& ctx) override {
    const RtTensor& X = ctx.t(node.inputs[0]);
    RtTensor& Y = ctx.t(node.outputs[0]);
    float* y = cpu::allocOut(Y, X.shape);   // sizes + allocates the output
    const float* x = X.host.f32();
    for (int64_t i = 0, n = X.elems(); i < n; ++i) y[i] = /* ... */ x[i];
  }
};
}  // namespace
VKNN_REGISTER_CPU_OP(OpType::Foo, FooCpu);
}  // namespace vknn
```

## Vulkan op pattern

`src/backend/vulkan/ops/foo.cpp` — `prepare()` (one-time: build the pipeline, read shapes) is split from
`record()` (hot path: bind buffers, dispatch). `shader("foo", env.useFp16)` resolves to `foo.spv` /
`foo_fp16.spv`.

A new `shaders/foo.comp` is picked up by the CMake glob and embedded into the binary — nothing to
register. Include `precision.glsl` and declare buffers with the `STORE` element type (read with
`float(buf[i])`, write with `TO_STORE(x)`): that include is what makes the build emit the
`foo_fp16.spv` variant (`-DUSE_FP16=1`), and the fp16-store contract (`store16.glsl` explicit
round-to-nearest-even stores) is enforced at configure time by `tools/check_shader_contracts.py`.

```cpp
#include "vk_op_common.h"

namespace vknn {
namespace {
struct FooOp : VulkanOp {
  std::shared_ptr<vk::ComputePipeline> pipe;
  uint32_t count = 0;
  void prepare(const Node& node, VkOpEnv& env) override {
    count = (uint32_t)packedElems(env.graph->desc(node.outputs[0]).shape);
    pipe = env.pipeline(shader("foo", env.useFp16), 2, sizeof(uint32_t), std::vector<uint32_t>{});
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* src = env.devBuf(node.inputs[0]);   // use operandBuf(...) if an input can be a constant
    vk::Buffer* dst = env.devBuf(node.outputs[0]);
    pipe->dispatch(cmd, {src->handle(), dst->handle()}, &count, sizeof(count), groups(count, 256));
  }
};
}  // namespace
VKNN_REGISTER_VK_OP(OpType::Foo, FooOp);
}  // namespace vknn
```

`supportsNode` in the Vulkan backend declares which shapes/dtypes the GPU kernel accepts; a declined
node falls back to the CPU oracle at a segment boundary.

### Layout notes

- The default GPU layout is **NC4HW4** (channels packed in vec4 blocks). `packedElems` accounts for
  channel padding; a flat (row-major) tensor is `numElements`-sized. The **flat path** (rank > 4,
  transformer shapes) dispatches over `numElements`, not `packedElems`. See `flat_ops.h` and the
  `shaders/flat_*.comp` kernels.
- `initFloats(graph, id)` reads weights so a model loaded from an **fp16 `.vxm`** works unchanged.
- `operandBuf(env, tensor, hold)` replaces `env.devBuf` when an input may be a constant initializer; it
  uploads the constant flat instead of dereferencing a null device buffer.

## Validate

Build host (CPU oracle) and Android (GPU), then check cosine against an onnxruntime golden. For a small
synthetic op, build a tiny ONNX + golden with `scripts/yonosplat/op_test.py` and run `vknn_run_io
--backend vulkan` vs `--backend cpu`. A logic bug surfaces in **fp32** where fp16 noise would mask it.
Cross-check the shape rule against `onnx.shape_inference` and the values against onnxruntime across the
op's **attribute matrix** (strides, kernels, `auto_pad`, `output_shape`, `output_padding`, dilation,
group), not a single config — a one-config test passes even when a variant is unhandled. Add a
self-contained CPU case under `tests/` (build a graph, run, assert against a reference; every
`tests/*.cpp` is globbed into `vknn_tests`).

GLSL never runs in the host tests. When the shader's arithmetic is not the CPU op's line for line, keep a
C++ transcription of the shader functions in the test file, sweep it against the CPU oracle, and add a
source test that reads `shaders/foo.comp` through `__FILE__` and requires the transcription (and the
push constants, bindings, spec-constant ids, named constants) to match — skipping only when the sources
are unreadable. Precedents: `ModOps.ShaderTranscriptionMatchesCompSource`,
`ArgExtremeShader.SourceMatchesTranscriptionAndInterface`. A pin or layout test calls
`planFlatLayoutAndStorage` (the session's load order) rather than one pass alone.

`scripts/yonosplat/op_validate.py` automates the per-op ORT-vs-VKNN-CPU compare.
`tools/check_support_consistency.py` (run by `scripts/ci_host.sh`) fails when an OpType mapped in
`opTypeFromOnnx` has no `VKNN_REGISTER_CPU_OP` registration.

```sh
./build.sh && ./build-host/vknn_tests        # host: passes
./build.sh --android                          # GPU: shaders compile
```

A new `.cpp` that is not compiled requires re-running `./build.sh` (CMake reconfigures + re-globs).
