# How to add an operator

Goal: import a new ONNX op, run it correctly on the CPU (the reference), and optionally accelerate it
on the GPU. The rule that matters: **one operator per file**. Full writeup:
[../docs/ADDING_AN_OPERATOR.md](../docs/ADDING_AN_OPERATOR.md).

## The five touch points

1. **`include/vknn/op.h`** — add an `OpType::kFoo` enumerator.
2. **`src/core/op.cpp`** — map the ONNX op name to `OpType::kFoo` in `opTypeFromOnnx` (and back, if you
   serialize). Read list/scalar attributes via the helpers there.
3. **`src/import/passes.cpp`** — add a shape rule for `kFoo` in `inferShapes` so the planner can size
   buffers. (`readI64Param` reads a param from an attribute *or* an initializer.) The Vulkan path
   needs concrete shapes at plan time.
4. **`src/backends/cpu/ops/foo.cpp`** — the CPU oracle (always required; it is the correctness ground
   truth and the fallback).
5. **`src/backends/vulkan/ops/foo.cpp`** + **`shaders/foo.comp`** — optional GPU kernel, gated by
   `supportsNode`.

## CPU op pattern

`src/backends/cpu/ops/foo.cpp` — one struct, one registration:

```cpp
#include "backends/cpu/cpu_backend.h"

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
VKNN_REGISTER_CPU_OP(OpType::kFoo, FooCpu);
}  // namespace vknn
```

## Vulkan op pattern

`src/backends/vulkan/ops/foo.cpp` — split `prepare()` (one-time: build the pipeline, read shapes) from
`record()` (hot path: bind buffers, dispatch). `shader("foo", env.useFp16)` resolves to `foo.spv` /
`foo_fp16.spv`.

```cpp
#include "vk_op_common.h"

namespace vknn {
namespace {
struct FooOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  uint32_t count = 0;
  void prepare(const Node& node, VkOpEnv& env) override {
    count = (uint32_t)packedElems(env.graph->desc(node.outputs[0]).shape);
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("foo", env.useFp16), 2,
                                                 sizeof(uint32_t), std::vector<uint32_t>{},
                                                 env.cache->handle());
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* src = env.devBuf(node.inputs[0]);   // use operandBuf(...) if an input can be a constant
    vk::Buffer* dst = env.devBuf(node.outputs[0]);
    pipe->dispatch(cmd, {src->handle(), dst->handle()}, &count, sizeof(count), groups(count, 256));
  }
};
}  // namespace
VKNN_REGISTER_VK_OP(OpType::kFoo, FooOp);
}  // namespace vknn
```

Override `supportsNode` in the Vulkan backend to declare which shapes/dtypes the GPU kernel actually
accepts; anything you decline falls back to the CPU oracle at a segment boundary.

### Layout notes

- The default GPU layout is **NC4HW4** (channels packed in vec4 blocks). `packedElems` accounts for
  channel padding; a flat (row-major) tensor is `numElements`-sized — for the **flat path** (rank > 4,
  transformer shapes) dispatch over `numElements`, not `packedElems`. See `flat_ops.h` and the
  `shaders/flat_*.comp` kernels.
- Read weights with `initFloats(graph, id)` so a model loaded from an **fp16 `.vxm`** works unchanged.
- Use `operandBuf(env, tensor, hold)` instead of `env.devBuf` when an input may be a constant
  initializer (it uploads the constant flat instead of dereferencing a null device buffer).

## Validate

Build host (CPU oracle) and Android (GPU), then check cosine against an onnxruntime golden. For a small
synthetic op, build a tiny ONNX + golden with `scripts/yonosplat/op_test.py` and run `vknn_run_io
--backend vulkan` vs `--backend cpu`. A logic bug shows up in **fp32** where fp16 noise would mask it.

```sh
./build.sh && ./build-host/vknn_tests        # host: passes
./build.sh --android                          # GPU: shaders compile
```

If a new `.cpp` is not compiled, re-run `./build.sh` (CMake reconfigures + re-globs).
