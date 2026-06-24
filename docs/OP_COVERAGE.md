# vxrt operator coverage vs MNN's Vulkan backend

Reference: MNN's Vulkan **buffer** (SSBO) backend op set, enumerated from
`source/backend/vulkan/buffer/execution/`. vxrt's device layout is NC4HW4; ops whose semantics move
the channel axis (Transpose, channel Slice/Split, general Reduce) are awkward to pack, so they run on
the CPU in canonical NCHW with automatic NC4HW4<->NCHW converts at the segment boundary (always
correct, GPU-accelerated neighbours unaffected). All ops below are verified against onnxruntime
(cosine ≥ 0.999 on synthetic + real models).

| Operator | vxrt | backend | notes |
|---|---|---|---|
| Conv / ConvDepthwise | ✅ | GPU+CPU | NC4HW4, split-K, residual+act fusion |
| Pooling (Max/Avg/Global) | ✅ | GPU+CPU | windowed + global |
| BinaryOp (Mul/Sub/Div/Max/Min/Pow/Add) | ✅ | GPU+CPU | same-shape + channel-broadcast (SE) |
| UnaryOp (Sigmoid/Tanh/HardSwish/HardSigmoid/Elu/Abs/Exp/Log/Sqrt/Floor/Ceil/…) | ✅ | GPU+CPU | type-coded family |
| ReLU/ReLU6/Clip | ✅ | GPU+CPU | also fused into conv |
| PRelu | ✅ | GPU+CPU | per-channel slope |
| Concat | ✅ | GPU+CPU | channel-axis NC4HW4 (4-aligned) on GPU, else CPU |
| Interp / Resize / Upsample | ✅ | GPU+CPU | nearest+bilinear, 4 coord modes, spatial |
| GridSample | ✅ | CPU | bilinear/nearest, zeros/border/reflection (grid can't be NC4HW4-packed) |
| Softmax | ✅ | CPU | |
| MatMul / Gemm | ✅ | GPU(Gemm)+CPU | |
| Transpose / Permute | ✅ | CPU | generic N-D |
| Slice | ✅ | CPU | starts/ends/axes/steps |
| Split | ✅ | CPU | |
| Pad | ✅ | CPU | constant/edge/reflect |
| Reduce (Sum/Max/Min/Prod/Mean) | ✅ | CPU | (Mean spatial → GlobalAvgPool on GPU) |
| Cast | ✅ | CPU | float<->int64 |
| BatchNorm / Scale | ✅ | folded | folded into conv |
| Reshape / Flatten / Gather / Unsqueeze / Shape / Constant | ✅ | GPU/CPU/const-fold | |
| **Not yet implemented** (specced, ready): Where/Select, ArgMax/ArgMin, LayerNorm, ConvTranspose/Deconvolution, ROIPooling, OneHot, Range, Loop/While | ⬜ | — | designs in the op-parity workflow output |

Adding any of the remaining ops is mechanical: enum in `include/vx/op.h`, ONNX name in
`src/core/op.cpp`, shape rule in `src/import/passes.cpp` inferShapes, a CPU oracle in
`src/backends/cpu/ops/`, and (when NC4HW4-clean) a Vulkan op + shader gated via
`Backend::supportsNode()`. See `docs/ADDING_AN_OPERATOR.md`.
