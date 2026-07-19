# ADR-0019: Real float64 as a CPU-oracle precision path

## Status
Accepted (2026-07-19).

## Context
An exported encoder carries a numerically-sensitive sub-path — a camera-head SVD whose determinant
and sign flip on rounding error — that the exporter marks `double`. Until now the importer narrowed
every ONNX `DOUBLE` tensor to fp32 silently, so that precision was lost before the graph ran, and the
determinant's sign could come out wrong (fp32 catastrophic cancellation). The engine needs to honor a
declared fp64 tensor as real double precision, not a relabeled fp32.

The primary backend is the GPU, and mobile Vulkan drivers (the target devices) expose no usable
`shaderFloat64` — genuine fp64 arithmetic on the GPU is not available. So "support float64" cannot
mean "run fp64 on the GPU"; it has to be a precision path with a home the hardware actually provides.

## Decision
`DType::Float64` is a real element type with native 8-byte host storage, confined to a small set of
**fp64-capable ops** whose CPU kernel computes in genuine `double`:

- **Cast** — the fp32↔fp64 bridge (a Cast to DOUBLE widens to real fp64; a Cast off a fp64 input
  reads every bit before narrowing).
- **Det** — templated cofactor/LU on the element type: the `float` instantiation stays bitwise-equal
  to the GPU `det_flat` kernel, the `double` one gives a real fp64 determinant.
- **Unary** (incl. `Sign`), **Binary**/**Add** — templated elementwise arithmetic in `double`, so a
  sign reflects the true value and a difference does not cancel to fp32 zero.
- **Transpose** and the metadata reshapes — dtype-preserving byte movement, so fp64 flows through
  structure losslessly.

Three properties make this safe and honest:

1. **Lossless at the boundaries.** A DOUBLE initializer keeps native 8-byte storage and round-trips
   byte-identically through the `.vxm` (dtype is an append-only `u32` field; no format-magic bump); a
   DOUBLE graph input/output is bound and read back as real fp64. The fp32 compute path still reads a
   fp64 initializer through `initFloats` (decode to fp32); a genuine fp64 reader takes `initDoubles`.

2. **Explicit narrowing, never a silent misread.** The `legalizeFp64` pass runs after inference and
   fusion: for every consumer that is *not* fp64-capable it inserts a real `Cast(fp64→fp32)` on that
   edge, so a fp64 tensor never reaches an fp32-only kernel. The narrowing is a visible graph node.
   Pointwise fusion already excludes fp64 (`pwFloatDtype`), and `eliminateFloatCast` now preserves a
   real fp32↔fp64 Cast instead of dropping it as a same-precision no-op.

3. **A named precision refusal, not a coverage gap.** `vkNodeGate` refuses any node that reads or
   writes a Float64 tensor with a reason that names it a precision choice, routing it to the CPU
   double kernel. This only fires on the path the model declared fp64, so a model with no fp64 tensors
   plans and runs exactly as before and keeps `fallbacks: 0`. The engine probes
   `VkPhysicalDeviceFeatures.shaderFloat64` and logs it (`fp64=0` on the current devices) as the gate a
   future fp64 GPU kernel would key on; none ships today.

## Consequences
The camera-head SVD determinant and sign compute in real double precision — device-verified on two
GPUs: the fp64 head returns `det = 414` exactly and `sign = +1`, with every fp64 op on the CPU by the
named refusal and `shaderFloat64` probed as absent. Basic fp64 arithmetic (Cast/Det/Unary/Binary/Add/
Transpose + reshapes) is genuinely double; any other op on a fp64 tensor is narrowed by an explicit
Cast and computes fp32 — a documented boundary, extendable by adding the op to the fp64-capable set and
its CPU kernel's double path. fp32 / fp16 / int models are unchanged (byte-identical, `fallbacks: 0`).

## Alternatives considered
- **Keep narrowing DOUBLE to fp32 (status quo).** Rejected: it loses the precision the exporter asked
  for and can produce the wrong determinant sign — a correctness failure, not just accuracy.
- **fp64 on the GPU.** Not available: no usable `shaderFloat64` on the target drivers. The capability
  probe is in place so a future device with fp64 could carry double-typed shaders for the critical ops.
- **Make every op fp64-capable.** Rejected as premature: the SVD path needs a small island, and the
  legalization safety net makes the rest correct (explicitly narrowed) without touching 60+ kernels.
