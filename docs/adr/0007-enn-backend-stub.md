# ADR-0007: ENN/NPU backend shipped as a runtime-probing stub

## Status
Accepted (2026-06-24)

## Context
The ENN runtime libraries are present on-device (`libenn_public_api_cpp.so`, `libenn_engine.so`,
`libenn_user_driver_gpu.so`, …) and `.nnc` model samples confirm NNC is the consumed format.
However: (1) no public ENN C++ headers are available to us, and (2) there is **no on-device NNC
compiler** — NNC is produced by an offline Samsung SDK tool we do not have. Building a real ENN
inference path would require both, and the task explicitly says not to block Vulkan work on ENN.

## Decision
Ship `EnnBackend` as a **complete, documented stub** that proves the plug-in path:
- It subclasses the same `Backend` base as Vulkan/CPU and registers in the backend registry.
- It is selectable via `config.backend = ENN` with no changes to core dispatch.
- At init it `dlopen`s the ENN libs and reports which are present (proving on-device probing).
- `supports()` returns false for all ops and `compile()/run()` returns a clear
  `kUnsupported: "ENN requires an NNC model; no on-device NNC compiler / public headers"`,
  triggering the configured fallback (Vulkan/CPU).
- The required offline flow (source ONNX → Samsung NNC compiler → `.nnc` → ENN runtime) is
  documented in `docs/ADDING_A_BACKEND.md` and `LIMITATIONS.md`.

## Consequences
- The pluggable-backend architecture is demonstrated end-to-end and the gap is honestly recorded.
- If headers + an `.nnc` model become available, only `EnnBackend`'s body changes.
