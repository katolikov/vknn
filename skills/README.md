# skills/ — focused how-to guides

Short, task-oriented guides for the things you actually do in this repo. Each one is a checklist
with the exact commands and code patterns; the deeper reference docs in [`../docs/`](../docs/) explain
the *why*. Start from [`../AGENTS.md`](../AGENTS.md) for the overall map.

| Guide | When you need it |
|---|---|
| [add-an-operator.md](add-an-operator.md) | Add a new ONNX op (CPU oracle + optional Vulkan kernel). |
| [add-a-backend.md](add-a-backend.md) | Add a new execution backend (new hardware/runtime). |
| [compile-and-run-a-model.md](compile-and-run-a-model.md) | Compile an ONNX to `.vxm` and run it on the device with the right `Config`. |
| [run-yonosplat.md](run-yonosplat.md) | Run the full YoNoSplat 3D Gaussian Splatting pipeline end to end. |

Companion reference docs: [ARCHITECTURE](../docs/ARCHITECTURE.md) ·
[ADDING_AN_OPERATOR](../docs/ADDING_AN_OPERATOR.md) · [ADDING_A_BACKEND](../docs/ADDING_A_BACKEND.md) ·
[CONFIG](../docs/CONFIG.md) · [OP_COVERAGE](../docs/OP_COVERAGE.md) · [BENCHMARK](../docs/BENCHMARK.md) ·
[LIMITATIONS](../docs/LIMITATIONS.md).
