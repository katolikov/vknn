# skills/ — focused how-to guides

Task-oriented guides for common operations in this repo. Each guide is a checklist with the exact
commands and code patterns; the reference docs in [`../docs/`](../docs/) cover the *why*.
[`../AGENTS.md`](../AGENTS.md) holds the overall map.

| Guide | When you need it |
|---|---|
| [add-an-operator.md](add-an-operator.md) | Add a new ONNX op (CPU oracle + optional Vulkan kernel). |
| [add-a-backend.md](add-a-backend.md) | Add a new execution backend (new hardware/runtime). |
| [compile-and-run-a-model.md](compile-and-run-a-model.md) | Compile an ONNX to `.vxm` and run it on the device with the right `Config`. |
| [benchmark-a-model.md](benchmark-a-model.md) | Convert/run/validate a model on the device from one JSON config (npy/raw I/O, golden metrics, profiling). |
| [run-yonosplat.md](run-yonosplat.md) | Run the full YoNoSplat 3D Gaussian Splatting pipeline end to end. |

Companion reference docs: [ARCHITECTURE](../docs/architecture.md) ·
[ADDING_AN_OPERATOR](../docs/adding-an-operator.md) · [ADDING_A_BACKEND](../docs/adding-a-backend.md) ·
[CONFIG](../docs/config.md) · [OP_COVERAGE](../docs/op-coverage.md) · [BENCHMARK](../docs/benchmark.md) ·
[LIMITATIONS](../docs/limitations.md).

Two further task guides live in [`../docs/`](../docs/): [running-an-llm.md](../docs/running-an-llm.md) (run an LLM end to end on the GPU) and [running-a-vlm.md](../docs/running-a-vlm.md) (run a VLM end to end on the GPU).
