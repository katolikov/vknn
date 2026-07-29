# Examples

Each example builds to a `vknn_<name>` binary (the target name is the file basename, independent of the
group folder). Build them with the host or Android build (`./build.sh` / `./build.sh --android`).

| Group | Example | Binary | What it shows |
|-------|---------|--------|---------------|
| `basics/` | `readme_quickstart.cpp` | `vknn_readme_quickstart` | Minimal load → run → read-back. |
| `basics/` | `probe.cpp` | `vknn_probe` | Print the Vulkan device, driver, compute limits, feature flags, and memory heaps (no model, no arguments). |
| `basics/` | `backend_switch.cpp` | `vknn_backend_switch` | Run the same model on the Vulkan and CPU backends. |
| `basics/` | `op_check.cpp` | `vknn_op_check` | Single-op CPU-vs-GPU numeric check. |
| `vision/` | `classify.cpp` | `vknn_classify` | Image classification with config-file + CLI flag layering. |
| `vision/` | `predict.cpp` / `predict_cache.cpp` | `vknn_predict` / `vknn_predict_cache` | Predict, and predict with a warm cache. |
| `vision/` | `image_bench.cpp` | `vknn_image_bench` | Compare SSBO vs storage-image vs sampler kernels on a 1×1 conv, each verified against a CPU reference. |
| `llm/` | `chat.cpp` | `vknn_chat` | On-device autoregressive decode loop for a Qwen2 with-past decoder. |
| `llm/` | `chat_host.py` | — | Terminal front-end that tokenizes and drives `vknn_chat` over adb. |
| `llm/` | `vlm.cpp` | `vknn_vlm` | Vision-language chat (SmolVLM2) over a one-file multi-graph `.vxm`: image encode → on-device embedding splice → prefill → streamed decode. |
| `llm/` | `vlm_host.py` | — | Front-end for `vknn_vlm`: HF processor (tokenizer + chat template) and image preprocessing over adb. |
| `splatting/` | `yonosplat.cpp` | `vknn_yonosplat` | YoNoSplat 3D Gaussian-Splatting encoder + the Vulkan rasterizer (`raster_core`). |
| `io/` | `run_io.cpp` | `vknn_run_io` | Run any model over positional input files (the workhorse harness). |
| `io/` | `dmabuf_fd_io.cpp` | `vknn_dmabuf_fd_io` | Zero-copy caller-owned DMA-BUF I/O. Linux/Android only (dma-buf kernel interface); not built on Windows. |
| `io/` | `zerocopy_cache.cpp` / `zerocopy_simple.cpp` | `vknn_zerocopy_cache` / `vknn_zerocopy_simple` | Zero-copy + unified cache paths. Linux/Android only; not built on Windows. |
| `bench/` | `profile.cpp` | `vknn_profile` | Per-op profiling summary. |
| `bench/` | `microbench.cpp` | `vknn_microbench` | Drives Vulkan kernels directly (Android/Vulkan only). |

For a full on-device app built on `llm/` and `splatting/` — Chat, VLM camera coach, and 3D Splat
capture over a model library — see [`../app-demo/`](../app-demo/).
