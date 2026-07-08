# Examples

Each example builds to a `vknn_<name>` binary (the target name is the file basename, independent of the
group folder). Build them with the host or Android build (`./build.sh` / `./build.sh --android`).

| Group | Example | Binary | What it shows |
|-------|---------|--------|---------------|
| `basics/` | `readme_quickstart.cpp` | `vknn_readme_quickstart` | Minimal load → run → read-back. |
| `basics/` | `probe.cpp` | `vknn_probe` | Report a model's inputs/outputs and per-node backend assignment. |
| `basics/` | `backend_switch.cpp` | `vknn_backend_switch` | Run the same model on the Vulkan and CPU backends. |
| `basics/` | `op_check.cpp` | `vknn_op_check` | Single-op CPU-vs-GPU numeric check. |
| `vision/` | `classify.cpp` | `vknn_classify` | Image classification with config-file + CLI flag layering. |
| `vision/` | `predict.cpp` / `predict_cache.cpp` | `vknn_predict` / `vknn_predict_cache` | Predict, and predict with a warm cache. |
| `vision/` | `image_bench.cpp` | `vknn_image_bench` | Image-model latency benchmark. |
| `llm/` | `chat.cpp` | `vknn_chat` | On-device autoregressive decode loop for a Qwen2 with-past decoder. |
| `llm/` | `chat_host.py` | — | Terminal front-end that tokenizes and drives `vknn_chat` over adb. |
| `splatting/` | `yonosplat.cpp` | `vknn_yonosplat` | YoNoSplat 3D Gaussian-Splatting encoder. |
| `io/` | `run_io.cpp` | `vknn_run_io` | Run any model over positional input files (the workhorse harness). |
| `io/` | `dmabuf_fd_io.cpp` | `vknn_dmabuf_fd_io` | Zero-copy caller-owned DMA-BUF I/O. |
| `io/` | `zerocopy_cache.cpp` / `zerocopy_simple.cpp` | `vknn_zerocopy_cache` / `vknn_zerocopy_simple` | Zero-copy + unified cache paths. |
| `bench/` | `profile.cpp` | `vknn_profile` | Per-op profiling summary. |
| `bench/` | `microbench.cpp` | `vknn_microbench` | Drives Vulkan kernels directly (Android/Vulkan only). |

For a full on-device chat app built on `llm/`, see [`../app-demo/`](../app-demo/).
