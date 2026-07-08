# app-demo — VKNN Qwen chat (Android, on-device GPU)

A small Android chat app that downloads the Qwen2.5-Coder-0.5B VKNN `.vxm` from HuggingFace, runs it
on the GPU (Vulkan) via VKNN, and streams a chat reply with live latency metrics. Samsung One UI dark
theme. Kotlin + Jetpack Compose + a JNI bridge over the VKNN runtime.

## Model source
`https://huggingface.co/katolikov/qwen-vknn` (uploaded this session):
`qwen2.5-coder-0.5b-decode-c256.vxm` (1.26 GB, fp16, C=256, 932/932 vulkan) + tokenizer files.
The tokenizer (`vocab.json` + `merges.txt`) is bundled into the APK assets; only the `.vxm` downloads.

## Units (each one file / clear boundary)
1. `app-demo/native/vknn_jni.cpp` → `libvknnchat.so` (JNI). Lifts the `examples/chat.cpp` decode loop
   into a callable class `Decoder`:
   - `nativeInit(vxmPath, cacheFile, precision) -> long`  (Runtime::load; read L/kvHeads/C/headDim/vocab;
     alloc persistent IOTensors in model input order). 0 on failure.
   - `nativeInfo(ptr) -> int[]{L,kvHeads,C,headDim,vocab}`
   - `nativeReset(ptr)` — p=0, zero the KV past buffers.
   - `nativeStep(ptr, tok) -> int` — set id/pos/mask, run() on GPU, copy present slot C → past slot p
     (per head), store logits, p++. Returns 0 ok / <0 error.
   - `nativeSample(ptr, temp, topK, topP, seedLo, seedHi) -> int` — greedy at temp<=0 else temp+top-k+
     top-p over the stored logits row (reuses chat.cpp's sampler).
   - `nativeFree(ptr)`.
   Built with the repo's static `vknn` lib via `add_subdirectory`, linked WHOLE_ARCHIVE (op registries
   self-register at static init), Vulkan/android/log. Shaders are embedded in `vknn` → self-contained.
2. `app-demo/native/CMakeLists.txt` + `build_native.sh` — configure with homebrew cmake (>=3.24 for the
   WHOLE_ARCHIVE genex) + NDK arm64-v8a toolchain; output copied to `app/src/main/jniLibs/arm64-v8a/`.
3. `Tokenizer.kt` — pure-Kotlin byte-level BPE from bundled `vocab.json`+`merges.txt`: bytes→unicode
   map, GPT-2/Qwen pre-tokenizer regex, ranked BPE merges, special tokens, Qwen chat template
   (`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`). EOS `<|im_end|>`=151645.
   Validated against the HF tokenizer on sample prompts (host cross-check) before trusting chat.
4. `VknnEngine.kt` — Kotlin `external fun` bindings + a thin lifecycle wrapper (load/reset/step/sample/free).
5. `ModelDownloader.kt` — stream the `.vxm` from HF resolve URL → `filesDir`, progress %, resumable-ish
   (skip if present + size matches).
6. `ChatViewModel.kt` — orchestrates encode→prefill(step per prompt token)→decode(sample→step) on
   Dispatchers.Default; streams tokens to the thread; captures metrics.
7. Compose UI (`MainActivity.kt`, `ui/…`) — download screen (button + progress), chat thread (bubbles),
   input row, top info+metrics bar, Samsung One UI dark theme.

## Metrics (info bar, above the thread)
TTFT (send→first token, incl. prefill), decode speed (tok/s), prefill time (ms), token count, and the
active temperature (a small chip; tap opens a temperature slider 0.0–1.2, 0 = greedy).

## Data flow (one send)
encode(prompt via chat template) → for each prompt token `nativeStep` (prefill; last token's logits
remain) → `nativeSample` = first token (stop TTFT clock) → loop: emit, `nativeStep(prev)`, `nativeSample`
until `<|im_end|>` or maxNewTokens or context (p==C) → update metrics.

## Theme (Samsung One UI dark, fixed)
bg #000000, surface #141416 / elevated #1B1B1E, primary One UI blue #3E9BFF, text #EDEDED / secondary
#8A8A8E; user bubble = blue, assistant bubble = surface; 20dp bubbles, airy spacing.

## Build + smoke test
`build_native.sh` → `libvknnchat.so`; `gradle :app:assembleDebug` → `app-debug.apk`; `adb install`;
`adb push` the local `.vxm` into the app files dir to skip the 1.26 GB download during the test; launch,
send "Write a python function to reverse a linked list.", confirm a coherent GPU-generated reply +
non-zero metrics on S25. Compare a greedy reply to the known-good stream.

## Scope / YAGNI
One model, C=256, arm64-v8a only, no history persistence, no markdown rendering (monospace text). Greedy
+ temperature sampling only. These are easy follow-ons.

## Risks
- Pure-Kotlin BPE correctness — cross-check ids vs HF before trusting chat (the one place bad output hides).
- 1.26 GB model on device: needs ~1.3 GB; fine on S25/S26 (prior session ran it).
- Gradle/NDK version friction — sidestepped by building the `.so` out-of-band with homebrew cmake.
