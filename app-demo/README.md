# VKNN Chat — on-device GPU LLM demo (Android)

![VKNN Chat running on a Galaxy device](docs/screenshot.png)

A small Android app that downloads a Qwen2.5-Coder-0.5B model in VKNN's `.vxm` format from
[huggingface.co/katolikov/qwen-vknn](https://huggingface.co/katolikov/qwen-vknn), runs it **entirely on
the GPU** (Vulkan) through the VKNN engine, and lets you chat with it — with live latency metrics
(time-to-first-token, decode tokens/sec, prefill time) and a temperature control. Samsung One UI dark
theme. Kotlin + Jetpack Compose over a JNI bridge; no cloud, no server.

## Layout
```
app-demo/
  native/                 JNI decoder bridge -> libvknnchat.so
    vknn_jni.cpp          lifts examples/chat.cpp's decode loop into a callable Decoder handle
    CMakeLists.txt        links the static vknn engine (WHOLE_ARCHIVE) + Vulkan; shaders are embedded
    build_native.sh       cross-builds the .so for arm64-v8a into app/src/main/jniLibs/
  app/                    the Android app (Kotlin + Compose)
    src/main/java/com/vknn/chat/
      NativeLib.kt          JNI bindings
      Tokenizer.kt          pure-Kotlin byte-level BPE (Qwen2 vocab)
      ModelDownloader.kt    streams the .vxm from HuggingFace
      ChatViewModel.kt      encode -> prefill -> stream decode, with metrics
      MainActivity.kt, ui/  Compose UI + Samsung One UI dark theme
    src/main/assets/        vocab.json + merges.txt (the tokenizer ships in the APK)
    src/test/               JVM test: the Kotlin BPE matches HuggingFace token-for-token
```

## Build
Prerequisites: Android SDK + NDK (r27), a cmake >= 3.24 on `PATH` (for the native step), and a JDK 17+.

```sh
# 1. build the native engine bridge into the app's jniLibs
app-demo/native/build_native.sh                 # -> app/src/main/jniLibs/arm64-v8a/libvknnchat.so

# 2. build the APK
cd app-demo && ./gradlew :app:assembleDebug     # -> app/build/outputs/apk/debug/app-debug.apk

# 3. install
adb install -r app-demo/app/build/outputs/apk/debug/app-debug.apk
```

Run the tokenizer check on the JVM (no device needed): `./gradlew :app:testDebugUnitTest`.

## How it works
The model is a with-past Qwen2 decoder compiled at a fixed context length **C = 256** — one plan serves
prefill (fed token by token) and every decode step; the KV cache lives in the `past_key_values` boundary
buffers. The app tokenizes your text (byte-level BPE), feeds the tokens through the native `Decoder`
(each `step` runs the plan on the GPU and appends the new key/value into the cache), then samples the
next token (greedy, or temperature + top-k/top-p) and streams it back. Every tensor-compute op runs on
Vulkan; only argmax/sampling and tokenization are CPU. Metrics are wall-clock around the native calls.

The compiled `.vxm` reproduces the HuggingFace greedy stream token-for-token (see the repo's
`docs/running-an-llm.md`). arm64-v8a only; requires a Vulkan-capable device.
