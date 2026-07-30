package com.vknn.chat

// Kotlin bindings for libvknnchat.so (see app-demo/native/vknn_jni.cpp). Each call maps to a JNI
// entry point on the singleton; the returned `long` handle is an opaque pointer to the native
// Decoder / Vlm / Splat. `backend` is "vulkan" or "cpu" (the engine's backendFromStr spelling).
object NativeLib {
    init {
        System.loadLibrary("vknnchat")
    }

    /**
     * Version of the engine compiled into libvknnchat.so, as "major.minor.patch". Read from the
     * loaded library, not a Kotlin constant, so it reports the .so the APK actually ships.
     */
    external fun nativeVknnVersion(): String

    /**
     * Load a .vxm decoder; returns a native handle, or 0 on failure. [greedyArgMax] registers the
     * decode logits for the engine-side argmax (an 8-byte per-token readback instead of the vocab
     * row); the registration holds for the session's lifetime, so it is requested only when the
     * decode is greedy — temperature sampling then needs a reload.
     */
    external fun nativeInit(vxmPath: String, cacheFile: String, precision: String, backend: String, greedyArgMax: Boolean): Long

    /** {L, kv_heads, C, head_dim, vocab, prefillS, engineArgMax}. */
    external fun nativeInfo(ptr: Long): IntArray

    /** Reset to position 0 with a cleared KV cache and reseed the sampler. */
    external fun nativeReset(ptr: Long, seed: Int)

    /** Feed one token at the current position (runs the plan). 0 ok, <0 error. */
    external fun nativeStep(ptr: Long, tok: Int): Int

    /**
     * Feed a whole prompt from the current position — one batched forward per prefill window when
     * the model carries a prefill bucket, per-token steps otherwise. 0 ok, -1 error, -2 context full.
     */
    external fun nativePrefill(ptr: Long, promptIds: LongArray): Int

    /** Sample the next token from the last forward's logits (greedy at temp<=0). */
    external fun nativeSample(ptr: Long, temp: Float, topK: Int, topP: Float): Int

    external fun nativeFree(ptr: Long)

    // --- VLM: multi-graph vision-language .vxm (vision + embed + prefill/decode buckets) ---

    /** Load a vision-language .vxm; returns a native handle, or 0 on failure. */
    external fun nativeVlmInit(vxmPath: String, cacheFile: String, precision: String, backend: String): Long

    /** {L, kvHeads, C, headDim, vocab, prefillS, imageRows, H, imageSide}. */
    external fun nativeVlmInfo(ptr: Long): IntArray

    /** Reset to position 0 with a cleared KV cache and reseed the sampler. */
    external fun nativeVlmReset(ptr: Long, seed: Int)

    /** Run the vision bucket on fp32 CHW pixels [3*imageSide*imageSide]; returns [imageRows*H] rows or null. */
    external fun nativeVisionEncode(ptr: Long, pixels: FloatArray): FloatArray?

    /**
     * Prefill one prompt turn: ids padded to the prefill window with [padId], each image-token row
     * spliced with the next row of [imageEmbeds] (nullable for text-only turns). 0 ok, <0 error.
     */
    external fun nativeVlmPrefill(ptr: Long, promptIds: LongArray, imageEmbeds: FloatArray?, imageToken: Int, padId: Int): Int

    /** Feed one token at the current position (embed + decode buckets). 0 ok, <0 error. */
    external fun nativeVlmStep(ptr: Long, tok: Int): Int

    /** Sample the next token from the last prefill/step logits (greedy at temp<=0). */
    external fun nativeVlmSample(ptr: Long, temp: Float, topK: Int, topP: Float): Int

    external fun nativeVlmFree(ptr: Long)

    // --- Splat: YoNoSplat encoder + the shared Vulkan rasterizer (raster_core) ---

    /**
     * Load the encoder .vxm once; returns a native handle, or 0 on failure. renderSize is the
     * square rasterizer output side, independent of the encoder input side (<= 0 falls back to
     * the encoder side).
     */
    external fun nativeSplatLoad(vxmPath: String, cacheFile: String, precision: String, backend: String, renderSize: Int): Long

    /** {gaussians, views, height, width}; gaussians is 0 until an encode succeeds. */
    external fun nativeSplatInfo(ptr: Long): IntArray

    /**
     * Run the encoder on fp32 images [1,views,3,H,W] (RGB in [0,1], CHW) + normalized intrinsics
     * [1,views,3,3]; the Gaussians upload to the rasterizer and the predicted camera poses are
     * kept native-side. 0 ok, <0 error.
     */
    external fun nativeSplatEncode(ptr: Long, images: FloatArray, intrinsics: FloatArray): Int

    /** Per-view camera-to-world matrices [views*16], row-major. */
    external fun nativeSplatPoses(ptr: Long): FloatArray

    /** Median Gaussian depth in view-0 camera space; the orbit viewer's look-at distance. */
    external fun nativeSplatPivotDepth(ptr: Long): Float

    /** Render from a row-major camera-to-world [16]; packed ARGB [renderSize*renderSize], or null on failure. */
    external fun nativeSplatRender(ptr: Long, cameraToWorld: FloatArray): IntArray?

    external fun nativeSplatFree(ptr: Long)
}
