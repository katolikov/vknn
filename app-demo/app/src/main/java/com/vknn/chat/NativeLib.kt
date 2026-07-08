package com.vknn.chat

// Kotlin bindings for libvknnchat.so (see app-demo/native/vknn_jni.cpp). Each call maps to a JNI
// entry point on the singleton; the returned `long` handle is an opaque pointer to the native Decoder.
object NativeLib {
    init {
        System.loadLibrary("vknnchat")
    }

    /** Load a .vxm decoder; returns a native handle, or 0 on failure. */
    external fun nativeInit(vxmPath: String, cacheFile: String, precision: String): Long

    /** {L, kv_heads, C, head_dim, vocab}. */
    external fun nativeInfo(ptr: Long): IntArray

    /** Reset to position 0 with a cleared KV cache and reseed the sampler. */
    external fun nativeReset(ptr: Long, seed: Int)

    /** Feed one token at the current position (runs the plan on the GPU). 0 ok, <0 error. */
    external fun nativeStep(ptr: Long, tok: Int): Int

    /** Sample the next token from the last step's logits (greedy at temp<=0). */
    external fun nativeSample(ptr: Long, temp: Float, topK: Int, topP: Float): Int

    external fun nativeFree(ptr: Long)
}
