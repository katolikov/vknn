package com.vknn.chat.vlm

// ARGB pixels -> fp32 CHW, normalized the way the SmolVLM2 processor does with image splitting off:
// the full photo is already resized to exactly side x side (aspect distortion intentional), then each
// channel maps x/255 rescale followed by (x-0.5)/0.5, i.e. v/127.5 - 1. Pure function, JVM-tested.
fun normalizeToChw(argb: IntArray, side: Int): FloatArray {
    val plane = side * side
    val out = FloatArray(3 * plane)
    for (i in 0 until plane) {
        val px = argb[i]
        out[i] = ((px shr 16 and 0xFF) / 127.5f) - 1f          // R
        out[plane + i] = ((px shr 8 and 0xFF) / 127.5f) - 1f   // G
        out[2 * plane + i] = ((px and 0xFF) / 127.5f) - 1f     // B
    }
    return out
}
