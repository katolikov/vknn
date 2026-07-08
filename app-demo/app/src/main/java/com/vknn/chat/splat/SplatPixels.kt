package com.vknn.chat.splat

// ARGB pixels -> fp32 CHW RGB in [0, 1] (a plain 1/255 rescale, no mean/std), the YoNoSplat
// encoder's input convention (scripts/yonosplat/fetch_re10k_test.py). Pure function, JVM-tested.
fun rgbToChw01(argb: IntArray, side: Int): FloatArray {
    val plane = side * side
    val out = FloatArray(3 * plane)
    for (i in 0 until plane) {
        val pixel = argb[i]
        out[i] = (pixel shr 16 and 0xFF) / 255f              // R
        out[plane + i] = (pixel shr 8 and 0xFF) / 255f       // G
        out[2 * plane + i] = (pixel and 0xFF) / 255f         // B
    }
    return out
}
