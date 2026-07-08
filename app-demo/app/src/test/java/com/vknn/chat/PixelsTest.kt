package com.vknn.chat

import com.vknn.chat.vlm.normalizeToChw
import org.junit.Assert.assertEquals
import org.junit.Test

// Validates the ARGB -> fp32 CHW normalization (x/127.5 - 1 per channel, planes in R,G,B order).
class PixelsTest {
    @Test
    fun normalizesChannelsIntoPlanes() {
        // 2x2: red, green, blue, mid-gray (0x80 = 128).
        val argb = intArrayOf(0xFFFF0000.toInt(), 0xFF00FF00.toInt(), 0xFF0000FF.toInt(), 0xFF808080.toInt())
        val out = normalizeToChw(argb, 2)
        assertEquals(12, out.size)
        val eps = 1e-6f
        // R plane
        assertEquals(1f, out[0], eps)
        assertEquals(-1f, out[1], eps)
        assertEquals(-1f, out[2], eps)
        assertEquals(128f / 127.5f - 1f, out[3], eps)
        // G plane
        assertEquals(-1f, out[4], eps)
        assertEquals(1f, out[5], eps)
        assertEquals(-1f, out[6], eps)
        assertEquals(128f / 127.5f - 1f, out[7], eps)
        // B plane
        assertEquals(-1f, out[8], eps)
        assertEquals(-1f, out[9], eps)
        assertEquals(1f, out[10], eps)
        assertEquals(128f / 127.5f - 1f, out[11], eps)
    }
}
