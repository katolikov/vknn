package com.vknn.chat

import com.vknn.chat.model.ModelCatalog
import com.vknn.chat.model.friendlyLoadError
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

// Validates the user-facing copy for failed model loads: a driver memory refusal (relayed from the
// engine's vknn::Error through the JNI guard) gets dedicated honest copy sized from the catalogue;
// every other reason is relayed verbatim.
class LoadErrorTest {
    @Test
    fun driverMemoryRefusalGetsDedicatedCopy() {
        val raw = "Internal: vkAllocateMemory(192 MiB) failed: VK_ERROR_OUT_OF_HOST_MEMORY"
        val friendly = friendlyLoadError(ModelCatalog.SMOLVLM2, raw)
        assertEquals(
            "This device's GPU driver cannot map enough memory for this model (~4.5 GB needed).",
            friendly,
        )
    }

    @Test
    fun deviceMemoryRefusalAlsoMatches() {
        val friendly = friendlyLoadError(ModelCatalog.DL3DV, "VK_ERROR_OUT_OF_DEVICE_MEMORY")
        assertTrue(friendly, friendly.contains("cannot map enough memory"))
        assertTrue(friendly, friendly.contains("2.3 GB"))
    }

    @Test
    fun otherReasonsAreRelayedVerbatim() {
        assertEquals(
            "Load failed: InvalidArgument: graph has a cycle",
            friendlyLoadError(ModelCatalog.QWEN, "InvalidArgument: graph has a cycle"),
        )
        assertEquals("Load failed: unknown error", friendlyLoadError(ModelCatalog.QWEN, null))
    }
}
