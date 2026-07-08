package com.vknn.chat

import com.vknn.chat.model.BackendPolicy
import com.vknn.chat.model.ModelCatalog
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

// Validates the CPU-backend admission logic: fp32 working set = 2x the fp16 catalogue size, budget =
// half of physical RAM. The measured ground truth: SmolVLM2 (4.5 GB fp16 -> ~9 GB fp32) kernel-OOM
// reboots a 12 GB phone, while Qwen and the dl3dv encoder run on CPU (slowly).
class BackendPolicyTest {
    // A "12 GB" phone reports slightly under 12 GiB of physical RAM.
    private val twelveGbPhoneRamBytes = 12_500_000_000L
    private val sixteenGbPhoneRamBytes = 16_500_000_000L

    @Test
    fun smolvlm2IsBlockedOnTwelveGbDevices() {
        val verdict = BackendPolicy.cpuVerdict(ModelCatalog.SMOLVLM2, twelveGbPhoneRamBytes)
        assertTrue(verdict is BackendPolicy.CpuVerdict.Blocked)
        val reason = (verdict as BackendPolicy.CpuVerdict.Blocked).reason
        assertTrue(reason, reason.contains("9.0 GB"))
        assertTrue(reason, reason.contains("Vulkan only"))
    }

    @Test
    fun smolvlm2StaysBlockedOnSixteenGbDevices() {
        // 9 GB fp32 against an 8.25 GB half-RAM budget.
        assertTrue(
            BackendPolicy.cpuVerdict(ModelCatalog.SMOLVLM2, sixteenGbPhoneRamBytes)
                is BackendPolicy.CpuVerdict.Blocked,
        )
    }

    @Test
    fun qwenAndDl3dvAreAllowedSlowOnTwelveGbDevices() {
        for (spec in listOf(ModelCatalog.QWEN, ModelCatalog.DL3DV)) {
            val verdict = BackendPolicy.cpuVerdict(spec, twelveGbPhoneRamBytes)
            assertTrue(spec.id, verdict is BackendPolicy.CpuVerdict.AllowedSlow)
            assertTrue((verdict as BackendPolicy.CpuVerdict.AllowedSlow).note.contains("slower"))
        }
    }

    @Test
    fun estimateIsTwiceTheCatalogueSize() {
        assertEquals(ModelCatalog.QWEN.approxBytes * 2, BackendPolicy.estimatedCpuBytes(ModelCatalog.QWEN))
    }
}
