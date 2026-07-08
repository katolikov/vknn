package com.vknn.chat.model

// CPU-backend admission. The CPU backend decodes fp16 weights to fp32 host-side, so a model needs
// roughly twice its file size in RAM — and the decode transiently holds parts of both copies while
// the OS and app need their own headroom, so the budget is half of physical RAM (measured: the
// ~9 GB fp32 working set of a 4.5 GB fp16 model kernel-OOM-reboots a 12 GB device). Everything is
// derived from the catalogue size; nothing is hardcoded per model.
object BackendPolicy {

    /** The fp32 host working set a CPU load of this model implies (2x the fp16 file). */
    fun estimatedCpuBytes(spec: ModelSpec): Long = spec.approxBytes * 2

    sealed interface CpuVerdict {
        /** CPU works but is far slower than Vulkan; [note] labels the trade. */
        data class AllowedSlow(val note: String) : CpuVerdict

        /** The fp32 working set cannot fit this device's RAM; [reason] is the one-line user copy. */
        data class Blocked(val reason: String) : CpuVerdict
    }

    fun cpuVerdict(spec: ModelSpec, totalDeviceRamBytes: Long): CpuVerdict {
        val workingSetBytes = estimatedCpuBytes(spec)
        val budgetBytes = totalDeviceRamBytes / 2
        return if (workingSetBytes > budgetBytes) {
            CpuVerdict.Blocked("needs ~${formatBytes(workingSetBytes)} RAM — Vulkan only on this device")
        } else {
            CpuVerdict.AllowedSlow("much slower — for validation")
        }
    }
}
