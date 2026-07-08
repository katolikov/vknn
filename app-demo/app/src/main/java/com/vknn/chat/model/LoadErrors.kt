package com.vknn.chat.model

// User-facing copy for a failed model load. The native bridge relays the engine's thrown reason
// (vknn::Error) as an exception message; a Vulkan memory refusal (some drivers cap per-process
// host mappings below what a large model needs) gets honest dedicated copy, everything else is
// relayed verbatim. Pure function, JVM-tested.
fun friendlyLoadError(spec: ModelSpec, rawMessage: String?): String {
    val message = rawMessage ?: "unknown error"
    return if (message.contains("OUT_OF_HOST_MEMORY") || message.contains("OUT_OF_DEVICE_MEMORY")) {
        "This device's GPU driver cannot map enough memory for this model (~${formatBytes(spec.approxBytes)} needed)."
    } else {
        "Load failed: $message"
    }
}
