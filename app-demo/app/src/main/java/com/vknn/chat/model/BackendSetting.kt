package com.vknn.chat.model

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

enum class InferenceBackend(val engineName: String, val label: String) {
    VULKAN("vulkan", "Vulkan (GPU)"),
    CPU("cpu", "CPU"),
}

// The global inference-backend choice (Vulkan default), persisted in SharedPreferences — the app's
// settings store. The choice applies at the NEXT model load: view models read it when they call
// their native init, and an already-loaded session keeps its backend until reloaded.
class BackendSetting(context: Context) {
    private val preferences: SharedPreferences =
        context.getSharedPreferences("vknn_settings", Context.MODE_PRIVATE)

    private val _backend = MutableStateFlow(readPersisted())
    val backend: StateFlow<InferenceBackend> = _backend.asStateFlow()

    fun current(): InferenceBackend = _backend.value

    fun set(value: InferenceBackend) {
        preferences.edit().putString(KEY_BACKEND, value.engineName).apply()
        _backend.value = value
    }

    private fun readPersisted(): InferenceBackend {
        val persisted = preferences.getString(KEY_BACKEND, InferenceBackend.VULKAN.engineName)
        return InferenceBackend.entries.firstOrNull { it.engineName == persisted } ?: InferenceBackend.VULKAN
    }

    private companion object {
        const val KEY_BACKEND = "inference_backend"
    }
}
