package com.vknn.chat

import android.app.ActivityManager
import android.app.Application
import com.vknn.chat.model.BackendSetting
import com.vknn.chat.model.ModelResidency
import com.vknn.chat.model.ModelSelection
import com.vknn.chat.model.ModelStore
import com.vknn.chat.model.PromptSettings

// Application-scoped holder for the model store and settings, so downloads keep running across
// activity recreation and every screen observes the same state.
class VknnApp : Application() {
    lateinit var models: ModelStore
        private set
    lateinit var settings: BackendSetting
        private set
    lateinit var prompts: PromptSettings
        private set
    lateinit var residency: ModelResidency
        private set
    lateinit var modelSelection: ModelSelection
        private set

    override fun onCreate() {
        super.onCreate()
        models = ModelStore(this)
        settings = BackendSetting(this)
        prompts = PromptSettings(this)
        residency = ModelResidency()
        modelSelection = ModelSelection(this, models)
    }

    /** Physical device RAM; the CPU-backend admission check budgets against it. */
    fun totalRamBytes(): Long {
        val memoryInfo = ActivityManager.MemoryInfo()
        getSystemService(ActivityManager::class.java)?.getMemoryInfo(memoryInfo)
        return memoryInfo.totalMem
    }
}
