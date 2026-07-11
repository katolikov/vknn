package com.vknn.chat.ui

import android.graphics.Bitmap
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material.icons.filled.ViewInAr
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.vknn.chat.UiState
import com.vknn.chat.model.BackendPolicy
import com.vknn.chat.model.InferenceBackend
import com.vknn.chat.model.ModelChoice
import com.vknn.chat.model.ModelSpec
import com.vknn.chat.model.ModelState
import com.vknn.chat.splat.OrbitCamera
import com.vknn.chat.splat.SplatUiState
import com.vknn.chat.vlm.VlmPromptMeasurement
import com.vknn.chat.vlm.VlmUiState

// The three mode screens plus the model library, switched by a flat bottom bar. Each mode screen
// gates on its model's library state and opens the Library to fetch it.
enum class AppTab(val label: String, val icon: ImageVector) {
    CHAT("Chat", Icons.AutoMirrored.Filled.Chat),
    VLM("VLM", Icons.Filled.CameraAlt),
    SPLAT("3D Splat", Icons.Filled.ViewInAr),
    LIBRARY("Library", Icons.Filled.Storage),
}

@Composable
fun AppShell(
    chatUi: UiState,
    vlmUi: VlmUiState,
    catalog: List<ModelSpec>,
    modelStates: Map<String, ModelState>,
    modelLoadErrors: Map<String, String>,
    freeBytes: () -> Long,
    isMetered: () -> Boolean,
    backend: InferenceBackend,
    onBackend: (InferenceBackend) -> Unit,
    cpuVerdictFor: (ModelSpec) -> BackendPolicy.CpuVerdict,
    onDownload: (ModelSpec) -> Unit,
    onPause: (ModelSpec) -> Unit,
    onDelete: (ModelSpec) -> Unit,
    chatModelName: String,
    chatModelChoices: () -> List<ModelChoice>,
    chatSelectedModelKey: String,
    onChatSelectModel: (String) -> Unit,
    onLoad: () -> Unit,
    onUnload: () -> Unit,
    onSend: (String) -> Unit,
    onReset: () -> Unit,
    onTemp: (Float) -> Unit,
    chatPromptTemplate: String,
    chatPromptTemplateDefault: String,
    chatPromptPresetLabel: String,
    onChatPromptTemplate: (String) -> Unit,
    vlmModelName: String,
    vlmModelChoices: () -> List<ModelChoice>,
    vlmSelectedModelKey: String,
    onVlmSelectModel: (String) -> Unit,
    onVlmLoad: () -> Unit,
    onVlmUnload: () -> Unit,
    onVlmCapture: (Bitmap) -> Unit,
    onVlmCancel: () -> Unit,
    onVlmRetake: () -> Unit,
    vlmQuestion: String,
    onVlmQuestion: (String) -> Unit,
    measureVlmPrompt: (String) -> VlmPromptMeasurement?,
    splatUi: SplatUiState,
    onSplatLoad: () -> Unit,
    onSplatUnload: () -> Unit,
    onSplatStartCapture: () -> Unit,
    onSplatFrame: (Bitmap, Float, Float) -> Unit,
    onSplatOrbit: (OrbitCamera) -> Unit,
    onSplatRecapture: () -> Unit,
) {
    var tab by rememberSaveable { mutableStateOf(AppTab.CHAT.ordinal) }
    val openLibrary = { tab = AppTab.LIBRARY.ordinal }
    Column(
        Modifier
            .fillMaxSize()
            .background(Bg)
            .statusBarsPadding()
            .imePadding(),
    ) {
        Box(Modifier.weight(1f)) {
            when (AppTab.entries[tab]) {
                AppTab.CHAT -> ChatScreen(
                    ui = chatUi,
                    modelName = chatModelName,
                    modelChoices = chatModelChoices,
                    selectedModelKey = chatSelectedModelKey,
                    onSelectModel = onChatSelectModel,
                    onLoad = onLoad,
                    onUnload = onUnload,
                    onSend = onSend,
                    onReset = onReset,
                    onTemp = onTemp,
                    onOpenLibrary = openLibrary,
                    promptTemplate = chatPromptTemplate,
                    promptTemplateDefault = chatPromptTemplateDefault,
                    promptPresetLabel = chatPromptPresetLabel,
                    onPromptTemplate = onChatPromptTemplate,
                )
                AppTab.VLM -> VlmScreen(
                    ui = vlmUi,
                    modelName = vlmModelName,
                    modelChoices = vlmModelChoices,
                    selectedModelKey = vlmSelectedModelKey,
                    onSelectModel = onVlmSelectModel,
                    onLoad = onVlmLoad,
                    onUnload = onVlmUnload,
                    onCapture = onVlmCapture,
                    onCancel = onVlmCancel,
                    onRetake = onVlmRetake,
                    onOpenLibrary = openLibrary,
                    question = vlmQuestion,
                    onQuestion = onVlmQuestion,
                    measurePrompt = measureVlmPrompt,
                )
                AppTab.SPLAT -> SplatScreen(
                    ui = splatUi,
                    onLoad = onSplatLoad,
                    onUnload = onSplatUnload,
                    onStartCapture = onSplatStartCapture,
                    onFrame = onSplatFrame,
                    onOrbit = onSplatOrbit,
                    onRecapture = onSplatRecapture,
                    onOpenLibrary = openLibrary,
                )
                AppTab.LIBRARY -> LibraryScreen(
                    catalog = catalog,
                    states = modelStates,
                    loadErrors = modelLoadErrors,
                    freeBytes = freeBytes(),
                    metered = isMetered(),
                    backend = backend,
                    onBackend = onBackend,
                    cpuVerdictFor = cpuVerdictFor,
                    onDownload = onDownload,
                    onPause = onPause,
                    onDelete = onDelete,
                )
            }
        }
        BottomNav(selected = tab, onSelect = { tab = it })
    }
}

@Composable
private fun BottomNav(selected: Int, onSelect: (Int) -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .background(Surface)
            .windowInsetsPadding(WindowInsets.navigationBars)
            .padding(vertical = 6.dp),
    ) {
        AppTab.entries.forEachIndexed { i, t ->
            val active = i == selected
            Column(
                Modifier
                    .weight(1f)
                    .clickableNoRipple { onSelect(i) }
                    .padding(vertical = 4.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Icon(t.icon, t.label, tint = if (active) Accent else TextSecondary, modifier = Modifier.size(22.dp))
                Spacer(Modifier.height(2.dp))
                Text(t.label, color = if (active) Accent else TextSecondary, fontSize = 10.sp)
            }
        }
    }
}
