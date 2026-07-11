package com.vknn.chat

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.lifecycle.viewmodel.compose.viewModel
import com.vknn.chat.model.BackendPolicy
import com.vknn.chat.model.ModelCatalog
import com.vknn.chat.splat.SplatViewModel
import com.vknn.chat.ui.AppShell
import com.vknn.chat.ui.VknnChatTheme
import com.vknn.chat.vlm.VlmViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val app = application as VknnApp
        val models = app.models
        setContent {
            VknnChatTheme {
                val vm: ChatViewModel = viewModel()
                val vlm: VlmViewModel = viewModel()
                val splat: SplatViewModel = viewModel()
                val chatUi by vm.ui.collectAsState()
                val vlmUi by vlm.ui.collectAsState()
                val splatUi by splat.ui.collectAsState()
                val catalog by app.catalog.catalog.collectAsState()
                val modelStates by models.states.collectAsState()
                val modelLoadErrors by models.loadErrors.collectAsState()
                val backend by app.settings.backend.collectAsState()
                val chatTemplateOverrides by app.prompts.chatTemplateOverrides.collectAsState()
                val vlmQuestion by app.prompts.vlmQuestion.collectAsState()
                val chatModelKey by app.modelSelection.chatKey.collectAsState()
                val vlmModelKey by app.modelSelection.vlmKey.collectAsState()
                // The prompt template follows the selected chat model's family: its default preset, and
                // any saved override, are per dialect (a Qwen ChatML template has no meaning under Llama-3).
                val chatSpec = app.modelSelection.specFor(chatModelKey, ModelCatalog.QWEN)
                val chatDialect = ChatPromptTemplate.dialect(chatSpec.chatDialectId)
                val chatPromptTemplate = chatTemplateOverrides[chatDialect.id] ?: chatDialect.instructPreset
                AppShell(
                    chatUi = chatUi,
                    vlmUi = vlmUi,
                    catalog = catalog,
                    modelStates = modelStates,
                    modelLoadErrors = modelLoadErrors,
                    freeBytes = models::freeBytes,
                    isMetered = models::isMetered,
                    backend = backend,
                    onBackend = app.settings::set,
                    cpuVerdictFor = { spec -> BackendPolicy.cpuVerdict(spec, app.totalRamBytes()) },
                    onDownload = models::start,
                    onPause = models::pause,
                    onDelete = models::delete,
                    chatModelName = chatSpec.displayName,
                    chatModelChoices = { app.modelSelection.choicesFor(ModelCatalog.QWEN.mode) },
                    chatSelectedModelKey = chatModelKey,
                    onChatSelectModel = vm::selectModel,
                    onLoad = vm::loadModel,
                    onUnload = vm::unloadModel,
                    onSend = vm::send,
                    onReset = vm::reset,
                    onTemp = vm::setTemperature,
                    chatPromptTemplate = chatPromptTemplate,
                    chatPromptTemplateDefault = chatDialect.instructPreset,
                    chatPromptPresetLabel = chatDialect.presetLabel,
                    onChatPromptTemplate = { app.prompts.setChatTemplate(chatDialect, it) },
                    vlmModelName = app.modelSelection.specFor(vlmModelKey, ModelCatalog.SMOLVLM2).displayName,
                    vlmModelChoices = { app.modelSelection.choicesFor(ModelCatalog.SMOLVLM2.mode) },
                    vlmSelectedModelKey = vlmModelKey,
                    onVlmSelectModel = vlm::selectModel,
                    onVlmLoad = vlm::loadModel,
                    onVlmUnload = vlm::unloadModel,
                    onVlmCapture = vlm::onCapture,
                    onVlmCancel = vlm::cancel,
                    onVlmRetake = vlm::retake,
                    vlmQuestion = vlmQuestion,
                    onVlmQuestion = vlm::setQuestion,
                    measureVlmPrompt = vlm::measurePrompt,
                    splatUi = splatUi,
                    onSplatLoad = splat::loadModel,
                    onSplatUnload = splat::unloadModel,
                    onSplatStartCapture = splat::startCapture,
                    onSplatFrame = splat::addFrame,
                    onSplatOrbit = splat::setOrbit,
                    onSplatRecapture = splat::recapture,
                )
            }
        }
    }
}
