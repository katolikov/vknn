package com.vknn.chat.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Eject
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.RestartAlt
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.vknn.chat.ChatPromptTemplate
import com.vknn.chat.Metrics
import com.vknn.chat.Msg
import com.vknn.chat.Phase
import com.vknn.chat.Role
import com.vknn.chat.UiState
import com.vknn.chat.model.ModelChoice
import kotlin.math.roundToInt

// Window insets (status bar, ime, nav bar) are handled by the AppShell around this screen.
@Composable
fun ChatScreen(
    ui: UiState,
    modelName: String,
    modelChoices: () -> List<ModelChoice>,
    selectedModelKey: String,
    onSelectModel: (String) -> Unit,
    onLoad: () -> Unit,
    onUnload: () -> Unit,
    onSend: (String) -> Unit,
    onReset: () -> Unit,
    onTemp: (Float) -> Unit,
    onOpenLibrary: () -> Unit,
    promptTemplate: String,
    onPromptTemplate: (String) -> Unit,
) {
    var editingTemplate by remember { mutableStateOf(false) }
    var pickingModel by remember { mutableStateOf(false) }
    if (pickingModel) {
        ModelPickerDialog(
            title = "Chat model",
            choices = modelChoices(),
            selectedKey = selectedModelKey,
            onSelect = onSelectModel,
            onOpenLibrary = onOpenLibrary,
            onDismiss = { pickingModel = false },
        )
    }
    // The template needs no loaded model: it is text, validated by inspection, applied on the next send.
    if (editingTemplate) {
        PromptEditorDialog(
            title = "Prompt template",
            explanation = "This model completes text rather than answering it, so a bare message gets continued. " +
                "Wrap the message in an instruct template and the same weights answer instead. Leave it empty " +
                "for raw completion. A templated message is sent as one fresh turn.",
            initialPrompt = promptTemplate,
            defaultPrompt = ChatPromptTemplate.DEFAULT,
            fieldPlaceholder = "Empty — the model continues your message",
            presets = listOf(PromptPreset("Instruct (ChatML)", ChatPromptTemplate.INSTRUCT_PRESET)),
            monospaceField = true,
            validate = ::validatePromptTemplate,
            onSave = onPromptTemplate,
            onDismiss = { editingTemplate = false },
        )
    }
    Column(
        Modifier
            .fillMaxSize()
            .background(Bg)
    ) {
        TopBar(
            modelName = modelName,
            onPickModel = { pickingModel = true },
            onEditPromptTemplate = { editingTemplate = true },
            showUnload = ui.phase == Phase.READY,
            onUnload = onUnload,
        )
        if (ui.phase == Phase.READY) {
            MetricsBar(ui.metrics, ui.temperature, ui.contextUsed, ui.contextMax, ui.backend, onTemp)
            MessageList(ui.messages, Modifier.weight(1f))
            ui.status?.let { StatusLine(it) }
            InputBar(ui.generating, onSend, onReset)
        } else {
            SetupPanel(Modifier.weight(1f), ui, modelName, onLoad, onOpenLibrary, onPickModel = { pickingModel = true })
        }
    }
}

// A template the editor will accept: empty (raw completion), or marking where the message goes.
internal fun validatePromptTemplate(draft: String): PromptValidation = when {
    !ChatPromptTemplate.isActive(draft) ->
        PromptValidation(counterText = "No template — the model continues your message")
    !ChatPromptTemplate.isApplicable(draft) ->
        PromptValidation(rejectionReason = "Add ${ChatPromptTemplate.PLACEHOLDER} where the message should go")
    else -> PromptValidation(counterText = "Template active — the model answers your message")
}

@Composable
private fun TopBar(modelName: String, onPickModel: () -> Unit, onEditPromptTemplate: () -> Unit, showUnload: Boolean, onUnload: () -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier
                .size(30.dp)
                .background(Surface, RoundedCornerShape(9.dp)),
            contentAlignment = Alignment.Center,
        ) { Icon(Icons.Filled.Memory, null, tint = Accent, modifier = Modifier.size(18.dp)) }
        Spacer(Modifier.width(9.dp))
        Column(Modifier.weight(1f).clickableNoRipple(onPickModel)) {
            Text(modelName, color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Medium, maxLines = 1)
            Text("on-device · tap to switch model", color = TextSecondary, fontSize = 11.sp)
        }
        if (showUnload) {
            TopBarIconButton(Icons.Filled.Eject, "unload model", onUnload)
            Spacer(Modifier.width(8.dp))
        }
        Box(
            Modifier
                .size(30.dp)
                .background(Surface, RoundedCornerShape(9.dp))
                .clickableNoRipple(onEditPromptTemplate),
            contentAlignment = Alignment.Center,
        ) { Icon(Icons.Filled.Edit, "edit prompt template", tint = TextSecondary, modifier = Modifier.size(16.dp)) }
        Spacer(Modifier.width(8.dp))
        Row(
            Modifier
                .background(AccentDim, RoundedCornerShape(999.dp))
                .padding(horizontal = 9.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(Icons.Filled.Bolt, null, tint = Accent, modifier = Modifier.size(13.dp))
            Spacer(Modifier.width(4.dp))
            Text("GPU", color = Accent, fontSize = 11.sp)
        }
    }
}

@Composable
private fun MetricsBar(m: Metrics, temp: Float, ctxUsed: Int, ctxMax: Int, backend: String, onTemp: (Float) -> Unit) {
    var showTemp by remember { mutableStateOf(false) }
    Column(Modifier.padding(horizontal = 12.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
            MetricCell(Modifier.weight(1f), "TTFT", if (m.ttftMs > 0) "${m.ttftMs} ms" else "—")
            MetricCell(Modifier.weight(1f), "Speed", if (m.tokPerSec > 0) "${((m.tokPerSec * 10).roundToInt() / 10.0)} t/s" else "—", value2Accent = true)
            MetricCell(Modifier.weight(1f), "Prefill", if (m.prefillMs > 0) "${m.prefillMs} ms" else "—")
            TempCell(Modifier.weight(1f), temp) { showTemp = !showTemp }
        }
        if (showTemp) {
            Row(Modifier.fillMaxWidth().padding(top = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("greedy", color = TextSecondary, fontSize = 11.sp)
                Slider(
                    value = temp,
                    onValueChange = onTemp,
                    valueRange = 0f..1.2f,
                    modifier = Modifier.weight(1f).padding(horizontal = 8.dp),
                )
                Text("1.2", color = TextSecondary, fontSize = 11.sp)
            }
        }
        if (ctxMax > 0) {
            Text(
                if (backend.isEmpty()) "context ${ctxUsed}/${ctxMax}" else "context ${ctxUsed}/${ctxMax} · $backend",
                color = TextSecondary,
                fontSize = 10.sp,
                modifier = Modifier.padding(top = 4.dp),
            )
        }
    }
}

@Composable
private fun MetricCell(mod: Modifier, label: String, value: String, value2Accent: Boolean = false) {
    Column(
        mod
            .background(Surface, RoundedCornerShape(12.dp))
            .padding(horizontal = 8.dp, vertical = 9.dp)
    ) {
        Text(label, color = TextSecondary, fontSize = 11.sp)
        Text(value, color = if (value2Accent) Accent else TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.Medium)
    }
}

@Composable
private fun TempCell(mod: Modifier, temp: Float, onClick: () -> Unit) {
    Column(
        mod
            .background(Surface, RoundedCornerShape(12.dp))
            .padding(horizontal = 8.dp, vertical = 9.dp)
    ) {
        Text("Temp", color = TextSecondary, fontSize = 11.sp)
        Text(
            if (temp <= 0f) "greedy" else "${(temp * 10).roundToInt() / 10.0}",
            color = Accent,
            fontSize = 15.sp,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.clickableNoRipple(onClick),
        )
    }
}

@Composable
private fun MessageList(messages: List<Msg>, mod: Modifier) {
    val state = rememberLazyListState()
    // True only when the last item's bottom is on screen. Auto-follow the stream only while stuck at the
    // bottom, so scrolling up to read earlier text mid-generation is not yanked back each token; scroll to
    // the item's END (Int.MAX_VALUE offset, clamped) so a message taller than the viewport shows its newest
    // text at the bottom instead of pinning to its top.
    // `messages` is a plain parameter, so the derived state reads it through a State holder: capturing it
    // directly would pin `lastIndex` to the list the enclosing `remember` first saw (empty at load).
    val latestMessages by rememberUpdatedState(messages)
    val atBottom by remember {
        derivedStateOf {
            val info = state.layoutInfo
            val last = info.visibleItemsInfo.lastOrNull() ?: return@derivedStateOf true
            last.index >= latestMessages.lastIndex && last.offset + last.size <= info.viewportEndOffset + 4
        }
    }
    LaunchedEffect(messages.size) { // a new message always jumps to the bottom
        if (messages.isNotEmpty()) state.scrollToItem(messages.lastIndex, Int.MAX_VALUE)
    }
    LaunchedEffect(messages.lastOrNull()?.text) { // streaming text follows the bottom only when stuck there
        if (atBottom && messages.isNotEmpty()) state.scrollToItem(messages.lastIndex, Int.MAX_VALUE)
    }
    LazyColumn(
        mod.fillMaxWidth().padding(horizontal = 14.dp),
        state = state,
        verticalArrangement = Arrangement.spacedBy(12.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(vertical = 10.dp),
    ) {
        itemsIndexed(messages) { _, msg -> Bubble(msg) }
    }
}

@Composable
private fun Bubble(msg: Msg) {
    val user = msg.role == Role.USER
    Row(Modifier.fillMaxWidth(), horizontalArrangement = if (user) Arrangement.End else Arrangement.Start) {
        Box(
            Modifier
                .widthIn(max = 300.dp)
                .background(
                    if (user) Accent else SurfaceHi,
                    if (user) RoundedCornerShape(20.dp, 20.dp, 6.dp, 20.dp) else RoundedCornerShape(20.dp, 20.dp, 20.dp, 6.dp),
                )
                .padding(horizontal = 13.dp, vertical = 9.dp)
        ) {
            Text(
                msg.text.ifEmpty { "…" },
                color = if (user) OnAccent else TextPrimary,
                fontSize = 13.sp,
                fontFamily = if (user) FontFamily.Default else FontFamily.Monospace,
                lineHeight = 19.sp,
            )
        }
    }
}

@Composable
private fun StatusLine(text: String) {
    Text(text, color = Color(0xFFE07A7A), fontSize = 12.sp, modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp))
}

@Composable
private fun InputBar(generating: Boolean, onSend: (String) -> Unit, onReset: () -> Unit) {
    var text by remember { mutableStateOf("") }
    Row(
        Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier
                .size(40.dp)
                .background(Surface, RoundedCornerShape(999.dp))
                .clickableNoRipple(onReset),
            contentAlignment = Alignment.Center,
        ) { Icon(Icons.Filled.RestartAlt, "reset", tint = TextSecondary, modifier = Modifier.size(20.dp)) }
        Spacer(Modifier.width(8.dp))
        Box(
            Modifier
                .weight(1f)
                .background(Surface, RoundedCornerShape(999.dp))
                .padding(horizontal = 16.dp, vertical = 11.dp),
        ) {
            if (text.isEmpty()) Text("Message", color = TextSecondary, fontSize = 13.sp)
            BasicTextField(
                value = text,
                onValueChange = { text = it },
                textStyle = TextStyle(color = TextPrimary, fontSize = 13.sp),
                cursorBrush = SolidColor(Accent),
                enabled = !generating,
                singleLine = false,
                keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = androidx.compose.foundation.text.KeyboardActions(onSend = {
                    if (text.isNotBlank()) { onSend(text); text = "" }
                }),
                modifier = Modifier.fillMaxWidth(),
            )
        }
        Spacer(Modifier.width(8.dp))
        Box(
            Modifier
                .size(40.dp)
                .background(if (generating) Surface else Accent, RoundedCornerShape(999.dp))
                .clickableNoRipple {
                    if (!generating && text.isNotBlank()) { onSend(text); text = "" }
                },
            contentAlignment = Alignment.Center,
        ) {
            if (generating) CircularProgressIndicator(Modifier.size(18.dp), color = Accent, strokeWidth = 2.dp)
            else Icon(Icons.Filled.ArrowUpward, "send", tint = OnAccent, modifier = Modifier.size(20.dp))
        }
    }
}

@Composable
private fun SetupPanel(mod: Modifier, ui: UiState, modelName: String, onLoad: () -> Unit, onOpenLibrary: () -> Unit, onPickModel: () -> Unit) {
    Column(
        mod.fillMaxWidth().padding(28.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        when (ui.phase) {
            Phase.MISSING -> {
                Text("Model not downloaded", color = TextPrimary, fontSize = 18.sp, fontWeight = FontWeight.Medium)
                Spacer(Modifier.height(6.dp))
                Text("Get $modelName from the Model Library, or pick another variant.", color = TextSecondary, fontSize = 13.sp)
                Spacer(Modifier.height(20.dp))
                PrimaryButton("Open Library", onOpenLibrary)
                Spacer(Modifier.height(10.dp))
                PillButton("Switch model variant", accent = false, onClick = onPickModel)
            }
            Phase.DOWNLOADED -> {
                Text("Model ready", color = TextPrimary, fontSize = 18.sp, fontWeight = FontWeight.Medium)
                Spacer(Modifier.height(6.dp))
                Text("Load $modelName onto the GPU to start chatting.", color = TextSecondary, fontSize = 13.sp)
                Spacer(Modifier.height(20.dp))
                PrimaryButton("Load on GPU", onLoad)
                Spacer(Modifier.height(10.dp))
                PillButton("Switch model variant", accent = false, onClick = onPickModel)
            }
            Phase.LOADING -> {
                CircularProgressIndicator(color = Accent)
                Spacer(Modifier.height(16.dp))
                Text("Loading on GPU…", color = TextSecondary, fontSize = 13.sp)
            }
            Phase.READY -> {}
        }
        ui.status?.let {
            Spacer(Modifier.height(14.dp))
            Text(it, color = Color(0xFFE07A7A), fontSize = 12.sp)
        }
    }
}
