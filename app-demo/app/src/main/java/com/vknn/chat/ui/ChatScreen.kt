package com.vknn.chat.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
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
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.RestartAlt
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
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
import com.vknn.chat.Metrics
import com.vknn.chat.Msg
import com.vknn.chat.Phase
import com.vknn.chat.Role
import com.vknn.chat.UiState
import kotlin.math.roundToInt

@Composable
fun ChatScreen(
    ui: UiState,
    onDownload: () -> Unit,
    onLoad: () -> Unit,
    onSend: (String) -> Unit,
    onReset: () -> Unit,
    onTemp: (Float) -> Unit,
) {
    Column(
        Modifier
            .fillMaxSize()
            .background(Bg)
            .statusBarsPadding()
    ) {
        TopBar()
        if (ui.phase == Phase.READY) {
            MetricsBar(ui.metrics, ui.temperature, ui.contextUsed, ui.contextMax, onTemp)
            MessageList(ui.messages, Modifier.weight(1f))
            ui.status?.let { StatusLine(it) }
            InputBar(ui.generating, onSend, onReset)
        } else {
            SetupPanel(Modifier.weight(1f), ui, onDownload, onLoad)
        }
    }
}

@Composable
private fun TopBar() {
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
        Column(Modifier.weight(1f)) {
            Text("Qwen2.5-Coder 0.5B", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Medium)
            Text("on-device", color = TextSecondary, fontSize = 11.sp)
        }
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
private fun MetricsBar(m: Metrics, temp: Float, ctxUsed: Int, ctxMax: Int, onTemp: (Float) -> Unit) {
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
                "context ${ctxUsed}/${ctxMax}",
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
    LaunchedEffect(messages.size, messages.lastOrNull()?.text) {
        if (messages.isNotEmpty()) state.animateScrollToItem(messages.size - 1)
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
            .navigationBarsPadding()
            .imePadding()
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
private fun SetupPanel(mod: Modifier, ui: UiState, onDownload: () -> Unit, onLoad: () -> Unit) {
    Column(
        mod.fillMaxWidth().padding(28.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        when (ui.phase) {
            Phase.NEED_DOWNLOAD -> {
                Text("Download the model", color = TextPrimary, fontSize = 18.sp, fontWeight = FontWeight.Medium)
                Spacer(Modifier.height(6.dp))
                Text("Qwen2.5-Coder 0.5B, ~1.2 GB, runs on the GPU.", color = TextSecondary, fontSize = 13.sp)
                Spacer(Modifier.height(20.dp))
                PrimaryButton("Download model", onDownload)
            }
            Phase.DOWNLOADING -> {
                Text("Downloading… ${ui.downloadPct}%", color = TextPrimary, fontSize = 16.sp)
                Spacer(Modifier.height(4.dp))
                Text("${ui.downloadMB} / ${ui.totalMB} MB", color = TextSecondary, fontSize = 12.sp)
                Spacer(Modifier.height(16.dp))
                LinearProgressIndicator(
                    progress = { ui.downloadPct / 100f },
                    modifier = Modifier.fillMaxWidth().height(6.dp),
                    color = Accent,
                    trackColor = Surface,
                )
            }
            Phase.DOWNLOADED -> {
                Text("Model ready", color = TextPrimary, fontSize = 18.sp, fontWeight = FontWeight.Medium)
                Spacer(Modifier.height(6.dp))
                Text("Load it onto the GPU to start chatting.", color = TextSecondary, fontSize = 13.sp)
                Spacer(Modifier.height(20.dp))
                PrimaryButton("Load on GPU", onLoad)
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

@Composable
private fun PrimaryButton(label: String, onClick: () -> Unit) {
    Box(
        Modifier
            .background(Accent, RoundedCornerShape(999.dp))
            .clickableNoRipple(onClick)
            .padding(horizontal = 28.dp, vertical = 13.dp),
        contentAlignment = Alignment.Center,
    ) { Text(label, color = OnAccent, fontSize = 15.sp, fontWeight = FontWeight.Medium) }
}

// A no-ripple clickable (One UI's flat feel) that avoids pulling the material ripple indication setup.
private fun Modifier.clickableNoRipple(onClick: () -> Unit): Modifier =
    this.clickable(
        interactionSource = MutableInteractionSource(),
        indication = null,
        onClick = onClick,
    )
