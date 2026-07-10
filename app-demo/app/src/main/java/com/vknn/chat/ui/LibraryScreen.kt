package com.vknn.chat.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.vknn.chat.model.BackendPolicy
import com.vknn.chat.model.InferenceBackend
import com.vknn.chat.model.ModelCatalog
import com.vknn.chat.model.ModelSpec
import com.vknn.chat.model.ModelState
import com.vknn.chat.model.formatBytes

// One card per catalogue model: what mode it powers, its size up front, and the download lifecycle
// (start, pause/resume across process death, verify, delete-to-free-space).
@Composable
fun LibraryScreen(
    states: Map<String, ModelState>,
    loadErrors: Map<String, String>,
    freeBytes: Long,
    metered: Boolean,
    backend: InferenceBackend,
    onBackend: (InferenceBackend) -> Unit,
    cpuVerdictFor: (ModelSpec) -> BackendPolicy.CpuVerdict,
    onDownload: (ModelSpec) -> Unit,
    onPause: (ModelSpec) -> Unit,
    onDelete: (ModelSpec) -> Unit,
) {
    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp),
    ) {
        Spacer(Modifier.height(14.dp))
        Text("Model Library", color = TextPrimary, fontSize = 20.sp, fontWeight = FontWeight.Medium)
        Spacer(Modifier.height(4.dp))
        Text("${formatBytes(freeBytes)} free on device", color = TextSecondary, fontSize = 12.sp)
        if (metered) {
            Spacer(Modifier.height(10.dp))
            Row(
                Modifier
                    .fillMaxWidth()
                    .background(Surface, RoundedCornerShape(12.dp))
                    .padding(horizontal = 12.dp, vertical = 9.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(Icons.Filled.Warning, null, tint = Warn, modifier = Modifier.size(15.dp))
                Spacer(Modifier.width(8.dp))
                Text(
                    "Metered connection — model downloads use mobile data.",
                    color = Warn,
                    fontSize = 12.sp,
                )
            }
        }
        Spacer(Modifier.height(14.dp))
        BackendSelectorRow(backend, onBackend)
        Spacer(Modifier.height(14.dp))
        for (spec in ModelCatalog.ALL) {
            ModelCard(spec, states[spec.id] ?: ModelState.Missing, loadErrors[spec.id], metered, backend, cpuVerdictFor, onDownload, onPause, onDelete)
            Spacer(Modifier.height(12.dp))
        }
    }
}

// The global inference-backend choice; a change applies at the next model load (a loaded session
// keeps its backend until it is reloaded — sessions are never torn down mid-generation).
@Composable
private fun BackendSelectorRow(backend: InferenceBackend, onBackend: (InferenceBackend) -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .background(Surface, RoundedCornerShape(16.dp))
            .padding(14.dp),
    ) {
        Text("Inference backend", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Medium)
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for (option in InferenceBackend.entries) {
                PillButton(option.label, accent = option == backend) { onBackend(option) }
            }
        }
        Spacer(Modifier.height(8.dp))
        Text("Applies when a model is next loaded.", color = TextSecondary, fontSize = 11.sp)
    }
}

@Composable
private fun ModelCard(
    spec: ModelSpec,
    state: ModelState,
    loadError: String?,
    metered: Boolean,
    backend: InferenceBackend,
    cpuVerdictFor: (ModelSpec) -> BackendPolicy.CpuVerdict,
    onDownload: (ModelSpec) -> Unit,
    onPause: (ModelSpec) -> Unit,
    onDelete: (ModelSpec) -> Unit,
) {
    Column(
        Modifier
            .fillMaxWidth()
            .background(Surface, RoundedCornerShape(16.dp))
            .padding(14.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(spec.displayName, color = TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.Medium)
                Text("${spec.mode} mode · ${spec.variant}", color = Accent, fontSize = 11.sp)
            }
            Text(formatBytes(spec.approxBytes), color = TextSecondary, fontSize = 12.sp)
        }
        Spacer(Modifier.height(6.dp))
        Text(spec.description, color = TextSecondary, fontSize = 12.sp, lineHeight = 17.sp)
        loadError?.let {
            Spacer(Modifier.height(6.dp))
            Text(it, color = Err, fontSize = 11.sp, lineHeight = 15.sp)
        }
        if (backend == InferenceBackend.CPU) {
            Spacer(Modifier.height(6.dp))
            when (val verdict = cpuVerdictFor(spec)) {
                is BackendPolicy.CpuVerdict.Blocked -> Text("CPU: ${verdict.reason}", color = Err, fontSize = 11.sp)
                is BackendPolicy.CpuVerdict.AllowedSlow -> Text("CPU: ${verdict.note}", color = Warn, fontSize = 11.sp)
            }
        }
        Spacer(Modifier.height(12.dp))
        when (state) {
            is ModelState.Missing -> Row(verticalAlignment = Alignment.CenterVertically) {
                PillButton("Download • ${formatBytes(spec.approxBytes)}", accent = true) { onDownload(spec) }
                if (metered) {
                    Spacer(Modifier.width(10.dp))
                    Text("uses mobile data", color = Warn, fontSize = 11.sp)
                }
            }
            is ModelState.Downloading -> {
                Progress(state.bytesDone, state.bytesTotal)
                Spacer(Modifier.height(8.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        "${pct(state.bytesDone, state.bytesTotal)}% • ${formatBytes(state.bytesDone)} / ${formatBytes(state.bytesTotal)}",
                        color = TextSecondary,
                        fontSize = 11.sp,
                        modifier = Modifier.weight(1f),
                    )
                    PillButton("Pause", accent = false) { onPause(spec) }
                }
            }
            is ModelState.Paused -> {
                Progress(state.bytesDone, state.bytesTotal)
                Spacer(Modifier.height(8.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        "Paused • ${formatBytes(state.bytesDone)} / ${formatBytes(state.bytesTotal)}",
                        color = TextSecondary,
                        fontSize = 11.sp,
                        modifier = Modifier.weight(1f),
                    )
                    PillButton("Resume", accent = true) { onDownload(spec) }
                    Spacer(Modifier.width(8.dp))
                    DeleteButton(spec, state, onDelete)
                }
            }
            is ModelState.Verifying -> {
                Progress(state.bytesDone, state.bytesTotal)
                Spacer(Modifier.height(8.dp))
                Text(
                    "Verifying checksum… ${pct(state.bytesDone, state.bytesTotal)}%",
                    color = TextSecondary,
                    fontSize = 11.sp,
                )
            }
            is ModelState.Ready -> Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.CheckCircle, null, tint = Accent, modifier = Modifier.size(15.dp))
                Spacer(Modifier.width(6.dp))
                Text("On device", color = TextPrimary, fontSize = 12.sp, modifier = Modifier.weight(1f))
                DeleteButton(spec, state, onDelete)
            }
            is ModelState.Failed -> {
                Text(state.reason, color = Err, fontSize = 11.sp, lineHeight = 15.sp)
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    PillButton("Retry", accent = true) { onDownload(spec) }
                    DeleteButton(spec, state, onDelete)
                }
            }
        }
    }
}

// Two-tap delete: the first tap arms the red confirm; any model-state change disarms it.
@Composable
private fun DeleteButton(spec: ModelSpec, state: ModelState, onDelete: (ModelSpec) -> Unit) {
    var arming by remember(spec.id, state) { mutableStateOf(false) }
    Text(
        if (arming) "Confirm delete" else "Delete",
        color = if (arming) OnAccent else TextSecondary,
        fontSize = 12.sp,
        fontWeight = FontWeight.Medium,
        modifier = Modifier
            .background(if (arming) Err else SurfaceHi, RoundedCornerShape(999.dp))
            .clickableNoRipple { if (arming) onDelete(spec) else arming = true }
            .padding(horizontal = 14.dp, vertical = 7.dp),
    )
}

@Composable
private fun Progress(done: Long, total: Long) {
    LinearProgressIndicator(
        progress = { if (total > 0) (done.toFloat() / total).coerceIn(0f, 1f) else 0f },
        modifier = Modifier.fillMaxWidth().height(6.dp),
        color = Accent,
        trackColor = SurfaceHi,
    )
}

private fun pct(done: Long, total: Long): Int = if (total > 0) (done * 100 / total).toInt() else 0
