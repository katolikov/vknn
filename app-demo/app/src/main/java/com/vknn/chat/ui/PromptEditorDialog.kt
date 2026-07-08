package com.vknn.chat.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties

/** How the counter under the field reads: informative, close to a hard limit, or past it. */
internal enum class PromptCounterTone { NEUTRAL, WARNING, ERROR }

/** A ready-made prompt the user can drop into the field. */
internal data class PromptPreset(val label: String, val prompt: String)

/**
 * The verdict on the prompt currently in the field, recomputed on every keystroke.
 * A non-null [rejectionReason] both disables Save and renders as the one-line explanation.
 */
internal data class PromptValidation(
    val counterText: String = "",
    val tone: PromptCounterTone = PromptCounterTone.NEUTRAL,
    val rejectionReason: String? = null,
) {
    val canSave: Boolean get() = rejectionReason == null
}

// The prompt editor shared by the Chat and VLM screens: a multiline field over the app's dark
// surfaces, a live counter, optional one-tap presets, and a Save gated on [validate]. The draft is
// local until Save, so Reset-then-Cancel leaves the persisted prompt untouched.
@Composable
internal fun PromptEditorDialog(
    title: String,
    explanation: String,
    initialPrompt: String,
    defaultPrompt: String,
    fieldPlaceholder: String,
    validate: (String) -> PromptValidation,
    onSave: (String) -> Unit,
    onDismiss: () -> Unit,
    presets: List<PromptPreset> = emptyList(),
    monospaceField: Boolean = false,
) {
    var draft by remember { mutableStateOf(initialPrompt) }
    val validation = validate(draft)
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Column(
            Modifier
                .padding(horizontal = 20.dp)
                .fillMaxWidth()
                .clip(RoundedCornerShape(20.dp))
                .background(Surface)
                .padding(18.dp),
        ) {
            Text(title, color = TextPrimary, fontSize = 16.sp, fontWeight = FontWeight.Medium)
            Spacer(Modifier.height(7.dp))
            Text(explanation, color = TextSecondary, fontSize = 12.sp, lineHeight = 17.sp)
            Spacer(Modifier.height(14.dp))

            Box(
                Modifier
                    .fillMaxWidth()
                    .heightIn(min = 92.dp, max = 200.dp)
                    .clip(RoundedCornerShape(14.dp))
                    .background(Bg)
                    .border(1.dp, Line, RoundedCornerShape(14.dp))
                    .padding(12.dp),
            ) {
                if (draft.isEmpty()) Text(fieldPlaceholder, color = TextSecondary, fontSize = 13.sp, lineHeight = 18.sp)
                BasicTextField(
                    value = draft,
                    onValueChange = { draft = it },
                    textStyle = TextStyle(
                        color = TextPrimary,
                        fontSize = 13.sp,
                        lineHeight = 18.sp,
                        fontFamily = if (monospaceField) FontFamily.Monospace else FontFamily.Default,
                    ),
                    cursorBrush = SolidColor(Accent),
                    modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState()),
                )
            }

            if (validation.counterText.isNotEmpty()) {
                Spacer(Modifier.height(7.dp))
                Text(
                    validation.counterText,
                    color = when (validation.tone) {
                        PromptCounterTone.NEUTRAL -> TextSecondary
                        PromptCounterTone.WARNING -> Warn
                        PromptCounterTone.ERROR -> Err
                    },
                    fontSize = 11.sp,
                )
            }
            validation.rejectionReason?.let {
                Spacer(Modifier.height(5.dp))
                Text(it, color = Err, fontSize = 11.sp, lineHeight = 15.sp)
            }

            if (presets.isNotEmpty()) {
                Spacer(Modifier.height(14.dp))
                Text("Presets", color = TextSecondary, fontSize = 11.sp)
                Spacer(Modifier.height(7.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    presets.forEach { preset ->
                        PillButton(preset.label, accent = false) { draft = preset.prompt }
                    }
                }
            }

            Spacer(Modifier.height(18.dp))
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "Reset to default",
                    color = TextSecondary,
                    fontSize = 12.sp,
                    modifier = Modifier.clickableNoRipple { draft = defaultPrompt }.padding(vertical = 7.dp),
                )
                Spacer(Modifier.weight(1f))
                PillButton("Cancel", accent = false, onClick = onDismiss)
                Spacer(Modifier.width(8.dp))
                SaveButton(enabled = validation.canSave) {
                    onSave(draft)
                    onDismiss()
                }
            }
        }
    }
}

// Save reads as unavailable — not merely inert — while the prompt is rejected.
@Composable
private fun SaveButton(enabled: Boolean, onClick: () -> Unit) {
    Text(
        "Save",
        color = if (enabled) OnAccent else TextSecondary,
        fontSize = 12.sp,
        fontWeight = FontWeight.Medium,
        modifier = Modifier
            .background(if (enabled) Accent else SurfaceHi, RoundedCornerShape(999.dp))
            .clickableNoRipple { if (enabled) onClick() }
            .padding(horizontal = 18.dp, vertical = 7.dp),
    )
}
