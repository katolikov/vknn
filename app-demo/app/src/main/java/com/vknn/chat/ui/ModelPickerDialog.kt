package com.vknn.chat.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
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
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.vknn.chat.model.ModelChoice
import com.vknn.chat.model.formatBytes

// The model-variant picker shared by the Chat and VLM screens: one row per selectable .vxm for the
// mode — catalogue variants (fp16 / int4) plus ad-hoc files already on device — with the weight
// format and size up front. Picking a variant persists it and swaps the resident session; a
// variant that is not on device yet points at the Library to download it.
@Composable
internal fun ModelPickerDialog(
    title: String,
    choices: List<ModelChoice>,
    selectedKey: String,
    onSelect: (String) -> Unit,
    onOpenLibrary: () -> Unit,
    onDismiss: () -> Unit,
) {
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
            Spacer(Modifier.height(4.dp))
            Text(
                "The choice is remembered and applies right away — switching unloads the current model.",
                color = TextSecondary,
                fontSize = 11.sp,
                lineHeight = 15.sp,
            )
            Spacer(Modifier.height(12.dp))
            Column(
                Modifier.verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                for (choice in choices) {
                    ChoiceRow(choice, selected = choice.key == selectedKey) {
                        onSelect(choice.key)
                        onDismiss()
                    }
                }
            }
            Spacer(Modifier.height(14.dp))
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "Manage downloads in Library",
                    color = TextSecondary,
                    fontSize = 12.sp,
                    modifier = Modifier
                        .clickableNoRipple {
                            onDismiss()
                            onOpenLibrary()
                        }
                        .padding(vertical = 7.dp),
                )
                Spacer(Modifier.weight(1f))
                PillButton("Close", accent = false, onClick = onDismiss)
            }
        }
    }
}

@Composable
private fun ChoiceRow(choice: ModelChoice, selected: Boolean, onClick: () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(if (selected) SurfaceHi else Bg)
            .border(1.dp, if (selected) Accent else Line, RoundedCornerShape(14.dp))
            .clickableNoRipple(onClick)
            .padding(horizontal = 12.dp, vertical = 10.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                choice.displayName,
                color = TextPrimary,
                fontSize = 13.sp,
                fontWeight = FontWeight.Medium,
                modifier = Modifier.weight(1f),
            )
            if (selected) {
                Spacer(Modifier.width(6.dp))
                Icon(Icons.Filled.CheckCircle, "selected", tint = Accent, modifier = Modifier.size(15.dp))
            }
        }
        Spacer(Modifier.height(3.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            VariantBadge(choice.variant)
            Spacer(Modifier.width(8.dp))
            Text(formatBytes(choice.sizeBytes), color = TextSecondary, fontSize = 11.sp)
            Spacer(Modifier.weight(1f))
            Text(
                if (choice.onDevice) "on device" else "download in Library",
                color = if (choice.onDevice) Accent else Warn,
                fontSize = 11.sp,
            )
        }
    }
}

@Composable
private fun VariantBadge(variant: String) {
    Text(
        variant,
        color = Accent,
        fontSize = 10.sp,
        fontWeight = FontWeight.Medium,
        modifier = Modifier
            .background(AccentDim, RoundedCornerShape(999.dp))
            .padding(horizontal = 8.dp, vertical = 2.dp),
    )
}
