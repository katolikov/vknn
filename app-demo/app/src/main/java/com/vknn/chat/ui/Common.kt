package com.vknn.chat.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// A no-ripple clickable (a flat, ripple-free feel) that avoids pulling the material ripple indication setup.
internal fun Modifier.clickableNoRipple(onClick: () -> Unit): Modifier =
    this.clickable(
        interactionSource = MutableInteractionSource(),
        indication = null,
        onClick = onClick,
    )

@Composable
internal fun PrimaryButton(label: String, onClick: () -> Unit) {
    Box(
        Modifier
            .background(Accent, RoundedCornerShape(999.dp))
            .clickableNoRipple(onClick)
            .padding(horizontal = 28.dp, vertical = 13.dp),
        contentAlignment = Alignment.Center,
    ) { Text(label, color = OnAccent, fontSize = 15.sp, fontWeight = FontWeight.Medium) }
}

@Composable
internal fun PillButton(label: String, accent: Boolean, onClick: () -> Unit) {
    Text(
        label,
        color = if (accent) OnAccent else TextPrimary,
        fontSize = 12.sp,
        fontWeight = FontWeight.Medium,
        modifier = Modifier
            .background(if (accent) Accent else SurfaceHi, RoundedCornerShape(999.dp))
            .clickableNoRipple(onClick)
            .padding(horizontal = 14.dp, vertical = 7.dp),
    )
}
