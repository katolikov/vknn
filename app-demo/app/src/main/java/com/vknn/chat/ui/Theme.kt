package com.vknn.chat.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// Dark palette: true-black canvas, elevated near-black surfaces, blue accent.
val Bg = Color(0xFF000000)
val Surface = Color(0xFF141416)
val SurfaceHi = Color(0xFF1B1B1E)
val Accent = Color(0xFF3E9BFF)
val AccentDim = Color(0xFF0E2338)
val OnAccent = Color(0xFFFFFFFF)
val TextPrimary = Color(0xFFEDEDED)
val TextSecondary = Color(0xFF8A8A8E)
val Line = Color(0xFF2A2A2C)
val Warn = Color(0xFFE0B34D)
val Err = Color(0xFFE07A7A)

private val Scheme = darkColorScheme(
    primary = Accent,
    onPrimary = OnAccent,
    background = Bg,
    onBackground = TextPrimary,
    surface = Surface,
    onSurface = TextPrimary,
    surfaceVariant = SurfaceHi,
    onSurfaceVariant = TextSecondary,
    outline = Line,
)

@Composable
fun VknnChatTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = Scheme, content = content)
}
