package dev.femustafa.kalaamscribe.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.sp

private val LightColorScheme = lightColorScheme(
    primary = Color(0xFF1A6DB5),
    onPrimary = Color(0xFFFFFFFF),
    primaryContainer = Color(0xFFE8F1FB),
    onPrimaryContainer = Color(0xFF0B3B5E),
    secondary = Color(0xFF0B3B5E),
    onSecondary = Color(0xFFFFFFFF),
    secondaryContainer = Color(0xFFE8F1FB),
    onSecondaryContainer = Color(0xFF0B3B5E),
    tertiary = Color(0xFF188038),
    onTertiary = Color(0xFFFFFFFF),
    tertiaryContainer = Color(0xFFE6F4EA),
    onTertiaryContainer = Color(0xFF0D652D),
    background = Color(0xFFF4F6F9),
    onBackground = Color(0xFF1A1C1E),
    surface = Color(0xFFFFFFFF),
    onSurface = Color(0xFF1A1C1E),
    surfaceVariant = Color(0xFFF8F9FB),
    onSurfaceVariant = Color(0xFF5F6368),
    outline = Color(0xFFCFD6DE),
    error = Color(0xFFB3261E),
    onError = Color(0xFFFFFFFF),
    errorContainer = Color(0xFFFCE8E6),
    onErrorContainer = Color(0xFFA50E0E),
)

private val DarkColorScheme = darkColorScheme(
    primary = Color(0xFF8EC9FF),
    onPrimary = Color(0xFF04263F),
    primaryContainer = Color(0xFF173A54),
    onPrimaryContainer = Color(0xFFCDE6FF),
    secondary = Color(0xFFCDE6FF),
    onSecondary = Color(0xFF04263F),
    secondaryContainer = Color(0xFF173A54),
    onSecondaryContainer = Color(0xFFCDE6FF),
    tertiary = Color(0xFF79CF8F),
    onTertiary = Color(0xFF04263F),
    tertiaryContainer = Color(0xFF123D2A),
    onTertiaryContainer = Color(0xFFBCE9CB),
    background = Color(0xFF0F1216),
    onBackground = Color(0xFFE6E8EB),
    surface = Color(0xFF171B20),
    onSurface = Color(0xFFE6E8EB),
    surfaceVariant = Color(0xFF1D2228),
    onSurfaceVariant = Color(0xFFA0A8B1),
    outline = Color(0xFF38414C),
    error = Color(0xFFF2B8B5),
    onError = Color(0xFF04263F),
    errorContainer = Color(0xFF3A1D1C),
    onErrorContainer = Color(0xFFF2B8B5),
)

private val AppTypography = Typography().let { base ->
    base.copy(
        headlineSmall = base.headlineSmall.copy(fontSize = 20.sp),
        titleLarge = base.titleLarge.copy(fontSize = 22.sp),
        bodyLarge = base.bodyLarge.copy(lineHeight = 22.sp),
    )
}

@Composable
fun KalaamTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme
    MaterialTheme(
        colorScheme = colorScheme,
        typography = AppTypography,
        content = content,
    )
}
