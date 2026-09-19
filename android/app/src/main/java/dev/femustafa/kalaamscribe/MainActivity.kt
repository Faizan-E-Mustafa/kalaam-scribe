package dev.femustafa.kalaamscribe

import android.os.Bundle
import android.os.SystemClock
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import dev.femustafa.kalaamscribe.ui.theme.KalaamTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        val splashScreen = installSplashScreen()
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        // Keep splash screen visible for at most 400ms to reduce the time it's shown.
        val startTime = SystemClock.uptimeMillis()
        splashScreen.setKeepOnScreenCondition {
            SystemClock.uptimeMillis() - startTime < 400
        }
        setContent {
            KalaamTheme {
                RootApp()
            }
        }
    }
}
