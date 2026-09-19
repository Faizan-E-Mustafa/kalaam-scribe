package dev.femustafa.kalaamscribe

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel

/**
 * Top-level composable: owns the [WhisperManager] (production wiring: AAR engine +
 * app-private storage) and the [ModelPickerViewModel], shows the dictation
 * surface, and hosts the settings bottom sheet over it.
 */
@Composable
fun RootApp() {
    val context = LocalContext.current.applicationContext
    val app = KalaamApp.from(context)
    val manager = app.whisper
    val pickerViewModel: ModelPickerViewModel = viewModel()
    val dictationViewModel: DictationViewModel = viewModel()

    var showSettings by remember { mutableStateOf(false) }
    // First run: ask the dictation language before showing the app, so the model
    // list can be filtered by it. Stays on-screen until the user taps Continue,
    // even if a language was already picked from the dropdown.
    var showOnboarding by remember { mutableStateOf(!app.onboardingComplete()) }

    if (showOnboarding) {
        LanguageOnboardingScreen(
            viewModel = pickerViewModel,
            onSelect = { entry -> pickerViewModel.select(entry, manager) },
            onDownload = { entry -> pickerViewModel.download(entry, manager) },
            onLanguageChange = { code -> app.setLanguageCode(code) },
            onContinue = {
                app.completeOnboarding()
                showOnboarding = false
            },
        )
    } else {
        DictationScreen(
            viewModel = dictationViewModel,
            onOpenSettings = {
                pickerViewModel.refresh()
                showSettings = true
            },
        )
        if (showSettings) {
            AdvancedSettingsSheet(
                viewModel = pickerViewModel,
                manager = manager,
                onClose = { showSettings = false },
            )
        }
    }
}