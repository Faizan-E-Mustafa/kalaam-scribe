package dev.femustafa.voicedictation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel

/**
 * Top-level composable: owns the [WhisperManager] (production wiring: AAR engine +
 * app-private storage) and the [ModelPickerViewModel], and switches between the
 * dictation surface and the model picker.
 */
@Composable
fun RootApp() {
    val context = LocalContext.current.applicationContext
    val app = VoiceDictationApp.from(context)
    val manager = app.whisper
    val pickerViewModel: ModelPickerViewModel = viewModel()
    val dictationViewModel: DictationViewModel = viewModel()

    var showPicker by remember { mutableStateOf(false) }
    var showSettings by remember { mutableStateOf(false) }
    // First run: ask the dictation language before showing the app, so the model
    // list can be filtered by it. Dismissed permanently once a language is chosen.
    var showOnboarding by remember { mutableStateOf(!app.hasSelectedLanguage()) }

    if (showOnboarding) {
        LanguageOnboardingScreen(
            onContinue = { code ->
                app.setLanguageCode(code)
                showOnboarding = false
            },
        )
    } else if (showPicker) {
        ModelPickerScreen(
            viewModel = pickerViewModel,
            onSelect = { entry ->
                pickerViewModel.select(entry, manager)
            },
            onDownload = { entry -> pickerViewModel.download(entry) },
            onClose = { showPicker = false },
        )
    } else if (showSettings) {
        SettingsScreen(onBack = { showSettings = false })
    } else {
        DictationScreen(
            viewModel = dictationViewModel,
            currentModel = manager.currentModel,
            onOpenPicker = {
                pickerViewModel.refresh()
                showPicker = true
            },
            onOpenSettings = { showSettings = true },
        )
    }
}
