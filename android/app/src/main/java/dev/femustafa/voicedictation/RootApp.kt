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
            viewModel = pickerViewModel,
            onSelect = { entry -> pickerViewModel.select(entry, manager) },
            onDownload = { entry -> pickerViewModel.download(entry, manager) },
            onLanguageChange = { code -> app.setLanguageCode(code) },
            onContinue = { showOnboarding = false },
        )
    } else if (showPicker) {
        ModelPickerScreen(
            viewModel = pickerViewModel,
            onSelect = { entry ->
                pickerViewModel.select(entry, manager)
            },
            onDownload = { entry -> pickerViewModel.download(entry, manager) },
            onClose = { showPicker = false },
        )
    } else if (showSettings) {
        SettingsScreen(onBack = { showSettings = false })
    } else {
        DictationScreen(
            viewModel = dictationViewModel,
            currentModelName = manager.currentModel?.let { ModelCatalog.byId(it.id)?.modelName },
            onOpenPicker = {
                pickerViewModel.refresh()
                showPicker = true
            },
            onOpenSettings = { showSettings = true },
        )
    }
}
