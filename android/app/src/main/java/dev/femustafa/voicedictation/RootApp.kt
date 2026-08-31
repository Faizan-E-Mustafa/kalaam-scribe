package dev.femustafa.voicedictation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.launch

/**
 * Top-level composable: owns the [WhisperManager] (production wiring: AAR engine +
 * app-private storage) and the [ModelPickerViewModel], and switches between the
 * dictation surface and the model picker.
 */
@Composable
fun RootApp() {
    val context = LocalContext.current.applicationContext
    val scope = rememberCoroutineScope()
    val manager = VoiceDictationApp.from(context).whisper
    val pickerViewModel: ModelPickerViewModel = viewModel()
    val dictationViewModel: DictationViewModel = viewModel()

    var showPicker by remember { mutableStateOf(false) }

    if (showPicker) {
        ModelPickerScreen(
            viewModel = pickerViewModel,
            onSelect = { entry ->
                scope.launch { pickerViewModel.select(entry, manager) }
            },
            onDownload = { entry -> pickerViewModel.download(entry) },
            onClose = { showPicker = false },
        )
    } else {
        DictationScreen(
            viewModel = dictationViewModel,
            currentModel = manager.currentModel,
            onOpenPicker = {
                pickerViewModel.refresh()
                showPicker = true
            },
        )
    }
}
