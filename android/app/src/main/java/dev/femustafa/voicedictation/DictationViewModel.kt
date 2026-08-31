package dev.femustafa.voicedictation

import android.app.Application
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelProvider.AndroidViewModelFactory.Companion.APPLICATION_KEY
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File

/**
 * Orchestrates the end-to-end dictation flow (record -> transcribe -> clipboard),
 * the port of dictate.sh. Owns [AudioRecorder] and drives [WhisperManager] with the
 * resident model, then writes the transcript to the clipboard and posts a
 * notification. Added to ticket 12; the line between recording and transcription
 * is the seam verified end-to-end with the resident model (no reload per dictation).
 */
class DictationViewModel(
    application: Application,
    private val whisper: WhisperManager,
) : AndroidViewModel(application) {

    private val filesDir = application.filesDir
    private val recorder = AudioRecorder()

    private val _recording = MutableStateFlow(false)
    val recording: StateFlow<Boolean> = _recording.asStateFlow()

    private val _transcribing = MutableStateFlow(false)
    val transcribing: StateFlow<Boolean> = _transcribing.asStateFlow()

    private val _transcript = MutableStateFlow<String?>(null)
    val transcript: StateFlow<String?> = _transcript.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    private var recordJob: Job? = null

    /** Toggle dictation. Mic permission must already be granted by the caller. */
    fun toggleRecording() {
        if (_recording.value) {
            stopAndTranscribe()
        } else {
            startRecording()
        }
    }

    private fun startRecording() {
        recordJob = viewModelScope.launch(Dispatchers.IO) {
            val out = File(filesDir, "dictation.wav")
            try {
                recorder.start(out)
                NotificationHelper.ensureChannel(getApplication())
                NotificationHelper.showRecording(getApplication())
                _error.value = null
                _transcript.value = null
                _recording.value = true
            } catch (t: Throwable) {
                _error.value = t.message ?: "recording failed"
            }
        }
    }

    private fun stopAndTranscribe() {
        recordJob = viewModelScope.launch(Dispatchers.IO) {
            recorder.stop()
            NotificationHelper.cancelRecording(getApplication())
            _recording.value = false

            if (whisper.currentModel == null) {
                _error.value = "No model loaded — pick a model first"
                _transcribing.value = false
                return@launch
            }

            val audioPath = File(filesDir, "dictation.wav").absolutePath
            _transcribing.value = true
            _error.value = null
            try {
                val text = whisper.transcribe(audioPath)
                handleTranscript(text)
            } catch (t: Throwable) {
                _error.value = t.message ?: "transcription failed"
            } finally {
                _transcribing.value = false
            }
        }
    }

    private fun handleTranscript(text: String) {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) {
            _error.value = "No speech detected — please try again"
            return
        }
        val clipboard = getApplication<Application>()
            .getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("dictation", trimmed))
        _transcript.value = trimmed
        NotificationHelper.showCopied(getApplication(), trimmed)
    }

    class Factory(
        private val whisper: WhisperManager,
    ) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
            val app = extras[APPLICATION_KEY]
                ?: error("no application in creation extras")
            return DictationViewModel(app, whisper) as T
        }
    }
}
