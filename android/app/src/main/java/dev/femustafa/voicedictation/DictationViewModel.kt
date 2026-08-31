package dev.femustafa.voicedictation

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File

/**
 * Backs the dictation surface: owns [AudioRecorder] and produces a recording file.
 * Transcription is added in a later ticket; for now the output is a recorded WAV
 * path. Recording runs off the main thread (the recorder's read loop is blocking).
 */
class DictationViewModel(application: Application) : AndroidViewModel(application) {

    private val filesDir = application.filesDir
    private val recorder = AudioRecorder()

    private val _recording = MutableStateFlow(false)
    val recording: StateFlow<Boolean> = _recording.asStateFlow()

    private val _lastFile = MutableStateFlow<String?>(null)
    val lastFile: StateFlow<String?> = _lastFile.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    /** Toggle capture. The caller must have already secured RECORD_AUDIO permission. */
    fun toggleRecording() {
        viewModelScope.launch(Dispatchers.IO) {
            if (_recording.value) {
                recorder.stop()
                _recording.value = false
                _lastFile.value = lastOutput?.absolutePath
            } else {
                val out = File(filesDir, "dictation.wav")
                try {
                    recorder.start(out)
                    lastOutput = out
                    _error.value = null
                    _recording.value = true
                } catch (t: Throwable) {
                    _error.value = t.message ?: "recording failed"
                }
            }
        }
    }

    private var lastOutput: File? = null
}
