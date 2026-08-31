package dev.femustafa.voicedictation

import android.app.Application
import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * App-scoped singleton holder (an idiomatic Android pattern for sharing one
 * resident model and a UI-observable state bus across the Activity and the
 * [DictationService]). Only one [WhisperManager] exists for the whole process,
 * so the resident model is never double-loaded by the UI and the service
 * (ADR 0004).
 *
 * The service drives [recording]/[transcribing]/[transcript]/[error]; the
 * Activity observes them so the dictation screen stays in sync even when the
 * service runs in the background on the A50.
 */
class VoiceDictationApp : Application() {

    /** The single shared resident-model manager, created lazily on first use. */
    val whisper: WhisperManager by lazy { WhisperManager(filesDir, AarWhisperEngine(this)) }

    private val _recording = MutableStateFlow(false)
    val recording: StateFlow<Boolean> = _recording.asStateFlow()

    private val _transcribing = MutableStateFlow(false)
    val transcribing: StateFlow<Boolean> = _transcribing.asStateFlow()

    private val _transcript = MutableStateFlow<String?>(null)
    val transcript: StateFlow<String?> = _transcript.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    fun setRecording(value: Boolean) {
        _recording.value = value
    }

    fun setTranscribing(value: Boolean) {
        _transcribing.value = value
    }

    fun setTranscript(value: String?) {
        _transcript.value = value
    }

    fun setError(value: String?) {
        _error.value = value
    }

    companion object {
        fun from(context: Context): VoiceDictationApp =
            context.applicationContext as VoiceDictationApp
    }
}