package dev.femustafa.voicedictation

import android.app.Application
import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * One entry in the history list shown on the dictation screen: the transcript
 * text plus the time it was dictated (HH:mm) and which Model produced it
 * (mirrors the `History` rows in ui-preview.html).
 */
data class DictationHistoryItem(
    val text: String,
    val time: String,
    val modelId: String,
)

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
    val whisper: WhisperManager by lazy {
        WhisperManager(filesDir, DualFormatWhisperEngine(this)).also { mgr ->
            // Reflect the user's last-chosen Model (or the catalog default) as the
            // resident Model when its file is already on disk, so the top chip shows
            // it and transcription can start without a picker tap. Load is lazy.
            mgr.setInitialModelIfNone(initialModel().model)
            // Restore the persisted language-code selection.
            mgr.setLanguageCode(prefs.getString(KEY_LANGUAGE_CODE, null))
        }
    }

    private val prefs: android.content.SharedPreferences
        get() = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    /** The last Model the user selected (persisted), or null if never chosen. */
    val persistedModelId: String?
        get() = prefs.getString(KEY_MODEL_ID, null)

    /** The Model to start with: the persisted user selection if known, else the catalog default. */
    fun initialModel(): CatalogEntry =
        persistedModelId?.let { ModelCatalog.byId(it) } ?: ModelCatalog.default

    /** Persist the user's chosen Model so it is restored on the next launch. */
    fun setSelectedModelId(id: String) {
        prefs.edit().putString(KEY_MODEL_ID, id).apply()
    }

    /** The user's chosen model-file format (GGML whisper.cpp default), or the default. */
    val modelFormat: ModelFormat
        get() {
            val stored = prefs.getString(KEY_MODEL_FORMAT, null)
                ?: return ModelFormat.GGML
            return try {
                ModelFormat.valueOf(stored)
            } catch (_: IllegalArgumentException) {
                ModelFormat.GGML
            }
        }

    /** Persist the chosen model-file format. */
    fun setModelFormat(format: ModelFormat) {
        prefs.edit().putString(KEY_MODEL_FORMAT, format.name).apply()
    }

    /** The user's chosen transcription mode (batch whole-clip default), or the default. */
    val transcriptionMode: TranscriptionMode
        get() {
            val stored = prefs.getString(KEY_TRANSCRIPTION_MODE, null)
                ?: return TranscriptionMode.Batch
            return try {
                TranscriptionMode.valueOf(stored)
            } catch (_: IllegalArgumentException) {
                TranscriptionMode.Batch
            }
        }

    /** Persist the chosen transcription mode. */
    fun setTranscriptionMode(mode: TranscriptionMode) {
        prefs.edit().putString(KEY_TRANSCRIPTION_MODE, mode.name).apply()
    }

    /** Persist + publish a user-picked whisper language code (null/blank = auto). */
    fun setLanguageCode(code: String?) {
        prefs.edit().putString(KEY_LANGUAGE_CODE, code).apply()
        whisper.setLanguageCode(code)
    }

    private val _recording = MutableStateFlow(false)
    val recording: StateFlow<Boolean> = _recording.asStateFlow()

    private val _transcribing = MutableStateFlow(false)
    val transcribing: StateFlow<Boolean> = _transcribing.asStateFlow()

    private val _transcript = MutableStateFlow<String?>(null)
    val transcript: StateFlow<String?> = _transcript.asStateFlow()

    private val _lastTranscriptionMs = MutableStateFlow<Long?>(null)
    val lastTranscriptionMs: StateFlow<Long?> = _lastTranscriptionMs.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    private val _history = MutableStateFlow<List<DictationHistoryItem>>(emptyList())
    val history: StateFlow<List<DictationHistoryItem>> = _history.asStateFlow()

    fun setRecording(value: Boolean) {
        _recording.value = value
    }

    fun setTranscribing(value: Boolean) {
        _transcribing.value = value
    }

    fun setTranscript(value: String?) {
        _transcript.value = value
    }

    fun setLastTranscriptionMs(ms: Long) {
        _lastTranscriptionMs.value = ms
    }

    /** Called when a fresh recording starts: clear the previous result + timing. */
    fun onRecordingStarted() {
        _lastTranscriptionMs.value = null
    }

    fun setError(value: String?) {
        _error.value = value
    }

    fun appendHistory(item: DictationHistoryItem) {
        _history.value = (listOf(item) + _history.value).take(10)
    }

    companion object {
        private const val PREFS_NAME = "voice_dictation"
        private const val KEY_LANGUAGE_CODE = "language_code"
        private const val KEY_MODEL_ID = "model_id"
        private const val KEY_MODEL_FORMAT = "model_format"
        private const val KEY_WHISPER_THREADS = "whisper_threads"
        private const val KEY_TRANSCRIPTION_MODE = "transcription_mode"
        /** Default thread count; auto-select on first run is capped at this value. */
        const val DEFAULT_WHISPER_THREADS = 4

        // VAD parameter defaults (aligned with sherpa-onnx reference)
        private const val KEY_VAD_THRESHOLD = "vad_threshold"
        private const val KEY_VAD_MIN_SILENCE = "vad_min_silence"
        private const val KEY_VAD_MIN_SPEECH = "vad_min_speech"
        private const val KEY_VAD_MAX_SPEECH = "vad_max_speech"
        private const val KEY_DOLPHIN_BEAM_SIZE = "dolphin_beam_size"
        const val DEFAULT_VAD_THRESHOLD = 0.25f
        const val DEFAULT_VAD_MIN_SILENCE = 0.5f
        const val DEFAULT_VAD_MIN_SPEECH = 0.5f
        const val DEFAULT_VAD_MAX_SPEECH = 30.0f
        const val DEFAULT_DOLPHIN_BEAM_SIZE = 1

        fun from(context: Context): VoiceDictationApp =
            context.applicationContext as VoiceDictationApp
    }

    /** The persisted thread count, auto-selecting on first run (capped at [DEFAULT_WHISPER_THREADS]). */
    fun whisperThreads(): Int {
        if (!prefs.contains(KEY_WHISPER_THREADS)) {
            val auto = minOf(DEFAULT_WHISPER_THREADS, Runtime.getRuntime().availableProcessors())
            prefs.edit().putInt(KEY_WHISPER_THREADS, auto).apply()
        }
        return prefs.getInt(KEY_WHISPER_THREADS, DEFAULT_WHISPER_THREADS)
    }

    /** Persist the user-chosen thread count (e.g. from the Settings screen). */
    fun setWhisperThreads(count: Int) {
        prefs.edit().putInt(KEY_WHISPER_THREADS, count).apply()
    }

    // ---- VAD parameters (persisted, shared across all engines) ----

    fun vadThreshold(): Float =
        prefs.getString(KEY_VAD_THRESHOLD, DEFAULT_VAD_THRESHOLD.toString())?.toFloat()
            ?: DEFAULT_VAD_THRESHOLD

    fun setVadThreshold(v: Float) {
        prefs.edit().putString(KEY_VAD_THRESHOLD, v.toString()).apply()
    }

    fun vadMinSilence(): Float =
        prefs.getString(KEY_VAD_MIN_SILENCE, DEFAULT_VAD_MIN_SILENCE.toString())?.toFloat()
            ?: DEFAULT_VAD_MIN_SILENCE

    fun setVadMinSilence(v: Float) {
        prefs.edit().putString(KEY_VAD_MIN_SILENCE, v.toString()).apply()
    }

    fun vadMinSpeech(): Float =
        prefs.getString(KEY_VAD_MIN_SPEECH, DEFAULT_VAD_MIN_SPEECH.toString())?.toFloat()
            ?: DEFAULT_VAD_MIN_SPEECH

    fun setVadMinSpeech(v: Float) {
        prefs.edit().putString(KEY_VAD_MIN_SPEECH, v.toString()).apply()
    }

    fun vadMaxSpeech(): Float =
        prefs.getString(KEY_VAD_MAX_SPEECH, DEFAULT_VAD_MAX_SPEECH.toString())?.toFloat()
            ?: DEFAULT_VAD_MAX_SPEECH

    fun setVadMaxSpeech(v: Float) {
        prefs.edit().putString(KEY_VAD_MAX_SPEECH, v.toString()).apply()
    }

    fun dolphinBeamSize(): Int =
        prefs.getInt(KEY_DOLPHIN_BEAM_SIZE, DEFAULT_DOLPHIN_BEAM_SIZE)

    fun setDolphinBeamSize(v: Int) {
        prefs.edit().putInt(KEY_DOLPHIN_BEAM_SIZE, v).apply()
    }
}
