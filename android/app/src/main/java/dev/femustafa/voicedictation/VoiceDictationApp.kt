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
        WhisperManager(filesDir, AarWhisperEngine(this)).also { mgr ->
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

        fun from(context: Context): VoiceDictationApp =
            context.applicationContext as VoiceDictationApp
    }
}