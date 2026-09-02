package dev.femustafa.voicedictation

import androidx.annotation.VisibleForTesting
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Guards the resident-model lifecycle (ADR 0004): one Model stays loaded in memory
 * across repeated dictations and is reused, never reloaded per inference. It loads
 * on cold start / first use, and reloads only on [switchTo]. The audio pipeline is
 * serialized so a transcription cannot start while one is already running.
 *
 * The manager talks only to the [WhisperEngine] seam, so its logic is unit-testable
 * on the dev machine without a device.
 */
class WhisperManager(
    private val baseDir: java.io.File,
    private val engine: WhisperEngine,
) {
    sealed interface Status {
        data object Idle : Status
        data object Loading : Status
        data object Transcribing : Status
    }

    private val _status = MutableStateFlow<Status>(Status.Idle)
    val status: StateFlow<Status> = _status.asStateFlow()

    private val _languageCode = MutableStateFlow<String?>(null)
    val languageCode: StateFlow<String?> = _languageCode.asStateFlow()

    private var resident: WhisperModelRef? = null
    private var residentModel: Model? = null

    private val mutation = Mutex()
    private val transcriptionMutex = Mutex()

    /** The currently resident [Model], or null before the first load. */
    val currentModel: Model?
        get() = residentModel

    /** Used at startup to reflect the default Model when its file already exists (lazy load). */
    fun setInitialModelIfNone(model: Model) {
        // For both formats the fileName is the loadable model file: a GGML .bin for
        // GGML entries, or the ONNX encoder .onnx/.int8.onnx for ONNX entries.
        val present = java.io.File(baseDir, model.fileName).exists()
        if (residentModel == null && present) {
            residentModel = model
        }
    }

    /**
     * Set the user-picked language code (a whisper-supported code, or null/blank
     * for auto-detect). Applies only to Models that [Model.canOverrideLanguage];
     * fixed-English and Roman-Urdu Models always use their declared mode.
     */
    fun setLanguageCode(code: String?) {
        _languageCode.value = code?.takeIf { it.isNotBlank() }
    }

    fun absolutePath(model: Model): String =
        java.io.File(baseDir, model.fileName).absolutePath

    private suspend fun doTranscribe(audioPath: String): String {
        val model = ensureLoaded() ?: error("no resident model to transcribe")
        val resident = residentModel!!
        val languageMode = resident.languageMode
        // A user code may override only multilingual (non-Roman-Urdu) Models.
        val language = resident.canOverrideLanguage.takeIf { it }?.let {
            _languageCode.value
        }
        _status.value = Status.Transcribing
        try {
            return engine.transcribe(model, audioPath, languageMode, language)
        } finally {
            _status.value = Status.Idle
        }
    }

    /**
     * Transcribe [audioPath] using the resident Model, loading it first if needed.
     * The resident Model is kept in memory and reused across calls: repeated
     * calls do NOT reload it. Serialized so concurrent transcribe calls queue up
     * rather than run in parallel, and a second dictation cannot start mid-run.
     */
    suspend fun transcribe(audioPath: String): String =
        transcriptionMutex.withLock { doTranscribe(audioPath) }

    /**
     * Switch the resident Model: unload the current one and load [newModel].
     * Noticeably slower than dictation (a new file is loaded). Waits for any
     * in-flight transcription to finish so a model is never unloaded mid-run.
     */
    suspend fun switchTo(newModel: Model) {
        transcriptionMutex.withLock {
            mutation.withLock {
                // Skip the expensive reload when the requested model is already
                // the resident one (e.g. the user re-taps the selected row).
                if (residentModel?.id == newModel.id && resident != null) return
                _status.value = Status.Loading
                try {
                    unloadLocked()
                    val ref = engine.load(absolutePath(newModel))
                    resident = ref
                    residentModel = newModel
                } finally {
                    _status.value = Status.Idle
                }
            }
        }
    }

    /** Unload the resident Model and free its memory (e.g. on app shutdown). */
    suspend fun shutdown() {
        mutation.withLock {
            unloadLocked()
        }
    }

    private suspend fun ensureLoaded(): WhisperModelRef? {
        mutation.withLock {
            if (resident != null) return resident
            val model = residentModel ?: return null
            _status.value = Status.Loading
            try {
                resident = engine.load(absolutePath(model))
            } finally {
                _status.value = Status.Idle
            }
            return resident
        }
    }

    private fun unloadLocked() {
        resident?.let { engine.release(it) }
        resident = null
    }

    @VisibleForTesting
    internal fun residentHandle(): WhisperModelRef? = resident
}
