package dev.femustafa.voicedictation

import androidx.annotation.VisibleForTesting
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Guards the resident-model lifecycle (ADR 0004): one Model stays loaded in memory
 * across repeated dictations and is reused, never reloaded per inference. It loads
 * on cold start / first use, and reloads only on [switchTo]. The audio pipeline is
 * serialized so a transcription cannot start while one is already running.
 *
 * The manager talks only to the [WhisperEngine] seam, so its logic is unit-testable
 * on the dev machine without a device.
 */
class WhisperManager internal constructor(
    private val baseDir: java.io.File,
    private val engine: WhisperEngine,
    /** Builds the VAD used by [transcribeVad] (null when the VAD model is missing). */
    private val vadFactory: (() -> VadLike?)? = null,
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

    /**
     * Held for the whole lifetime of a streaming [TranscriptionSession]:
     * [switchTo]/[shutdown] acquire it too, so a resident model can never be unloaded
     * while a session that borrows it is still open (the service holds the session for
     * the full recording).
     */
    private val sessionMutex = Mutex()

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
        val (languageMode, language) = residentLanguage()
        _status.value = Status.Transcribing
        try {
            return engine.transcribe(model, audioPath, languageMode, language)
        } finally {
            _status.value = Status.Idle
        }
    }

    /**
     * VAD-segmented batch decode ([TranscriptionMode.BatchVad], vad-accuracy spec):
     * split [audioPath] with the VAD, merge consecutive utterances into <=[MERGE_MAX_SAMPLES]
     * (28 s) chunks, pad each chunk ([PAD_SAMPLES] both sides), and decode the chunks in
     * order with the resident engine. Longer per-decode context for long clips than a single
     * whole-clip pass, with chunk boundaries at real silence gaps.
     *
     * Falls back to the plain whole-clip [transcribe] when the VAD model is unavailable or
     * the audio cannot be read.
     */
    suspend fun transcribeVad(audioPath: String): String =
        transcriptionMutex.withLock { doTranscribeVad(audioPath) }

    private suspend fun doTranscribeVad(audioPath: String): String {
        val model = ensureLoaded() ?: error("no resident model to transcribe")
        val (languageMode, language) = residentLanguage()
        val vad = vadFactory?.invoke()
            ?: return engine.transcribe(model, audioPath, languageMode, language)
        val samples = try {
            AudioDecoder.decodeToMono16kSamples(audioPath)
        } catch (t: Throwable) {
            return engine.transcribe(model, audioPath, languageMode, language)
        }
        val chunks = segmentAudioWithVad(
            samples = samples,
            vad = vad,
            padding = PAD_SAMPLES,
            maxChunkSamples = MERGE_MAX_SAMPLES,
        )
        if (chunks.isEmpty()) return engine.transcribe(model, audioPath, languageMode, language)

        _status.value = Status.Transcribing
        val texts = mutableListOf<String>()
        try {
            chunks.forEachIndexed { index, chunk ->
                val chunkFile = java.io.File(baseDir, "vad_chunk_$index.wav")
                try {
                    AudioDecoder.writePcm16Wav(chunkFile, chunk.samples, chunk.sampleRate)
                    val text = engine.transcribe(model, chunkFile.absolutePath, languageMode, language).trim()
                    if (text.isNotBlank()) texts += text
                } finally {
                    chunkFile.delete()
                }
            }
        } finally {
            _status.value = Status.Idle
        }
        return texts.joinToString(" ")
    }

    /** The resident model's language mode plus the user's language override, if any. */
    private fun residentLanguage(): Pair<LanguageMode, String?> {
        val resident = residentModel!!
        val languageMode = resident.languageMode
        val language = resident.canOverrideLanguage.takeIf { it }?.let {
            _languageCode.value?.takeIf { code -> WhisperLanguages.supports(code) }
        }
        return languageMode to language
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
     * Open a live transcription session for the resident model (loading it first if
     * needed), or null when the model's backend cannot stream or no model is set. The
     * session borrows the resident model for its whole lifetime; [switchTo]/[shutdown]
     * wait until it is flushed or closed.
     */
    suspend fun createSession(): TranscriptionSession? {
        sessionMutex.lock()
        val session = try {
            val ref = ensureLoaded()
            if (ref == null) {
                null
            } else {
                val resident = residentModel!!
                val languageMode = resident.languageMode
                // Same language resolution as [doTranscribe]: only multilingual Models
                // take the user-selected code (and only real whisper codes; the synthetic
                // "Urdu (Roman)" code is never passed to an engine).
                val language = resident.canOverrideLanguage.takeIf { it }?.let {
                    _languageCode.value?.takeIf { code -> WhisperLanguages.supports(code) }
                }
                engine.createSession(ref, languageMode, language)
            }
        } catch (t: Throwable) {
            sessionMutex.unlock()
            throw t
        }
        if (session == null) {
            sessionMutex.unlock()
            return null
        }
        return SessionEndGuard(session) { sessionMutex.unlock() }
    }

    /**
     * Switch the resident Model: unload the current one and load [newModel].
     * Noticeably slower than dictation (a new file is loaded). Waits for any
     * in-flight transcription OR open streaming session to finish so a model is
     * never unloaded mid-run.
     */
    suspend fun switchTo(newModel: Model) {
        sessionMutex.withLock {
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
    }

    /** Unload the resident Model and free its memory (e.g. on app shutdown). */
    suspend fun shutdown() {
        sessionMutex.withLock {
            mutation.withLock {
                unloadLocked()
            }
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

/**
 * Releases [onEnd] exactly once when the wrapped session is flushed or closed, so
 * [WhisperManager.sessionMutex] never stays held by an abandoned recording.
 */
private class SessionEndGuard(
    private val delegate: TranscriptionSession,
    private val onEnd: () -> Unit,
) : TranscriptionSession {
    private val ended = AtomicBoolean(false)

    override fun accept(samples: ShortArray) = delegate.accept(samples)

    override val partials: SharedFlow<String> = delegate.partials

    override suspend fun flush(partialsSnapshot: String): String {
        val text = delegate.flush(partialsSnapshot)
        end()
        return text
    }

    override fun close() {
        delegate.close()
        end()
    }

    private fun end() {
        if (ended.compareAndSet(false, true)) onEnd()
    }
}
