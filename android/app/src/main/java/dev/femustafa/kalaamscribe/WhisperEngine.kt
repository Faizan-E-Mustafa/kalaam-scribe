package dev.femustafa.kalaamscribe

import kotlinx.coroutines.flow.SharedFlow

/**
 * Opaque handle to a loaded engine model. The WhisperManager never sees the
 * concrete engine/model type, so it is unit-testable on the JVM without a device.
 */
interface WhisperModelRef

/**
 * A live, recording-spanning transcription session (simulated streaming ASR, ticket 31):
 * mic samples are pushed in via [accept] while recording, and each completed VAD utterance
 * is decoded independently and surfaced through [partials] in capture order. [flush] is
 * called at Stop; it decodes the trailing utterance(s) and returns the FULL joined
 * transcript (what the UI copies). [close] frees the session when recording is abandoned.
 *
 * Threading contract: [accept] must be cheap and non-blocking (it only feeds the VAD and
 * enqueues closed segments) so it can run on the mic capture thread; all model decode work
 * happens on the session's own worker, never on the capture thread. Implementations must
 * never touch the shared recognizer/ORT sessions from [accept].
 */
interface TranscriptionSession {
    /** Push captured samples (16 kHz mono 16-bit PCM) into the session. Non-blocking. */
    fun accept(samples: ShortArray)

    /**
     * The text of each completed utterance, in capture order. Never completes on its own;
     * the service collects it to grow the on-screen transcript while recording.
     */
    val partials: SharedFlow<String>

    /**
     * End of input: decodes the trailing in-progress utterance (VAD flush tail) and returns
     * the full joined transcript. [partialsSnapshot] is the transcript already accumulated
     * from [partials] emissions during recording — flush appends only the tail segments
     * that the decode worker has not yet emitted. Safe to call once; after returning the
     * session is done.
     */
    suspend fun flush(partialsSnapshot: String): String

    /** Abandon the session and free resources without a final decode. */
    fun close()
}

/**
 * The seam around the transcription engine. The production implementation wraps
 * the whisper.cpp AAR (`dev.ffmpegkit.whisper.Whisper`); unit tests supply a fake
 * that counts loads/transcribes so the lifecycle can be verified without a device.
 */
interface WhisperEngine {
    /** Load a model file into memory and return a handle to it. */
    suspend fun load(modelPath: String): WhisperModelRef

    /**
     * Transcribe [audioPath] with the given resident [model], applying the model's
     * [LanguageMode]. Returns the transcript text.
     *
     * @param language A user-picked, whisper-supported language code, or null to
     *   fall back to the model's declared mode (auto-detect or fixed English).
     */
    suspend fun transcribe(
        model: WhisperModelRef,
        audioPath: String,
        languageMode: LanguageMode,
        language: String?,
    ): String

    /**
     * Open a live [TranscriptionSession] for [model], or return null when the backend
     * cannot stream (e.g. the whisper.cpp GGML backend — its AAR has no incremental API).
     * The session borrows the resident model for its whole lifetime and must be closed
     * (or flushed) before the model is unloaded.
     */
    suspend fun createSession(
        model: WhisperModelRef,
        languageMode: LanguageMode,
        language: String?,
    ): TranscriptionSession? = null

    /** Unload [model] and free its memory. */
    fun release(model: WhisperModelRef)
}
