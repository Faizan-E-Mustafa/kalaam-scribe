package dev.femustafa.voicedictation

/**
 * Opaque handle to a loaded engine model. The WhisperManager never sees the
 * concrete engine/model type, so it is unit-testable on the JVM without a device.
 */
interface WhisperModelRef

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
     */
    suspend fun transcribe(
        model: WhisperModelRef,
        audioPath: String,
        languageMode: LanguageMode,
    ): String

    /** Unload [model] and free its memory. */
    fun release(model: WhisperModelRef)
}
