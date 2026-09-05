package dev.femustafa.voicedictation

/**
 * Interface for a component that accumulates audio samples and can produce
 * a complete Wave file representation.
 */
interface WaveWriter {
    /**
     * Accepts a chunk of audio samples to be added to the accumulated audio.
     * @param samples The audio samples (16 kHz mono 16-bit PCM).
     */
    fun accept(samples: ShortArray)

    /**
     * Flushes any accumulated audio and returns it as a [Wave] object.
     * After this call, the writer should be in a state to accept new audio
     * or be considered finalized.
     * @return A [Wave] object containing the accumulated samples and sample rate, or null if no audio was accumulated.
     */
    fun flush(): Wave?

    /**
     * Represents a simple Wave audio container.
     * @param samples The audio samples as a FloatArray.
     * @param sampleRate The sample rate of the audio.
     */
    data class Wave(val samples: FloatArray, val sampleRate: Int)
}
