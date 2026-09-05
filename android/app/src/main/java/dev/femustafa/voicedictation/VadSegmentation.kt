package dev.femustafa.voicedictation

import com.k2fsa.sherpa.onnx.Vad
import kotlin.math.min

/**
 * A speech utterance returned by a [VadLike]: its start sample index (16 kHz) and the
 * normalized [-1,1] float samples (the scale sherpa's Silero pipeline consumes).
 */
internal data class SpeechUtterance(val start: Int, val samples: FloatArray) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SpeechUtterance) return false
        return start == other.start && samples.contentEquals(other.samples)
    }

    override fun hashCode(): Int = 31 * start + samples.contentHashCode()
}

/**
 * The minimal surface of sherpa-onnx's `com.k2fsa.sherpa.onnx.Vad` used by
 * [segmentAudioWithVad], so JVM unit tests can drive segmentation with a fake without
 * loading the sherpa native lib. The production impl is [SherpaVad].
 */
internal interface VadLike {
    fun acceptWaveform(samples: FloatArray)
    fun isSpeechDetected(): Boolean
    fun isEmpty(): Boolean
    fun front(): SpeechUtterance?
    fun pop()
    fun flush()
}

/**
 * Production [VadLike] wrapping the sherpa-onnx Silero [Vad]. Loads from an absolute
 * file path in filesDir (like the other engines' models), so the AssetManager arg is
 * null (matching `OfflineRecognizer(null, ...)`).
 */
internal class SherpaVad(private val vad: Vad) : VadLike {
    override fun acceptWaveform(samples: FloatArray) = vad.acceptWaveform(samples)
    override fun isSpeechDetected(): Boolean = vad.isSpeechDetected()
    override fun isEmpty(): Boolean = vad.empty()
    override fun front(): SpeechUtterance? =
        vad.front()?.let { SpeechUtterance(it.start, it.samples) }
    override fun pop() = vad.pop()
    override fun flush() = vad.flush()
}

/**
 * Feed a full PCM clip to [vad] in [window]-sample chunks and collect the speech
 * utterances it emits — both live (drained the moment [VadLike.isSpeechDetected]
 * reports one) and via the trailing [VadLike.flush] for buffered tail audio.
 *
 * Faithful port of the canonical sherpa-onnx java VAD loop
 * (`VadNonStreamingDolphinCtc.java`): push 512-sample windows, drain any queued
 * utterances, then flush once at the end and drain again.
 */
internal fun segmentAudioWithVad(
    samples: FloatArray,
    vad: VadLike,
    window: Int = VAD_WINDOW,
): List<SpeechUtterance> {
    val out = mutableListOf<SpeechUtterance>()
    var i = 0
    while (i < samples.size) {
        val end = min(i + window, samples.size)
        vad.acceptWaveform(samples.copyOfRange(i, end))
        if (vad.isSpeechDetected()) {
            while (!vad.isEmpty()) {
                vad.front()?.let { out += it }
                vad.pop()
            }
        }
        i = end
    }
    vad.flush()
    while (!vad.isEmpty()) {
        vad.front()?.let { out += it }
        vad.pop()
    }
    return out
}

/** The Silero VAD window: the graph consumes this many samples per `acceptWaveform`. */
internal const val VAD_WINDOW = 512