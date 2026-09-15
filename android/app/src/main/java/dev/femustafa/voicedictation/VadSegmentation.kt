package dev.femustafa.voicedictation

import com.k2fsa.sherpa.onnx.Vad
import kotlin.math.min

/**
 * A speech utterance returned by a [VadLike]: its start sample index (16 kHz) and the
 * normalized [-1,1] float samples (the scale sherpa's pipeline consumes).
 *
 * With segment padding enabled, [samples] is the *padded* region handed to the ASR
 * decoder (VAD boundary ± pre/post padding) and [start] is the sample index of the
 * padded region's first sample within the source audio.
 */
internal data class SpeechSegment(val start: Int, val samples: FloatArray, val sampleRate: Int = SAMPLE_RATE) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SpeechSegment) return false
        return start == other.start && samples.contentEquals(other.samples) && sampleRate == other.sampleRate
    }

    override fun hashCode(): Int {
        var result = start
        result = 31 * result + samples.contentHashCode()
        result = 31 * result + sampleRate
        return result
    }
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
    fun front(): SpeechSegment?
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
    override fun front(): SpeechSegment? = vad.front()?.let { SpeechSegment(it.start, it.samples, SAMPLE_RATE) }
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
 *
 * When [padding] > 0, each drained segment is expanded by [padding] samples on
 * both sides (clamped to the clip) so words at the VAD's exact silence-boundary
 * cuts are not clipped by the decoder (ADR 0006 deems this a future optimization;
 * see the vad-accuracy spec). When [maxChunkSamples] > 0, consecutive segments are
 * first merged into chunks under that many samples (WhisperX cut-and-merge), then
 * padded once at each merged chunk's outer edges. Defaults keep the historical
 * behavior.
 */
internal fun segmentAudioWithVad(
    samples: FloatArray,
    vad: VadLike,
    window: Int = VAD_WINDOW,
    padding: Int = 0,
    maxChunkSamples: Int = 0,
): List<SpeechSegment> {
    val out = mutableListOf<SpeechSegment>()
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
    // Pipeline: drain raw -> merge into <= maxChunkSamples chunks -> pad each chunk.
    val merged = if (maxChunkSamples > 0) mergeSegments(out, samples, maxChunkSamples) else out
    return if (padding > 0) padSegments(merged, samples, padding) else merged
}

/**
 * Expand each [SpeechSegment] in [segments] by [padding] samples of surrounding
 * source audio on both sides, clamped to [source]'s bounds. The resulting segment
 * starts at the padded slice's first sample and its samples come from [source]
 * (so post-padding holds the real trailing audio, not just the VAD's buffer).
 */
internal fun padSegments(
    segments: List<SpeechSegment>,
    source: FloatArray,
    padding: Int,
): List<SpeechSegment> =
    segments.map { seg ->
        val from = maxOf(0, seg.start - padding)
        val to = minOf(source.size, seg.start + seg.samples.size + padding)
        SpeechSegment(from, source.copyOfRange(from, to), seg.sampleRate)
    }

/**
 * Merge consecutive VAD [segments] into one continuous, source-backed chunk while
 * the chunk's total span stays under [maxSamples]. Chunk boundaries therefore fall
 * at real silence gaps (the gap between two merged segments is included in the
 * chunk's audio). A single segment that already exceeds [maxSamples] passes through
 * whole — merging never splits.
 *
 * The returned chunk is `SpeechSegment(first.start, source[start until lastEnd])`,
 * i.e. the full span from the first segment's start to the last segment's end,
 * clamped to [source]. Ordered; never splits or reorders. Greedy forward: an
 * oversized neighbor closes the current chunk rather than being absorbed.
 */
internal fun mergeSegments(
    segments: List<SpeechSegment>,
    source: FloatArray,
    maxSamples: Int,
): List<SpeechSegment> {
    if (maxSamples <= 0 || segments.size < 2) return segments
    val out = ArrayList<SpeechSegment>(segments.size)
    var i = 0
    while (i < segments.size) {
        val first = segments[i]
        var j = i + 1
        var last = first
        while (j < segments.size) {
            val next = segments[j]
            val spanEnd = next.start + next.samples.size
            if (spanEnd - first.start > maxSamples) break
            last = next
            j++
        }
        val to = minOf(source.size, last.start + last.samples.size)
        out += SpeechSegment(first.start, source.copyOfRange(first.start, to), first.sampleRate)
        i = j
    }
    return out
}

/**
 * Stateful live drainer for streaming VAD: feeds one window at a time via [push] and
 * immediately returns any segments the VAD closed during that chunk. Use [flushAndDrain]
 * at end-of-input to surface the trailing tail segment(s).
 *
 * This is the core of simulated streaming ASR (ticket 31): the capture thread calls
 * [push] with each mic frame and the decoded segments are emitted in capture order.
 *
 * When [padding] > 0 the drainer keeps a rolling window of the pushed source audio
 * (absolute-indexed) so each closed segment can be expanded by [padding] samples on
 * both sides before being handed to the decoder — pre-padding from samples that
 * arrived before the detected boundary, post-padding from samples that arrived after
 * it (best-effort: only already-pushed audio is available, so a tail segment closed
 * at end-of-input may have less than [padding] post-padding). Keeping this in the
 * shared drainer makes all three backends (sherpa Whisper, Dolphin CTC, Dolphin
 * attention) pad identically.
 */
internal class LiveVadDrainer(
    private val vad: VadLike,
    private val padding: Int = 0,
) {

    // Rolling source audio window (absolute index of source.buf[0] = sourceStart).
    // Only maintained when padding > 0. Pruned after each drain to the previous
    // segment's end minus padding, so memory tracks roughly one utterance at a time.
    private val source = GrowingFloatArray()
    private var sourceStart = 0

    /** Push one window (up to [VAD_WINDOW] samples) to the VAD and return any
     *  segments it just closed. Empty list = no closed segments this window. */
    fun push(window: FloatArray): List<SpeechSegment> {
        if (window.isEmpty()) return emptyList()
        if (padding > 0) source.append(window)
        vad.acceptWaveform(window)
        if (!vad.isSpeechDetected()) return emptyList()
        return drainQueue()
    }

    /** Signal end-of-input: flush the VAD's internal buffer and return the
     *  trailing tail segment(s), if any. */
    fun flushAndDrain(): List<SpeechSegment> {
        vad.flush()
        return drainQueue()
    }

    private fun drainQueue(): List<SpeechSegment> {
        val out = mutableListOf<SpeechSegment>()
        var keepFrom = -1
        while (!vad.isEmpty()) {
            vad.front()?.let { raw ->
                out += pad(raw)
                // Pre-padding for the NEXT segment needs up to [padding] samples before
                // its start (>= raw.end), so retain from raw.end - padding onward.
                keepFrom = maxOf(keepFrom, raw.start + raw.samples.size - padding)
            }
            vad.pop()
        }
        if (padding > 0 && keepFrom > sourceStart) {
            val dropCount = min(keepFrom - sourceStart, source.size)
            source.drop(dropCount)
            sourceStart += dropCount
        }
        return out
    }

    /** Expand [seg] with up to [padding] samples before and after it from [source]. */
    private fun pad(seg: SpeechSegment): SpeechSegment {
        if (padding <= 0) return seg
        val segEnd = seg.start + seg.samples.size
        val preFrom = maxOf(sourceStart, seg.start - padding)
        val postTo = minOf(sourceStart + source.size, segEnd + padding)
        val preCount = (seg.start - preFrom).coerceAtLeast(0)
        val postCount = (postTo - segEnd).coerceAtLeast(0)
        val out = FloatArray(preCount + seg.samples.size + postCount)
        var o = 0
        for (i in preFrom until seg.start) out[o++] = source.buf[i - sourceStart]
        seg.samples.copyInto(out, o)
        o += seg.samples.size
        for (i in segEnd until postTo) out[o++] = source.buf[i - sourceStart]
        return SpeechSegment(preFrom, out, seg.sampleRate)
    }
}

/** A growable (intrusive) float buffer for rolling VAD source audio. */
private class GrowingFloatArray(initialCapacity: Int = 8192) {
    var buf = FloatArray(initialCapacity)
    var size = 0

    fun append(a: FloatArray) {
        if (a.isEmpty()) return
        val need = size + a.size
        if (need > buf.size) buf = buf.copyOf(maxOf(buf.size * 2, need))
        a.copyInto(buf, size)
        size += a.size
    }

    /** Drop the first [count] samples, shifting the tail down. */
    fun drop(count: Int) {
        if (count <= 0) return
        buf.copyInto(buf, 0, count, size)
        size -= count
    }
}

/**
 * Simple in-memory [WaveWriter] for accumulating audio samples.
 */
internal class InMemoryWaveWriter(private val sampleRate: Int = 16000) : WaveWriter {
    private val accumulatedSamples = mutableListOf<Float>()

    override fun accept(samples: ShortArray) {
        accumulatedSamples.addAll(samples.toFloatArray(sampleRate).toList())
    }

    override fun flush(): WaveWriter.Wave? {
        if (accumulatedSamples.isEmpty()) return null
        return WaveWriter.Wave(accumulatedSamples.toFloatArray(), sampleRate)
    }
}

/** The Silero VAD window: the graph consumes this many samples per `acceptWaveform`. */
internal const val VAD_WINDOW = 512

/** The sample rate used by the VAD and the app's mic capture. */
internal const val SAMPLE_RATE = 16000

/** Pre/post padding (ms) added around a VAD segment before decode, so words at the
 *  detected speech boundary are not clipped. See the vad-accuracy spec (ADR 0006 note). */
internal const val PAD_MS = 200

/** Pre/post padding in samples at 16 kHz: 200 ms × 16 kHz. */
internal const val PAD_SAMPLES = PAD_MS * SAMPLE_RATE / 1000

/** A merged segment chunk may span at most this many ms: Whisper/Dolphin's ~30 s
 *  design limit minus headroom (vad-accuracy spec, Decision 2). */
internal const val MERGE_MAX_MS = 28_000

/** The merge cap in samples at 16 kHz: 28 s × 16 kHz = 448,000. */
internal const val MERGE_MAX_SAMPLES = MERGE_MAX_MS * SAMPLE_RATE / 1000

// Extension function to convert ShortArray to FloatArray
internal fun ShortArray.toFloatArray(sampleRate: Int): FloatArray {
    val floatArray = FloatArray(this.size) {
        this[it].toFloat() / Short.MAX_VALUE.toFloat()
    }
    return floatArray
}
