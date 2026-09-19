package dev.femustafa.voicedictation

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.ArrayDeque

/**
 * Tests for VAD segment padding used by the batch chunking path: [padSegments]
 * and how pre/post padding composes with [segmentAudioWithVad]. Driven by fake
 * [VadLike]s whose windows carry their absolute index as their sample value, so a
 * padded segment can be asserted sample-by-sample against the source stream.
 */
class SegmentPaddingTest {

    private class FakeVad(
        private val segmentPlan: Map<Int, List<SpeechSegment>> = emptyMap(),
        private val flushSegments: List<SpeechSegment> = emptyList(),
    ) : VadLike {
        private val queued = ArrayDeque<SpeechSegment>()
        private var speechOnLastChunk = false
        private var chunkIndex = 0

        override fun acceptWaveform(samples: FloatArray) {
            speechOnLastChunk = segmentPlan.containsKey(chunkIndex)
            segmentPlan[chunkIndex]?.let { queued.addAll(it) }
            chunkIndex++
        }

        override fun isSpeechDetected(): Boolean = speechOnLastChunk
        override fun isEmpty(): Boolean = queued.isEmpty()
        override fun front(): SpeechSegment? = queued.firstOrNull()
        override fun pop() {
            if (!queued.isEmpty()) queued.removeFirst()
        }

        override fun flush() {
            queued.addAll(flushSegments)
        }
    }

    @Test
    fun liveNoPaddingKeepsSegmentUntouched() {
        // Default padding = 0: drainer must be a pure pass-through (no source stream).
        val seg = SpeechSegment(start = 512, samples = FloatArray(10) { 0.5f })
        val vad = FakeVad(segmentPlan = mapOf(1 to listOf(seg)))
        val drainer = LiveVadDrainer(vad)

        drainer.push(FloatArray(512) { 0f })
        val result = drainer.push(FloatArray(512) { 0f })

        assertEquals(1, result.size)
        assertEquals(512, result[0].start)
        assertEquals(10, result[0].samples.size)
        assertEquals(seg, result[0])
    }

    // ---------------------------------------------------------------------
    // Batch padSegments + segmentAudioWithVad
    // ---------------------------------------------------------------------

    @Test
    fun padSegmentsExpandsBothSidesAndClamps() {
        val source = FloatArray(1000) { it.toFloat() }
        val segs = listOf(
            SpeechSegment(start = 100, samples = FloatArray(10) { (100 + it).toFloat() }),
            SpeechSegment(start = 0, samples = FloatArray(10) { it.toFloat() }),     // pre-clamp
            SpeechSegment(start = 990, samples = FloatArray(10) { (990 + it).toFloat() }), // post-clamp
        )
        val padded = padSegments(segs, source, padding = 50)

        assertEquals(3, padded.size)
        // Mid-stream: full 50/50 padding.
        assertEquals(50, padded[0].start)
        assertEquals(110, padded[0].samples.size)
        assertEquals(50f, padded[0].samples[0])
        assertEquals(159f, padded[0].samples[109])
        // Start edge: pre-padding clamped to 0.
        assertEquals(0, padded[1].start)
        assertEquals(60, padded[1].samples.size)
        // End edge: post-padding clamped to source.size.
        assertEquals(940, padded[2].start)
        assertEquals(60, padded[2].samples.size)
        assertEquals(999f, padded[2].samples[59])
    }

    @Test
    fun batchSegmentationAppliesPadding() {
        val seg = SpeechSegment(start = 100, samples = FloatArray(10) { (100 + it).toFloat() })
        val vad = FakeVad(segmentPlan = mapOf(0 to listOf(seg)))
        val clip = FloatArray(512 * 2) { it.toFloat() }

        val padded = segmentAudioWithVad(clip, vad, padding = 50)

        assertEquals(1, padded.size)
        assertEquals(50, padded[0].start)
        assertEquals(110, padded[0].samples.size) // 50 + 10 + 50
        assertEquals(50f, padded[0].samples[0])
        assertEquals(159f, padded[0].samples[109])
    }

    @Test
    fun batchSegmentationWithoutPaddingIsUnchanged() {
        val seg = SpeechSegment(start = 100, samples = FloatArray(10) { 0.25f })
        val vad = FakeVad(segmentPlan = mapOf(0 to listOf(seg)))
        val clip = FloatArray(512 * 2)

        val out = segmentAudioWithVad(clip, vad)

        assertEquals(1, out.size)
        assertEquals(100, out[0].start)
        assertEquals(10, out[0].samples.size)
        assertTrue(out[0].samples.all { it == 0.25f })
    }
}