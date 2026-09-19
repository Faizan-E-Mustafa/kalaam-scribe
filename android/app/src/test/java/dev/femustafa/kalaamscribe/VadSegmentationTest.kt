package dev.femustafa.kalaamscribe

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.ArrayDeque

/**
 * Unit tests for [segmentAudioWithVad] driven by a fake [VadLike] (no sherpa native
 * lib needed on the JVM). Verifies the windowing, the drain-when-speech-detected loop,
 * and the trailing flush() drain — the logic that turns a whole clip into per-utterance
 * segments for the Dolphin attention engine.
 */
class VadSegmentationTest {

    /** One FloatArray of the given size filled with a constant. */
    private fun tones(size: Int): FloatArray = FloatArray(size) { 0.01f }

    /** A [VadLike] whose behavior matches the sherpa contract used by the helper. */
    private class FakeVad(
        /** Segments to enqueue at a given accepted-chunk index (isSpeechDetected → true). */
        private val segmentPlan: Map<Int, List<SpeechSegment>> = emptyMap(),
        /** Segments enqueued by [flush] (tail audio surfaced). */
        private val flushSegments: List<SpeechSegment> = emptyList(),
    ) : VadLike {
        /** Sample counts of every window the helper accepted. */
        val acceptedSizes = mutableListOf<Int>()
        var flushCalled = false

        private val queued = ArrayDeque<SpeechSegment>()
        private var speechOnLastChunk = false
        private var chunkIndex = 0

        override fun acceptWaveform(samples: FloatArray) {
            acceptedSizes += samples.size
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
            flushCalled = true
            queued.addAll(flushSegments)
        }
    }

    @Test
    fun emptyAudioAcceptsNothingAndFlushes() {
        val vad = FakeVad()
        val out = segmentAudioWithVad(tones(0), vad)
        assertTrue(out.isEmpty())
        assertTrue(vad.acceptedSizes.isEmpty())
        assertTrue(vad.flushCalled)
    }

    @Test
    fun audioIsPushedIn512SampleWindowsWithPartialTail() {
        val vad = FakeVad()
        segmentAudioWithVad(tones(512 * 3 + 100), vad)
        assertEquals(listOf(512, 512, 512, 100), vad.acceptedSizes)
    }

    @Test
    fun collectsSegmentsInOrderWhenSpeechDetected() {
        val plan = mapOf(
            1 to listOf(
                SpeechSegment(start = 512, samples = tones(10)),
                SpeechSegment(start = 600, samples = tones(20)),
            ),
            3 to listOf(SpeechSegment(start = 512 * 3, samples = tones(5))),
        )
        val vad = FakeVad(segmentPlan = plan)
        val out = segmentAudioWithVad(tones(512 * 4), vad)
        assertEquals(3, out.size)
        // Order is preserved as emitted.
        assertEquals(512, out[0].start)
        assertEquals(10, out[0].samples.size)
        assertEquals(600, out[1].start)
        assertEquals(20, out[1].samples.size)
        assertEquals(512 * 3, out[2].start)
        assertEquals(5, out[2].samples.size)
    }

    @Test
    fun noSegmentsWhenNoSpeechDetected() {
        val out = segmentAudioWithVad(tones(512 * 3), FakeVad())
        assertTrue(out.isEmpty())
    }

    @Test
    fun flushSurfacesBufferedTailUntilDrained() {
        val tail = SpeechSegment(start = 512 * 2, samples = tones(8))
        // Speech detected on chunk 0 so a segment is drained live; then nothing until
        // flush enqueues the tail segment, which must also be drained.
        val vad = FakeVad(
            segmentPlan = mapOf(0 to listOf(SpeechSegment(0, tones(3)))),
            flushSegments = listOf(tail),
        )
        val out = segmentAudioWithVad(tones(512 * 2), vad)
        assertEquals(2, out.size)
        assertEquals(3, out[0].samples.size)
        assertEquals(8, out[1].samples.size)
        assertEquals(tail, out[1])
        assertTrue(vad.flushCalled)
    }
}