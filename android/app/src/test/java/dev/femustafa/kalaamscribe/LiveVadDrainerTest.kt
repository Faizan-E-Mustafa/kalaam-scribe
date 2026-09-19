package dev.femustafa.kalaamscribe

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.ArrayDeque

/**
 * Tests for the [LiveVadDrainer] — a stateful wrapper around a [VadLike]
 * that returns closed segments one window at a time instead of draining a
 * full clip. Driven by a [FakeVad] (no sherpa native lib on the JVM).
 */
class LiveVadDrainerTest {

    private fun tones(size: Int): FloatArray = FloatArray(size) { 0.01f }

    /**
     * A [VadLike] whose acceptWaveform enqueues pre-planned segments and
     * reports isSpeechDetected() based on the plan for the given chunk index.
     */
    private class FakeVad(
        private val segmentPlan: Map<Int, List<SpeechSegment>> = emptyMap(),
        private val flushSegments: List<SpeechSegment> = emptyList(),
    ) : VadLike {
        val acceptedSizes = mutableListOf<Int>()
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
            queued.addAll(flushSegments)
        }
    }

    @Test
    fun emptyWindowReturnsNothing() {
        val vad = FakeVad()
        val drainer = LiveVadDrainer(vad)
        val out = drainer.push(tones(0))
        assertTrue(out.isEmpty())
    }

    @Test
    fun noSpeechReturnsNothingPerWindow() {
        val vad = FakeVad()
        val drainer = LiveVadDrainer(vad)
        val a = drainer.push(tones(512))
        val b = drainer.push(tones(512))
        assertTrue(a.isEmpty())
        assertTrue(b.isEmpty())
    }

    @Test
    fun segmentDrainedImmediatelyWhenSpeechDetected() {
        val seg = SpeechSegment(start = 0, samples = tones(10))
        val vad = FakeVad(segmentPlan = mapOf(1 to listOf(seg)))
        val drainer = LiveVadDrainer(vad)

        val a = drainer.push(tones(512))  // chunk 0 — no speech
        val b = drainer.push(tones(512))  // chunk 1 — speech → segment queued
        val c = drainer.push(tones(512))  // chunk 2 — idle

        assertTrue(a.isEmpty())
        assertEquals(1, b.size)
        assertEquals(10, b[0].samples.size)
        assertTrue(c.isEmpty())
    }

    @Test
    fun multipleSegmentsFromSingleChunkAreDrainedInOrder() {
        val seg1 = SpeechSegment(start = 512, samples = tones(10))
        val seg2 = SpeechSegment(start = 600, samples = tones(20))
        val vad = FakeVad(segmentPlan = mapOf(1 to listOf(seg1, seg2)))
        val drainer = LiveVadDrainer(vad)

        drainer.push(tones(512))  // chunk 0
        val result = drainer.push(tones(512))  // chunk 1 — speech
        val after = drainer.push(tones(512))  // chunk 2

        assertEquals(2, result.size)
        assertEquals(10, result[0].samples.size)
        assertEquals(20, result[1].samples.size)
        assertTrue(after.isEmpty())
    }

    @Test
    fun flushSurfacesTailSegment() {
        val tail = SpeechSegment(start = 512 * 3, samples = tones(7))
        val vad = FakeVad(flushSegments = listOf(tail))
        val drainer = LiveVadDrainer(vad)

        drainer.push(tones(512))
        drainer.push(tones(512))
        val flushed = drainer.flushAndDrain()

        assertEquals(1, flushed.size)
        assertEquals(7, flushed[0].samples.size)
        assertEquals(tail, flushed[0])
    }

    @Test
    fun flushReturnsNothingWhenNoTail() {
        val vad = FakeVad()
        val drainer = LiveVadDrainer(vad)
        val flushed = drainer.flushAndDrain()
        assertTrue(flushed.isEmpty())
    }

    @Test
    fun segmentAfterFlushTailIsReturnedFromFlush() {
        val mid = SpeechSegment(start = 512, samples = tones(10))
        val tail = SpeechSegment(start = 512 * 2, samples = tones(5))
        val vad = FakeVad(
            segmentPlan = mapOf(1 to listOf(mid)),
            flushSegments = listOf(tail),
        )
        val drainer = LiveVadDrainer(vad)

        val during = drainer.push(tones(512))  // chunk 0 — nothing
        val atSpeech = drainer.push(tones(512))  // chunk 1 — speech → mid drained
        drainer.push(tones(512))  // chunk 2 — nothing
        val flushed = drainer.flushAndDrain()

        assertTrue(during.isEmpty())
        assertEquals(1, atSpeech.size)
        assertEquals(10, atSpeech[0].samples.size)
        assertEquals(1, flushed.size)
        assertEquals(tail, flushed[0])
    }

    @Test
    fun acceptingMultipleWindowsRespectedInOrder() {
        val vad = FakeVad()
        val drainer = LiveVadDrainer(vad)
        drainer.push(tones(512))
        drainer.push(tones(512))
        drainer.push(tones(100))
        assertEquals(listOf(512, 512, 100), vad.acceptedSizes)
    }
}
