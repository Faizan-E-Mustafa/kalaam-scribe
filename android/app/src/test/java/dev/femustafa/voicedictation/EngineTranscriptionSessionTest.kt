package dev.femustafa.voicedictation

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.yield
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.ArrayDeque

/**
 * Tests for [EngineTranscriptionSession] ordering: mic samples pushed through [accept]
 * must be decoded in capture order, the VAD flush tail must be decoded last (never
 * raced ahead of queued segments), and a session with no speech must fall back to a
 * whole-clip decode. Driven by a fake [VadLike] + fake decode lambdas (no native lib).
 */
class EngineTranscriptionSessionTest {

    private fun shortsOf(id: Int, n: Int): ShortArray = ShortArray(n) { id.toShort() }

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

    /** Labels a segment by its sample count (length encodes identity). */
    private fun decodeByLength(samples: FloatArray): String = "len${samples.size}"

    @Test
    fun segmentsDecodedInCaptureOrderAndJoined() = runBlocking {
        val seg1 = SpeechSegment(0, FloatArray(100) { 1f })
        val seg2 = SpeechSegment(600, FloatArray(200) { 2f })
        val vad = FakeVad(segmentPlan = mapOf(1 to listOf(seg1), 2 to listOf(seg2)))
        val session = EngineTranscriptionSession(vad, ::decodeByLength, { "whole" })

        val got = mutableListOf<String>()
        val collector = launch { session.partials.collect { got += it } }

        // 1024 shorts = two 512 windows: chunk 0 (no speech), chunk 1 (seg1).
        session.accept(shortsOf(1, 1024))
        // One 512 window: chunk 2 (seg2).
        session.accept(shortsOf(2, 512))
        val full = session.flush("")
        // flush() joins the worker, so both segments are decoded; drain the collector
        // (same runBlocking thread, so yield() guarantees delivery) before asserting.
        withTimeout(1000) { while (got.size < 2) yield() }

        collector.cancel()
        assertEquals(listOf("len100", "len200"), got)
        assertEquals("len100 len200", full)
    }

    @Test
    fun flushTailSegmentIsDecodedLastNotRacedAhead() = runBlocking {
        val early = SpeechSegment(0, FloatArray(100) { 1f })
        val tail = SpeechSegment(900, FloatArray(300) { 3f })
        // Speech detected on chunk 0 (early segment); flush surfaces the tail.
        val vad = FakeVad(
            segmentPlan = mapOf(0 to listOf(early)),
            flushSegments = listOf(tail),
        )
        val session = EngineTranscriptionSession(vad, ::decodeByLength, { "whole" })

        val got = mutableListOf<String>()
        val collector = launch { session.partials.collect { got += it } }

        session.accept(shortsOf(1, 512))
        val full = session.flush("")
        // Same same-thread drain as the segmentsInOrder test: two decoded partials.
        withTimeout(1000) { while (got.size < 2) yield() }

        collector.cancel()
        // Tail segment decoded AFTER the early segment — always last.
        assertEquals(listOf("len100", "len300"), got)
        assertEquals("len100 len300", full)
    }

    @Test
    fun noSpeechFallsBackToWholeClipDecode() = runBlocking {
        var wholeAudioSize = -1
        val vad = FakeVad()
        val session = EngineTranscriptionSession(vad, ::decodeByLength, { audio ->
            wholeAudioSize = audio.size
            "FALLBACK"
        }, logger = { _, _ -> })

        val got = mutableListOf<String>()
        val collector = launch { session.partials.collect { got += it } }

        session.accept(shortsOf(1, 512))
        session.accept(shortsOf(2, 256))
        val full = session.flush("")

        collector.cancel()
        assertEquals("FALLBACK", full)
        // The fallback decode saw the whole accepted audio (512 + 256).
        assertEquals(768, wholeAudioSize)
        assertTrue(got.isEmpty())
    }

    @Test
    fun blankSegmentsFallBackToWholeClipDecode() = runBlocking {
        val seg = SpeechSegment(0, FloatArray(100) { 1f })
        val vad = FakeVad(segmentPlan = mapOf(0 to listOf(seg)))
        // Every segment decodes to blank → whole-clip fallback (ticket-30 semantics).
        val session = EngineTranscriptionSession(vad, { "" }, { "FALLBACK" }, logger = { _, _ -> })

        session.accept(shortsOf(1, 512))
        val full = session.flush("")

        assertEquals("FALLBACK", full)
    }

    @Test
    fun acceptYieldsToWorkerBeforeFlushJoins() = runBlocking {
        val seg = SpeechSegment(0, FloatArray(100) { 1f })
        val vad = FakeVad(segmentPlan = mapOf(0 to listOf(seg)))
        val session = EngineTranscriptionSession(vad, ::decodeByLength, { "whole" })

        session.accept(shortsOf(1, 512))
        yield()
        val full = session.flush("")

        assertEquals("len100", full)
    }
}