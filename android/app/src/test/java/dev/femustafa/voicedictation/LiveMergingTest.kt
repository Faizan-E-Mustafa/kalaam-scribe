package dev.femustafa.voicedictation

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.ArrayDeque

/**
 * Tests for the [LiveVadDrainer] live-merge mode (ticket 03): closed utterances are
 * held and only released once a settle run of silent windows arrives (so a phrase
 * that pauses mid-sentence can rejoin its follow-up), when the held span hits the
 * merge cap, or at [LiveVadDrainer.flushAndDrain]. Release builds one continuous,
 * source-backed, once-padded chunk — the streaming twin of the batch [mergeSegments]
 * pipeline. Uses the same windowed [FakeVad] pattern as [LiveVadDrainerTest].
 */
class LiveMergingTest {

    /** Window [n] (1-based) covers absolute samples `(n-1)*512 .. n*512`. */
    private fun window(n: Int): FloatArray =
        FloatArray(VAD_WINDOW) { i -> ((n - 1) * VAD_WINDOW + i).toFloat() }

    /** A [VadLike] that enqueues pre-planned segments on specific push ordinals (1 =
     *  first [acceptWaveform] call) and reports isSpeechDetected() on a separate set. */
    private class FakeVad(
        private val segmentsByPush: Map<Int, List<SpeechSegment>> = emptyMap(),
        private val speechPushes: Set<Int> = emptySet(),
        private val flushSegments: List<SpeechSegment> = emptyList(),
    ) : VadLike {
        private val queued = ArrayDeque<SpeechSegment>()
        private var speechOnLastPush = false
        private var push = 0

        override fun acceptWaveform(samples: FloatArray) {
            push++
            speechOnLastPush = speechPushes.contains(push) || segmentsByPush.containsKey(push)
            segmentsByPush[push]?.let { queued.addAll(it) }
        }

        override fun isSpeechDetected(): Boolean = speechOnLastPush

        override fun isEmpty(): Boolean = queued.isEmpty()

        override fun front(): SpeechSegment? = queued.firstOrNull()

        override fun pop() {
            if (queued.isNotEmpty()) queued.removeFirst()
        }

        override fun flush() {
            queued.addAll(flushSegments)
        }
    }

    private fun seg(start: Int, size: Int): SpeechSegment =
        SpeechSegment(start, FloatArray(size) { (start + it).toFloat() })

    private val mergeSettle = 3

    @Test
    fun twoRemoteUtterancesMergeIntoOnePaddedChunk() {
        val vad = FakeVad(
            segmentsByPush = mapOf(2 to listOf(seg(512, 10)), 4 to listOf(seg(1536, 10))),
            speechPushes = setOf(2, 4),
        )
        val drainer = LiveVadDrainer(vad, padding = 32, mergeSettleWindows = mergeSettle)

        assertTrue(drainer.push(window(1)).isEmpty()) // silent 1
        assertTrue(drainer.push(window(2)).isEmpty()) // A held
        assertTrue(drainer.push(window(3)).isEmpty()) // silent 1
        assertTrue(drainer.push(window(4)).isEmpty()) // B appended to A, speech reset
        assertTrue(drainer.push(window(5)).isEmpty()) // silent 1
        assertTrue(drainer.push(window(6)).isEmpty()) // silent 2
        val merged = drainer.push(window(7))          // silent 3 -> emit

        assertEquals(1, merged.size)
        val chunk = merged[0]
        // A.start (512) minus pre-pad, one padded chunk spanning A.samples..B.samples.
        assertEquals(512 - 32, chunk.start)
        assertEquals((1546 - 512) + 32 + 32, chunk.samples.size)
        // Probe real gap audio inside the merged chunk: absolute sample 1000.
        assertEquals(1000f, chunk.samples[1000 - chunk.start], 0f)
    }

    @Test
    fun lonelyUtteranceEmitsAfterSettleWithoutPadding() {
        val vad = FakeVad(
            segmentsByPush = mapOf(2 to listOf(seg(512, 10))),
            speechPushes = setOf(2),
        )
        val drainer = LiveVadDrainer(vad, padding = 0, mergeSettleWindows = mergeSettle)

        assertTrue(drainer.push(window(1)).isEmpty())
        assertTrue(drainer.push(window(2)).isEmpty())
        assertTrue(drainer.push(window(3)).isEmpty())
        assertTrue(drainer.push(window(4)).isEmpty())
        val emitted = drainer.push(window(5))

        assertEquals(1, emitted.size)
        assertEquals(512, emitted[0].start)
        assertEquals(10, emitted[0].samples.size)
    }

    @Test
    fun flushReleasesHeldChunk() {
        val vad = FakeVad(
            segmentsByPush = mapOf(2 to listOf(seg(512, 10))),
            speechPushes = setOf(2),
        )
        val drainer = LiveVadDrainer(vad, padding = 32, mergeSettleWindows = 100)

        assertTrue(drainer.push(window(1)).isEmpty())
        assertTrue(drainer.push(window(2)).isEmpty())

        val emitted = drainer.flushAndDrain()

        assertEquals(1, emitted.size)
        assertEquals(512 - 32, emitted[0].start)
        // Both margins were pushed before the flush(2 + 1 windows cover 512..1578).
        assertEquals(32 + 10 + 32, emitted[0].samples.size)
    }

    @Test
    fun mergeCapForcesEmitDuringDrain() {
        val vad = FakeVad(
            segmentsByPush = mapOf(2 to listOf(seg(512, 10)), 5 to listOf(seg(2048, 10))),
            speechPushes = setOf(2, 3, 4, 5), // continuous speech, B emits at push 5
        )
        val drainer = LiveVadDrainer(vad, padding = 32, mergeSettleWindows = 100, maxChunkSamples = 1000)

        assertTrue(drainer.push(window(1)).isEmpty())
        assertTrue(drainer.push(window(2)).isEmpty())
        assertTrue(drainer.push(window(3)).isEmpty())
        assertTrue(drainer.push(window(4)).isEmpty())
        val forced = drainer.push(window(5))

        // A(512)+B(end 2058) spans 1546 >= the 1000 max -> released atomically.
        assertEquals(1, forced.size)
        assertEquals(512 - 32, forced[0].start)
        assertEquals((2058 - 512) + 32 + 32, forced[0].samples.size)
    }

    @Test
    fun mergeOffStillEmitsEachUtteranceImmediately() {
        val vad = FakeVad(
            segmentsByPush = mapOf(2 to listOf(seg(512, 10)), 3 to listOf(seg(1536, 10))),
            speechPushes = setOf(2, 3),
        )
        val drainer = LiveVadDrainer(vad, padding = 32)

        assertTrue(drainer.push(window(1)).isEmpty())
        val a = drainer.push(window(2))
        assertEquals(1, a.size)
        assertEquals(512 - 32, a[0].start)
        assertEquals(32 + 10 + 32, a[0].samples.size) // post-pad already pushed

        val b = drainer.push(window(3))
        assertEquals(1, b.size)
        assertEquals(1536 - 32, b[0].start)
        assertEquals(32 + 10, b[0].samples.size) // post-pad not yet available
    }

    @Test
    fun prePadSurvivesAnEarlierEmit() {
        val vad = FakeVad(
            segmentsByPush = mapOf(2 to listOf(seg(512, 10)), 6 to listOf(seg(2560, 10))),
            speechPushes = setOf(2, 6),
        )
        val drainer = LiveVadDrainer(vad, padding = 32, mergeSettleWindows = mergeSettle)

        // First chunk: A alone, emitted after the settle.
        assertTrue(drainer.push(window(1)).isEmpty())
        assertTrue(drainer.push(window(2)).isEmpty())
        assertTrue(drainer.push(window(3)).isEmpty())
        assertTrue(drainer.push(window(4)).isEmpty())
        val first = drainer.push(window(5))
        assertEquals(1, first.size)
        assertEquals(512 - 32, first[0].start)

        // Second utterance; its pre-pad must still be sourced after the first emit.
        assertTrue(drainer.push(window(6)).isEmpty())
        assertTrue(drainer.push(window(7)).isEmpty())
        assertTrue(drainer.push(window(8)).isEmpty())
        val second = drainer.push(window(9))

        assertEquals(1, second.size)
        assertEquals(2560 - 32, second[0].start)
        assertEquals(32 + 10 + 32, second[0].samples.size)
    }
}