package dev.femustafa.kalaamscribe

import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.ArrayDeque

/**
 * Tests for VAD segment merging (vad-accuracy ticket 02): the pure [mergeSegments]
 * batch pass and how it composes with [padSegments] in [segmentAudioWithVad].
 * Driven by fake [VadLike]s whose sample values equal their absolute source index,
 * so merged spans can be asserted sample-by-sample.
 */
class SegmentMergingTest {

    /** A 512-sample window whose values equal the absolute indices [c*512, c*512+512). */
    private fun window(c: Int): FloatArray = FloatArray(512) { idx -> (c * 512 + idx).toFloat() }

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

    /** Source whose sample values equal their index, like the window generator. */
    private fun source(n: Int): FloatArray = FloatArray(n) { it.toFloat() }

    private fun segment(start: Int, size: Int): SpeechSegment =
        SpeechSegment(start, FloatArray(size) { (start + it).toFloat() })

    // ---------------------------------------------------------------------
    // mergeSegments (pure)
    // ---------------------------------------------------------------------

    @Test
    fun mergesConsecutiveSegmentsIntoFullSourceSpanIncludingSilence() {
        // A ends at 522, B starts at 1024 — the 502 samples of silence between them
        // are part of the merged chunk's span (context for the decoder).
        val src = source(4096)
        val out = mergeSegments(listOf(segment(512, 10), segment(1024, 10)), src, maxSamples = 1000)

        assertEquals(1, out.size)
        val chunk = out[0]
        assertEquals(512, chunk.start)
        // Span from A.start (512) to B.end (1034): includes the gap.
        assertEquals(1034 - 512, chunk.samples.size)
        assertEquals(512f, chunk.samples[0])
        assertEquals(1033f, chunk.samples[chunk.samples.size - 1])
    }

    @Test
    fun oversizedNeighborClosesCurrentChunkAtSilenceGap() {
        // A+B fits under the cap (span 512..1100); adding C would blow it (512..2100),
        // so C closes chunk 1 and becomes chunk 2 alone.
        val src = source(4096)
        val segments = listOf(segment(512, 10), segment(600, 500), segment(2000, 100))
        val out = mergeSegments(segments, src, maxSamples = 1000)

        assertEquals(2, out.size)
        assertEquals(512, out[0].start)
        assertEquals(1100 - 512, out[0].samples.size)
        assertEquals(2000, out[1].start)
        assertEquals(100, out[1].samples.size)
        assertEquals(2000f, out[1].samples[0])
    }

    @Test
    fun oversizedSingleSegmentPassesThroughWhole() {
        // A multi-second segment is never split, even when it alone exceeds the cap.
        val src = source(4096)
        val out = mergeSegments(listOf(segment(512, 2000)), src, maxSamples = 1000)

        assertEquals(1, out.size)
        assertEquals(512, out[0].start)
        assertEquals(2000, out[0].samples.size)
    }

    @Test
    fun mergeWithNoCapIsIdentity() {
        val segments = listOf(segment(512, 10), segment(1024, 10))
        assertSame(segments, mergeSegments(segments, source(4096), maxSamples = 0))
        assertTrue(mergeSegments(segments, source(4096), maxSamples = -1) == segments)
    }

    @Test
    fun mergeClampsMergedSpanToSourceBounds() {
        // The merged span (512..750) runs past the 700-sample source end; the slice is
        // clamped to 700 rather than reading out of bounds.
        val src = source(700)
        val out = mergeSegments(listOf(segment(512, 10), segment(650, 100)), src, maxSamples = 1000)

        assertEquals(1, out.size)
        assertEquals(512, out[0].start)
        assertEquals(700 - 512, out[0].samples.size)
        assertEquals(699f, out[0].samples[out[0].samples.size - 1])
    }

    // ---------------------------------------------------------------------
    // segmentAudioWithVad integration: merge then pad
    // ---------------------------------------------------------------------

    @Test
    fun batchMergeThenPadPadsMergedEdgesOnce() {
        // 8 windows: A at chunk 1, B at chunk 2; flush adds tail C at 2048.
        val clip = source(8 * 512)
        val vad = FakeVad(
            segmentPlan = mapOf(
                1 to listOf(segment(512, 10)),
                2 to listOf(segment(1024, 10)),
            ),
            flushSegments = listOf(segment(2048, 10)),
        )

        // Merge into <= 1000-sample chunks (A+B), then pad each chunk by 50.
        val out = segmentAudioWithVad(clip, vad, padding = 50, maxChunkSamples = 1000)

        assertEquals(2, out.size)
        // Chunk 1 = merged A+B (512..1034), padded 50 each side -> 462..1084 (pad+span+pad).
        assertEquals(462, out[0].start)
        assertEquals(50 + (1034 - 512) + 50, out[0].samples.size)
        assertEquals(462f, out[0].samples[0])
        assertEquals(1083f, out[0].samples[out[0].samples.size - 1])
        // Chunk 2 = tail C alone (2048..2058), padded -> 1998..2108.
        assertEquals(1998, out[1].start)
        assertEquals(50 + 10 + 50, out[1].samples.size)
        assertEquals(1998f, out[1].samples[0])
        assertEquals(2107f, out[1].samples[out[1].samples.size - 1])
    }

    @Test
    fun batchWithoutMergeKeepsEverySegmentSeparate() {
        val clip = source(8 * 512)
        val vad = FakeVad(
            segmentPlan = mapOf(
                1 to listOf(segment(512, 10)),
                2 to listOf(segment(1024, 10)),
            ),
            flushSegments = listOf(segment(2048, 10)),
        )

        val out = segmentAudioWithVad(clip, vad, padding = 50)

        // No merging: three separately padded segments.
        assertEquals(3, out.size)
        assertEquals(listOf(512, 1024, 2048), out.map { it.start + 50 })
    }
}