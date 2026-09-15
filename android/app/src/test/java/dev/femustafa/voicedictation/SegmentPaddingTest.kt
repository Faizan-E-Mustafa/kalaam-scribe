package dev.femustafa.voicedictation

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.ArrayDeque

/**
 * Tests for VAD segment padding (vad-accuracy ticket 01): [padSegments] (batch) and
 * pre/post padding inside [LiveVadDrainer] (live). Driven by fake [VadLike]s whose
 * windows carry their absolute index as their sample value, so a padded segment can
 * be asserted sample-by-sample against the source stream.
 */
class SegmentPaddingTest {

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

    // ---------------------------------------------------------------------
    // LiveVadDrainer
    // ---------------------------------------------------------------------

    @Test
    fun liveSegmentIsPaddedOnBothSidesFromSource() {
        // Segment starts mid-stream at absolute index 512 (chunk 1); 3 windows of
        // source have been pushed when it closes, so full pre+post padding is available.
        val seg = SpeechSegment(start = 512, samples = FloatArray(10) { (512 + it).toFloat() })
        val vad = FakeVad(segmentPlan = mapOf(2 to listOf(seg)))
        val drainer = LiveVadDrainer(vad, padding = 100)

        drainer.push(window(0))
        drainer.push(window(1))
        val result = drainer.push(window(2)) // closes the segment

        assertEquals(1, result.size)
        val padded = result[0]
        assertEquals(412, padded.start)                    // 512 - 100
        assertEquals(100 + 10 + 100, padded.samples.size)  // pre + segment + post
        assertEquals(412f, padded.samples[0])
        assertEquals(621f, padded.samples[padded.samples.size - 1])
        // Interior is the VAD segment itself, still in place.
        assertEquals(512f, padded.samples[100])
    }

    @Test
    fun livePrePaddingClampsAtSourceStart() {
        // Segment begins 100 samples in; only 100 pre-samples exist, so pre-pad is
        // exactly the samples from index 0..100. Post-pad is fully available.
        val seg = SpeechSegment(start = 100, samples = FloatArray(10) { (100 + it).toFloat() })
        val vad = FakeVad(segmentPlan = mapOf(0 to listOf(seg)))
        val drainer = LiveVadDrainer(vad, padding = 200)

        val result = drainer.push(window(0)) // only 512 samples pushed

        assertEquals(1, result.size)
        val padded = result[0]
        assertEquals(0, padded.start)
        // 100 pre + 10 segment + 200 post (post clamped at the 512 pushed samples).
        assertEquals(310, padded.samples.size)
        assertEquals(0f, padded.samples[0])
        assertEquals(309f, padded.samples[padded.samples.size - 1])
    }

    @Test
    fun livePostPaddingUsesOnlyArrivedSamples() {
        // Segment closes late in the stream; post-padding is limited to the samples
        // already pushed (best-effort, like a tail segment at end-of-input).
        val seg = SpeechSegment(start = 400, samples = FloatArray(50) { (400 + it).toFloat() })
        val vad = FakeVad(flushSegments = listOf(seg))
        val drainer = LiveVadDrainer(vad, padding = 200)

        drainer.push(window(0)) // 512 samples total
        val flushed = drainer.flushAndDrain()

        assertEquals(1, flushed.size)
        val padded = flushed[0]
        assertEquals(200, padded.start)
        // 200 pre + 50 segment + (512 - 450) post (only 62 arrived samples available).
        assertEquals(312, padded.samples.size)
        assertEquals(200f, padded.samples[0])
        assertEquals(511f, padded.samples[padded.samples.size - 1])
    }

    @Test
    fun livePaddingRetainsHistoryForNextSegment() {
        // After the first segment is drained, enough history must survive for the
        // second segment's pre-padding even though both are drained at different times.
        val first = SpeechSegment(start = 512, samples = FloatArray(10) { (512 + it).toFloat() })
        val second = SpeechSegment(start = 512 * 2, samples = FloatArray(10) { (512 * 2 + it).toFloat() })
        val vad = FakeVad(segmentPlan = mapOf(1 to listOf(first), 3 to listOf(second)))
        val drainer = LiveVadDrainer(vad, padding = 100)

        drainer.push(window(0))
        drainer.push(window(1)) // closes first
        // Idle traffic between utterances; source is pruned after the drain and must
        // still cover the second segment's pre-pad.
        drainer.push(window(2))
        val result = drainer.push(window(3)) // closes second

        assertEquals(1, result.size)
        val padded = result[0]
        assertEquals(512 * 2 - 100, padded.start)
        // Full pre + interior + post padding (idle traffic between utterances must not
        // prune the pre-source away).
        assertEquals(100 + 10 + 100, padded.samples.size)
        assertEquals(924f, padded.samples[0])
    }

    @Test
    fun liveNoPaddingKeepsSegmentUntouched() {
        // Default padding = 0: drainer must be a pure pass-through (no source stream).
        val seg = SpeechSegment(start = 512, samples = FloatArray(10) { 0.5f })
        val vad = FakeVad(segmentPlan = mapOf(1 to listOf(seg)))
        val drainer = LiveVadDrainer(vad)

        drainer.push(window(0))
        val result = drainer.push(window(1))

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