package dev.femustafa.voicedictation

import ai.onnxruntime.OnnxJavaType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.yield
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.ArrayDeque
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Tests that streaming sessions (DolphinAttnSession, SherpaOfflineSession) do not drop
 * segments when the decode worker falls behind. The sessions use Channel.UNLIMITED so
 * the capture thread can enqueue freely; the worker drains at its own pace.
 *
 * Before the fix: Channel(BUFFERED) of size 5 with trySend+drop silently discarded
 * segments 6+ when the worker was slow. After: Channel.UNLIMITED and flush() joins the
 * decode worker, so no enqueued segment (live or tail) is ever dropped.
 *
 * These tests push more segments than the old bounded size and verify ALL of them
 * appear in the final transcript.
 */
class StreamingSessionNoDropTest {

    private fun tones(size: Int): ShortArray = ShortArray(size) { 0 }

    /** Poll until the async partials collector has drained [expected] items. */
    private suspend fun awaitPartialCount(got: List<String>, expected: Int) {
        repeat(200) {
            if (got.size >= expected) return
            withContext(Dispatchers.Default) { yield() }
        }
    }

    /**
     * Let the collector coroutine actually start collecting before the worker emits.
     * MutableSharedFlow(replay=0) drops values emitted before a collector subscribes, so
     * without this barrier the fast (faked) decode worker can emit everything into the
     * void and the test would measure 0 partials.
     */
    private suspend fun awaitCollectorSubscribed() {
        repeat(5) { withContext(Dispatchers.Default) { yield() } }
    }

    /** FakeVad (mirrors the pattern from LiveVadDrainerTest). */
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

    // -------------------------------------------------------------------------
    // DolphinAttnSession tests
    // -------------------------------------------------------------------------

    /** A [DolphinRecognizerLike] whose beamSearch/decodeTokens are faked, bypassing ONNX. */
    private class FakeDolphinDecoder : DolphinRecognizerLike {

        /** Map segment identity (audio length) to a token id so each segment
         *  is distinguishable in the output. The fake omits the 5-token
         *  language/prefix tokens the real engine prepends. */
        override fun beamSearch(
            ref: DolphinAttnEngine.DolphinAttnModelRef,
            audio: ShortArray,
        ): IntArray = intArrayOf(audio.size)

        /** Decode the fake's bare content token back to a text label (`<audioSize>`). */
        override fun decodeTokens(
            ref: DolphinAttnEngine.DolphinAttnModelRef,
            tokens: IntArray,
        ): String = tokens.filter { it != 0 && it != 39999 && it != 40000 && it != 324 }
            .joinToString(",")
    }

    /** Reusable fake model ref (nullable fields are fine — our fake decoder ignores it). */
    private class FakeModelRef : DolphinAttnEngine.DolphinAttnModelRef(
        env = null, encoder = null, decoder = null, tokenList = emptyList(),
        nl = 0, headDim = 64, dModel = 512, kvDtype = OnnxJavaType.FLOAT16,
        langStartTensor = null, langEndTensor = null, maskTensor = null, vad = null,
    )

    /** [DolphinAttnSession] driven by the fake decode seam (no engine/Context needed). */
    private class TestDolphinAttnSession(vad: VadLike) : DolphinAttnSession(
        recognizerLike = FakeDolphinDecoder(),
        modelRef = FakeModelRef(),
        waveWriter = InMemoryWaveWriter(16000),
        vadLike = vad,
    )

    /**
     * The old Channel(BUFFERED) with size 5 would drop segments 6+. The unlimited
     * channel must keep all 7 segments alive until the worker drains them.
     */
    @Test
    fun dolphinAttnSessionDoesNotDropSegmentsBeyondOldBoundedSize() = runBlocking {
        val N = 7
        val segments = (0 until N).map { id ->
            SpeechSegment(start = id * 512, samples = FloatArray(100 + id * 10) { 1f })
        }
        val vad = FakeVad(segmentPlan = (0 until N).associateWith { listOf(segments[it]) })
        val session = TestDolphinAttnSession(vad)

        val got = CopyOnWriteArrayList<String>()
        val collector = launch(Dispatchers.Default) { session.partials.collect { got += it } }

        awaitCollectorSubscribed()

        repeat(N) { session.accept(tones(512)) }
        yield()
        // Simulate the live UI: it has already shown (and snapshots) the partials.
        // flush() must NOT prepend this snapshot — decodedTexts already contains every
        // live segment, so prepending would duplicate them.
        val liveSnapshot = segments.joinToString(" ") { it.samples.size.toString() }
        val full = session.flush(liveSnapshot)

        awaitPartialCount(got, N)
        collector.cancel()
        assertEquals("all $N segments must appear in partials", N, got.size)
        assertEquals("final transcript must contain all $N segment texts, no duplicates", N, full.split(" ").size)
    }

    /**
     * Tail segments (from flush) are also preserved, not dropped. The VAD tail is routed
     * into the worker, which is joined by flush() so every enqueued segment is decoded.
     * Both the live segments and the tail surface as partials and in the final transcript.
     */
    @Test
    fun dolphinAttnSessionPreservesTailSegments() = runBlocking {
        val early = (0 until 3).map { id ->
            SpeechSegment(start = id * 512, samples = FloatArray(50 + id * 10) { 1f })
        }
        val tail = listOf(
            SpeechSegment(start = 2000, samples = FloatArray(30) { 2f }),
            SpeechSegment(start = 2100, samples = FloatArray(20) { 3f }),
        )
        val vad = FakeVad(
            segmentPlan = (0 until 3).associateWith { listOf(early[it]) },
            flushSegments = tail,
        )
        val session = TestDolphinAttnSession(vad)

        val got = CopyOnWriteArrayList<String>()
        val collector = launch(Dispatchers.Default) { session.partials.collect { got += it } }

        awaitCollectorSubscribed()

        repeat(3) { session.accept(tones(512)) }
        yield()
        // Snapshot = the live-only text the UI showed before Stop. flush() must not
        // prepend it (would duplicate the early segments); tail segments only come from
        // the worker now.
        val liveSnapshot = early.joinToString(" ") { it.samples.size.toString() }
        val full = session.flush(liveSnapshot)

        awaitPartialCount(got, early.size + tail.size)
        collector.cancel()
        assertEquals("live + tail segments must all appear as partials", early.size + tail.size, got.size)
        assertEquals("tail must appear in the final transcript", early.size + tail.size, full.split(" ").size)
    }

    /** A [DolphinRecognizerLike] that records every audio sizes handed to beamSearch. */
    private class RecordingDolphinDecoder : DolphinRecognizerLike {
        val beamedSizes = mutableListOf<Int>()

        override fun beamSearch(
            ref: DolphinAttnEngine.DolphinAttnModelRef,
            audio: ShortArray,
        ): IntArray {
            beamedSizes += audio.size
            return intArrayOf(audio.size)
        }

        override fun decodeTokens(
            ref: DolphinAttnEngine.DolphinAttnModelRef,
            tokens: IntArray,
        ): String = tokens.filter { it != 0 && it != 39999 && it != 40000 && it != 324 }
            .joinToString(",")
    }

    /**
     * A live VAD utterance may exceed the decoder's 66-token kernel ceiling (the VAD's
     * max-speech bound is 30 s; ~120 tokens at the measured 4.1 tok/s). The streaming
     * session must split such a segment into <=14 s pieces and decode each, so the
     * utterance's tail is never truncated mid-word.
     */
    @Test
    fun dolphinStreamingSplitsAnOverCapSegmentSoItsTailIsNotTruncated() = runBlocking {
        // 300,000 samples is over the 224,000-sample (14 s) cap. A constant signal means
        // no quiet pause, so the splitter falls back to the hard cap: 224,000 + 76,000.
        val segment = SpeechSegment(start = 0, samples = FloatArray(300_000) { 1f })
        val vad = FakeVad(segmentPlan = mapOf(0 to listOf(segment)))
        val decoder = RecordingDolphinDecoder()
        val session = DolphinAttnSession(
            recognizerLike = decoder,
            modelRef = FakeModelRef(),
            waveWriter = InMemoryWaveWriter(16000),
            vadLike = vad,
        )

        val got = CopyOnWriteArrayList<String>()
        val collector = launch(Dispatchers.Default) { session.partials.collect { got += it } }

        awaitCollectorSubscribed()
        session.accept(tones(512))
        yield()
        val full = session.flush("")

        awaitPartialCount(got, 1)
        collector.cancel()
        assertEquals("over-cap segment split into two pieces", listOf(224_000, 76_000), decoder.beamedSizes)
        assertEquals("both pieces decoded and joined, tail preserved", "224000 76000", full)
    }

    /** A short live utterance is still one decode (the highest-quality path). */
    @Test
    fun dolphinStreamingKeepsAShortSegmentAsOneDecode() = runBlocking {
        // 30,000 samples must NOT collide with a special token id (EOS=40000), which the
        // fake's decodeTokens filters out (that would blank the text and trigger the
        // whole-clip fallback).
        val segment = SpeechSegment(start = 0, samples = FloatArray(30_000) { 1f })
        val vad = FakeVad(segmentPlan = mapOf(0 to listOf(segment)))
        val decoder = RecordingDolphinDecoder()
        val session = DolphinAttnSession(
            recognizerLike = decoder,
            modelRef = FakeModelRef(),
            waveWriter = InMemoryWaveWriter(16000),
            vadLike = vad,
        )

        val got = CopyOnWriteArrayList<String>()
        val collector = launch(Dispatchers.Default) { session.partials.collect { got += it } }

        awaitCollectorSubscribed()
        session.accept(tones(512))
        yield()
        val full = session.flush("")

        awaitPartialCount(got, 1)
        collector.cancel()
        assertEquals(listOf(30_000), decoder.beamedSizes)
        assertEquals("30000", full)
    }

    // -------------------------------------------------------------------------
    // SherpaOfflineSession tests
    // -------------------------------------------------------------------------

    /** A [SherpaRecognizerLike] that records every segment decoded. */
    private class FakeRecognizer : SherpaRecognizerLike {
        val decoded = mutableListOf<FloatArray>()
        override fun decode(samples: FloatArray, sampleRate: Int): String {
            decoded.add(samples)
            return "len${samples.size}"
        }
    }

    /** [SherpaOfflineSession] using the test seam (SherpaRecognizerLike, no Log). */
    private class TestSherpaOfflineSession(vad: VadLike) : SherpaOfflineSession(
        recognizerLike = FakeRecognizer(),
        waveWriter = InMemoryWaveWriter(16000),
        vadLike = vad,
    )

    @Test
    fun sherpaSessionDoesNotDropSegmentsBeyondOldBoundedSize() = runBlocking {
        val N = 8
        val segments = (0 until N).map { id ->
            SpeechSegment(start = id * 512, samples = FloatArray(80 + id * 5) { 1f })
        }
        val vad = FakeVad(segmentPlan = (0 until N).associateWith { listOf(segments[it]) })
        val session = TestSherpaOfflineSession(vad)

        val got = CopyOnWriteArrayList<String>()
        val collector = launch(Dispatchers.Default) { session.partials.collect { got += it } }

        awaitCollectorSubscribed()

        repeat(N) { session.accept(tones(512)) }
        yield()
        val liveSnapshot = segments.joinToString(" ") { "len${it.samples.size}" }
        val full = session.flush(liveSnapshot)

        awaitPartialCount(got, N)
        collector.cancel()
        assertEquals("all $N segments must appear in partials", N, got.size)
        assertEquals("final transcript must contain all $N segment texts, no duplicates", N, full.split(" ").size)
    }

    @Test
    fun sherpaSessionPreservesTailSegments() = runBlocking {
        val early = (0 until 3).map { id ->
            SpeechSegment(start = id * 512, samples = FloatArray(60 + id * 5) { 1f })
        }
        val tail = listOf(
            SpeechSegment(start = 2000, samples = FloatArray(40) { 2f }),
            SpeechSegment(start = 2100, samples = FloatArray(30) { 3f }),
        )
        val vad = FakeVad(
            segmentPlan = (0 until 3).associateWith { listOf(early[it]) },
            flushSegments = tail,
        )
        val session = TestSherpaOfflineSession(vad)

        val got = CopyOnWriteArrayList<String>()
        val collector = launch(Dispatchers.Default) { session.partials.collect { got += it } }

        awaitCollectorSubscribed()

        repeat(3) { session.accept(tones(512)) }
        yield()
        val liveSnapshot = early.joinToString(" ") { "len${it.samples.size}" }
        val full = session.flush(liveSnapshot)

        awaitPartialCount(got, early.size + tail.size)
        collector.cancel()
        assertEquals("live + tail segments must all appear as partials", early.size + tail.size, got.size)
        assertEquals("tail must appear in the final transcript", early.size + tail.size, full.split(" ").size)
    }
}