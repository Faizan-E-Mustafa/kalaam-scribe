package dev.femustafa.voicedictation

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch
import kotlin.math.min

/**
 * Shared implementation of the simulated-streaming [TranscriptionSession] (ticket 31):
 *
 *  - [accept] (mic capture thread) converts the incoming shorts to normalized floats,
 *    feeds them window-by-window into the [LiveVadDrainer], and enqueues each closed VAD
 *    segment onto an unbounded channel. It never decodes — the VAD is the only native
 *    work it touches.
 *  - A single background worker decodes segments in FIFO order via [decodeSegment],
 *    appending each non-blank text (in order) to the running join and emitting it on
 *    [partials].
 *  - [flush] (service coroutine, after the recorder has stopped) drains the VAD tail,
 *    closes the channel, awaits the worker, and returns the joined transcript. If no
 *    segment produced text, it falls back to a whole-clip decode of the accumulated
 *    audio via [decodeWhole] — preserving the ticket-30 fallback semantics.
 *
 * The decoder hooks are constructor lambdas, so subclasses stay thin and the ordering
 * logic is JVM-testable without any native model.
 */
internal class EngineTranscriptionSession(
    private val vad: VadLike,
    private val decodeSegment: suspend (FloatArray) -> String,
    private val decodeWhole: suspend (FloatArray) -> String,
    private val logger: (String, String) -> Unit = { tag, msg -> android.util.Log.w(tag, msg) },
) : TranscriptionSession {

    private val drainer = LiveVadDrainer(vad)

    /** Worker owns all decoding; unbounded so no segment is ever dropped (loss would
     *  corrupt the final joined transcript). Decode outpaces capture in practice. */
    private val segmentChannel = Channel<FloatArray>(Channel.UNLIMITED)

    private val _partials = MutableSharedFlow<String>(
        extraBufferCapacity = PARTIAL_BUFFER,
        replay = PARTIAL_BUFFER,
    )
    override val partials: SharedFlow<String> = _partials.asSharedFlow()

    private val decodedTexts = ArrayList<String>()

    /** Whole accepted audio so far (normalized floats), for the no-speech fallback. */
    private var whole = FloatArray(INITIAL_WHOLE)
    private var wholeSize = 0

    @Volatile
    private var closed = false

    /** Owned by this session; cancelled in [flush] (session done) or [close]. */
    private val jobScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private val worker = jobScope.launch {
        for (segment in segmentChannel) {
            try {
                val text = decodeSegment(segment)
                if (text.isNotBlank()) {
                    decodedTexts += text
                    // Best-effort UI delivery; the authoritative text lives in decodedTexts.
                    _partials.tryEmit(text)
                }
            } catch (t: Throwable) {
                logger(TAG, "per-segment decode failed, continuing: ${t.message}")
            }
        }
    }

    override fun accept(samples: ShortArray) {
        if (closed || samples.isEmpty()) return
        val floats = FloatArray(samples.size) { samples[it] / 32768f }
        appendWhole(floats)
        var i = 0
        while (i < floats.size) {
            val end = min(i + VAD_WINDOW, floats.size)
            for (utterance in drainer.push(floats.copyOfRange(i, end))) {
                segmentChannel.trySend(utterance.samples)
            }
            i = end
        }
    }

    override suspend fun flush(partialsSnapshot: String): String {
        // The recorder has already stopped, so no accept() can run concurrently here.
        // Drain any trailing VAD tail into the channel for the worker to process.
        for (utterance in drainer.flushAndDrain()) {
            segmentChannel.trySend(utterance.samples)
        }
        segmentChannel.close()
        worker.join()
        // decodedTexts holds everything the worker emitted. partialsSnapshot is the text
        // the UI already showed. Combine them — the worker may have emitted segments
        // after we captured partialsSnapshot, but not before.
        val joined = decodedTexts.joinToString(" ")
        val full = when {
            partialsSnapshot.isNotBlank() && joined.isNotBlank() -> "$partialsSnapshot $joined"
            partialsSnapshot.isNotBlank() -> partialsSnapshot
            joined.isNotBlank() -> joined
            else -> {
                logger(TAG, "no VAD segment produced text; whole-clip fallback decode")
                decodeWhole(wholeAudio())
            }
        }
        jobScope.cancel()
        return full
    }

    override fun close() {
        closed = true
        segmentChannel.close()
        jobScope.cancel()
    }

    private fun appendWhole(chunk: FloatArray) {
        if (wholeSize + chunk.size > whole.size) {
            val newSize = maxOf(whole.size * 2, wholeSize + chunk.size)
            whole = whole.copyOf(newSize)
        }
        chunk.copyInto(whole, wholeSize)
        wholeSize += chunk.size
    }

    private fun wholeAudio(): FloatArray = whole.copyOf(wholeSize)

    private companion object {
        const val TAG = "EngineTranscriptionSession"
        const val PARTIAL_BUFFER = 16
        const val INITIAL_WHOLE = 8192
    }
}