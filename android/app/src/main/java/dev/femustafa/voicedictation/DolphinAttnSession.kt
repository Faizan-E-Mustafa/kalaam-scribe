package dev.femustafa.voicedictation

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineName
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * A [TranscriptionSession] for the Dolphin Attention model.
 * This session uses a resident VAD and a decode worker to process audio segments.
 * It uses the existing beamSearch and decodeTokens from DolphinAttnEngine.
 *
 * @param engine The *resident* engine whose beamSearch/decodeTokens decode segments.
 * @param modelRef The *resident* model ref containing encoder/decoder sessions and VAD.
 * @param waveWriter A [WaveWriter] for accumulating audio for potential whole-clip fallback.
 * @param context Android Context for accessing app-specific resources and threading.
 * @param vadLike The VAD instance to use for segmentation.
 *
 * Testing: the secondary ctor accepts a [DolphinRecognizerLike] so tests can drive
 * the session with a fake decode without the ONNX runtime.
 */
internal open class DolphinAttnSession(
    private val engine: DolphinAttnEngine?,
    private val recognizerLike: DolphinRecognizerLike?,
    private val modelRef: DolphinAttnEngine.DolphinAttnModelRef?,
    private val waveWriter: WaveWriter,
    private val context: Context?,
    private val vadLike: VadLike,
) : TranscriptionSession {

    /** Production ctor: keeps the existing call site unchanged. The engine is a
     *  [DolphinRecognizerLike], so it doubles as the decode hook. */
    constructor(
        engine: DolphinAttnEngine,
        modelRef: DolphinAttnEngine.DolphinAttnModelRef,
        waveWriter: WaveWriter,
        context: Context?,
        vadLike: VadLike,
    ) : this(
        engine = engine,
        recognizerLike = null,
        modelRef = modelRef,
        waveWriter = waveWriter,
        context = context,
        vadLike = vadLike,
    )

    /** Test ctor: skip the engine and drive decoding through a fake [DolphinRecognizerLike]. */
    constructor(
        recognizerLike: DolphinRecognizerLike,
        modelRef: DolphinAttnEngine.DolphinAttnModelRef,
        waveWriter: WaveWriter,
        vadLike: VadLike,
    ) : this(
        engine = null,
        recognizerLike = recognizerLike,
        modelRef = modelRef,
        waveWriter = waveWriter,
        context = null,
        vadLike = vadLike,
    )

    /** Resolve the decode hook: prefer the test seam if set, else the engine. */
    private val decodeHook: DolphinRecognizerLike by lazy {
        recognizerLike ?: requireNotNull(engine) { "no engine in this session" }
    }

    // Coroutine scope for the decode worker. Cancelled on [close].
    private val scope = CoroutineScope(Dispatchers.Default + CoroutineName("DolphinAttnSession"))

    // Channel for VAD segments. Runs on the capture thread. UNLIMITED so no
    // segment is ever dropped on a slow decode worker — loss would corrupt the
    // final joined transcript. Decode outpaces capture in practice.
    private val segments = Channel<SpeechSegment>(Channel.UNLIMITED)

    // Shared flow for partial transcription results.
    // Large buffer + SUSPEND strategy so bursty emits are not silently dropped —
    // the collector will catch up and the UI will always see every partial.
    private val _partials = MutableSharedFlow<String>(
        replay = 0,
        extraBufferCapacity = 32,
        onBufferOverflow = kotlinx.coroutines.channels.BufferOverflow.SUSPEND,
    )
    override val partials: SharedFlow<String> = _partials

    /** Every non-blank text the worker decoded, in capture order. The authoritative
     *  source for [flush]'s final transcript — the async [partials] flow is best-effort. */
    private val decodedTexts = ArrayList<String>()

    // Mutex for state protection during flush/close.
    private val mutex = Mutex()

    // Flag to ensure session is not used after closing.
    @Volatile
    private var isClosed = false

    // Live VAD drainer for streaming segmentation
    private val vadDrainer = LiveVadDrainer(vadLike)

    // The decode worker. Kept so [flush] can join it and thus never drop segments
    // still queued when recording stops.
    private val worker: Job = scope.launch {
        Log.i(TAG, "decode worker started")
        for (segment in segments) {
            val text = try {
                decodeSegment(segment)
            } catch (e: Throwable) {
                Log.e(TAG, "decode worker: segment decode failed", e)
                null
            }
            if (text?.isNotBlank() == true) {
                decodedTexts += text
                // Best-effort UI delivery (the authoritative text is decodedTexts, which
                // flush reads after joining the worker). Emitting during flush is fine —
                // the service has already cancelled the UI collector by then.
                _partials.tryEmit(text)
                Log.i(TAG, "emitted partial: " + text)
            }
        }
        Log.i(TAG, "decode worker stopped")
    }

    override fun accept(samples: ShortArray) {
        if (isClosed) return

        // Feed samples to VAD and drain any closed segments.
        val floatSamples = samples.toFloatArray(16000)
        waveWriter.accept(samples)
        val newSegments = vadDrainer.push(floatSamples)
        newSegments.forEach { segment ->
            segments.trySend(segment)
        }
    }

    // partialsSnapshot is accepted to match the TranscriptionSession signature but is
    // intentionally unused: decodedTexts (complete after worker.join) is authoritative.
    override suspend fun flush(_partialsSnapshot: String): String = mutex.withLock {
        if (isClosed) return ""
        isClosed = true
        Log.i(TAG, "flush: draining tail segments into the worker, then joining it")

        // Drain the VAD tail into the channel so the worker decodes it in order.
        vadDrainer.flushAndDrain().forEach { segments.trySend(it) }

        // Close then JOIN the worker: every queued segment (live + tail) is decoded.
        // Cancelling without a join would drop segments still buffered on a slow worker —
        // exactly the loss this channel/worker design must prevent.
        segments.close()
        worker.join()
        scope.cancel()

        // decodedTexts is the authoritative, complete transcript: the worker was joined, so
        // it contains every live AND tail segment in capture order. partialsSnapshot only
        // reflects what the UI showed live (a subset of decodedTexts), so prepending it would
        // duplicate segments — decodedTexts already includes them. We ignore it here.
        val finalTranscript =
            if (decodedTexts.isNotEmpty()) decodedTexts.joinToString(" ")
            else {
                Log.w(TAG, "No VAD segments, falling back to whole-clip decode.")
                val fullWave = waveWriter.flush()
                if (fullWave != null) decodeSegment(SpeechSegment(0, fullWave.samples)) else ""
            }

        Log.i(TAG, "flush completed, final transcript: " + finalTranscript)
        return finalTranscript
    }

    private fun decodeSegment(segment: SpeechSegment): String {
        val shortArray = ShortArray(segment.samples.size) {
            (segment.samples[it] * 32767f).toInt().toShort()
        }
        val tokens = decodeHook.beamSearch(modelRef!!, shortArray)
        return decodeHook.decodeTokens(modelRef!!, tokens).trim()
    }

    override fun close() {
        // Use tryLock so close() remains a non-suspend function.
        if (!mutex.tryLock()) return
        try {
            if (isClosed) return
            isClosed = true
            Log.i(TAG, "close: cancelling decode worker")
            scope.cancel() // Cancel decode worker scope
            segments.close() // Close channel
        } finally {
            mutex.unlock()
        }
    }

    private companion object {
        const val TAG = "DolphinAttnSession"
    }
}
