package dev.femustafa.voicedictation

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineRecognizer
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
 * A [TranscriptionSession] for sherpa-onnx [OfflineRecognizer]s (Whisper, Dolphin CTC).
 * It runs a [SherpaVad] internally, feeding it mic samples on the capture thread, then
 * queuing closed segments to an internal channel. A single decode worker drains the
 * channel, decodes segments, and emits them to [partials].
 *
 * @param recognizer The *resident* recognizer to use for decoding segments.
 * @param waveWriter A [WaveWriter] for accumulating all audio; used for whole-clip
 *   fallback (when the VAD produces zero segments) and WAV file saving.
 * @param language The BCP-47 language tag to use for Whisper models ("" = auto-detect).
 * @param modelRef The underlying sherpa-onnx model ref; only needed for [applyLanguage].
 * @param context Android Context to derive the app singleton for threads.
 * @param vadLike The VAD instance to use for segmentation.
 *
 * Testing: the secondary ctor accepts a [SherpaRecognizerLike] so tests can drive
 * the session with a fake without needing the sherpa native lib.
 */

/**
 * Minimal surface of sherpa-onnx's [OfflineRecognizer] used by [SherpaOfflineSession]
 * to decode a single segment. Extracting this seam lets unit tests drive the
 * session with a fake recognizer (the sherpa native lib isn't loadable on the JVM).
 */
internal interface SherpaRecognizerLike {
    /** Decode [samples] and return the transcript. One-shot per call. */
    fun decode(samples: FloatArray, sampleRate: Int): String
}

internal open class SherpaOfflineSession(
    private val recognizer: OfflineRecognizer?,
    private val recognizerLike: SherpaRecognizerLike?,
    private val waveWriter: WaveWriter,
    private val language: String,
    private val modelRef: SherpaWhisperEngine.SherpaModelRef?, // Null for Dolphin CTC
    private val context: Context?,
    private val vadLike: VadLike,
) : TranscriptionSession {

    /** Production ctor: keep the existing call site unchanged by wrapping a real
     *  [OfflineRecognizer] in a [SherpaRecognizerAdapter]. */
    constructor(
        recognizer: OfflineRecognizer,
        waveWriter: WaveWriter,
        language: String,
        modelRef: SherpaWhisperEngine.SherpaModelRef?,
        context: Context,
        vadLike: VadLike,
    ) : this(
        recognizer = recognizer,
        recognizerLike = null,
        waveWriter = waveWriter,
        language = language,
        modelRef = modelRef,
        context = context,
        vadLike = vadLike,
    )

    /** Test ctor: skip the language-apply init, no recognizer needed. */
    constructor(
        recognizerLike: SherpaRecognizerLike,
        waveWriter: WaveWriter,
        vadLike: VadLike,
    ) : this(
        recognizer = null,
        recognizerLike = recognizerLike,
        waveWriter = waveWriter,
        language = "",
        modelRef = null,
        context = null,
        vadLike = vadLike,
    )

    private inner class SherpaRecognizerAdapter : SherpaRecognizerLike {
        override fun decode(samples: FloatArray, sampleRate: Int): String {
            val rec = requireNotNull(recognizer) { "no recognizer in this session" }
            val stream = rec.createStream()
            return try {
                stream.acceptWaveform(samples, sampleRate)
                rec.decode(stream)
                rec.getResult(stream).text.trim()
            } finally {
                stream.release()
            }
        }
    }

    /** Resolve the decode hook: prefer the test seam if set, else wrap the sherpa recognizer. */
    private val decodeHook: SherpaRecognizerLike by lazy {
        recognizerLike ?: SherpaRecognizerAdapter()
    }

    // Internal coroutine scope for the decode worker. Cancelled on [close].
    private val scope = CoroutineScope(Dispatchers.Default + CoroutineName("SherpaOfflineSession"))

    // VAD runs on the capture thread and pushes segments here. UNLIMITED so no
    // segment is ever dropped on a slow decode worker — loss would corrupt the
    // final joined transcript. Decode outpaces capture in practice.
    private val segments = Channel<SpeechSegment>(Channel.UNLIMITED)

    // The decode worker emits completed phrases here, for the UI to collect.
    // Large buffer + SUSPEND strategy so bursty emits (e.g. several utterances
    // decoded back-to-back after a long silence) are not silently dropped — the
    // collector will catch up and the UI will always see every partial.
    private val _partials = MutableSharedFlow<String>(
        replay = 0,
        extraBufferCapacity = 32,
        onBufferOverflow = kotlinx.coroutines.channels.BufferOverflow.SUSPEND,
    )
    override val partials: SharedFlow<String> = _partials

    // Mutex to protect state during flush/close.
    private val mutex = Mutex()

    // Flag to ensure flush/close are called only once.
    @Volatile
    private var isClosed = false

    /** Every non-blank text the worker decoded, in capture order. The authoritative
     *  source for [flush]'s final transcript — the async [partials] flow is best-effort. */
    private val decodedTexts = ArrayList<String>()

    // Live VAD drainer for streaming segmentation
    private val vadDrainer = LiveVadDrainer(vadLike)

    // The decode worker. Kept so [flush] can join it and thus never drop segments
    // still queued when recording stops.
    private val worker: Job = scope.launch {
        Log.i(TAG, "decode worker started")
        for (segment in segments) {
            val text = decodeHook.decode(segment.samples, segment.sampleRate)
            if (text.isNotBlank()) {
                decodedTexts += text
                // Best-effort UI delivery (the authoritative text is decodedTexts, which
                // flush reads after joining the worker). Emitting during flush is fine —
                // the service has already cancelled the UI collector by then.
                _partials.emit(text)
                Log.i(TAG, "emitted partial: " + text)
            }
        }
        Log.i(TAG, "decode worker stopped")
    }

    init {
        // Apply language for Whisper models (Dolphin CTC ignores language).
        // Skip when the test seam was used (recognizer == null).
        if (recognizer != null && modelRef != null) {
            val sherpaEngine = SherpaWhisperEngine(context!!)
            sherpaEngine.applyLanguage(modelRef, language)
        }
    }

    override fun accept(samples: ShortArray) {
        if (isClosed) return // Already closed, ignore samples

        // Feed samples to VAD and drain any closed segments.
        // Assuming 16kHz sample rate for the ShortArray conversion.
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
                // No segments decoded at all — fall back to whole-clip decode.
                Log.w(TAG, "No VAD segments, falling back to whole-clip decode.")
                val fullWave = waveWriter.flush()
                if (fullWave != null) {
                    decodeHook.decode(fullWave.samples, fullWave.sampleRate)
                } else ""
            }

        Log.i(TAG, "flush completed, final transcript: " + finalTranscript)
        return finalTranscript
    }

    override fun close() {
        // Use tryLock so close() remains a non-suspend function.
        if (!mutex.tryLock()) return
        try {
            if (isClosed) return
            isClosed = true
            Log.i(TAG, "close: cancelling decode worker")
            scope.cancel() // Cancel the decode worker's scope
            segments.close() // Close the channel
        } finally {
            mutex.unlock()
        }
    }

    private companion object {
        const val TAG = "SherpaOfflineSession"
    }
}