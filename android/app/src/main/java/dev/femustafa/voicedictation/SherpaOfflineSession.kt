package dev.femustafa.voicedictation

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import kotlinx.coroutines.CoroutineName
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
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
 */
internal class SherpaOfflineSession(
    private val recognizer: OfflineRecognizer,
    private val waveWriter: WaveWriter,
    private val language: String,
    private val modelRef: SherpaWhisperEngine.SherpaModelRef?, // Null for Dolphin CTC
    private val context: Context,
    private val vadLike: VadLike,
) : TranscriptionSession {

    // Internal coroutine scope for the decode worker. Cancelled on [close].
    private val scope = CoroutineScope(Dispatchers.Default + CoroutineName("SherpaOfflineSession"))

    // VAD runs on the capture thread and pushes segments here.
    private val segments = Channel<SpeechSegment>(VAD_SEGMENT_QUEUE_SIZE)

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
    private var isClosed = false

    // Live VAD drainer for streaming segmentation
    private val vadDrainer = LiveVadDrainer(vadLike)

    init {
        // If this is a Whisper model, apply the language. Dolphin CTC ignores it.
        modelRef?.let {
            // Re-create SherpaWhisperEngine to access its internal methods like applyLanguage
            // This is a workaround as applyLanguage is currently private/internal to SherpaWhisperEngine
            val sherpaEngine = SherpaWhisperEngine(context)
            sherpaEngine.applyLanguage(it, language)
        }

        // Launch the decode worker.
        scope.launch {
            Log.i(TAG, "decode worker started")
            for (segment in segments) {
                // Decode segment and emit to partials.
                // We use a separate stream per segment to avoid state leakage and
                // simplify threading, as per sherpa-onnx examples.
                val stream = recognizer.createStream()
                try {
                    stream.acceptWaveform(segment.samples, segment.sampleRate)
                    recognizer.decode(stream)
                    val text = recognizer.getResult(stream).text.trim()
                    // Only emit if the session is still open (not flushed/closed).
                    // This prevents the decode worker from emitting stale partials
                    // after flush() has taken over and cancelled this coroutine.
                    if (text.isNotBlank() && !isClosed) {
                        _partials.emit(text)
                        Log.i(TAG, "emitted partial: " + text)
                    }
                } finally {
                    stream.release()
                }
            }
            Log.i(TAG, "decode worker stopped")
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
            if (!segments.trySend(segment).isSuccess) {
                Log.w(TAG, "VAD segment queue full, dropping segment.")
            }
        }
    }

    override suspend fun flush(partialsSnapshot: String): String = mutex.withLock {
        if (isClosed) return ""
        isClosed = true
        Log.i(TAG, "flush: stopping decode worker, appending tail segments to partials snapshot")

        // Close the channel FIRST so the decode worker's next receive() throws immediately,
        // and the worker exits without fetching a new segment. The worker has already
        // emitted all segments currently in the channel — we only decode the TAIL here.
        segments.close()
        scope.cancel()

        // Drain the VAD tail and decode it directly.
        val tailSegments = vadDrainer.flushAndDrain()
        val tailText = decodeSegments(tailSegments)

        // Combine: partialsSnapshot has what the UI already showed (all worker-emitted
        // partials); tailText has the trailing utterance(s) the worker hasn't seen yet.
        val finalTranscript = when {
            partialsSnapshot.isNotBlank() && tailText.isNotBlank() -> "$partialsSnapshot $tailText"
            partialsSnapshot.isNotBlank() -> partialsSnapshot
            tailText.isNotBlank() -> tailText
            else -> {
                // No segments decoded at all — fall back to whole-clip decode.
                Log.w(TAG, "No VAD segments, falling back to whole-clip decode.")
                val fullWave = waveWriter.flush()
                if (fullWave != null) {
                    val stream = recognizer.createStream()
                    try {
                        stream.acceptWaveform(fullWave.samples, fullWave.sampleRate)
                        recognizer.decode(stream)
                        recognizer.getResult(stream).text.trim()
                    } finally {
                        stream.release()
                    }
                } else ""
            }
        }

        Log.i(TAG, "flush completed, final transcript: " + finalTranscript)
        return finalTranscript
    }

    private fun decodeSegments(segments: List<SpeechSegment>): String {
        if (segments.isEmpty()) return ""
        val parts = mutableListOf<String>()
        for (segment in segments) {
            val stream = recognizer.createStream()
            try {
                stream.acceptWaveform(segment.samples, segment.sampleRate)
                recognizer.decode(stream)
                val text = recognizer.getResult(stream).text.trim()
                if (text.isNotBlank()) parts.add(text)
            } finally {
                stream.release()
            }
        }
        return parts.joinToString(" ")
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
        const val VAD_SEGMENT_QUEUE_SIZE = 5
    }
}