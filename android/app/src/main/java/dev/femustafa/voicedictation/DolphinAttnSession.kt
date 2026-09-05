package dev.femustafa.voicedictation

import android.content.Context
import android.util.Log
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
 * A [TranscriptionSession] for the Dolphin Attention model.
 * This session uses a resident VAD and a decode worker to process audio segments.
 * It uses the existing beamSearch and decodeTokens from DolphinAttnEngine.
 *
 * @param modelRef The *resident* model ref containing encoder/decoder sessions and VAD.
 * @param waveWriter A [WaveWriter] for accumulating audio for potential whole-clip fallback.
 * @param context Android Context for accessing app-specific resources and threading.
 * @param vadLike The VAD instance to use for segmentation.
 */
internal class DolphinAttnSession(
    private val engine: DolphinAttnEngine,
    private val modelRef: DolphinAttnEngine.DolphinAttnModelRef,
    private val waveWriter: WaveWriter,
    private val context: Context,
    private val vadLike: VadLike,
) : TranscriptionSession {

    // Coroutine scope for the decode worker. Cancelled on [close].
    private val scope = CoroutineScope(Dispatchers.Default + CoroutineName("DolphinAttnSession"))

    // Channel for VAD segments. Runs on the capture thread.
    private val segments = Channel<SpeechSegment>(VAD_SEGMENT_QUEUE_SIZE)

    // Shared flow for partial transcription results.
    // Large buffer + SUSPEND strategy so bursty emits are not silently dropped —
    // the collector will catch up and the UI will always see every partial.
    private val _partials = MutableSharedFlow<String>(
        replay = 0,
        extraBufferCapacity = 32,
        onBufferOverflow = kotlinx.coroutines.channels.BufferOverflow.SUSPEND,
    )
    override val partials: SharedFlow<String> = _partials

    // Mutex for state protection during flush/close.
    private val mutex = Mutex()

    // Flag to ensure session is not used after closing.
    private var isClosed = false

    // Live VAD drainer for streaming segmentation
    private val vadDrainer = LiveVadDrainer(vadLike)

    init {
        // Launch the decode worker.
        scope.launch {
            Log.i(TAG, "decode worker started")
            for (segment in segments) {
                // Decode segment using the existing beamSearch and decodeTokens.
                val shortArray = ShortArray(segment.samples.size) {
                    (segment.samples[it] * 32767f).toInt().toShort()
                }
                val tokens = engine.beamSearch(modelRef, shortArray)
                val text = engine.decodeTokens(modelRef, tokens).trim()
                // Only emit if the session is still open. This prevents stale partials
                // from being emitted after flush() has taken over.
                if (text.isNotBlank() && !isClosed) {
                    _partials.emit(text)
                    Log.i(TAG, "emitted partial: " + text)
                }
            }
            Log.i(TAG, "decode worker stopped")
        }
    }

    override fun accept(samples: ShortArray) {
        if (isClosed) return

        // Feed samples to VAD and drain any closed segments.
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

        // Close the channel FIRST so the decode worker's next receive() throws immediately.
        segments.close()
        scope.cancel()

        // Drain the VAD tail and decode it directly.
        val tailSegments = vadDrainer.flushAndDrain()
        val tailText = decodeSegments(tailSegments)

        val finalTranscript = when {
            partialsSnapshot.isNotBlank() && tailText.isNotBlank() -> "$partialsSnapshot $tailText"
            partialsSnapshot.isNotBlank() -> partialsSnapshot
            tailText.isNotBlank() -> tailText
            else -> {
                Log.w(TAG, "No VAD segments, falling back to whole-clip decode.")
                val fullWave = waveWriter.flush()
                if (fullWave != null) {
                    val shortArray = ShortArray(fullWave.samples.size) {
                        (fullWave.samples[it] * 32767f).toInt().toShort()
                    }
                    val tokens = engine.beamSearch(modelRef, shortArray)
                    engine.decodeTokens(modelRef, tokens).trim()
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
            val shortArray = ShortArray(segment.samples.size) {
                (segment.samples[it] * 32767f).toInt().toShort()
            }
            val tokens = engine.beamSearch(modelRef, shortArray)
            val text = engine.decodeTokens(modelRef, tokens).trim()
            if (text.isNotBlank()) parts.add(text)
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
            scope.cancel() // Cancel decode worker scope
            segments.close() // Close channel
        } finally {
            mutex.unlock()
        }
    }

    private companion object {
        const val TAG = "DolphinAttnSession"
        const val VAD_SEGMENT_QUEUE_SIZE = 5
    }
}