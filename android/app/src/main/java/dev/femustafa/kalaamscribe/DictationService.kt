package dev.femustafa.kalaamscribe

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

/**
 * Foreground service that owns a dictation while the app may be in the
 * background (the A50 background-reliability fix a Termux process could not
 * provide). It is NEVER always-on: it starts when recording begins, holds the
 * mic + resident model hot, and stops itself once transcription is done and the
 * result is copied. Only one resident [WhisperManager] exists (in
 * [KalaamApp]), so the model is not reloaded when the service takes over.
 *
 * The recording notification is the stop surface: its action stops the capture
 * and starts transcription even with the app closed.
 *
 * Streaming (simulated streaming ASR, ticket 31): when the active model supports
 * a [TranscriptionSession], the service opens one at recording start, wires the
 * mic's [frameListener] to push samples into the session, and collects [partials]
 * to grow the on-screen transcript in real time. At Stop, the session is flushed
 * and its final text is copied/notified exactly like the batch path.
 */
class DictationService : Service() {

    private val app: KalaamApp by lazy { KalaamApp.from(this) }
    private val recorder = AudioRecorder()

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /** Live streaming session when the active backend supports it; null → batch fallback. */
    private var streamingSession: TranscriptionSession? = null

    /** Background job that collects [TranscriptionSession.partials] to grow the transcript. */
    private var partialsJob: Job? = null

    /**
     * Live transcript accumulated from streaming partials. All reads and writes are
     * synchronized to avoid races between [partialsJob] (appending) and
     * [stopAndTranscribe] (snapshotting at Stop).
     */
    private val liveTranscript = StringBuilder()
    private val liveTranscriptLock = Any()

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        NotificationHelper.ensureChannel(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_RECORDING -> startRecording()
            ACTION_STOP_RECORDING -> stopAndTranscribe()
        }
        return START_NOT_STICKY
    }

    /** Start capturing (once), showing the foreground recording notification. */
    private fun startRecording() {
        if (app.recording.value) return
        startForegroundCompat()
        app.setError(null)
        app.setTranscript(null)
        app.onRecordingStarted()
        // Set recording=true IMMEDIATELY so the UI flips to the "Listening…" state
        // without waiting for the model to load. The mic + session are wired up in
        // the coroutine below; while that runs, the user may already be speaking.
        app.setRecording(true)

        // Open a live streaming session only when the user chose simulated streaming
        // AND the backend supports it. Batch mode always takes the whole-clip path;
        // a backend that cannot stream (e.g. GGML) silently falls back to batch too.
        val wantStreaming = app.transcriptionMode == TranscriptionMode.SimulatedStreaming
        serviceScope.launch {
            val session = if (wantStreaming) {
                try {
                    app.whisper.createSession()
                } catch (t: Throwable) {
                    Log.w(TAG, "createSession failed; falling back to batch: ${t.message}")
                    null
                }
            } else {
                null
            }
            streamingSession = session

            if (session != null) {
                synchronized(liveTranscriptLock) { liveTranscript.clear() }
                // Wire the mic capture to feed the session with each PCM frame.
                recorder.frameListener = { samples -> session.accept(samples) }
                // Collect partials on the service scope and grow the on-screen transcript
                // as each VAD-closed utterance decodes.
                partialsJob = serviceScope.launch {
                    session.partials.collectLatest { partial ->
                        // Append to the running transcript atomically so stopAndTranscribe can
                        // safely snapshot it without a race.
                        synchronized(liveTranscriptLock) {
                            if (liveTranscript.isNotEmpty()) {
                                liveTranscript.append(' ')
                            }
                            liveTranscript.append(partial)
                            app.setTranscript(liveTranscript.toString())
                        }
                        Log.i(TAG, "live partial: $partial")
                    }
                }
            } else {
                recorder.frameListener = null
                if (wantStreaming) {
                    Log.e(TAG, "streaming session unavailable (missing VAD or engine without streaming support); fell back to whole-clip batch")
                }
            }

            try {
                recorder.start(java.io.File(filesDir, "dictation.wav"))
            } catch (t: Throwable) {
                app.setError(t.message ?: "recording failed")
                app.setRecording(false)
                stopSelf()
            }
        }
    }

    /** Stop the mic and transcribe with the resident model; copy + notify. */
    private fun stopAndTranscribe() {
        serviceScope.launch {
            recorder.stop()
            app.setRecording(false)

            val whisper = app.whisper
            if (whisper.currentModel == null) {
                app.setError("No model loaded — pick a model first")
                stopSelf()
                return@launch
            }

            app.setTranscribing(true)
            try {
                val start = SystemClock.elapsedRealtime()
                val text: String = streamingSession?.let { session ->
                    // Snapshot the partials accumulated during recording. This is read atomically
                    // from the synchronized liveTranscript, so we get exactly what the UI showed.
                    val partialsSnapshot: String
                    synchronized(liveTranscriptLock) {
                        partialsSnapshot = liveTranscript.toString()
                    }
                    // Cancel the collector, clear the frame listener, flush the session
                    // (appending only the tail segments the decode worker hasn't emitted yet),
                    // then close.
                    partialsJob?.cancel()
                    partialsJob = null
                    recorder.frameListener = null
                    val flushed = session.flush(partialsSnapshot)
                    try {
                        session.close()
                    } catch (t: Throwable) {
                        Log.w(TAG, "session.close() failed: ${t.message}")
                    }
                    streamingSession = null
                    flushed
                } ?: run {
                    // Batch fallback: no live session, run the whole-clip path.
                    whisper.transcribe(java.io.File(filesDir, "dictation.wav").absolutePath)
                }
                app.setLastTranscriptionMs(SystemClock.elapsedRealtime() - start)
                app.setTranscript(text)
                val trimmed = text.trim()
                if (trimmed.isEmpty()) {
                    app.setError("No speech detected — please try again")
                } else {
                    app.copyToClipboard(trimmed)
                    val modelId = whisper.currentModel?.id ?: "unknown"
                    app.appendHistory(
                        DictationHistoryItem(
                            text = trimmed,
                            time = java.time.LocalTime.now()
                                .format(java.time.format.DateTimeFormatter.ofPattern("HH:mm")),
                            modelId = modelId,
                        ),
                    )
                    notifyCopied(trimmed)
                }
            } catch (t: Throwable) {
                app.setError(t.message ?: "transcription failed")
            } finally {
                app.setTranscribing(false)
            }
            stopSelf()
        }
    }

    private fun notifyCopied(text: String) {
        NotificationHelper.showCopied(this, text)
    }

    private fun startForegroundCompat() {
        val stopIntent = PendingIntent.getService(this, 0, stopRecordingIntent(), PendingIntent.FLAG_IMMUTABLE)
        val notification = NotificationCompat.Builder(this, NotificationHelper.CHANNEL_ID)
            .setContentTitle("Kalaam Scribe")
            .setContentText("Recording… tap Stop to transcribe")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(stopIntent)
            .addAction(0, "Stop", stopIntent)
            .setOngoing(true)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(FOREGROUND_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
        } else {
            startForeground(FOREGROUND_ID, notification)
        }
    }

    private fun stopRecordingIntent(): Intent =
        Intent(this, DictationService::class.java).setAction(ACTION_STOP_RECORDING)

    override fun onDestroy() {
        recorder.stop()
        // If the service is killed while recording, drop the live session cleanly so
        // the resident model is not held hostage by an abandoned session.
        partialsJob?.cancel()
        streamingSession?.close()
        streamingSession = null
        NotificationHelper.cancelRecording(this)
        serviceScope.cancel()
        super.onDestroy()
    }

    companion object {
        const val ACTION_START_RECORDING = "dev.femustafa.kalaamscribe.action.START_RECORDING"
        const val ACTION_STOP_RECORDING = "dev.femustafa.kalaamscribe.action.STOP_RECORDING"
        private const val FOREGROUND_ID = 1
        private const val TAG = "DictationService"
    }
}