package dev.femustafa.voicedictation

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Foreground service that owns a dictation while the app may be in the
 * background (the A50 background-reliability fix a Termux process could not
 * provide). It is NEVER always-on: it starts when recording begins, holds the
 * mic + resident model hot, and stops itself once transcription is done and the
 * result is copied. Only one resident [WhisperManager] exists (in
 * [VoiceDictationApp]), so the model is not reloaded when the service takes over.
 *
 * The recording notification is the stop surface: its action stops the capture
 * and starts transcription even with the app closed.
 */
class DictationService : Service() {

    private val app: VoiceDictationApp by lazy { VoiceDictationApp.from(this) }
    private val recorder = AudioRecorder()

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

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
        serviceScope.launch {
            try {
                recorder.start(java.io.File(filesDir, "dictation.wav"))
                app.setRecording(true)
            } catch (t: Throwable) {
                app.setError(t.message ?: "recording failed")
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
                val text = whisper.transcribe(java.io.File(filesDir, "dictation.wav").absolutePath)
                app.setTranscript(text)
                val trimmed = text.trim()
                if (trimmed.isEmpty()) {
                    app.setError("No speech detected — please try again")
                } else {
                    copyToClipboard(trimmed)
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

    private fun copyToClipboard(text: String) {
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("dictation", text))
    }

    private fun notifyCopied(text: String) {
        NotificationHelper.showCopied(this, text)
    }

    private fun startForegroundCompat() {
        val stopIntent = PendingIntent.getService(this, 0, stopRecordingIntent(), PendingIntent.FLAG_IMMUTABLE)
        val notification = NotificationCompat.Builder(this, NotificationHelper.CHANNEL_ID)
            .setContentTitle("Voice Dictation")
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
        NotificationHelper.cancelRecording(this)
        serviceScope.cancel()
        super.onDestroy()
    }

    companion object {
        const val ACTION_START_RECORDING = "dev.femustafa.voicedictation.action.START_RECORDING"
        const val ACTION_STOP_RECORDING = "dev.femustafa.voicedictation.action.STOP_RECORDING"
        private const val FOREGROUND_ID = 1
    }
}