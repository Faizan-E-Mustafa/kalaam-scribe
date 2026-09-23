package dev.femustafa.kalaamscribe

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat

/**
 * Posts the informational notifications called for by the dictation flow (port of
 * dictate.sh's notify): a "Recording… tap Stop" notice while capturing, and a
 * "Copied" notice with the transcript when done. Plain notifications — no
 * foreground service, no automessaging (ADR 0001 boundary: manual send only).
 */
object NotificationHelper {
    const val CHANNEL_ID = "dictation"
    private const val NOTIF_RECORDING = 1
    private const val NOTIF_COPIED = 2
    private const val TAIL_CHARS = 140

    /** Idempotent: creates the channel once on API 26+. */
    fun ensureChannel(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID, "Kalaam Scribe", NotificationManager.IMPORTANCE_LOW,
            )
            val nm = context.getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(channel)
        }
    }

    fun showCopied(context: Context, transcript: String) {
        // The collapsed notification shows only ~2 lines of contentText, so putting the
        // TAIL there proves the dictation ended with real words (the head is visible on
        // the screen anyway). The expanded BigTextStyle still carries the full text.
        val tail = if (transcript.length <= TAIL_CHARS) transcript else {
            val cut = transcript.lastIndexOf(' ', transcript.length - TAIL_CHARS)
            "…" + transcript.substring(if (cut > 0) cut + 1 else transcript.length - TAIL_CHARS)
        }
        post(context, NOTIF_COPIED, "Copied (${transcript.length} chars)", tail, transcript)
    }

    fun cancelRecording(context: Context) {
        NotificationManagerCompat.from(context).cancel(NOTIF_RECORDING)
    }

    private fun post(context: Context, id: Int, title: String, text: String, bigText: String) {
        if (Build.VERSION.SDK_INT >= 33 &&
            context.checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) !=
            android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            return // best effort: notification just won't show without the permission
        }
        val notification = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(bigText))
            .setOnlyAlertOnce(true)
            .build()
        NotificationManagerCompat.from(context).notify(id, notification)
    }
}
