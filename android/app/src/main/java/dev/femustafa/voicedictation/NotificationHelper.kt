package dev.femustafa.voicedictation

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

    /** Idempotent: creates the channel once on API 26+. */
    fun ensureChannel(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID, "Voice Dictation", NotificationManager.IMPORTANCE_LOW,
            )
            val nm = context.getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(channel)
        }
    }

    fun showCopied(context: Context, transcript: String) {
        post(context, NOTIF_COPIED, "Copied", transcript)
    }

    fun cancelRecording(context: Context) {
        NotificationManagerCompat.from(context).cancel(NOTIF_RECORDING)
    }

    private fun post(context: Context, id: Int, title: String, text: String) {
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
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setOnlyAlertOnce(true)
            .build()
        NotificationManagerCompat.from(context).notify(id, notification)
    }
}
