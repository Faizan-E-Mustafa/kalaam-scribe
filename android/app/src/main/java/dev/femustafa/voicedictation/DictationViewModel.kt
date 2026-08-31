package dev.femustafa.voicedictation

import android.app.Application
import android.content.Intent
import androidx.lifecycle.AndroidViewModel
import kotlinx.coroutines.flow.StateFlow

/**
 * Launcher for dictation on the [DictationService] (a foreground service, so
 * recording and transcription survive switching to WhatsApp). The mic ownership
 * and the resident-model work live in the service; this ViewModel only starts it
 * and mirrors the shared state the service publishes on [VoiceDictationApp].
 *
 * We use [VoiceDictationApp]'s [StateFlow]s directly — no duplicated state.
 */
class DictationViewModel(
    application: Application,
) : AndroidViewModel(application) {

    private val app: VoiceDictationApp
        get() = VoiceDictationApp.from(getApplication())

    val recording: StateFlow<Boolean> = app.recording
    val transcribing: StateFlow<Boolean> = app.transcribing
    val transcript: StateFlow<String?> = app.transcript
    val error: StateFlow<String?> = app.error

    /** Toggle dictation. The caller must have already secured the mic permission. */
    fun toggleRecording() {
        val ctx = getApplication<Application>()
        val action = if (app.recording.value)
            DictationService.ACTION_STOP_RECORDING
        else
            DictationService.ACTION_START_RECORDING

        val intent = Intent(ctx, DictationService::class.java).setAction(action)
        if (app.recording.value) {
            ctx.startService(intent)
        } else {
            ctx.startForegroundService(intent)
        }
    }
}