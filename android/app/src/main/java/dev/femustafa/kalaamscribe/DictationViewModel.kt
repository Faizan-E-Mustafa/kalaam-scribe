package dev.femustafa.kalaamscribe

import android.app.Application
import android.content.Intent
import android.net.Uri
import android.os.SystemClock
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import java.io.File
import java.io.FileOutputStream
import java.time.LocalTime
import java.time.format.DateTimeFormatter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Launcher for dictation on the [DictationService] (a foreground service, so
 * recording and transcription survive switching apps). The mic ownership
 * and the resident-model work live in the service; this ViewModel only starts it
 * and mirrors the shared state the service publishes on [KalaamApp].
 *
 * Also transcribes a user-picked WAV/MP3 file in-app (no service): the file is
 * decoded to the canonical 16 kHz PCM format and run through the resident model,
 * with results landing in the same shared state as dictation.
 *
 * We use [KalaamApp]'s [StateFlow]s directly — no duplicated state.
 */
class DictationViewModel(
    application: Application,
) : AndroidViewModel(application) {

    private val app: KalaamApp
        get() = KalaamApp.from(getApplication())

    val recording: StateFlow<Boolean> = app.recording
    val transcribing: StateFlow<Boolean> = app.transcribing
    val transcript: StateFlow<String?> = app.transcript
    val lastTranscriptionMs: StateFlow<Long?> = app.lastTranscriptionMs
    val error: StateFlow<String?> = app.error
    val history: StateFlow<List<DictationHistoryItem>> = app.history

    /**
     * Transcribe a user-picked audio file ([uri], WAV or MP3) with the resident
     * model. The picked file is copied into app storage, decoded to a 16 kHz mono
     * PCM WAV by [AudioDecoder], then transcribed on a background dispatcher.
     * Mirrors [DictationService]'s result handling: transcript card + clipboard +
     * history. The picker grants read access to the chosen file, so no extra
     * permission is needed.
     */
    fun transcribeFile(uri: Uri) {
        val ctx = getApplication<Application>()
        viewModelScope.launch {
            app.setError(null)
            app.setTranscript(null)
            if (app.recording.value || app.transcribing.value) {
                app.setError("Wait for the current dictation to finish")
                return@launch
            }
            if (app.whisper.currentModel == null) {
                app.setError("No model loaded — pick a model first")
                return@launch
            }
            app.setTranscribing(true)
            try {
                val start = SystemClock.elapsedRealtime()
                val text = withContext(Dispatchers.IO) {
                    val src = File(ctx.filesDir, "upload-source")
                    val input = ctx.contentResolver.openInputStream(uri)
                        ?: throw java.io.IOException("Could not open the selected file")
                    input.use { srcInput ->
                        FileOutputStream(src).use { srcOutput -> srcInput.copyTo(srcOutput) }
                    }
                    val decoded = AudioDecoder.decodeToMono16kWav(
                        src.absolutePath,
                        File(ctx.filesDir, "upload_16k.wav").absolutePath,
                    )
                    app.whisper.transcribe(decoded)
                }
                app.setLastTranscriptionMs(SystemClock.elapsedRealtime() - start)
                app.setTranscript(text)
                val trimmed = text.trim()
                if (trimmed.isEmpty()) {
                    app.setError("No speech detected in the selected file")
                } else {
                    app.copyToClipboard(trimmed)
                    app.appendHistory(
                        DictationHistoryItem(
                            text = trimmed,
                            time = LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm")),
                            modelId = app.whisper.currentModel?.id ?: "unknown",
                        ),
                    )
                }
            } catch (t: Throwable) {
                app.setError(t.message ?: "failed to transcribe file")
            } finally {
                app.setTranscribing(false)
            }
        }
    }

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