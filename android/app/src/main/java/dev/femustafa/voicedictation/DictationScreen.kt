package dev.femustafa.voicedictation

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat

/**
 * Dictation surface: shows the resident Model, Record/Stop control, and the
 * transcript (or an error). Requests the mic runtime permission on first use, and
 * the notification permission on API 33+. Stopping transcribes via the resident
 * model and copies the result to the clipboard (ticket 12).
 */
@Composable
fun DictationScreen(
    viewModel: DictationViewModel,
    currentModel: Model?,
    onOpenPicker: () -> Unit,
) {
    val recording by viewModel.recording.collectAsState()
    val transcribing by viewModel.transcribing.collectAsState()
    val transcript by viewModel.transcript.collectAsState()
    val error by viewModel.error.collectAsState()

    val context = LocalContext.current

    val micLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted -> if (granted) viewModel.toggleRecording() }

    val notifLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { /* best effort */ }

    fun maybeRequestNotifications() {
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            notifLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    fun onRecordPressed() {
        val granted = ContextCompat.checkSelfPermission(
            context, Manifest.permission.RECORD_AUDIO,
        ) == PackageManager.PERMISSION_GRANTED
        if (granted) {
            maybeRequestNotifications()
            viewModel.toggleRecording()
        } else {
            micLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(text = "Voice Dictation", style = MaterialTheme.typography.headlineMedium)
        Text(
            text = "Model: ${currentModel?.id ?: "none"}",
            modifier = Modifier.padding(top = 8.dp),
            style = MaterialTheme.typography.bodyMedium,
        )

        Button(
            onClick = { onRecordPressed() },
            modifier = Modifier.padding(top = 24.dp),
        ) {
            Text(
                when {
                    recording -> "Stop"
                    transcribing -> "…"
                    else -> "Record"
                },
            )
        }

        when {
            recording -> Text(
                text = "Recording… tap Stop to transcribe",
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.padding(top = 12.dp),
            )
            transcribing -> Text(
                text = "Transcribing…",
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.padding(top = 12.dp),
            )
            transcript != null -> Text(
                text = "Copied: $transcript",
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.padding(top = 12.dp),
            )
            error != null -> Text(
                text = error.orEmpty(),
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(top = 12.dp),
            )
        }

        Button(
            onClick = onOpenPicker,
            modifier = Modifier.padding(top = 16.dp),
        ) {
            Text("Models")
        }
    }
}
