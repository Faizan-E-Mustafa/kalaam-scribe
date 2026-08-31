package dev.femustafa.voicedictation

import android.Manifest
import android.content.pm.PackageManager
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
 * Dictation surface: shows the resident Model, a Record/Stop control that requests
 * the mic runtime permission on first use, and the last recorded file. Transcription
 * of the recorded clip is added in a later ticket.
 */
@Composable
fun DictationScreen(
    viewModel: DictationViewModel,
    currentModel: Model?,
    onOpenPicker: () -> Unit,
) {
    val recording by viewModel.recording.collectAsState()
    val lastFile by viewModel.lastFile.collectAsState()
    val error by viewModel.error.collectAsState()

    val context = LocalContext.current

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) viewModel.toggleRecording()
    }

    fun onRecordPressed() {
        val granted = ContextCompat.checkSelfPermission(
            context, Manifest.permission.RECORD_AUDIO,
        ) == PackageManager.PERMISSION_GRANTED
        if (granted) {
            viewModel.toggleRecording()
        } else {
            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
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
            Text(if (recording) "Stop" else "Record")
        }

        if (recording) {
            Text(text = "Recording…", style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(top = 12.dp))
        }
        lastFile?.let {
            Text(text = "Saved: $it", style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 8.dp))
        }
        error?.let {
            Text(text = it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 8.dp))
        }

        Button(
            onClick = onOpenPicker,
            modifier = Modifier.padding(top = 16.dp),
        ) {
            Text("Models")
        }
    }
}
