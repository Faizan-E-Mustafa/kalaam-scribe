package dev.femustafa.voicedictation

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp

private val THREAD_OPTIONS = listOf(2, 4, 6, 8)

@Composable
fun SettingsScreen(
    onBack: () -> Unit,
) {
    val context = LocalContext.current
    val app = VoiceDictationApp.from(context)
    val currentThreads = app.whisperThreads()
    var selected by remember { mutableStateOf(currentThreads) }

    Column(
        modifier = Modifier.padding(16.dp),
    ) {
        Text(
            text = stringResource(id = R.string.settings_whisper_threads),
            style = MaterialTheme.typography.headlineMedium,
        )
        Text(
            text = stringResource(id = R.string.settings_threads_summary),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Row(
            modifier = Modifier.padding(vertical = 12.dp),
        ) {
            THREAD_OPTIONS.forEach { n ->
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(horizontal = 8.dp),
                ) {
                    RadioButton(
                        selected = n == selected,
                        onClick = { selected = n },
                    )
                    Text("$n")
                }
            }
        }
        Button(
            onClick = {
                app.setWhisperThreads(selected)
                onBack()
            },
            modifier = Modifier.align(Alignment.CenterHorizontally),
        ) {
            Text("Save")
        }
    }
}
