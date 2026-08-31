package dev.femustafa.voicedictation

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch

@Composable
fun DictationScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var running by remember { mutableStateOf(false) }
    var result by remember { mutableStateOf("") }

    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(text = "Voice Dictation", style = MaterialTheme.typography.headlineMedium)
        Text(text = "SPIKE build (ticket 08): whisper AAR dry-run.", modifier = Modifier.padding(top = 8.dp))
        Button(
            onClick = {
                running = true
                result = "loading…"
                scope.launch {
                    result = try {
                        SpikeRunner.run(context)
                    } catch (t: Throwable) {
                        "ERROR: ${t.message}"
                    } finally {
                        running = false
                    }
                }
            },
            modifier = Modifier.padding(top = 24.dp),
        ) {
            Text(if (running) "Running…" else "Run spike (jfk.wav)")
        }
        Text(
            text = result,
            modifier = Modifier.padding(top = 16.dp).verticalScroll(rememberScrollState()),
            style = MaterialTheme.typography.bodyMedium,
        )
    }
}

@Preview(showBackground = true)
@Composable
fun DictationScreenPreview() {
    DictationScreen()
}
