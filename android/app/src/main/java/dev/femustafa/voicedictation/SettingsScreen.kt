package dev.femustafa.voicedictation

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

private val THREAD_OPTIONS = listOf(2, 4, 6, 8)

@Composable
fun SettingsScreen(
    onBack: () -> Unit,
) {
    val context = LocalContext.current
    val app = VoiceDictationApp.from(context)

    var selectedThreads by remember { mutableStateOf(app.whisperThreads()) }
    var transcriptionMode by remember { mutableStateOf(app.transcriptionMode) }
    val streamingActive = transcriptionMode == TranscriptionMode.SimulatedStreaming
    var threshold by remember { mutableStateOf(app.vadThreshold()) }
    var minSilence by remember { mutableStateOf(app.vadMinSilence()) }
    var minSpeech by remember { mutableStateOf(app.vadMinSpeech()) }
    var maxSpeech by remember { mutableStateOf(app.vadMaxSpeech()) }
    var beamSize by remember { mutableStateOf(app.dolphinBeamSize().toFloat()) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(12.dp)
            .verticalScroll(rememberScrollState()),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(text = "Settings", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.weight(1f))
        }

        Spacer(modifier = Modifier.height(8.dp))
        SectionHeader(text = stringResource(id = R.string.settings_whisper_threads))
        Row(modifier = Modifier.padding(vertical = 4.dp)) {
            THREAD_OPTIONS.forEach { n ->
                Row(verticalAlignment = Alignment.CenterVertically) {
                    RadioButton(selected = n == selectedThreads, onClick = { selectedThreads = n })
                    Text("$n", fontSize = 14.sp)
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))
        SectionHeader(text = "Transcription Mode")
        Row(modifier = Modifier.padding(vertical = 4.dp)) {
            TranscriptionMode.entries.forEach { mode ->
                Row(verticalAlignment = Alignment.CenterVertically) {
                    RadioButton(
                        selected = mode == transcriptionMode,
                        onClick = { transcriptionMode = mode },
                    )
                    Text(
                        when (mode) {
                            TranscriptionMode.Batch -> "Batch"
                            TranscriptionMode.SimulatedStreaming -> "Simulated streaming"
                        },
                        fontSize = 14.sp,
                    )
                }
            }
        }
        Text(
            text = if (streamingActive) {
                "Show each utterance as you speak. Requires a streaming-capable model; others fall back to batch."
            } else {
                "Decode the whole clip after you stop. Best for simple/stable results; works with all models."
            },
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Spacer(modifier = Modifier.height(8.dp))
        SectionHeader(text = "VAD Settings (streaming only)")
        val vadDim = Modifier.alpha(if (streamingActive) 1f else 0.4f)
        Column(modifier = vadDim) {
            CompactSlider("Threshold", threshold, { threshold = it }, 0f..1f, "%.2f".format(threshold))
            CompactSlider("Min Silence", minSilence, { minSilence = it }, 0.1f..5f, "%.1fs".format(minSilence))
            CompactSlider("Min Speech", minSpeech, { minSpeech = it }, 0.1f..5f, "%.1fs".format(minSpeech))
            CompactSlider("Max Speech", maxSpeech, { maxSpeech = it }, 5f..60f, "%.0fs".format(maxSpeech))
        }

        Spacer(modifier = Modifier.height(8.dp))
        SectionHeader(text = "Dolphin Beam Size")
        CompactSlider(
            "Beam", beamSize, { beamSize = it }, 1f..5f, "%.0f".format(beamSize), steps = 3,
            modifier = Modifier.alpha(if (streamingActive) 1f else 0.4f),
        )

        Spacer(modifier = Modifier.height(16.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally),
        ) {
            OutlinedButton(
                onClick = {
                    selectedThreads = VoiceDictationApp.DEFAULT_WHISPER_THREADS
                    transcriptionMode = TranscriptionMode.Batch
                    threshold = VoiceDictationApp.DEFAULT_VAD_THRESHOLD
                    minSilence = VoiceDictationApp.DEFAULT_VAD_MIN_SILENCE
                    minSpeech = VoiceDictationApp.DEFAULT_VAD_MIN_SPEECH
                    maxSpeech = VoiceDictationApp.DEFAULT_VAD_MAX_SPEECH
                    beamSize = VoiceDictationApp.DEFAULT_DOLPHIN_BEAM_SIZE.toFloat()
                },
            ) {
                Text("Reset", fontSize = 14.sp)
            }

            Button(
                onClick = {
                    app.setWhisperThreads(selectedThreads)
                    app.setTranscriptionMode(transcriptionMode)
                    app.setVadThreshold(threshold)
                    app.setVadMinSilence(minSilence)
                    app.setVadMinSpeech(minSpeech)
                    app.setVadMaxSpeech(maxSpeech)
                    app.setDolphinBeamSize(beamSize.toInt())
                    onBack()
                },
            ) {
                Text("Save", fontSize = 14.sp)
            }
        }
    }
}

@Composable
private fun SectionHeader(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.titleSmall,
        modifier = Modifier.padding(vertical = 4.dp),
    )
}

@Composable
private fun CompactSlider(
    label: String,
    value: Float,
    onValueChange: (Float) -> Unit,
    range: ClosedFloatingPointRange<Float>,
    display: String,
    steps: Int = 0,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, fontSize = 14.sp, modifier = Modifier.weight(0.3f))
        Text(display, fontSize = 14.sp, color = MaterialTheme.colorScheme.primary, modifier = Modifier.weight(0.15f))
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = range,
            steps = steps,
            modifier = Modifier.weight(0.55f),
        )
    }
}
