package dev.femustafa.voicedictation

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat

/**
 * B — Transcript-first layout (chosen from ui-preview.html):
 * big transcript card with Copy/Share, live waveform + chip while recording,
 * and a short history list. Keeps the same permission gating as before.
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
    val lastTranscriptionMs by viewModel.lastTranscriptionMs.collectAsState()
    val error by viewModel.error.collectAsState()
    val history by viewModel.history.collectAsState()

    val context = LocalContext.current

    val micLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted -> if (granted) viewModel.toggleRecording() }

    val notifLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { }

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

    fun copy(text: String) {
        val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        cm.setPrimaryClip(ClipData.newPlainText("dictation", text))
    }

    Column(
        modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.safeDrawing).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        // Top chips
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            when {
                recording -> AssistChip(
                    onClick = {},
                    label = { Text("Recording") },
                    leadingIcon = {
                        Box(
                            Modifier.size(8.dp).clip(RoundedCornerShape(50)).background(MaterialTheme.colorScheme.error),
                        )
                    },
                    colors = AssistChipDefaults.assistChipColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                )
                transcribing -> AssistChip(
                    onClick = {},
                    label = { Text("Transcribing…") },
                )
                transcript != null -> AssistChip(
                    onClick = {},
                    label = { Text("Copied") },
                    colors = AssistChipDefaults.assistChipColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
                )
                else -> AssistChip(onClick = {}, label = { Text("Ready") })
            }
            AssistChip(
                onClick = onOpenPicker,
                label = { Text(currentModel?.id ?: "no model") },
            )
        }

        if (recording) {
            Waveform(modifier = Modifier.fillMaxWidth().height(28.dp))
        }

        // Main action
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
            Button(
                onClick = { onRecordPressed() },
                modifier = Modifier.weight(1f),
                colors = if (recording) ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                else ButtonDefaults.buttonColors(),
            ) {
                Text(if (recording) "■ Stop" else "● Record")
            }
            FilledTonalButton(onClick = onOpenPicker) { Text("Models") }
        }

        // Transcript card
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = if (transcript != null) MaterialTheme.colorScheme.secondaryContainer
                else MaterialTheme.colorScheme.surfaceVariant,
            ),
            shape = RoundedCornerShape(16.dp),
        ) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    text = when {
                        recording -> "Listening…"
                        transcribing -> "Transcribing…"
                        transcript != null -> "Copied to clipboard"
                        error != null -> "Error"
                        else -> "Transcript"
                    },
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    text = when {
                        recording -> "Speak now — tap Stop when done (works in background)"
                        transcribing -> "Running on-device…"
                        transcript != null -> transcript!!
                        error != null -> error!!
                        else -> "No dictation yet — tap Record"
                    },
                    style = MaterialTheme.typography.bodyMedium,
                    color = if (error != null) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface,
                )
                if (transcript != null) {
                    val lastMs = lastTranscriptionMs
                    if (lastMs != null) {
                        Text(
                            text = "Transcribed in ${"%.1f".format(lastMs / 1000.0)} s",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    FilledTonalButton(onClick = { copy(transcript!!) }) { Text("Copy") }
                }
                if (error != null && transcript == null && !recording && !transcribing) {
                    Text(
                        text = "Tip: check mic permission and that a model is selected",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        // History
        if (history.isNotEmpty()) {
            Text("History", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            LazyColumn(verticalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.weight(1f, fill = false)) {
                items(history) { item ->
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                        shape = RoundedCornerShape(12.dp),
                    ) {
                        Row(
                            Modifier.fillMaxWidth().padding(12.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(item, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f).padding(end = 8.dp))
                            FilledTonalButton(onClick = { copy(item) }, modifier = Modifier.height(32.dp)) { Text("Copy", style = MaterialTheme.typography.labelSmall) }
                        }
                    }
                }
            }
        } else {
            Spacer(Modifier.weight(1f))
            Text(
                "History appears here — last 10 dictations",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun Waveform(modifier: Modifier = Modifier) {
    val t = rememberInfiniteTransition(label = "wave")
    val h1 by t.animateFloat(initialValue = 8f, targetValue = 26f, animationSpec = infiniteRepeatable(tween(380, easing = LinearEasing), RepeatMode.Reverse), label = "h1")
    val h2 by t.animateFloat(initialValue = 14f, targetValue = 22f, animationSpec = infiniteRepeatable(tween(420, easing = LinearEasing), RepeatMode.Reverse), label = "h2")
    val h3 by t.animateFloat(initialValue = 10f, targetValue = 28f, animationSpec = infiniteRepeatable(tween(360, easing = LinearEasing), RepeatMode.Reverse), label = "h3")
    Row(modifier = modifier, horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically) {
        listOf(h1, h2, h3, h2, h1).forEach { h ->
            Box(Modifier.padding(horizontal = 3.dp).width(4.dp).height(h.dp).clip(RoundedCornerShape(50)).background(MaterialTheme.colorScheme.primary))
        }
    }
}
