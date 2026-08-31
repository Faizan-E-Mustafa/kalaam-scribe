package dev.femustafa.voicedictation

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * Model picker: lists the six-Model catalog with download state, lets the user
 * select one (calling [onSelect]) or download it (calling [onDownload]).
 */
@Composable
fun ModelPickerScreen(
    viewModel: ModelPickerViewModel,
    onSelect: (CatalogEntry) -> Unit,
    onDownload: (CatalogEntry) -> Unit,
    onClose: () -> Unit,
) {
    val state by viewModel.state.collectAsState()
    val selectedId by viewModel.selectedId.collectAsState()

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(text = "Models", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.weight(1f))
            Button(onClick = onClose) { Text("Done") }
        }
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(state, key = { it.first.model.id }) { (entry, dlState) ->
                ModelRow(
                    entry = entry,
                    dlState = dlState,
                    selected = entry.model.id == selectedId,
                    onSelect = { onSelect(entry) },
                    onDownload = { onDownload(entry) },
                )
                HorizontalDivider()
            }
        }
    }
}

@Composable
private fun ModelRow(
    entry: CatalogEntry,
    dlState: ModelPickerViewModel.DownloadState,
    selected: Boolean,
    onSelect: () -> Unit,
    onDownload: () -> Unit,
) {
    val meta = buildString {
        append(entry.model.languageMode.label)
        if (entry.approxSizeMb > 0) append(" · ${entry.approxSizeMb} MB")
        if (entry.isDefault) append(" · default")
    }
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        RadioButton(selected = selected, onClick = onSelect)
        Column(modifier = Modifier.weight(1f).padding(horizontal = 8.dp)) {
            Text(text = entry.displayName, style = MaterialTheme.typography.bodyLarge)
            Text(text = meta, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            when (dlState) {
                is ModelPickerViewModel.DownloadState.Downloading -> {
                    LinearProgressIndicator(progress = { dlState.progress }, modifier = Modifier.fillMaxWidth().padding(top = 4.dp))
                }
                is ModelPickerViewModel.DownloadState.Failed -> {
                    Text("Download failed: ${dlState.message}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
                }
                else -> {}
            }
        }
        when (dlState) {
            ModelPickerViewModel.DownloadState.Ready -> Text("Ready", color = MaterialTheme.colorScheme.primary)
            is ModelPickerViewModel.DownloadState.Downloading -> Text("…", style = MaterialTheme.typography.bodySmall)
            ModelPickerViewModel.DownloadState.NotDownloaded -> Button(onClick = onDownload, enabled = entry.sourceUrl != null) { Text("Download") }
            is ModelPickerViewModel.DownloadState.Failed -> Button(onClick = onDownload) { Text("Retry") }
        }
    }
}

private val LanguageMode.label: String
    get() = when (this) {
        LanguageMode.Auto -> "Auto-detect"
        LanguageMode.English -> "English"
    }
