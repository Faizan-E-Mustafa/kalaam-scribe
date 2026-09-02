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
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * Model picker: lists the Model catalog with download state, lets the user
 * select one (calling [onSelect]) or download it (calling [onDownload]), and
 * hosts the app-level whisper language selection (its "language" setting).
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
    val loadingId by viewModel.loadingId.collectAsState()
    val languageCode by viewModel.languageCode.collectAsState()
    val format by viewModel.modelFormat.collectAsState()

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(text = "Models", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.weight(1f))
            Button(onClick = onClose) { Text("Done") }
        }
        // App-level model-file format: GGML (whisper.cpp) or ONNX (sherpa-onnx).
        FormatSelector(
            format = format,
            onSelect = viewModel::setModelFormat,
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
        )
        // App-level language setting: chosen once, persisted, applied to
        // multilingual Models when transcribing.
        LanguagePicker(
            selectedCode = languageCode,
            onSelect = viewModel::setLanguageCode,
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
        )
        LazyColumn(
            modifier = Modifier.weight(1f).fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            items(state, key = { it.first.model.id }) { (entry, dlState) ->
                ModelRow(
                    entry = entry,
                    dlState = dlState,
                    format = format,
                    selected = entry.model.id == selectedId,
                    loading = entry.model.id == loadingId,
                    onSelect = { onSelect(entry) },
                    onDownload = { onDownload(entry) },
                )
                HorizontalDivider()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun FormatSelector(
    format: ModelFormat,
    onSelect: (ModelFormat) -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(text = "Format", style = MaterialTheme.typography.bodyLarge)
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            ModelFormat.entries.forEach { candidate ->
                val selected = candidate == format
                val label = if (selected) "${candidate.name} ✓" else candidate.name
                if (selected) {
                    Button(onClick = { onSelect(candidate) }, modifier = Modifier.weight(1f)) {
                        Text(label)
                    }
                } else {
                    OutlinedButton(onClick = { onSelect(candidate) }, modifier = Modifier.weight(1f)) {
                        Text(label)
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun LanguagePicker(
    selectedCode: String?,
    onSelect: (String?) -> Unit,
    modifier: Modifier = Modifier,
) {
    var expanded by remember { mutableStateOf(false) }

    val label = selectedCode?.let { code ->
        val name = WhisperLanguages.nameOf(code)
        if (name != null) "$name ($code)" else code
    } ?: "Auto-detect"

    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = it },
        modifier = modifier,
    ) {
        OutlinedTextField(
            value = label,
            onValueChange = {},
            readOnly = true,
            label = { Text("Language") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier.menuAnchor(MenuAnchorType.PrimaryNotEditable).fillMaxWidth(),
        )
        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
        ) {
            DropdownMenuItem(
                text = { Text("Auto-detect") },
                onClick = {
                    expanded = false
                    onSelect(null)
                },
            )
            WhisperLanguages.entries.forEach { (code, name) ->
                DropdownMenuItem(
                    text = { Text("$name ($code)") },
                    onClick = {
                        expanded = false
                        onSelect(code)
                    },
                )
            }
        }
    }
}

@Composable
private fun ModelRow(
    entry: CatalogEntry,
    dlState: ModelPickerViewModel.DownloadState,
    format: ModelFormat,
    selected: Boolean,
    loading: Boolean,
    onSelect: () -> Unit,
    onDownload: () -> Unit,
) {
    val meta = buildString {
        append(format.name)
        entry.precision?.let { append(" · ${it.label}") }
        append(" · ")
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
                if (loading) {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth().padding(top = 4.dp))
                }
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
            ModelPickerViewModel.DownloadState.NotDownloaded -> Button(
                onClick = onDownload,
                enabled = entry.sourceUrl != null || entry.onnxSourceUrl != null,
            ) { Text("Download") }
            is ModelPickerViewModel.DownloadState.Failed -> Button(onClick = onDownload) { Text("Retry") }
        }
    }
}

private val LanguageMode.label: String
    get() = when (this) {
        LanguageMode.Auto -> "Auto-detect"
        LanguageMode.English -> "English"
        LanguageMode.RomanUrdu -> "Roman Urdu"
    }
