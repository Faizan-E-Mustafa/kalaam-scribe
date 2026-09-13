package dev.femustafa.voicedictation

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
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
    // Only list Models that support the selected language; Auto-detect (null)
    // shows the whole catalog. Recomputes live when the language dropdown changes.
    val code = languageCode
    val visible = remember(state, code) {
        if (code == null) state else state.filter { (entry, _) -> entry.supportsLanguage(code) }
    }

    Column(modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.safeDrawing).padding(16.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(text = "Models", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.weight(1f))
            Button(onClick = onClose) { Text("Done") }
        }
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
            // Keep the last row off the screen edge (edge-to-edge: the window bottom
            // is flush with the gesture/nav bar, so insets alone may report 0 on some
            // devices and the final model row gets cut off).
            contentPadding = PaddingValues(bottom = 16.dp),
        ) {
            items(visible, key = { it.first.model.id }) { (entry, dlState) ->
                ModelRow(
                    entry = entry,
                    dlState = dlState,
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
    selected: Boolean,
    loading: Boolean,
    onSelect: () -> Unit,
    onDownload: () -> Unit,
) {
    val meta = buildString {
        // Tech subtext: engine title · precision · size (no format/language jargon).
        append(entry.modelName)
        entry.precision?.let { append(" · ${it.label}") }
        if (entry.approxSizeMb > 0) append(" · ${entry.approxSizeMb} MB")
        if (ModelCatalog.isOmnilingual(entry)) append(" · auto-detect · 1600+ languages")
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
