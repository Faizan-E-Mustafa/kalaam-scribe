package dev.femustafa.voicedictation

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Download
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

/**
 * One row in a model list (welcome screen and model picker): name, tech subtext,
 * download/ready state with progress, and a radio to select it. [isRecommended]
 * marks the row the catalog recommends for the current language.
 */
@Composable
internal fun ModelRow(
    entry: CatalogEntry,
    dlState: ModelPickerViewModel.DownloadState,
    selected: Boolean,
    loading: Boolean,
    isRecommended: Boolean = false,
    onSelect: () -> Unit,
    onDownload: () -> Unit,
) {
    val meta = catalogEntryMeta(entry)
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        RadioButton(selected = selected, onClick = onSelect)
        Column(modifier = Modifier.weight(1f).padding(horizontal = 8.dp)) {
            Text(
                text = entry.displayName,
                style = MaterialTheme.typography.bodyLarge,
                maxLines = 1,
                softWrap = false,
                overflow = TextOverflow.Ellipsis,
            )
            if (isRecommended) {
                RecommendedBadge()
            }
            Text(
                text = meta,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                softWrap = false,
                overflow = TextOverflow.Ellipsis,
            )
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
            is ModelPickerViewModel.DownloadState.Downloading -> Text(
                "${(dlState.progress * 100).toInt()}%",
                style = MaterialTheme.typography.bodySmall,
            )
            ModelPickerViewModel.DownloadState.NotDownloaded -> IconButton(
                onClick = onDownload,
                enabled = entry.sourceUrl != null || entry.onnxSourceUrl != null,
            ) {
                Icon(
                    imageVector = Icons.Filled.Download,
                    contentDescription = "Download ${entry.displayName}",
                )
            }
            is ModelPickerViewModel.DownloadState.Failed -> IconButton(
                onClick = onDownload,
            ) {
                Icon(
                    imageVector = Icons.Filled.Download,
                    contentDescription = "Retry download ${entry.displayName}",
                )
            }
        }
    }
}

/**
 * Tech subtext for a model row/dropdown item: engine title · precision · size
 * (no format/language jargon).
 */
internal fun catalogEntryMeta(entry: CatalogEntry): String = buildString {
    append(entry.modelName)
    entry.precision?.let { append(" · ${it.label}") }
    if (entry.approxSizeMb > 0) append(" · ${entry.approxSizeMb} MB")
    if (ModelCatalog.isOmnilingual(entry)) append(" · auto-detect · 1600+ languages")
}

/** Compact pill marking the catalog's recommended model for the current language. */
@Composable
internal fun RecommendedBadge() {
    Surface(
        shape = RoundedCornerShape(percent = 50),
        color = MaterialTheme.colorScheme.secondaryContainer,
    ) {
        Text(
            text = "Recommended",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSecondaryContainer,
            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
        )
    }
}