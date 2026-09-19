package dev.femustafa.kalaamscribe

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material.icons.filled.TrackChanges
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
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        RadioButton(selected = selected, onClick = onSelect)
        Column(modifier = Modifier.weight(1f).padding(horizontal = 8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = entry.displayName,
                    style = MaterialTheme.typography.bodyLarge,
                    maxLines = 1,
                    softWrap = false,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f, fill = false),
                )
                if (isRecommended) {
                    RecommendedBadge(modifier = Modifier.padding(start = 6.dp))
                }
            }
            ModelMetaLine(entry)
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
 * Tech subtext for a model row/dropdown item: size label (tiny/base/small) followed by
 * precision · size (no format/language jargon).
 */
internal fun catalogEntryMeta(entry: CatalogEntry): String = buildString {
    val sizeLabel = entry.sizeLabel()
    if (!sizeLabel.isNullOrEmpty()) {
        append(sizeLabel)
        append(" · ")
    }
    entry.precision?.let { append("${it.label} · ") }
    if (entry.approxSizeMb > 0) append("${entry.approxSizeMb} MB")
    if (ModelCatalog.isOmnilingual(entry)) append("auto-detect · 1600+ languages")
}

/**
 * The user-facing quality tier of a catalog entry — Fast (fast, lower accuracy),
 * Balanced (middle), or High Accuracy (slow, most accurate).
 */
internal enum class ModelTier(
    val speed: Int,
    val accuracy: Int,
) {
    FAST(speed = 3, accuracy = 1),
    BALANCED(speed = 2, accuracy = 2),
    HIGH_ACCURACY(speed = 1, accuracy = 3),
}

/**
 * Map a catalog entry to its [ModelTier] by model size (tiny → Fast,
 * small → High Accuracy, everything else → Balanced).
 */
internal fun CatalogEntry.tier(): ModelTier = when {
    model.id.contains("tiny") -> ModelTier.FAST
    model.id.contains("small") || model.id.contains("roman-urdu") -> ModelTier.HIGH_ACCURACY
    else -> ModelTier.BALANCED
}

/**
 * Returns a short size label (tiny/base/small) derived from the model id,
 * or null if the id does not contain a known size token.
 */
internal fun CatalogEntry.sizeLabel(): String? = when {
    model.id.contains("tiny") -> "tiny"
    model.id.contains("small") -> "small"
    model.id.contains("base") -> "base"
    model.id == "roman-urdu-int8" || model.id == "roman-urdu-fp32" -> "small"
    else -> null
}

/**
 * The meta line under a model name: tier speed/accuracy icons (theme-tinted,
 * so they read on any background) followed by the tech subtext from
 * [catalogEntryMeta].
 */
@Composable
internal fun ModelMetaLine(
    entry: CatalogEntry,
    modifier: Modifier = Modifier,
) {
    val tier = entry.tier()
    Row(
        modifier = modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        repeat(tier.speed) {
            Icon(
                imageVector = Icons.Filled.FlashOn,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(14.dp),
            )
        }
        repeat(tier.accuracy) {
            Icon(
                imageVector = Icons.Filled.TrackChanges,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(14.dp),
            )
        }
        Spacer(modifier = Modifier.width(6.dp))
        Text(
            text = catalogEntryMeta(entry),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 1,
            softWrap = false,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

/** Compact pill marking the catalog's recommended model for the current language. */
@Composable
internal fun RecommendedBadge(modifier: Modifier = Modifier) {
    Surface(
        shape = RoundedCornerShape(percent = 50),
        color = MaterialTheme.colorScheme.secondaryContainer,
        modifier = modifier,
    ) {
        Text(
            text = "Recommended",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSecondaryContainer,
            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
        )
    }
}