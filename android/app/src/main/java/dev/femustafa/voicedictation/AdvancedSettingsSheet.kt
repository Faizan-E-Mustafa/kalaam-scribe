package dev.femustafa.voicedictation

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch

private val THREAD_OPTIONS = listOf(2, 4, 6, 8)

/**
 * The single settings surface: a modal bottom sheet over the dictation screen.
 * Shows the app-level Language dropdown and a collapsible Model selector up top,
 * with the power-user knobs (threads, transcription mode, VAD, dolphin beam)
 * tucked behind a collapsible "Advanced settings" section.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AdvancedSettingsSheet(
    viewModel: ModelPickerViewModel,
    manager: WhisperManager,
    onClose: () -> Unit,
) {
    val context = LocalContext.current
    val app = VoiceDictationApp.from(context)

    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    val scope = rememberCoroutineScope()
    fun dismiss() {
        scope.launch { sheetState.hide() }.invokeOnCompletion { onClose() }
    }

    var advancedExpanded by remember { mutableStateOf(false) }
    // Set when the language changes to a recommended model that isn't downloaded
    // yet; drives the "download the recommended model" confirmation dialog.
    var pendingDownload by remember { mutableStateOf<CatalogEntry?>(null) }

    var selectedThreads by remember { mutableStateOf(app.whisperThreads()) }
    var transcriptionMode by remember { mutableStateOf(app.transcriptionMode) }
    val streamingActive = transcriptionMode == TranscriptionMode.SimulatedStreaming
    var threshold by remember { mutableStateOf(app.vadThreshold()) }
    var minSilence by remember { mutableStateOf(app.vadMinSilence()) }
    var minSpeech by remember { mutableStateOf(app.vadMinSpeech()) }
    var maxSpeech by remember { mutableStateOf(app.vadMaxSpeech()) }
    var beamSize by remember { mutableStateOf(app.dolphinBeamSize().toFloat()) }

    val state by viewModel.state.collectAsState()
    val selectedId by viewModel.selectedId.collectAsState()
    val loadingId by viewModel.loadingId.collectAsState()
    val languageCode by viewModel.languageCode.collectAsState()

    // Only list Models that support the selected language; Auto-detect (null)
    // shows the whole catalog. The catalog's recommended model for the language
    // is pinned first and badged, matching the onboarding list.
    val code = languageCode
    val recommended = ModelCatalog.recommendedForLanguage(code)
    val visible = remember(state, code, recommended) {
        (if (code == null) state else state.filter { (entry, _) -> entry.supportsLanguage(code) })
            .sortedWith(compareByDescending { it.first.model.id == recommended?.model?.id })
    }

    ModalBottomSheet(
        onDismissRequest = onClose,
        sheetState = sheetState,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .fillMaxHeight()
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(
                    imageVector = Icons.Filled.Settings,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                )
                Text(
                    text = "Settings",
                    style = MaterialTheme.typography.titleLarge,
                    modifier = Modifier.padding(start = 8.dp),
                )
            }

            InlineLanguageSearch(
                selectedCode = languageCode,
                includeAutoDetect = true,
                onSelect = { code ->
                    viewModel.setLanguageCode(code)
                    // The recommended model for the new language becomes the default
                    // selection; if it isn't downloaded yet, ask the user to grab it.
                    val recommended = viewModel.updateSelectionForLanguage(manager)
                    pendingDownload = if (recommended != null && !viewModel.isDownloaded(recommended)) recommended else null
                },
            )

            HorizontalDivider()

            // Dropdown matching [LanguagePicker]: only the current model is shown
            // until the user expands it, so the whole catalog is never exposed up front.
            ModelSelector(
                selectedId = selectedId,
                models = visible,
                loadingId = loadingId,
                recommendedId = recommended?.model?.id,
                onSelect = { viewModel.select(it, manager) },
                onDownload = { viewModel.download(it, manager) },
            )

            HorizontalDivider()

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
                text = "Streaming shows each utterance as you speak.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            HorizontalDivider()

            // Collapsible power-user settings, collapsed by default.
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { advancedExpanded = !advancedExpanded }
                    .padding(vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = "Advanced settings",
                    style = MaterialTheme.typography.titleSmall,
                    modifier = Modifier.weight(1f),
                )
                val rotation by animateFloatAsState(if (advancedExpanded) 180f else 0f, label = "advancedArrow")
                Icon(
                    imageVector = Icons.Filled.ArrowDropDown,
                    contentDescription = if (advancedExpanded) "Hide advanced settings" else "Show advanced settings",
                    modifier = Modifier.rotate(rotation),
                )
            }
            AnimatedVisibility(visible = advancedExpanded) {
                Column(modifier = Modifier.fillMaxWidth()) {
                    SectionHeader(text = "Whisper threads")
                    Row(modifier = Modifier.padding(vertical = 4.dp)) {
                        THREAD_OPTIONS.forEach { n ->
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                RadioButton(selected = n == selectedThreads, onClick = { selectedThreads = n })
                                Text("$n", fontSize = 14.sp)
                            }
                        }
                    }

                    SectionHeader(text = "VAD Settings (streaming only)")
                    val vadDim = Modifier.alpha(if (streamingActive) 1f else 0.4f)
                    Column(modifier = vadDim) {
                        CompactSlider("Threshold", threshold, { threshold = it }, 0f..1f, "%.2f".format(threshold))
                        CompactSlider("Min Silence", minSilence, { minSilence = it }, 0.1f..5f, "%.1fs".format(minSilence))
                        CompactSlider("Min Speech", minSpeech, { minSpeech = it }, 0.1f..5f, "%.1fs".format(minSpeech))
                        CompactSlider("Max Speech", maxSpeech, { maxSpeech = it }, 5f..60f, "%.0fs".format(maxSpeech))
                    }

                    HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

                    SectionHeader(text = "Dolphin Beam Size")
                    CompactSlider(
                        "Beam", beamSize, { beamSize = it }, 1f..5f, "%.0f".format(beamSize), steps = 3,
                    )
                }
            }

            Spacer(modifier = Modifier.height(4.dp))
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
                        dismiss()
                    },
                ) {
                    Text("Save", fontSize = 14.sp)
                }
            }
            Spacer(modifier = Modifier.height(24.dp))
        }
    }

    pendingDownload?.let { recommended ->
        AlertDialog(
            onDismissRequest = { pendingDownload = null },
            title = { Text("Download model?") },
            text = {
                Text(
                    "${recommended.displayName} is the recommended model for " +
                        "${languageCode?.let { WhisperLanguages.nameOf(it) } ?: "Auto-detect"}. " +
                        "Download it now to start dictating with this language.",
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        pendingDownload = null
                        viewModel.download(recommended, manager)
                    },
                ) {
                    Text("Download")
                }
            },
            dismissButton = {
                TextButton(onClick = { pendingDownload = null }) {
                    Text("Not now")
                }
            },
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ModelSelector(
    selectedId: String,
    models: List<Pair<CatalogEntry, ModelPickerViewModel.DownloadState>>,
    loadingId: String?,
    recommendedId: String?,
    onSelect: (CatalogEntry) -> Unit,
    onDownload: (CatalogEntry) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }

    val label = ModelCatalog.byId(selectedId)?.displayName ?: "no model"

    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = it },
    ) {
        OutlinedTextField(
            value = label,
            onValueChange = {},
            readOnly = true,
            label = { Text("Model") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier.menuAnchor(MenuAnchorType.PrimaryNotEditable).fillMaxWidth(),
        )
        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
        ) {
            models.forEach { (entry, dlState) ->
                val selected = entry.model.id == selectedId
                val loading = entry.model.id == loadingId
                val ready = dlState is ModelPickerViewModel.DownloadState.Ready
                DropdownMenuItem(
                    text = {
                        Column {
                            Text(entry.displayName)
                            if (entry.model.id == recommendedId) RecommendedBadge()
                            Text(
                                text = catalogEntryMeta(entry),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    },
                    onClick = {
                        if (ready) {
                            expanded = false
                            onSelect(entry)
                        }
                    },
                    leadingIcon = {
                        if (selected) {
                            Icon(
                                imageVector = Icons.Filled.Check,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                            )
                        }
                    },
                    trailingIcon = {
                        when (dlState) {
                            ModelPickerViewModel.DownloadState.Ready -> {}
                            is ModelPickerViewModel.DownloadState.Downloading -> Text("${(dlState.progress * 100).toInt()}%")
                            is ModelPickerViewModel.DownloadState.Failed -> IconButton(onClick = { onDownload(entry) }) {
                                Icon(Icons.Filled.Download, contentDescription = "Retry download")
                            }
                            ModelPickerViewModel.DownloadState.NotDownloaded -> IconButton(onClick = { onDownload(entry) }) {
                                Icon(Icons.Filled.Download, contentDescription = "Download ${entry.displayName}")
                            }
                        }
                        if (loading) {
                            Text("loading…", color = MaterialTheme.colorScheme.primary)
                        }
                    },
                )
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

/**
 * Searchable language picker for the settings sheet. Unlike the popup-based
 * [SearchableLanguageDropdown] used on onboarding, the matching languages render
 * inline below the field inside the sheet's scrollable column, so the keyboard
 * (via the sheet's ime/safeDrawing padding) can never cover the input box while
 * typing. Selecting a row collapses the list back down.
 */
@Composable
private fun InlineLanguageSearch(
    selectedCode: String?,
    includeAutoDetect: Boolean,
    onSelect: (String?) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    var query by remember { mutableStateOf("") }

    val selection = selectedCode?.let { code ->
        val name = WhisperLanguages.nameOf(code)
        if (name != null) "$name ($code)" else code
    } ?: if (includeAutoDetect) "Auto-detect" else ""

    val value = if (expanded) query else selection

    val filtered = remember(query) {
        if (query.isBlank()) WhisperLanguages.entries
        else WhisperLanguages.entries.filter { (code, name) ->
            code.contains(query, ignoreCase = true) || name.contains(query, ignoreCase = true)
        }
    }

    Column(modifier = Modifier.fillMaxWidth()) {
        OutlinedTextField(
            value = value,
            onValueChange = {
                query = it
                expanded = true
            },
            label = { Text("Language") },
            singleLine = true,
            trailingIcon = {
                IconButton(onClick = { expanded = !expanded }) {
                    Icon(
                        imageVector = Icons.Filled.ArrowDropDown,
                        contentDescription = if (expanded) "Collapse language list" else "Expand language list",
                    )
                }
            },
            modifier = Modifier.fillMaxWidth(),
        )

        if (expanded) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(max = 280.dp)
                    .verticalScroll(rememberScrollState())
                    .padding(vertical = 4.dp),
            ) {
                if (includeAutoDetect) {
                    LanguageRow(
                        label = "Auto-detect",
                        selected = selectedCode == null,
                        onClick = {
                            expanded = false
                            query = ""
                            onSelect(null)
                        },
                    )
                }
                filtered.forEach { (code, name) ->
                    LanguageRow(
                        label = "$name ($code)",
                        selected = code == selectedCode,
                        onClick = {
                            expanded = false
                            query = ""
                            onSelect(code)
                        },
                    )
                }
                if (query.isNotBlank() && filtered.isEmpty()) {
                    Text(
                        text = "No languages match \"$query\"",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(vertical = 8.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun LanguageRow(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(text = label, modifier = Modifier.weight(1f))
        if (selected) {
            Icon(
                imageVector = Icons.Filled.Check,
                contentDescription = "Selected",
                tint = MaterialTheme.colorScheme.primary,
            )
        }
    }
}