package dev.femustafa.voicedictation

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
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
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/** Quick-pick language chips on the onboarding screen ([LanguageOnboardingScreen]). */
private val QUICK_LANGUAGES = listOf("en" to "English", "ur" to "Urdu")

/**
 * First-run screen: asks the user which language they plan to dictate in and,
 * once picked, presents the Models that support it — the catalog's recommended
 * one first with a "Recommended" tag — so the user can download and select a
 * model right away. Only Models for [WhisperLanguages] support a chosen
 * concrete language (no Auto-detect / skip) so the model list is meaningful
 * from the start; the choice stays editable later via the Models screen's
 * Language dropdown.
 *
 * Language changes are published immediately via [onLanguageChange] so the
 * filter, the recommended model, and the picker's default highlight stay in
 * sync. [onContinue] is called once the language is chosen AND its active
 * (selected) model is downloaded; callers dismiss the screen.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LanguageOnboardingScreen(
    viewModel: ModelPickerViewModel,
    onSelect: (CatalogEntry) -> Unit,
    onDownload: (CatalogEntry) -> Unit,
    onLanguageChange: (String) -> Unit,
    onContinue: () -> Unit,
) {
    var selected by remember { mutableStateOf<String?>(null) }
    val state by viewModel.state.collectAsState()
    val selectedId by viewModel.selectedId.collectAsState()
    val loadingId by viewModel.loadingId.collectAsState()
    val code = selected

    val recommended = ModelCatalog.recommendedForLanguage(code)
    val visible = remember(state, code, recommended) {
        if (code == null) emptyList()
        else state
            .filter { (entry, _) -> entry.supportsLanguage(code) }
            .sortedWith(compareByDescending { it.first.model.id == recommended?.model?.id })
    }
    // The radio shows the recommended model once a language is picked; before any
    // download its row is just pre-highlighted and tapping it is a no-op, but after
    // the recommended download the ViewModel auto-selects it for real.
    val radioSelectedId = recommended?.model?.id ?: selectedId
    val activeReady = visible.any { (entry, dl) ->
        entry.model.id == radioSelectedId && dl is ModelPickerViewModel.DownloadState.Ready
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .windowInsetsPadding(WindowInsets.safeDrawing)
            .padding(16.dp),
    ) {
        Spacer(modifier = Modifier.height(24.dp))
        Text(text = "Welcome", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Which language will you dictate in? Only models that support it " +
                "are shown below — the recommended one is enough to get started.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Spacer(modifier = Modifier.height(20.dp))
        Text(text = "Quick pick", style = MaterialTheme.typography.titleSmall)
        Spacer(modifier = Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            QUICK_LANGUAGES.forEach { (code, name) ->
                FilterChip(
                    selected = selected == code,
                    onClick = {
                        selected = code
                        onLanguageChange(code)
                    },
                    label = { Text(name) },
                )
            }
        }

        Spacer(modifier = Modifier.height(20.dp))
        LanguagePicker(
            selectedCode = selected,
            onSelect = {
                selected = it
                onLanguageChange(it)
            },
            modifier = Modifier.fillMaxWidth(),
        )

        if (selected != null) {
            Spacer(modifier = Modifier.height(20.dp))
            Text(
                text = "Models for ${WhisperLanguages.nameOf(selected!!) ?: selected!!}",
                style = MaterialTheme.typography.titleSmall,
            )
            // The list scrolls independently so the Continue button stays on screen
            // no matter how many models the language has.
            LazyColumn(
                modifier = Modifier.weight(1f).fillMaxWidth().padding(top = 4.dp),
                contentPadding = PaddingValues(bottom = 8.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                items(visible, key = { it.first.model.id }) { (entry, dlState) ->
                    ModelRow(
                        entry = entry,
                        dlState = dlState,
                        selected = entry.model.id == radioSelectedId,
                        loading = entry.model.id == loadingId,
                        isRecommended = entry.model.id == recommended?.model?.id,
                        onSelect = { onSelect(entry) },
                        onDownload = { onDownload(entry) },
                    )
                    HorizontalDivider()
                }
            }
        } else {
            Spacer(modifier = Modifier.weight(1f))
        }

        Spacer(modifier = Modifier.height(12.dp))
        if (selected != null && !activeReady) {
            Text(
                text = "Download the recommended model to continue",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(modifier = Modifier.height(8.dp))
        }
        Button(
            onClick = { if (selected != null && activeReady) onContinue() },
            enabled = selected != null && activeReady,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Continue")
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun LanguagePicker(
    selectedCode: String?,
    onSelect: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    var expanded by remember { mutableStateOf(false) }

    val label = selectedCode?.let { code ->
        val name = WhisperLanguages.nameOf(code)
        if (name != null) "$name ($code)" else code
    } ?: "Choose a language"

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