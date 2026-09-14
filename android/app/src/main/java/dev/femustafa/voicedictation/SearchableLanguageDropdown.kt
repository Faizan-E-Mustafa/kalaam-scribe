package dev.femustafa.voicedictation

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier

/**
 * Searchable language picker shared by the onboarding Welcome screen and the
 * settings sheet: an outlined dropdown whose field doubles as a filter box, so
 * the 100+ language list narrows by code or name as you type. [includeAutoDetect]
 * puts the "Auto-detect" entry on top (settings only — onboarding always picks a
 * concrete language). Selecting resets the query and closes the menu.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun SearchableLanguageDropdown(
    selectedCode: String?,
    onSelect: (String?) -> Unit,
    modifier: Modifier = Modifier,
    includeAutoDetect: Boolean = false,
) {
    var expanded by remember { mutableStateOf(false) }
    var query by remember { mutableStateOf("") }

    val selection = selectedCode?.let { code ->
        val name = WhisperLanguages.nameOf(code)
        if (name != null) "$name ($code)" else code
    } ?: if (includeAutoDetect) "Auto-detect" else ""

    // While the menu is open the field shows the live search query; otherwise it
    // shows the current selection.
    val value = if (expanded && query.isNotEmpty()) query else selection

    val filtered = remember(query) {
        if (query.isBlank()) WhisperLanguages.entries
        else WhisperLanguages.entries.filter { (code, name) ->
            code.contains(query, ignoreCase = true) || name.contains(query, ignoreCase = true)
        }
    }

    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = {
            expanded = it
            if (it) query = ""
        },
        modifier = modifier,
    ) {
        OutlinedTextField(
            value = value,
            onValueChange = {
                query = it
                if (!expanded) expanded = true
            },
            label = { Text("Language") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier.menuAnchor(MenuAnchorType.PrimaryEditable).fillMaxWidth(),
        )
        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
        ) {
            if (includeAutoDetect) {
                DropdownMenuItem(
                    text = { Text("Auto-detect") },
                    onClick = {
                        expanded = false
                        query = ""
                        onSelect(null)
                    },
                )
            }
            filtered.forEach { (code, name) ->
                DropdownMenuItem(
                    text = { Text("$name ($code)") },
                    onClick = {
                        expanded = false
                        query = ""
                        onSelect(code)
                    },
                )
            }
            if (query.isNotBlank() && filtered.isEmpty()) {
                DropdownMenuItem(
                    text = { Text("No languages match \"$query\"") },
                    onClick = {},
                    enabled = false,
                )
            }
        }
    }
}