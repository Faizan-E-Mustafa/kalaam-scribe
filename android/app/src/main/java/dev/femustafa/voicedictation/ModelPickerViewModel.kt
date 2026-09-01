package dev.femustafa.voicedictation

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Backs the model picker. Tracks per-model download state (persisted by checking
 * whether the file already exists in app-private storage), the user's selection,
 * and drives [ModelDownloader]. Selection calls [WhisperManager.switchTo] so the
 * resident model and the picker stay consistent.
 */
class ModelPickerViewModel(application: Application) : AndroidViewModel(application) {

    sealed interface DownloadState {
        data object NotDownloaded : DownloadState
        data class Downloading(val progress: Float) : DownloadState
        data class Failed(val message: String) : DownloadState
        data object Ready : DownloadState
    }

    private val baseDir: File = getApplication<Application>().filesDir
    private val app: VoiceDictationApp
        get() = getApplication()

    private val downloader = ModelDownloader(baseDir)

    private val _entries = MutableStateFlow(ModelCatalog.models.map { entry ->
        entry to initialState(entry)
    })
    val state: StateFlow<List<Pair<CatalogEntry, DownloadState>>> = _entries.asStateFlow()

    private val _selectedId = MutableStateFlow(defaultOrPersistedId())
    val selectedId: StateFlow<String> = _selectedId.asStateFlow()

    /** The user's app-level language selection (persisted; null = auto-detect). */
    val languageCode: StateFlow<String?> = app.whisper.languageCode

    fun setLanguageCode(code: String?) {
        app.setLanguageCode(code)
    }

    private fun initialState(entry: CatalogEntry): DownloadState =
        if (File(baseDir, entry.model.fileName).exists()) DownloadState.Ready else DownloadState.NotDownloaded

    /** Re-scan storage so Ready models are recognised on later launches. */
    fun refresh() {
        _entries.value = ModelCatalog.models.map { entry ->
            entry to initialState(entry)
        }
    }

    suspend fun select(entry: CatalogEntry, manager: WhisperManager) {
        if (!File(baseDir, entry.model.fileName).exists()) return // must download first
        _selectedId.value = entry.model.id
        app.setSelectedModelId(entry.model.id)
        manager.switchTo(entry.model)
    }

    fun download(entry: CatalogEntry) {
        if (entry.sourceUrl == null) {
            setState(entry, DownloadState.Failed("Not hosted; copy the file manually into app storage"))
            return
        }
        setState(entry, DownloadState.Downloading(0f))
        viewModelScope.launch(Dispatchers.IO) {
            val result = downloader.download(entry) { progress ->
                setState(entry, DownloadState.Downloading(progress))
            }
            when (result) {
                is ModelDownloader.Result.Success -> setState(entry, DownloadState.Ready)
                is ModelDownloader.Result.Failure -> setState(entry, DownloadState.Failed(result.message))
            }
        }
    }

    private fun setState(entry: CatalogEntry, state: DownloadState) {
        _entries.value = _entries.value.map { (e, s) -> if (e.model.id == entry.model.id) e to state else e to s }
    }

    private fun defaultOrPersistedId(): String {
        // The user's last-selected Model, or the catalog default. If that Model's
        // file is missing from storage, fall back to the catalog default so the
        // radio group never preselects a Model that can't be used yet.
        val intended = app.initialModel()
        return if (File(baseDir, intended.model.fileName).exists()) intended.model.id
        else ModelCatalog.default.model.id
    }
}
