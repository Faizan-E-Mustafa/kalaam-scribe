package dev.femustafa.voicedictation

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
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
 * resident model and the picker stay consistent. Disabled model families (GGML
 * and Dolphin CTC) are retained in code but never offered, so the picker always
 * lists [ModelCatalog.activeCatalog] (Whisper ONNX + Dolphin attention).
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

    private val _entries = MutableStateFlow(catalogFor(app.modelFormat).map { entry ->
        entry to initialState(entry)
    })
    val state: StateFlow<List<Pair<CatalogEntry, DownloadState>>> = _entries.asStateFlow()

    private val _selectedId = MutableStateFlow(defaultOrPersistedId())
    val selectedId: StateFlow<String> = _selectedId.asStateFlow()

    /** The id of the model currently loading into memory (or null). */
    private val _loadingId = MutableStateFlow<String?>(null)
    val loadingId: StateFlow<String?> = _loadingId.asStateFlow()

    /** The in-flight model-load job, cancelled when the user picks a different model. */
    private var _loadJob: Job? = null

    /** The user's app-level language selection (persisted; null = auto-detect). */
    val languageCode: StateFlow<String?> = app.whisper.languageCode

    fun setLanguageCode(code: String?) {
        app.setLanguageCode(code)
    }

    /** The [CatalogEntry]s to show for the current format. */
    private fun catalogFor(format: ModelFormat): List<CatalogEntry> = when (format) {
        // GGML is disabled (retained in [ModelCatalog.ggmlModels] but never offered).
        ModelFormat.GGML -> ModelCatalog.ggmlModels
        // Dolphin CTC and Dolphin attention models load through ONNX files on phone
        // (sherpa-onnx / onnxruntime-android), so they show up alongside the Whisper
        // ONNX models.
        ModelFormat.ONNX -> ModelCatalog.activeCatalog
    }

    private fun initialState(entry: CatalogEntry): DownloadState =
        if (downloader.isDownloaded(entry, app.modelFormat)) DownloadState.Ready else DownloadState.NotDownloaded

    /** Re-scan storage so Ready models are recognised on later launches. */
    fun refresh() {
        _entries.value = catalogFor(app.modelFormat).map { entry ->
            entry to initialState(entry)
        }
    }

    fun select(entry: CatalogEntry, manager: WhisperManager) {
        if (!downloader.isDownloaded(entry, app.modelFormat)) return // must download first
        _selectedId.value = entry.model.id
        app.setSelectedModelId(entry.model.id)
        // Show the row as selected immediately, then load the model into memory
        // in the background so transcription can use it. Loading is cancelled if
        // the user picks a different model before it finishes.
        _loadJob?.cancel()
        // Load the model on a background dispatcher: constructing the sherpa-onnx
        // OfflineRecognizer (or the GGML engine) loads the model into memory and
        // blocks the calling thread. Running it on the main thread would freeze
        // the UI (and make the radio toggle appear to lag). Use the IO pool so the
        // heavy JNI/native load never touches the main thread.
        _loadJob = viewModelScope.launch(Dispatchers.IO) {
            _loadingId.value = entry.model.id
            try {
                manager.switchTo(entry.model)
            } finally {
                _loadingId.value = null
            }
        }
    }

    fun download(entry: CatalogEntry) {
        // Each catalog entry maps to exactly one format download path.
        val format = entryFormat(entry)
        setState(entry, DownloadState.Downloading(0f))
        viewModelScope.launch(Dispatchers.IO) {
            val result = downloader.download(entry, format) { progress ->
                setState(entry, DownloadState.Downloading(progress))
            }
            when (result) {
                is ModelDownloader.Result.Success -> setState(entry, DownloadState.Ready)
                is ModelDownloader.Result.Failure -> setState(entry, DownloadState.Failed(result.message))
            }
        }
    }

    /** The format that [entry] belongs to (on the ONNX catalog vs the GGML catalog). */
    private fun entryFormat(entry: CatalogEntry): ModelFormat =
        if (entry.onnxSourceUrl != null) ModelFormat.ONNX else ModelFormat.GGML

    private fun setState(entry: CatalogEntry, state: DownloadState) {
        _entries.value = _entries.value.map { (e, s) -> if (e.model.id == entry.model.id) e to state else e to s }
    }

    /**
     * The model id to highlight as selected. This must belong to the *current
     * catalog*; disabled GGML entries are never selectable. Prefer the user's
     * persisted selection when it is still active and already downloaded; otherwise
     * fall back to the first downloaded entry, else the catalog's default.
     */
    private fun defaultOrPersistedId(): String {
        val catalog = catalogFor(app.modelFormat)
        val persisted = app.persistedModelId?.let { ModelCatalog.byId(it) }
        val candidates = buildList {
            if (persisted != null) add(persisted)
            addAll(catalog)
        }
        // First downloaded entry, preferring the persisted one; else the default.
        val fallbackDefault =
            if (app.modelFormat == ModelFormat.ONNX) ModelCatalog.onnxDefault else ModelCatalog.ggmlDefault
        return (candidates.firstOrNull { it in catalog && downloader.isDownloaded(it, app.modelFormat) } ?: fallbackDefault)
            .model.id
    }
}
