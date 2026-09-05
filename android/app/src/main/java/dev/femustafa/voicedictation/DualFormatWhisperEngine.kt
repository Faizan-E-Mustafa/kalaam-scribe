package dev.femustafa.voicedictation

import android.content.Context
import android.util.Log
import dev.femustafa.voicedictation.WhisperEngine.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * [WhisperEngine] that dispatches to one of several backends:
 *  - [ModelFormat.GGML] → [AarWhisperEngine] (whisper.cpp .bin files)
 *  - [ModelFormat.ONNX] → [SherpaWhisperEngine] (sherpa-onnx Whisper .onnx + .tokens)
 *  - Dolphin CTC → [DolphinCtcEngine] (sherpa-onnx OfflineDolphinModelConfig,
 *    auto-detect language)
 *  - Dolphin attention → [DolphinAttnEngine] (onnxruntime-android encoder+decoder,
 *    honors the ur/PK language pin)
 *
 * Lets a user choose between GGML and ONNX for the same underlying Whisper model
 * and have both work (Ticket 21). The chosen format is read at load time from
 * [VoiceDictationApp.modelFormat] for Whisper models, so switching format +
 * re-selecting a model reloads it with the matching backend. Dolphin CTC and
 * Dolphin attention models are always routed to their dedicated engines regardless
 * of the Whisper format.
 *
 * The backends return different, unrelated handle types, so this class wraps every
 * handle with its own [DualModelRef] carrying the backend; that lets
 * [transcribe]/[release] route the call to exactly the backend that owns it.
 */
class DualFormatWhisperEngine(
    private val context: Context,
) : WhisperEngine {

    private enum class Backend { GGML, ONNX, DOLPHIN_CTC, DOLPHIN_ATTN }

    private class DualModelRef(
        val backend: Backend,
        val delegate: WhisperModelRef,
    ) : WhisperModelRef

    private val ggmlEngine = AarWhisperEngine(context)
    private val onnxEngine = SherpaWhisperEngine(context)
    private val dolphinCtcEngine = DolphinCtcEngine(context)
    private val dolphinAttnEngine = DolphinAttnEngine(context)

    private fun engineFor(format: ModelFormat): WhisperEngine = when (format) {
        ModelFormat.GGML -> ggmlEngine
        ModelFormat.ONNX -> onnxEngine
    }

    /** The format the user currently has chosen (persisted). */
    private fun currentFormat(): ModelFormat =
        VoiceDictationApp.from(context).modelFormat

    override suspend fun load(modelPath: String): WhisperModelRef {
        val fileName = modelPath.substringAfterLast('/')
        // Dolphin Attn models load through their own ORT-based engine, independent of
        // the user's Whisper GGML/ONNX format choice.
        if (ModelCatalog.isDolphinAttnFileName(fileName)) {
            Log.i(TAG, "loading Dolphin attention model: $modelPath")
            return DualModelRef(Backend.DOLPHIN_ATTN, dolphinAttnEngine.load(modelPath))
        }
        // Dolphin CTC models load through their own engine, also format-independent.
        if (ModelCatalog.isDolphinCtcFileName(fileName)) {
            Log.i(TAG, "loading Dolphin CTC model: $modelPath")
            return DualModelRef(Backend.DOLPHIN_CTC, dolphinCtcEngine.load(modelPath))
        }
        val format = currentFormat()
        Log.i(TAG, "loading model with format $format: $modelPath")
        return DualModelRef(
            when (format) {
                ModelFormat.GGML -> Backend.GGML
                ModelFormat.ONNX -> Backend.ONNX
            },
            engineFor(format).load(modelPath),
        )
    }

    override suspend fun transcribe(
        model: WhisperModelRef,
        audioPath: String,
        languageMode: LanguageMode,
        language: String?,
    ): String {
        val real = model as? DualModelRef
            ?: throw IllegalArgumentException("unexpected model handle")
        return when (real.backend) {
            Backend.GGML -> ggmlEngine.transcribe(real.delegate, audioPath, languageMode, language)
            Backend.ONNX -> onnxEngine.transcribe(real.delegate, audioPath, languageMode, language)
            Backend.DOLPHIN_CTC -> dolphinCtcEngine.transcribe(real.delegate, audioPath, languageMode, language)
            Backend.DOLPHIN_ATTN -> dolphinAttnEngine.transcribe(real.delegate, audioPath, languageMode, language)
        }
    }

    private val sessionMutex = Mutex()

    override suspend fun createSession(
        model: WhisperModelRef,
        languageMode: LanguageMode,
        language: String?,
    ): TranscriptionSession? {
        val real = model as? DualModelRef
            ?: throw IllegalArgumentException("unexpected model handle")

        // Serialise session creation so the resident model isn't borrowed twice
        // (WhisperManager.sessionMutex already gates this across the app, but
        // we also hold the engine's own mutex to prevent races between concurrent
        // release()/createSession() calls).
        return sessionMutex.withLock {
            when (real.backend) {
                // The whisper.cpp GGML backend has no incremental API — batch only.
                Backend.GGML -> null
                Backend.ONNX -> onnxEngine.createSession(real.delegate, languageMode, language)
                Backend.DOLPHIN_CTC -> dolphinCtcEngine.createSession(real.delegate, languageMode, language)
                Backend.DOLPHIN_ATTN -> dolphinAttnEngine.createSession(real.delegate, languageMode, language)
            }
        }
    }

    override fun release(model: WhisperModelRef) {
        val real = model as? DualModelRef ?: return
        when (real.backend) {
            Backend.GGML -> ggmlEngine.release(real.delegate)
            Backend.ONNX -> onnxEngine.release(real.delegate)
            Backend.DOLPHIN_CTC -> dolphinCtcEngine.release(real.delegate)
            Backend.DOLPHIN_ATTN -> dolphinAttnEngine.release(real.delegate)
        }
        Log.i(TAG, "released model (backend ${real.backend})")
    }

    private companion object {
        const val TAG = "DualFormatWhisperEngine"
    }
}