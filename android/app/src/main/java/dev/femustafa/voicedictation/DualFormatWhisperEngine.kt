package dev.femustafa.voicedictation

import android.content.Context
import android.util.Log
import dev.femustafa.voicedictation.WhisperEngine.*

/**
 * [WhisperEngine] that dispatches to one of two backends based on the app-wide
 * model-file [ModelFormat] chosen in settings:
 *  - [ModelFormat.GGML] → [AarWhisperEngine] (whisper.cpp .bin files)
 *  - [ModelFormat.ONNX] → [SherpaWhisperEngine] (sherpa-onnx .onnx + .tokens files)
 *
 * Lets a user choose between GGML and ONNX for the same underlying Whisper model
 * and have both work (Ticket 21). The chosen format is read at load time from
 * [VoiceDictationApp.modelFormat], so switching format + re-selecting a model
 * reloads it with the matching backend.
 *
 * The two backends return different, unrelated handle types, so this class wraps
 * every handle with its own [DualModelRef] carrying the backend format; that lets
 * [transcribe]/[release] route the call to exactly the backend that owns it.
 */
class DualFormatWhisperEngine(
    private val context: Context,
) : WhisperEngine {

    private class DualModelRef(
        val format: ModelFormat,
        val delegate: WhisperModelRef,
    ) : WhisperModelRef

    private val ggmlEngine = AarWhisperEngine(context)
    private val onnxEngine = SherpaWhisperEngine(context)

    private fun engineFor(format: ModelFormat): WhisperEngine = when (format) {
        ModelFormat.GGML -> ggmlEngine
        ModelFormat.ONNX -> onnxEngine
    }

    /** The format the user currently has chosen (persisted). */
    private fun currentFormat(): ModelFormat =
        VoiceDictationApp.from(context).modelFormat

    override suspend fun load(modelPath: String): WhisperModelRef {
        val format = currentFormat()
        Log.i(TAG, "loading model with format $format: $modelPath")
        val delegate = engineFor(format).load(modelPath)
        return DualModelRef(format, delegate)
    }

    override suspend fun transcribe(
        model: WhisperModelRef,
        audioPath: String,
        languageMode: LanguageMode,
        language: String?,
    ): String {
        val real = model as? DualModelRef
            ?: throw IllegalArgumentException("unexpected model handle")
        return engineFor(real.format).transcribe(real.delegate, audioPath, languageMode, language)
    }

    override fun release(model: WhisperModelRef) {
        val real = model as? DualModelRef ?: return
        engineFor(real.format).release(real.delegate)
        Log.i(TAG, "released model (format ${real.format})")
    }

    private companion object {
        const val TAG = "DualFormatWhisperEngine"
    }
}