package dev.femustafa.voicedictation

import android.content.Context
import android.util.Log
import dev.ffmpegkit.whisper.Whisper
import dev.ffmpegkit.whisper.WhisperConfig
import dev.ffmpegkit.whisper.WhisperModel

/**
 * Production [WhisperEngine] backed by the whisper.cpp AAR
 * (`dev.ffmpegkit.whisper.Whisper`). The AAR's API is suspend and needs a Context
 * and an on-device GGML file; this is verified on-device (ticket 08), not in
 * unit tests.
 */
class AarWhisperEngine(
    private val context: Context,
) : WhisperEngine {

    private class AarModelRef(val model: WhisperModel) : WhisperModelRef

    override suspend fun load(modelPath: String): WhisperModelRef {
        val model = Whisper.loadModel(context, modelPath)
        Log.i(TAG, "loaded model from $modelPath")
        return AarModelRef(model)
    }

    override suspend fun transcribe(
        model: WhisperModelRef,
        audioPath: String,
        languageMode: LanguageMode,
        language: String?,
    ): String {
        val real = requireNotNull(model as? AarModelRef) { "unexpected model handle" }
        val config = WhisperConfig(language = language ?: languageMode.whisperLanguage)
        val result = Whisper.transcribe(real.model, audioPath, config)
        return result.text?.takeIf { it.isNotBlank() } ?: ""
    }

    override fun release(model: WhisperModelRef) {
        val real = model as? AarModelRef ?: return
        Whisper.releaseModel(real.model)
        Log.i(TAG, "released model")
    }

    private val LanguageMode.whisperLanguage: String
        get() = when (this) {
            LanguageMode.Auto -> "auto"
            LanguageMode.English -> "en"
            LanguageMode.RomanUrdu -> "en"
        }

    private companion object {
        const val TAG = "AarWhisperEngine"
    }
}
