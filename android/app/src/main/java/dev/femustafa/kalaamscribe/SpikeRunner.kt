package dev.femustafa.kalaamscribe

import android.content.Context
import android.util.Log
import dev.ffmpegkit.whisper.WhisperConfig
import dev.ffmpegkit.whisper.WhisperModel
import dev.ffmpegkit.whisper.Whisper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

object SpikeRunner {
    private const val TAG = "WhisperSpike"

    suspend fun run(context: Context): String = withContext(Dispatchers.IO) {
        val filesDir = context.filesDir
        val modelPath = "$filesDir/ggml-base-q8_0.bin"
        val audioPath = "$filesDir/jfk.wav"

        val modelFile = java.io.File(modelPath)
        val audioFile = java.io.File(audioPath)
        if (!modelFile.exists()) {
            return@withContext "MISSING model at $modelPath"
        }
        if (!audioFile.exists()) {
            return@withContext "MISSING audio at $audioPath"
        }

        var model: WhisperModel? = null
        try {
            model = Whisper.loadModel(context, modelPath)
            Log.i(TAG, "model loaded ok")
            val config = WhisperConfig(language = "en", threads = 4)
            val result = Whisper.transcribe(model, audioPath, config)
            val text = result.text?.takeIf { it.isNotBlank() } ?: "(no text)"
            Log.i(TAG, "transcription: $text")
            text
        } catch (t: Throwable) {
            Log.e(TAG, "spike error", t)
            "ERROR: ${t.message}"
        } finally {
            if (model != null) {
                try {
                    Whisper.releaseModel(model)
                } catch (t: Throwable) {
                    Log.w(TAG, "releaseModel failed", t)
                }
            }
            Log.i(TAG, "spike complete")
        }
    }
}
