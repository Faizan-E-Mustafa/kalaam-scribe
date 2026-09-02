package dev.femustafa.voicedictation

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineDolphinModelConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import dev.femustafa.voicedictation.WhisperEngine.*

/**
 * [WhisperEngine] backed by sherpa-onnx for the Dolphin CTC multilingual model
 * family (DataoceanAI). Unlike [SherpaWhisperEngine] (which loads Whisper's
 * encoder+decoder+tokens), Dolphin CTC is a single ONNX model file plus a
 * `tokens.txt`; it is loaded via sherpa-onnx's `OfflineDolphinModelConfig`.
 *
 * Dolphin CTC supports 40+ Eastern languages including Urdu, but sherpa-onnx does
 * not expose a language switch for it (auto-detect only), so [transcribe] ignores
 * the [language]/[languageMode] arguments and always lets the model detect.
 *
 * Models are downloaded by [ModelDownloader] into the app's filesDir as
 * `<id>-model.onnx` + `<id>-tokens.txt`.
 */
class DolphinCtcEngine(
    private val context: Context,
) : WhisperEngine {

    private class DolphinModelRef(
        val recognizer: OfflineRecognizer
    ) : WhisperModelRef

    override suspend fun load(modelPath: String): WhisperModelRef {
        // Adopt the catalog entry whose model file matches the passed path so we
        // can derive the sibling tokens file path (and per-identity naming).
        val entry = ModelCatalog.dolphinCtcModels.firstOrNull { entry ->
            modelPath.endsWith(entry.model.fileName)
        } ?: throw java.io.FileNotFoundException(
            "No Dolphin CTC model entry for path: $modelPath"
        )

        Log.i(TAG, "loading Dolphin CTC model: $modelPath")

        // The model path passed by the manager is already the app-private absolute
        // path to `<id>-model.onnx`; derive the sibling tokens path from the entry.
        val tokensPath = java.io.File(context.filesDir, dolphinTokensName(entry)).absolutePath

        Log.i(TAG, "Dolphin CTC model : $modelPath")
        Log.i(TAG, "Dolphin CTC tokens: $tokensPath")

        // Both files must be present; sherpa-onnx crashes on a sparse config, so
        // throw a descriptive error and let the caller surface it instead.
        val modelExists = java.io.File(modelPath).exists()
        val tokensExists = java.io.File(tokensPath).exists()
        if (!(modelExists && tokensExists)) {
            val missing = buildList {
                if (!modelExists) add("model")
                if (!tokensExists) add("tokens")
            }.joinToString(", ")
            Log.w(TAG, "Dolphin CTC files missing for ${entry.model.id}: $missing")
            throw java.io.FileNotFoundException(
                "Dolphin CTC model files missing for ${entry.model.id} ($missing). " +
                    "Download the model in the Models screen first."
            )
        }

        val dolphinConfig = OfflineDolphinModelConfig(modelPath)

        val modelConfig = OfflineModelConfig()
        modelConfig.dolphin = dolphinConfig
        modelConfig.tokens = tokensPath
        modelConfig.numThreads = VoiceDictationApp.from(context).whisperThreads()
        modelConfig.debug = true

        val recognizerConfig = OfflineRecognizerConfig()
        recognizerConfig.modelConfig = modelConfig
        // CTC models decode with greedy search.
        recognizerConfig.decodingMethod = "greedy_search"

        // Models are loaded from absolute paths in filesDir, so assetManager must
        // be null (sherpa-onnx aborts otherwise — github.com/k2-fsa/sherpa-onnx#2562).
        val recognizer = OfflineRecognizer(null, recognizerConfig)
        return DolphinModelRef(recognizer)
    }

    override suspend fun transcribe(
        model: WhisperModelRef,
        audioPath: String,
        languageMode: LanguageMode,
        language: String?,
    ): String {
        val real = model as? DolphinModelRef ?: throw IllegalArgumentException("unexpected model handle")
        val wave = readWave(audioPath)
            ?: throw java.io.IOException("Failed to read wave file: $audioPath")
        val stream = real.recognizer.createStream()
        try {
            stream.acceptWaveform(wave.samples, wave.sampleRate)
            real.recognizer.decode(stream)
            // Dolphin auto-detects the language; the language args are ignored.
            return real.recognizer.getResult(stream).text.trim()
        } finally {
            stream.release()
        }
    }

    override fun release(model: WhisperModelRef) {
        val real = model as? DolphinModelRef ?: return
        try {
            real.recognizer.release()
            Log.i(TAG, "released Dolphin CTC model")
        } catch (e: Exception) {
            Log.w(TAG, "error releasing model: ${e.message}")
        }
    }

    /**
     * Parse a PCM 16-bit mono Wave file into float samples + sample rate. Mirrors
     * the app's own recorder output (16 kHz mono 16-bit), which is what the
     * dictation pipeline produces. Returns null on a file we can't read.
     */
    private fun readWave(path: String): Wave? {
        val f = java.io.File(path)
        if (!f.exists()) return null
        val len = f.length()
        if (len < 44) return null

        // Verify RIFF/WAVE + find the data chunk, using actual file length rather
        // than the (sometimes stale) data-size header field.
        val bytes = f.readBytes()
        if (bytes[0] != 'R'.code.toByte() || bytes[1] != 'I'.code.toByte() ||
            bytes[2] != 'F'.code.toByte() || bytes[3] != 'F'.code.toByte() ||
            bytes[8] != 'W'.code.toByte() || bytes[9] != 'A'.code.toByte() ||
            bytes[10] != 'V'.code.toByte() || bytes[11] != 'E'.code.toByte()
        ) {
            return null
        }

        var offset = 12
        var sampleRate = 16000
        var nChannels = 1
        var bitsPerSample = 16
        var dataStart = -1
        var dataLen = 0
        while (offset + 8 <= len) {
            val fourCC = String(bytes, offset, 4)
            val size = le32(bytes, offset + 4)
            if (fourCC == "fmt ") {
                sampleRate = le32(bytes, offset + 12)
                nChannels = le16(bytes, offset + 10)
                bitsPerSample = le16(bytes, offset + 22)
            } else if (fourCC == "data") {
                dataStart = offset + 8
                val remaining = (len - dataStart).toInt().coerceAtLeast(0)
                dataLen = size.coerceAtMost(remaining)
                break
            }
            offset += 8 + size
        }
        if (dataStart < 0) return null
        if (bitsPerSample != 16) return null

        val frameBytes = nChannels * 2
        val sampleCount = dataLen / frameBytes
        if (sampleCount <= 0) return null

        val samples = FloatArray(sampleCount)
        var bi = dataStart
        for (i in 0 until sampleCount) {
            val sample = ((bytes[bi + 1].toInt() and 0xFF) shl 8) or (bytes[bi].toInt() and 0xFF)
            samples[i] = sample.toShort().toFloat() / 32768f
            bi += frameBytes
        }
        return Wave(samples, sampleRate)
    }

    private fun le16(b: ByteArray, off: Int): Int =
        (b[off].toInt() and 0xFF) or ((b[off + 1].toInt() and 0xFF) shl 8)

    private fun le32(b: ByteArray, off: Int): Int =
        (b[off].toInt() and 0xFF) or
            ((b[off + 1].toInt() and 0xFF) shl 8) or
            ((b[off + 2].toInt() and 0xFF) shl 16) or
            ((b[off + 3].toInt() and 0xFF) shl 24)

    private data class Wave(val samples: FloatArray, val sampleRate: Int)

    companion object {
        private const val TAG = "DolphinCtcEngine"

        /** Dolphin CTC tokens filename for an entry, e.g. `<id>-tokens.txt`. */
        fun dolphinTokensName(entry: CatalogEntry): String = "${entry.model.id}-tokens.txt"
    }
}
