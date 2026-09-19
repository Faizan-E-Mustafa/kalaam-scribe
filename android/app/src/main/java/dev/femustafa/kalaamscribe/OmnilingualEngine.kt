package dev.femustafa.kalaamscribe

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineOmnilingualAsrCtcModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import dev.femustafa.kalaamscribe.WhisperEngine.*

/**
 * [WhisperEngine] backed by sherpa-onnx for the Omnilingual (Meta OmniASR CTC)
 * model family. The sherpa-onnx port is the 300M/1B int8 CTC variant; it is a
 * single ONNX model file plus a `tokens.txt`, loaded via
 * `OfflineOmnilingualAsrCtcModelConfig`.
 *
 * Omnilingual supports 1600+ zero-shot languages but sherpa-onnx exposes **no
 * way to pin an output language** (language conditioning exists only in the
 * LLM variants, which sherpa-onnx does not support), so [transcribe] ignores
 * the [language]/[languageMode] arguments and always lets the model detect.
 *
 * The model is downloaded by [ModelDownloader] into the app's filesDir as
 * `<id>-model.onnx` + `<id>-tokens.txt`.
 */
class OmnilingualEngine(
    private val context: Context,
) : WhisperEngine {

    private class OmnilingualModelRef(
        val recognizer: OfflineRecognizer
    ) : WhisperModelRef

    override suspend fun load(modelPath: String): WhisperModelRef {
        val entry = ModelCatalog.omnilingualModels.firstOrNull { entry ->
            modelPath.endsWith(entry.model.fileName)
        } ?: throw java.io.FileNotFoundException(
            "No Omnilingual model entry for path: $modelPath"
        )

        Log.i(TAG, "loading Omnilingual model: $modelPath")

        val tokensPath = java.io.File(context.filesDir, omnilingualTokensName(entry)).absolutePath

        Log.i(TAG, "Omnilingual model : $modelPath")
        Log.i(TAG, "Omnilingual tokens: $tokensPath")

        val modelExists = java.io.File(modelPath).exists()
        val tokensExists = java.io.File(tokensPath).exists()
        if (!(modelExists && tokensExists)) {
            val missing = buildList {
                if (!modelExists) add("model")
                if (!tokensExists) add("tokens")
            }.joinToString(", ")
            Log.w(TAG, "Omnilingual files missing for ${entry.model.id}: $missing")
            throw java.io.FileNotFoundException(
                "Omnilingual model files missing for ${entry.model.id} ($missing). " +
                    "Download the model in the Models screen first."
            )
        }

        val omniConfig = OfflineOmnilingualAsrCtcModelConfig(modelPath)

        val modelConfig = OfflineModelConfig()
        modelConfig.omnilingual = omniConfig
        modelConfig.tokens = tokensPath
        modelConfig.numThreads = KalaamApp.from(context).whisperThreads()
        modelConfig.debug = true

        val recognizerConfig = OfflineRecognizerConfig()
        recognizerConfig.modelConfig = modelConfig
        recognizerConfig.decodingMethod = "greedy_search"

        val recognizer = OfflineRecognizer(null, recognizerConfig)
        return OmnilingualModelRef(recognizer)
    }

    override suspend fun transcribe(
        model: WhisperModelRef,
        audioPath: String,
        languageMode: LanguageMode,
        language: String?,
    ): String {
        val real = model as? OmnilingualModelRef ?: throw IllegalArgumentException("unexpected model handle")
        val wave = readWave(audioPath)
            ?: throw java.io.IOException("Failed to read wave file: $audioPath")
        val stream = real.recognizer.createStream()
        try {
            stream.acceptWaveform(wave.samples, wave.sampleRate)
            real.recognizer.decode(stream)
            // Omnilingual auto-detects the language; the language args are ignored.
            return real.recognizer.getResult(stream).text.trim()
        } finally {
            stream.release()
        }
    }

    override suspend fun createSession(
        model: WhisperModelRef,
        languageMode: LanguageMode,
        language: String?,
    ): TranscriptionSession? {
        val real = model as? OmnilingualModelRef ?: return null
        val waveWriter = InMemoryWaveWriter(16000)
        val sileroPath = java.io.File(context.filesDir, "silero_vad.onnx").absolutePath
        if (!java.io.File(sileroPath).exists()) {
            Log.w(TAG, "Silero VAD model not found at $sileroPath, cannot create streaming session")
            return null
        }
        val silero = com.k2fsa.sherpa.onnx.SileroVadModelConfig()
        silero.model = sileroPath
        val app = KalaamApp.from(context)
        silero.threshold = app.vadThreshold()
        silero.minSilenceDuration = app.vadMinSilence()
        silero.minSpeechDuration = app.vadMinSpeech()
        silero.windowSize = 512
        silero.maxSpeechDuration = app.vadMaxSpeech()
        val vadConfig = com.k2fsa.sherpa.onnx.VadModelConfig()
        vadConfig.sileroVadModelConfig = silero
        vadConfig.sampleRate = 16000
        vadConfig.numThreads = KalaamApp.from(context).whisperThreads()
        vadConfig.provider = "cpu"
        val vad = com.k2fsa.sherpa.onnx.Vad(null, vadConfig)
        val vadLike = SherpaVad(vad)
        // Omnilingual doesn't use language override, pass empty string
        return SherpaOfflineSession(real.recognizer, waveWriter, "", null, context, vadLike)
    }

    override fun release(model: WhisperModelRef) {
        val real = model as? OmnilingualModelRef ?: return
        try {
            real.recognizer.release()
            Log.i(TAG, "released Omnilingual model")
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
        private const val TAG = "OmnilingualEngine"

        /** Omnilingual tokens filename for an entry, e.g. `<id>-tokens.txt`. */
        fun omnilingualTokensName(entry: CatalogEntry): String = "${entry.model.id}-tokens.txt"
    }
}