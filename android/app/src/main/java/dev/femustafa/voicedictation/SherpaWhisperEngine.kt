package dev.femustafa.voicedictation

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineWhisperModelConfig
import dev.femustafa.voicedictation.WhisperEngine.*

/**
 * Production [WhisperEngine] backed by sherpa-onnx.
 * Uses ONNX model format (.onnx + .tokens) for transcription.
 * Falls back to GGML format (.bin) when ONNX models are not available.
 * 
 * Ticket 21 notes:
 * - ONNX .onnx models are supported by this engine (new) - deferred until models are converted
 * - GGML .bin models are supported by the old AarWhisperEngine
 * - Non-Roman-Urdu models (indices 3-10) will be converted first
 * - Roman-Urdu models (indices 1-2) conversion deferred
 * - initial_prompt: workaround via language="en" (GitHub #2295)
 * - VAD: handled via separate Vad class with SileroVadModelConfig
 * 
 * Models are expected to be in the app's filesDir (downloaded by [ModelDownloader]).
 * For sherpa-onnx whisper models, we need encoder.onnx + decoder.onnx + tokens.txt.
 * If only a GGML .bin file is present (e.g. Roman-Urdu), fall back to basic loading.
 */
class SherpaWhisperEngine(
    private val context: Context,
) : WhisperEngine {

    class SherpaModelRef(
        val recognizer: OfflineRecognizer,
        val encoderPath: String,
        val decoderPath: String,
        val tokensPath: String,
    ) : WhisperModelRef

    override suspend fun load(modelPath: String): WhisperModelRef {
        // Find the ONNX catalog entry whose encoder filename matches the passed
        // model path (for ONNX, WhisperManager.absolutePath returns the encoder file).
        val modelEntry = ModelCatalog.onnxModels.firstOrNull { entry ->
            modelPath.endsWith(entry.model.fileName) || modelPath.endsWith(onnxEncoderName(entry))
        }

        Log.i(TAG, "loading sherpa-onnx model: $modelPath")

        if (modelEntry != null) {
            // Build ONNX model config with encoder/decoder/tokens paths. We do NOT
            // bake a language in here: the language is applied per-transcription in
            // [applyLanguage] so only the user-selected code for multilingual models
            // reaches the recognizer. An empty language means auto-detect.
            val whisperConfig = OfflineWhisperModelConfig()
            whisperConfig.language = ""
            whisperConfig.task = "transcribe"
            whisperConfig.tailPaddings = 0

            // Derive the ONNX encoder/decoder/tokens file paths from the model entry.
            // sherpa-onnx auto-detects int8 from `int8` in the filename.
            val encoderName = onnxEncoderName(modelEntry)
            val decoderName = onnxDecoderName(modelEntry)
            val tokensName = onnxTokensName(modelEntry)
            val encoderPath = java.io.File(context.filesDir, encoderName).absolutePath
            val decoderPath = java.io.File(context.filesDir, decoderName).absolutePath
            val tokensPath = java.io.File(context.filesDir, tokensName).absolutePath

            Log.i(TAG, "ONNX encoder: $encoderPath")
            Log.i(TAG, "ONNX decoder: $decoderPath")
            Log.i(TAG, "ONNX tokens : $tokensPath")

            // All three ONNX files must be present. sherpa-onnx segfaults
            // ("Please provide a model") on an empty config, so never construct a
            // recognizer without all files — throw a descriptive error instead and
            // let the caller surface it (e.g. WhisperManager status).
            val encoderExists = java.io.File(encoderPath).exists()
            val decoderExists = java.io.File(decoderPath).exists()
            val tokensExists = java.io.File(tokensPath).exists()

            if (!(encoderExists && decoderExists && tokensExists)) {
                val missing = buildList {
                    if (!encoderExists) add("encoder")
                    if (!decoderExists) add("decoder")
                    if (!tokensExists) add("tokens")
                }.joinToString(", ")
                Log.w(TAG, "ONNX files missing for ${modelEntry.model.id}: $missing")
                throw java.io.FileNotFoundException(
                    "ONNX model files missing for ${modelEntry.model.id} ($missing). " +
                        "Download the model in the Models screen first."
                )
            }

            whisperConfig.encoder = encoderPath
            whisperConfig.decoder = decoderPath

            val modelConfig = OfflineModelConfig()
            modelConfig.whisper = whisperConfig
            modelConfig.tokens = tokensPath
            modelConfig.numThreads = VoiceDictationApp.from(context).whisperThreads()
            modelConfig.debug = true

            val recognizerConfig = OfflineRecognizerConfig()
            recognizerConfig.modelConfig = modelConfig
            recognizerConfig.decodingMethod = "greedy_search"

            Log.i(TAG, "loading ONNX sherpa-onnx model with full config")
            // Models are loaded from absolute paths in filesDir, so assetManager must
            // be null (sherpa-onnx aborts otherwise — github.com/k2-fsa/sherpa-onnx#2562).
            val recognizer = OfflineRecognizer(null, recognizerConfig)
            return SherpaModelRef(recognizer, encoderPath, decoderPath, tokensPath)
        }

        throw java.io.FileNotFoundException(
            "No sherpa-onnx source for model path: $modelPath (no ONNX model entry found)."
        )
    }

    override suspend fun transcribe(
        model: WhisperModelRef,
        audioPath: String,
        languageMode: LanguageMode,
        language: String?,
    ): String {
        val real = model as? SherpaModelRef ?: throw IllegalArgumentException("unexpected model handle")
        val wave = readWave(audioPath)
            ?: throw java.io.IOException("Failed to read wave file: $audioPath")

        // Apply the per-transcription language via recognizer.setConfig(), which
        // is the officially supported way to change Whisper language at runtime
        // (k2-fsa/sherpa-onnx#1116). We use the user-selected code only for
        // multilingual (auto-detect) Models; fixed-English and Roman-Urdu Models
        // never take an override (their output must stay in Latin script). When no
        // language is selected we pass "" so Whisper auto-detects.
        applyLanguage(real, resolveLanguage(languageMode, language))

        val stream = real.recognizer.createStream()
        try {
            stream.acceptWaveform(wave.samples, wave.sampleRate)
            real.recognizer.decode(stream)
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
        val real = model as? SherpaModelRef ?: return null
        val resolvedLanguage = resolveLanguage(languageMode, language)
        val waveWriter = InMemoryWaveWriter(16000)
        // Create a SherpaVad instance for this session
        val sileroPath = java.io.File(context.filesDir, "silero_vad.onnx").absolutePath
        if (!java.io.File(sileroPath).exists()) {
            Log.w(TAG, "Silero VAD model not found at $sileroPath, cannot create streaming session")
            return null
        }
        val silero = com.k2fsa.sherpa.onnx.SileroVadModelConfig()
        silero.model = sileroPath
        silero.threshold = 0.25f
        silero.minSilenceDuration = 0.5f
        silero.minSpeechDuration = 0.5f
        silero.windowSize = 512
        silero.maxSpeechDuration = 30.0f
        val vadConfig = com.k2fsa.sherpa.onnx.VadModelConfig()
        vadConfig.sileroVadModelConfig = silero
        vadConfig.sampleRate = 16000
        vadConfig.numThreads = VoiceDictationApp.from(context).whisperThreads()
        vadConfig.provider = "cpu"
        val vad = com.k2fsa.sherpa.onnx.Vad(null, vadConfig)
        val vadLike = SherpaVad(vad)
        return SherpaOfflineSession(real.recognizer, waveWriter, resolvedLanguage, real, context, vadLike)
    }

    /**
     * Update the whisper sub-config of the resident recognizer so subsequent
     * decodes use [lang] ("" = auto-detect). The C++ whisper impl's SetConfig only
     * reads the whisper sub-config, so we rebuild just that portion.
     */
    internal fun applyLanguage(real: SherpaModelRef, lang: String) {
        val whisperConfig = OfflineWhisperModelConfig()
        whisperConfig.encoder = real.encoderPath
        whisperConfig.decoder = real.decoderPath
        whisperConfig.language = lang
        whisperConfig.task = "transcribe"
        whisperConfig.tailPaddings = 0

        val modelConfig = OfflineModelConfig()
        modelConfig.whisper = whisperConfig
        modelConfig.tokens = real.tokensPath
        modelConfig.numThreads = VoiceDictationApp.from(context).whisperThreads()
        modelConfig.debug = true

        val recognizerConfig = OfflineRecognizerConfig()
        recognizerConfig.modelConfig = modelConfig
        recognizerConfig.decodingMethod = "greedy_search"

        Log.i(TAG, "setting whisper language to: '${lang}'")
        real.recognizer.setConfig(recognizerConfig)
    }

    /**
     * Resolve the whisper language code to use for a transcription. Only
     * multilingual ([LanguageMode.Auto]) Models may take the user-selected
     * [language]; fixed-English and Roman-Urdu Models always resolve to "en" so
     * their output stays in Latin script and never takes an override. A missing
     * [language] on a multilingual Model resolves to "" (auto-detect).
     */
    private fun resolveLanguage(languageMode: LanguageMode, language: String?): String =
        when (languageMode) {
            LanguageMode.Auto -> language ?: ""
            LanguageMode.English -> "en"
            LanguageMode.RomanUrdu -> "en"
        }

    /**
     * Parse a PCM 16-bit mono Wave file into float samples + sample rate.
     * Handles the app's own recorder output (16 kHz mono 16-bit), which is what
     * the dictation pipeline produces. Returns null on a file we can't read.
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

        // Walk chunks to find "data". RIFF at byte 12 ("fmt ") is 16 bytes for PCM,
        // so data typically starts at byte 36/44. We parse defensively: read the
        // sample rate + block align from fmt, then locate the data chunk.
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
                // dataLen may be stale; clamp to actual bytes remaining in file.
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

    override fun release(model: WhisperModelRef) {
        val real = model as? SherpaModelRef ?: return
        try {
            real.recognizer.release()
            Log.i(TAG, "released sherpa-onnx model")
        } catch (e: Exception) {
            Log.w(TAG, "error releasing model: ${e.message}")
        }
    }

    private companion object {
        const val TAG = "SherpaWhisperEngine"

        /** ONNX encoder filename for an entry, e.g. `<id>-encoder.onnx` / `<id>-encoder.int8.onnx`. */
        fun onnxEncoderName(entry: CatalogEntry): String =
            if (entry.precision == ModelPrecision.INT8) "${entry.model.id}-encoder.int8.onnx"
            else "${entry.model.id}-encoder.onnx"

        /** ONNX decoder filename for an entry, e.g. `<id>-decoder.onnx` / `<id>-decoder.int8.onnx`. */
        fun onnxDecoderName(entry: CatalogEntry): String =
            if (entry.precision == ModelPrecision.INT8) "${entry.model.id}-decoder.int8.onnx"
            else "${entry.model.id}-decoder.onnx"

        /** ONNX tokens filename for an entry, e.g. `<id>-tokens.txt` (shared by fp32 and int8). */
        fun onnxTokensName(entry: CatalogEntry): String = "${entry.model.id}-tokens.txt"
    }
}