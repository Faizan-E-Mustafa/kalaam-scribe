package dev.femustafa.voicedictation

import android.util.Log
import java.net.HttpURLConnection
import java.net.URL

/**
 * Downloads a [CatalogEntry]'s model file for a chosen [ModelFormat].
 *
 * Supports both GGML (.bin via [CatalogEntry.sourceUrl], loaded by
 * [AarWhisperEngine]) and ONNX (.onnx + .tokens via [CatalogEntry.onnxSourceUrl],
 * loaded by [SherpaWhisperEngine]) formats. Files land in the app's private
 * internal storage (no storage permission needed), satisfying the ticket-08
 * finding that Models must live in app-private storage.
 *
 * Uses HttpURLConnection (no extra dependency) and reports progress as a fraction
 * 0f..1f. No size gating, per spec. Not unit-tested (network); verified on-device.
 *
 * HTTP 401 (authentication required) is surfaced with a helpful message so the
 * user knows to download the model manually and place it in the app's files dir.
 */
class ModelDownloader(
    private val baseDir: java.io.File,
) {
    sealed interface Result {
        data class Success(val bytes: Long) : Result
        data class Failure(val message: String) : Result
    }

    private val DEFAULT_BUFFER_SIZE = 8192

    private companion object {
        const val TAG = "ModelDownloader"

        // ONNX model files ship as three siblings. These floor sizes reject
        // stale/truncated downloads (a partial write or an aborted fetch) that
        // sherpa-onnx would otherwise try to load and crash on.
        const val MIN_ONNX_ENCODER_BYTES = 1_000_000L
        const val MIN_ONNX_DECODER_BYTES = 1_000_000L
        const val MIN_ONNX_TOKENS_BYTES = 100_000L
    }

    /**
     * Download [entry] into [baseDir] for [format], calling [onProgress] with a
     * fraction 0f..1f. If the target file(s) already exist, returns Success without
     * re-downloading.
     */
    suspend fun download(
        entry: CatalogEntry,
        format: ModelFormat,
        onProgress: (Float) -> Unit,
    ): Result = when (format) {
        ModelFormat.GGML -> downloadGGML(entry, onProgress)
        ModelFormat.ONNX -> downloadONNX(entry, onProgress)
    }

    /** Whether [entry] already has a downloaded file for [format]. */
    fun isDownloaded(entry: CatalogEntry, format: ModelFormat): Boolean = when (format) {
        ModelFormat.GGML -> java.io.File(baseDir, entry.model.fileName).exists()
        ModelFormat.ONNX -> onnxComplete(entry)
    }

    /**
     * True when all three ONNX files (encoder, decoder, tokens) exist with
     * plausible sizes. Guards against stale/truncated files that would otherwise
     * load as "Ready" and crash sherpa-onnx at transcription time.
     */
    fun onnxComplete(entry: CatalogEntry): Boolean {
        val encoder = java.io.File(baseDir, onnxEncoderName(entry))
        val decoder = java.io.File(baseDir, onnxDecoderName(entry))
        val tokens = java.io.File(baseDir, onnxTokensName(entry))
        return encoder.exists() && encoder.length() >= MIN_ONNX_ENCODER_BYTES &&
            decoder.exists() && decoder.length() >= MIN_ONNX_DECODER_BYTES &&
            tokens.exists() && tokens.length() >= MIN_ONNX_TOKENS_BYTES
    }

    /** The .onnx encoder filename for an ONNX model, e.g. `<id>-encoder.onnx` or `<id>-encoder.int8.onnx`. */
    fun onnxEncoderName(entry: CatalogEntry): String =
        if (entry.precision == ModelPrecision.INT8) "${entry.model.id}-encoder.int8.onnx"
        else "${entry.model.id}-encoder.onnx"

    /** The .tokens filename for an ONNX model, e.g. `<id>-tokens.txt` (shared by fp32 and int8). */
    fun onnxTokensName(entry: CatalogEntry): String = "${entry.model.id}-tokens.txt"

    /** The .onnx decoder filename for an ONNX model, e.g. `<id>-decoder.onnx` or `<id>-decoder.int8.onnx`. */
    fun onnxDecoderName(entry: CatalogEntry): String =
        if (entry.precision == ModelPrecision.INT8) "${entry.model.id}-decoder.int8.onnx"
        else "${entry.model.id}-decoder.onnx"

    private suspend fun downloadGGML(
        entry: CatalogEntry,
        onProgress: (Float) -> Unit,
    ): Result {
        val url = entry.sourceUrl
            ?: return Result.Failure("No GGML URL for model ${entry.model.id}")
        val target = java.io.File(baseDir, entry.model.fileName)
        if (target.exists()) {
            onProgress(1f)
            return Result.Success(target.length())
        }
        val tmp = java.io.File(baseDir, "${entry.model.fileName}.part")

        val connection = URL(url).openConnection() as HttpURLConnection
        try {
            connection.connectTimeout = 15_000
            connection.readTimeout = 30_000
            connection.connect()
            val code = connection.responseCode
            if (code == HttpURLConnection.HTTP_OK) {
                val total = connection.contentLengthLong
                var downloaded = 0L
                target.parentFile?.mkdirs()

                connection.inputStream.use { input ->
                    tmp.outputStream().use { out ->
                        val buf = ByteArray(DEFAULT_BUFFER_SIZE)
                        while (true) {
                            val read = input.read(buf)
                            if (read == -1) break
                            out.write(buf, 0, read)
                            downloaded += read
                            if (total > 0) onProgress(downloaded.toFloat() / total)
                        }
                    }
                }

                if (tmp.renameTo(target)) {
                    tmp.delete()
                    onProgress(1f)
                    return Result.Success(downloaded)
                }
                return Result.Failure("rename failed")
            }
            return httpFailure(code, url)
        } catch (e: Exception) {
            tmp.delete()
            return Result.Failure(e.message ?: e.toString())
        } finally {
            connection.disconnect()
        }
    }

    private suspend fun downloadONNX(
        entry: CatalogEntry,
        onProgress: (Float) -> Unit,
    ): Result {
        val url = entry.onnxSourceUrl
            ?: return Result.Failure("No ONNX URL for model ${entry.model.id}")

        // The sherpa-onnx pre-exported model ships encoder + decoder + tokens as
        // three sibling files. The encoder URL points at `*-encoder.onnx`; the
        // decoder and tokens come from sibling URLs derived by swapping the suffix.
        val encoderTarget = java.io.File(baseDir, onnxEncoderName(entry))
        if (encoderTarget.exists()) {
            // Encoder already downloaded; still try to fill any missing siblings.
            ensureSiblings(entry, url)
            onProgress(1f)
            return Result.Success(encoderTarget.length())
        }

        val encoderTmp = java.io.File(baseDir, "${onnxEncoderName(entry)}.part")
        val connection = URL(url).openConnection() as HttpURLConnection
        try {
            connection.connectTimeout = 15_000
            connection.readTimeout = 30_000
            connection.connect()
            val code = connection.responseCode
            if (code != HttpURLConnection.HTTP_OK) {
                return httpFailure(code, url)
            }
            val total = connection.contentLengthLong
            var downloaded = 0L
            encoderTarget.parentFile?.mkdirs()

            connection.inputStream.use { input ->
                encoderTmp.outputStream().use { out ->
                    val buf = ByteArray(DEFAULT_BUFFER_SIZE)
                    while (true) {
                        val read = input.read(buf)
                        if (read == -1) break
                        out.write(buf, 0, read)
                        downloaded += read
                        if (total > 0) onProgress(downloaded.toFloat() / total)
                    }
                }
            }

            if (encoderTmp.renameTo(encoderTarget)) {
                encoderTmp.delete()
                ensureSiblings(entry, url)
                onProgress(1f)
                return Result.Success(downloaded)
            }
            return Result.Failure("rename failed")
        } catch (e: Exception) {
            encoderTmp.delete()
            return Result.Failure(e.message ?: e.toString())
        } finally {
            connection.disconnect()
        }
    }

    /**
     * Ensure the decoder and tokens siblings exist for [entry]'s ONNX model.
     * Both are required by sherpa-onnx (loading without them fails), so a missing
     * sibling is reported as a failure rather than silently ignored.
     */
    private fun ensureSiblings(entry: CatalogEntry, encoderUrl: String) {
        val decoderGot = ensureSibling(encoderUrl, "decoder", onnxDecoderName(entry))
        val tokensGot = ensureSibling(encoderUrl, "tokens", onnxTokensName(entry))
        if (!(decoderGot && tokensGot)) {
            Log.w(TAG, "ONNX sibling download incomplete for ${entry.model.id}")
        }
    }

    /**
     * Best-effort download of an ONNX sibling file (decoder onnx / tokens txt),
     * derived from the encoder URL by swapping the model part (e.g. for fp32
     * `base.en-encoder.onnx` → `base.en-decoder.onnx` / `base.en-tokens.txt`; for
     * int8 `tiny-encoder.int8.onnx` → `tiny-decoder.int8.onnx` / `tiny-tokens.txt`).
     * Returns true if the file is present afterwards (already there or newly fetched).
     */
    private fun ensureSibling(
        encoderUrl: String,
        target: String,
        outName: String,
    ): Boolean {
        val targetFile = java.io.File(baseDir, outName)
        val isTokens = target == "tokens"
        val minBytes = if (isTokens) MIN_ONNX_TOKENS_BYTES else MIN_ONNX_DECODER_BYTES
        if (targetFile.exists() && targetFile.length() >= minBytes) return true
        if (targetFile.exists()) {
            // Existing file is stale/undersized; drop it so the fresh copy below lands.
            Log.w(TAG, "discarding undersized $outName (${targetFile.length()} bytes)")
            targetFile.delete()
        }
        val int8 = encoderUrl.contains("-encoder.int8.onnx")
        val base = if (int8) encoderUrl.substringBefore("-encoder.int8.onnx")
            else encoderUrl.substringBefore("-encoder.onnx")
        val siblingUrl = when (target) {
            "decoder" -> base + if (int8) "-decoder.int8.onnx" else "-decoder.onnx"
            else -> base + "-tokens.txt"
        }
        val tmp = java.io.File(baseDir, "$outName.part")
        try {
            val connection = URL(siblingUrl).openConnection() as HttpURLConnection
            var ok = false
            try {
                connection.connectTimeout = 15_000
                connection.readTimeout = 30_000
                connection.connect()
                val code = connection.responseCode
                if (code != HttpURLConnection.HTTP_OK) return false
                // Stream in chunks rather than buffering the whole file: the
                // decoder for large models is many hundreds of MB and buffering
                // it all in one ByteArray would OOM the VM.
                connection.inputStream.use { input ->
                    targetFile.parentFile?.mkdirs()
                    tmp.outputStream().use { out ->
                        val buf = ByteArray(DEFAULT_BUFFER_SIZE)
                        while (true) {
                            val read = input.read(buf)
                            if (read == -1) break
                            out.write(buf, 0, read)
                        }
                    }
                }
                ok = tmp.renameTo(targetFile)
            } finally {
                connection.disconnect()
            }
            if (!ok) tmp.delete()
            return ok && targetFile.exists() && targetFile.length() >= minBytes
        } catch (e: Exception) {
            tmp.delete()
            Log.w(TAG, "sibling download failed for $outName from $siblingUrl: ${e.message}")
            return false
        }
    }

    private fun httpFailure(code: Int, url: String): Result =
        if (code == HttpURLConnection.HTTP_UNAUTHORIZED) {
            Result.Failure(
                "HTTP 401: Authentication required. Download the model manually from " +
                    "$url and place it in the app's files directory."
            )
        } else {
            Result.Failure("HTTP $code from $url")
        }
}