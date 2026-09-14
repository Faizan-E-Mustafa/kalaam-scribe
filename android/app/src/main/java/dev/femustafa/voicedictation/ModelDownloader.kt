package dev.femustafa.voicedictation

import android.util.Log
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.async

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

    companion object {
        const val TAG = "ModelDownloader"

        // ONNX model files ship as three siblings. These floor sizes reject
        // stale/truncated downloads (a partial write or an aborted fetch) that
        // sherpa-onnx would otherwise try to load and crash on.
        const val MIN_ONNX_ENCODER_BYTES = 1_000_000L
        const val MIN_ONNX_DECODER_BYTES = 1_000_000L
        const val MIN_ONNX_TOKENS_BYTES = 100_000L
        // Dolphin CTC model.onnx is a few tens of MB at minimum; this floor
        // rejects stale/truncated downloads before sherpa-onnx tries to load them.
        const val MIN_DOLPHIN_MODEL_BYTES = 1_000_000L
        // Omnilingual (Meta OmniASR CTC) tokens.txt is smaller (~84 KB) than the
        // generic ONNX tokens floor; use a lower threshold for it.
        const val MIN_OMNILINGUAL_TOKENS_BYTES = 50_000L
        // Dolphin attention encoders/decoders are tens to hundreds of MB fp16; this
        // floor rejects truncated downloads the ORT engine would otherwise fail on.
        const val MIN_ATTN_ENC_DEC_BYTES = 1_000_000L
        // units.txt (the id→symbol vocab) is ~500 KB.
        const val MIN_ATTN_UNITS_BYTES = 100_000L
        // silero_vad.onnx is ~630 KB; this floor rejects truncated downloads the
        // sherpa VAD would otherwise fail to load (falls back to whole-clip decode).
        const val MIN_VAD_BYTES = 400_000L

        /** The decoder filename for a Dolphin attention model, e.g. `<id>-decoder.onnx`. */
        fun dolphinAttnDecoderName(entry: CatalogEntry): String = "${entry.model.id}-decoder.onnx"

        /** The shared units.txt vocab filename for all Dolphin attention models. */
        fun dolphinAttnUnitsName(): String = "dolphin-attn-units.txt"

        /** The shared Silero VAD model filename (one file, used by every Dolphin attention model). */
        fun vadFileName(): String = "silero_vad.onnx"
    }

    /**
     * Combines progress across the several files that make up one model download
     * (e.g. encoder + decoder + tokens + VAD). The total is the sum of every
     * file's Content-Length, registered as each connection response arrives, and
     * each emitted fraction is the cumulative bytes written so far divided by
     * that total. This keeps the percentage honest: it never reaches 100% while
     * a sibling (the decoder) is still streaming.
     */
    private class AggregateProgress(private val onProgress: (Float) -> Unit) {
        private val totalBytes = AtomicLong(0)
        private val downloadedBytes = AtomicLong(0)
        private var lastPercent = -1

        /** Register the size of a file that is about to be downloaded. */
        fun addFile(bytes: Long) {
            if (bytes > 0) totalBytes.addAndGet(bytes)
        }

        /** Report that [bytes] more bytes were written; emits a throttled combined fraction. */
        fun addDownloaded(bytes: Long) {
            if (bytes <= 0) return
            val total = totalBytes.get()
            if (total <= 0) return
            val fraction = downloadedBytes.addAndGet(bytes).toFloat() / total
            val percent = (fraction * 100).toInt().coerceIn(0, 100)
            if (percent != lastPercent) {
                lastPercent = percent
                onProgress(fraction.coerceAtMost(1f))
            }
        }

        /** Force the final 100% once every file in the set is done. */
        fun finish() = onProgress(1f)
    }

    /**
     * Download [entry] into [baseDir] for [format], calling [onProgress] with a
     * fraction 0f..1f. If the target file(s) already exist, returns Success without
     * re-downloading. Dolphin CTC and Dolphin attention entries are routed to their
     * own download paths regardless of [format].
     */
    suspend fun download(
        entry: CatalogEntry,
        format: ModelFormat,
        onProgress: (Float) -> Unit,
    ): Result = when {
        ModelCatalog.isDolphinCtc(entry) -> downloadDolphinCtc(entry, onProgress)
        ModelCatalog.isOmnilingual(entry) -> downloadOmnilingual(entry, onProgress)
        ModelCatalog.isDolphinAttn(entry) -> downloadDolphinAttn(entry, onProgress)
        format == ModelFormat.GGML -> downloadGGML(entry, onProgress)
        else -> downloadONNX(entry, onProgress)
    }

    /** Whether [entry] already has a downloaded file for [format]. */
    fun isDownloaded(entry: CatalogEntry, format: ModelFormat): Boolean = when {
        ModelCatalog.isDolphinCtc(entry) -> dolphinComplete(entry)
        ModelCatalog.isOmnilingual(entry) -> omnilingualComplete(entry)
        ModelCatalog.isDolphinAttn(entry) -> dolphinAttnComplete(entry)
        format == ModelFormat.GGML -> java.io.File(baseDir, entry.model.fileName).exists()
        else -> onnxComplete(entry)
    }

    /**
     * True when both Dolphin CTC files (model.onnx + tokens.txt) plus the shared
     * silero_vad.onnx exist with plausible sizes. Guards against stale/truncated
     * files that would otherwise load as "Ready" and crash sherpa-onnx at
     * transcription time. VAD absence only downgrades to whole-clip decode, but
     * keeping it in the completeness gate drives the download so streaming works.
     */
    fun dolphinComplete(entry: CatalogEntry): Boolean {
        val model = java.io.File(baseDir, entry.model.fileName)
        val tokens = java.io.File(baseDir, DolphinCtcEngine.dolphinTokensName(entry))
        val vad = java.io.File(baseDir, vadFileName())
        return model.exists() && model.length() >= MIN_DOLPHIN_MODEL_BYTES &&
            tokens.exists() && tokens.length() >= MIN_ONNX_TOKENS_BYTES &&
            vad.exists() && vad.length() >= MIN_VAD_BYTES
    }

    /**
     * True when the Omnilingual files (model.int8.onnx + tokens.txt) plus the shared
     * silero_vad.onnx exist with plausible sizes. Same shape as a Dolphin CTC model
     * (single model + tokens); VAD is gated so streaming works.
     */
    fun omnilingualComplete(entry: CatalogEntry): Boolean {
        val model = java.io.File(baseDir, entry.model.fileName)
        val tokens = java.io.File(baseDir, OmnilingualEngine.omnilingualTokensName(entry))
        val vad = java.io.File(baseDir, vadFileName())
        return model.exists() && model.length() >= MIN_DOLPHIN_MODEL_BYTES &&
            tokens.exists() && tokens.length() >= MIN_OMNILINGUAL_TOKENS_BYTES &&
            vad.exists() && vad.length() >= MIN_VAD_BYTES
    }

    /**
     * True when all four Dolphin attention files (encoder.onnx, decoder.onnx, the
     * shared units.txt and the shared silero_vad.onnx) exist with plausible sizes, plus
     * the encoder and decoder actually differ (downloading a decoder into the encoder
     * slot would pass the size floor but crash the ORT session). Guards against
     * stale/truncated files that would otherwise load as "Ready" and crash the engine
     * at transcription time. VAD absence only downgrades to whole-clip decode, but
     * keeping it in the completeness gate drives the download so the feature works.
     */
    fun dolphinAttnComplete(entry: CatalogEntry): Boolean {
        val encoder = java.io.File(baseDir, dolphinAttnEncoderName(entry))
        val decoder = java.io.File(baseDir, dolphinAttnDecoderName(entry))
        val units = java.io.File(baseDir, dolphinAttnUnitsName())
        val vad = java.io.File(baseDir, vadFileName())
        return encoder.exists() && encoder.length() >= MIN_ATTN_ENC_DEC_BYTES &&
            decoder.exists() && decoder.length() >= MIN_ATTN_ENC_DEC_BYTES &&
            units.exists() && units.length() >= MIN_ATTN_UNITS_BYTES &&
            vad.exists() && vad.length() >= MIN_VAD_BYTES &&
            encoder.length() != decoder.length()
    }

    /**
     * True when all three ONNX files (encoder, decoder, tokens) plus the shared
     * silero_vad.onnx exist with plausible sizes. Guards against stale/truncated
     * files that would otherwise load as "Ready" and crash sherpa-onnx at
     * transcription time. VAD absence only downgrades to whole-clip decode, but
     * keeping it in the completeness gate drives the download so streaming works.
     */
    fun onnxComplete(entry: CatalogEntry): Boolean {
        val encoder = java.io.File(baseDir, onnxEncoderName(entry))
        val decoder = java.io.File(baseDir, onnxDecoderName(entry))
        val tokens = java.io.File(baseDir, onnxTokensName(entry))
        val vad = java.io.File(baseDir, vadFileName())
        return encoder.exists() && encoder.length() >= MIN_ONNX_ENCODER_BYTES &&
            decoder.exists() && decoder.length() >= MIN_ONNX_DECODER_BYTES &&
            tokens.exists() && tokens.length() >= MIN_ONNX_TOKENS_BYTES &&
            vad.exists() && vad.length() >= MIN_VAD_BYTES
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

    /** The encoder filename for a Dolphin attention model, e.g. `<id>-encoder.onnx`. */
    fun dolphinAttnEncoderName(entry: CatalogEntry): String = entry.model.fileName

    /** The decoder sibling URL for a Dolphin attention encoder URL (same dir, `decoder.onnx`). */
    fun dolphinAttnDecoderUrl(encoderUrl: String): String =
        encoderUrl.substringBefore("/encoder.onnx") + "/decoder.onnx"

    /** The shared units.txt URL (the parent-of-variant `dolphin-attn/` dir, or the variant dir itself). */
    fun dolphinAttnUnitsUrl(encoderUrl: String): String =
        encoderUrl.substringBeforeLast('/').substringBeforeLast('/') + "/units.txt"

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

        val progress = AggregateProgress(onProgress)
        // The sherpa-onnx pre-exported model ships encoder + decoder + tokens as
        // three sibling files. The encoder URL points at `*-encoder.onnx`; the
        // decoder and tokens come from sibling URLs derived by swapping the suffix.
        val encoderTarget = java.io.File(baseDir, onnxEncoderName(entry))
        if (encoderTarget.exists()) {
            // Encoder already downloaded; still fill any missing siblings, reporting
            // their progress so the bar doesn't sit frozen while they stream.
            ensureSiblings(entry, url, progress)
            progress.finish()
            return Result.Success(encoderTarget.length())
        }

        val encoderTmp = java.io.File(baseDir, "${onnxEncoderName(entry)}.part")
        return coroutineScope {
            // Stream the decoder/tokens/VAD siblings concurrently with the encoder:
            // every Content-Length is registered up front, so the aggregate
            // percentage advances monotonically instead of resetting when the
            // decoder joins after the encoder finishes.
            val siblingsDeferred = async { ensureSiblings(entry, url, progress) }
            val result = try {
                val connection = URL(url).openConnection() as HttpURLConnection
                try {
                    connection.connectTimeout = 15_000
                    connection.readTimeout = 30_000
                    connection.connect()
                    val code = connection.responseCode
                    if (code != HttpURLConnection.HTTP_OK) {
                        httpFailure(code, url)
                    } else {
                        progress.addFile(connection.contentLengthLong)
                        var downloaded = 0L
                        encoderTarget.parentFile?.mkdirs()

                        connection.inputStream.use { input ->
                            encoderTmp.outputStream().use { out ->
                                val buf = ByteArray(DEFAULT_BUFFER_SIZE)
                                while (true) {
                                    val read = input.read(buf)
                                    if (read == -1) break
                                    out.write(buf, 0, read)
                                    downloaded += read.toLong()
                                    progress.addDownloaded(read.toLong())
                                }
                            }
                        }

                        if (encoderTmp.renameTo(encoderTarget)) {
                            encoderTmp.delete()
                            Result.Success(downloaded)
                        } else {
                            encoderTmp.delete()
                            Result.Failure("rename failed")
                        }
                    }
                } finally {
                    connection.disconnect()
                }
            } catch (e: Exception) {
                encoderTmp.delete()
                Result.Failure(e.message ?: e.toString())
            }
            siblingsDeferred.await()
            progress.finish()
            result
        }
    }

    /**
     * Download a Dolphin CTC model: its single `model.onnx`/`model.int8.onnx`
     * (from [CatalogEntry.onnxSourceUrl]) plus the sibling `tokens.txt`, both
     * renamed to identity-unique files (`<id>-model.onnx` + `<id>-tokens.txt`).
     * Loading without either file crashes sherpa-onnx, so a missing sibling is
     * reported as a failure rather than silently ignored.
     */
    private suspend fun downloadDolphinCtc(
        entry: CatalogEntry,
        onProgress: (Float) -> Unit,
    ): Result {
        val url = entry.onnxSourceUrl
            ?: return Result.Failure("No ONNX URL for model ${entry.model.id}")
        return downloadSingleModelWithTokens(
            entry, url,
            tokensName = DolphinCtcEngine.dolphinTokensName(entry),
            label = "Dolphin CTC",
            onProgress = onProgress,
            minTokensBytes = MIN_ONNX_TOKENS_BYTES,
        )
    }

    /**
     * Download an Omnilingual model: its single `model.int8.onnx` (from
     * [CatalogEntry.onnxSourceUrl]) plus the sibling `tokens.txt`, renamed to
     * `<id>-model.onnx` + `<id>-tokens.txt`. Same shape as Dolphin CTC.
     */
    private suspend fun downloadOmnilingual(
        entry: CatalogEntry,
        onProgress: (Float) -> Unit,
    ): Result {
        val url = entry.onnxSourceUrl
            ?: return Result.Failure("No ONNX URL for model ${entry.model.id}")
        return downloadSingleModelWithTokens(
            entry, url,
            tokensName = OmnilingualEngine.omnilingualTokensName(entry),
            label = "Omnilingual",
            onProgress = onProgress,
            minTokensBytes = MIN_OMNILINGUAL_TOKENS_BYTES,
        )
    }

    /**
     * Download a single-model + sibling-tokens catalog entry (Dolphin CTC,
     * Omnilingual): the model file from [modelUrl] renamed to `<id>-model.onnx`,
     * then the then-sibling `tokens.txt` derived by URL substitution.
     */
    private suspend fun downloadSingleModelWithTokens(
        entry: CatalogEntry,
        modelUrl: String,
        tokensName: String,
        label: String,
        onProgress: (Float) -> Unit,
        minTokensBytes: Long = MIN_ONNX_TOKENS_BYTES,
    ): Result {
        val progress = AggregateProgress(onProgress)
        val modelTarget = java.io.File(baseDir, entry.model.fileName)
        if (modelTarget.exists() && modelTarget.length() >= MIN_DOLPHIN_MODEL_BYTES) {
            ensureTokens(entry, modelUrl, tokensName, minTokensBytes, progress)
            ensureDolphinAttnSibling(ModelCatalog.SHERPA_SILERO_VAD, vadFileName(), MIN_VAD_BYTES, progress)
            progress.finish()
            return Result.Success(modelTarget.length())
        }

        val modelTmp = java.io.File(baseDir, "${entry.model.fileName}.part")
        return coroutineScope {
            // Fetch the tokens + VAD siblings concurrently with the model file so
            // every Content-Length registers up front and the percentage stays
            // monotonic across the whole download.
            val siblingsDeferred = async {
                ensureTokens(entry, modelUrl, tokensName, minTokensBytes, progress)
                ensureDolphinAttnSibling(ModelCatalog.SHERPA_SILERO_VAD, vadFileName(), MIN_VAD_BYTES, progress)
            }
            val result = try {
                val connection = URL(modelUrl).openConnection() as HttpURLConnection
                try {
                    connection.connectTimeout = 15_000
                    connection.readTimeout = 30_000
                    connection.connect()
                    val code = connection.responseCode
                    if (code != HttpURLConnection.HTTP_OK) {
                        httpFailure(code, modelUrl)
                    } else {
                        progress.addFile(connection.contentLengthLong)
                        var downloaded = 0L
                        modelTarget.parentFile?.mkdirs()

                        connection.inputStream.use { input ->
                            modelTmp.outputStream().use { out ->
                                val buf = ByteArray(DEFAULT_BUFFER_SIZE)
                                while (true) {
                                    val read = input.read(buf)
                                    if (read == -1) break
                                    out.write(buf, 0, read)
                                    downloaded += read.toLong()
                                    progress.addDownloaded(read.toLong())
                                }
                            }
                        }

                        if (modelTmp.renameTo(modelTarget)) {
                            modelTmp.delete()
                            Result.Success(downloaded)
                        } else {
                            modelTmp.delete()
                            Result.Failure("rename failed")
                        }
                    }
                } finally {
                    connection.disconnect()
                }
            } catch (e: Exception) {
                modelTmp.delete()
                Result.Failure(e.message ?: e.toString())
            }
            siblingsDeferred.await()
            progress.finish()
            result
        }
    }

    /**
     * Download a Dolphin attention model: its encoder.onnx (from
     * [CatalogEntry.onnxSourceUrl], renamed to `<id>-encoder.onnx`), the graph-surgeried
     * `decoder.onnx` sibling, and the shared `units.txt` vocab. Loading without any file
     * crashes the ORT engine, so a missing sibling is reported rather than ignored.
     */
    private suspend fun downloadDolphinAttn(
        entry: CatalogEntry,
        onProgress: (Float) -> Unit,
    ): Result {
        val url = entry.onnxSourceUrl
            ?: return Result.Failure("No ONNX URL for model ${entry.model.id}")
        val progress = AggregateProgress(onProgress)
        val encoderTarget = java.io.File(baseDir, dolphinAttnEncoderName(entry))
        if (encoderTarget.exists() && encoderTarget.length() >= MIN_ATTN_ENC_DEC_BYTES) {
            ensureDolphinAttnSiblings(entry, url, progress)
            progress.finish()
            return Result.Success(encoderTarget.length())
        }

        val encoderTmp = java.io.File(baseDir, "${dolphinAttnEncoderName(entry)}.part")
        return coroutineScope {
            // Stream the decoder/units/VAD siblings concurrently with the encoder so
            // the aggregate percentage stays monotonic across the whole download.
            val siblingsDeferred = async { ensureDolphinAttnSiblings(entry, url, progress) }
            val result = try {
                val connection = URL(url).openConnection() as HttpURLConnection
                try {
                    connection.connectTimeout = 15_000
                    connection.readTimeout = 30_000
                    connection.connect()
                    val code = connection.responseCode
                    if (code != HttpURLConnection.HTTP_OK) {
                        httpFailure(code, url)
                    } else {
                        progress.addFile(connection.contentLengthLong)
                        var downloaded = 0L
                        encoderTarget.parentFile?.mkdirs()

                        connection.inputStream.use { input ->
                            encoderTmp.outputStream().use { out ->
                                val buf = ByteArray(DEFAULT_BUFFER_SIZE)
                                while (true) {
                                    val read = input.read(buf)
                                    if (read == -1) break
                                    out.write(buf, 0, read)
                                    downloaded += read.toLong()
                                    progress.addDownloaded(read.toLong())
                                }
                            }
                        }

                        if (encoderTmp.renameTo(encoderTarget)) {
                            encoderTmp.delete()
                            Result.Success(downloaded)
                        } else {
                            encoderTmp.delete()
                            Result.Failure("rename failed")
                        }
                    }
                } finally {
                    connection.disconnect()
                }
            } catch (e: Exception) {
                encoderTmp.delete()
                Result.Failure(e.message ?: e.toString())
            }
            siblingsDeferred.await()
            progress.finish()
            result
        }
    }

    /** Ensure the decoder.onnx + units.txt (+ shared silero_vad.onnx) siblings exist. */
    private suspend fun ensureDolphinAttnSiblings(
        entry: CatalogEntry,
        encoderUrl: String,
        progress: AggregateProgress,
    ) {
        val (decoderGot, unitsGot, vadGot) = coroutineScope {
            val decoderDeferred = async { ensureDolphinAttnSibling(dolphinAttnDecoderUrl(encoderUrl), dolphinAttnDecoderName(entry), MIN_ATTN_ENC_DEC_BYTES, progress) }
            val unitsDeferred = async { ensureDolphinAttnSibling(dolphinAttnUnitsUrl(encoderUrl), dolphinAttnUnitsName(), MIN_ATTN_UNITS_BYTES, progress) }
            val vadDeferred = async { ensureDolphinAttnSibling(ModelCatalog.SHERPA_SILERO_VAD, vadFileName(), MIN_VAD_BYTES, progress) }
            Triple(decoderDeferred.await(), unitsDeferred.await(), vadDeferred.await())
        }
        if (!(decoderGot && unitsGot && vadGot)) {
            Log.w(TAG, "Dolphin attention sibling download incomplete for ${entry.model.id}")
        }
    }

    /** Best-effort download of one Dolphin attention sibling, streaming to `.part` + rename. */
    private fun ensureDolphinAttnSibling(
        siblingUrl: String,
        outName: String,
        minBytes: Long,
        progress: AggregateProgress? = null,
    ): Boolean =
        downloadFile(siblingUrl, outName, minBytes, progress, "attention sibling")

    /**
     * Best-effort download of a single-model catalog entry's sibling `tokens.txt`,
     * derived from the model URL by swapping `model.onnx`/`model.int8.onnx` → `tokens.txt`.
     * Returns true if the file is present afterwards (already there or newly fetched).
     */
    private fun ensureTokens(
        entry: CatalogEntry,
        modelUrl: String,
        tokensName: String,
        minTokensBytes: Long = MIN_ONNX_TOKENS_BYTES,
        progress: AggregateProgress? = null,
    ): Boolean {
        val int8 = modelUrl.endsWith("model.int8.onnx")
        val base = if (int8) modelUrl.substringBefore("model.int8.onnx")
            else modelUrl.substringBefore("model.onnx")
        return downloadFile(base + "tokens.txt", tokensName, minTokensBytes, progress, "tokens")
    }

    /**
     * Ensure the decoder and tokens siblings exist for [entry]'s ONNX model, plus the
     * shared silero_vad.onnx (required for streaming). Both ONNX siblings are required
     * by sherpa-onnx (loading without them fails), so a missing sibling is reported as
     * a failure rather than silently ignored. The siblings are downloaded concurrently
     * to cut total download time.
     */
    private suspend fun ensureSiblings(
        entry: CatalogEntry,
        encoderUrl: String,
        progress: AggregateProgress,
    ) {
        val (decoderGot, tokensGot, vadGot) = coroutineScope {
            val decoderDeferred = async { ensureSibling(encoderUrl, "decoder", onnxDecoderName(entry), progress) }
            val tokensDeferred = async { ensureSibling(encoderUrl, "tokens", onnxTokensName(entry), progress) }
            val vadDeferred = async { ensureDolphinAttnSibling(ModelCatalog.SHERPA_SILERO_VAD, vadFileName(), MIN_VAD_BYTES, progress) }
            Triple(decoderDeferred.await(), tokensDeferred.await(), vadDeferred.await())
        }
        if (!(decoderGot && tokensGot && vadGot)) {
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
        progress: AggregateProgress? = null,
    ): Boolean {
        val int8 = encoderUrl.contains("-encoder.int8.onnx")
        val base = if (int8) encoderUrl.substringBefore("-encoder.int8.onnx")
            else encoderUrl.substringBefore("-encoder.onnx")
        val siblingUrl = when (target) {
            "decoder" -> base + if (int8) "-decoder.int8.onnx" else "-decoder.onnx"
            else -> base + "-tokens.txt"
        }
        val minBytes = if (target == "tokens") MIN_ONNX_TOKENS_BYTES else MIN_ONNX_DECODER_BYTES
        return downloadFile(siblingUrl, outName, minBytes, progress, "sibling")
    }

    /**
     * Stream one remote file into [baseDir]/[outName] (via a `.part` temp + rename),
     * feeding its bytes into [progress] so a multi-file download reports overall
     * progress. Returns true if the target exists with at least [minBytes] afterwards
     * (skips the fetch if it already does). [logTag] names the file in log messages.
     */
    private fun downloadFile(
        url: String,
        outName: String,
        minBytes: Long,
        progress: AggregateProgress?,
        logTag: String,
    ): Boolean {
        val targetFile = java.io.File(baseDir, outName)
        if (targetFile.exists() && targetFile.length() >= minBytes) return true
        if (targetFile.exists()) {
            Log.w(TAG, "discarding undersized $outName (${targetFile.length()} bytes)")
            targetFile.delete()
        }
        val tmp = java.io.File(baseDir, "$outName.part")
        try {
            val connection = URL(url).openConnection() as HttpURLConnection
            var ok = false
            try {
                connection.connectTimeout = 15_000
                connection.readTimeout = 30_000
                connection.connect()
                val code = connection.responseCode
                if (code != HttpURLConnection.HTTP_OK) return false
                progress?.addFile(connection.contentLengthLong)
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
                            progress?.addDownloaded(read.toLong())
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
            Log.w(TAG, "$logTag download failed for $outName from $url: ${e.message}")
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