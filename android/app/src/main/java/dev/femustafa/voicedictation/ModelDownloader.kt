package dev.femustafa.voicedictation

import java.net.HttpURLConnection
import java.net.URL

/**
 * Downloads a [CatalogEntry]'s model file from its [CatalogEntry.sourceUrl] into the
 * app's private internal storage (so it needs no storage permission and satisfies
 * the ticket-08 finding that Models must live in app-private storage).
 *
 * Uses HttpURLConnection (no extra dependency) and reports progress as a fraction
 * 0f..1f. No size gating, per spec. Not unit-tested (network); verified on-device.
 */
class ModelDownloader(
    private val baseDir: java.io.File,
) {
    sealed interface Result {
        data class Success(val bytes: Long) : Result
        data class Failure(val message: String) : Result
    }

    /**
     * Download [entry] into [baseDir] as [entry.model.fileName], calling
     * [onProgress] with a fraction 0f..1f. Returns whether the download succeeded.
     */
    suspend fun download(
        entry: CatalogEntry,
        onProgress: (Float) -> Unit,
    ): Result {
        val url = requireNotNull(entry.sourceUrl) {
            "Model ${entry.model.id} is not publicly hosted (local conversion)"
        }
        val target = java.io.File(baseDir, entry.model.fileName)
        val tmp = java.io.File(baseDir, "${entry.model.fileName}.part")

        val connection = URL(url).openConnection() as HttpURLConnection
        try {
            connection.connectTimeout = 15_000
            connection.readTimeout = 30_000
            connection.connect()
            val code = connection.responseCode
            if (code != HttpURLConnection.HTTP_OK) {
                return Result.Failure("HTTP $code")
            }

            val total = connection.contentLengthLong
            var downloaded = 0L
            target.parentFile?.mkdirs()

            tmp.outputStream().use { out ->
                connection.inputStream.use { input ->
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
        } catch (e: Exception) {
            tmp.delete()
            return Result.Failure(e.message ?: e.toString())
        } finally {
            connection.disconnect()
        }
    }
}
