package dev.femustafa.voicedictation

import javazoom.jl.decoder.Bitstream
import javazoom.jl.decoder.Decoder
import javazoom.jl.decoder.Header
import javazoom.jl.decoder.SampleBuffer
import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

/**
 * Decodes a user-picked audio file (WAV or MP3) into the app's canonical
 * transcription format: 16 kHz mono 16-bit PCM WAV — the same format the mic
 * recorder writes ([AudioRecorder]). All engines already read that format, so a
 * picked file can flow through [WhisperManager.transcribe] unchanged.
 *
 * Pure-JVM (no Android framework types), so it is unit-testable on the dev
 * machine. MP3 is decoded with the pure-Java JLayer library
 * (javazoom.jl.decoder), which is why no on-device ffmpeg is required.
 */
object AudioDecoder {

    const val TARGET_SAMPLE_RATE = 16000

    /** Thrown for a file we cannot decode: not a WAV/MP3, or a malformed/unsupported variant. */
    class UnsupportedAudioFormatException(message: String) : java.io.IOException(message)

    /**
     * Decode [srcPath] (WAV or MP3) into a 16 kHz mono 16-bit PCM WAV at
     * [dstPath], returning the absolute path to [dstPath].
     */
    fun decodeToMono16kWav(srcPath: String, dstPath: String): String {
        val decoded = decodeSamples(srcPath)
        val mono16k = resampleToMono16k(decoded.samples, decoded.sampleRate)
        writePcm16Wav(File(dstPath), mono16k, TARGET_SAMPLE_RATE)
        return dstPath
    }

    /** Decode [srcPath] into mono FloatArray samples at their native sample rate. */
    fun decodeSamples(srcPath: String): DecodedAudio =
        when (detectFormat(srcPath)) {
            Format.WAV -> decodeWav(File(srcPath))
            Format.MP3 -> decodeMp3(srcPath)
            Format.UNKNOWN -> throw UnsupportedAudioFormatException(
                "Unsupported audio format — pick a .wav or .mp3 file",
            )
        }

    /** Mono float samples plus the sample rate they were captured at. */
    data class DecodedAudio(val samples: FloatArray, val sampleRate: Int)

    private enum class Format { WAV, MP3, UNKNOWN }

    private fun detectFormat(path: String): Format {
        val f = File(path)
        if (!f.exists() || f.length() < 4) return Format.UNKNOWN
        val n = minOf(12L, f.length()).toInt()
        val head = ByteArray(n)
        DataInputStream(FileInputStream(f)).use { it.readFully(head) }

        // RIFF .... WAVE
        if (head.size >= 12 &&
            head[0] == 'R'.code.toByte() && head[1] == 'I'.code.toByte() &&
            head[2] == 'F'.code.toByte() && head[3] == 'F'.code.toByte() &&
            head[8] == 'W'.code.toByte() && head[9] == 'A'.code.toByte() &&
            head[10] == 'V'.code.toByte() && head[11] == 'E'.code.toByte()
        ) {
            return Format.WAV
        }
        // MP3 with an ID3 tag, or a bare frame sync (0xFF + top-3-bits-set).
        if (head.size >= 3 &&
            head[0] == 'I'.code.toByte() && head[1] == 'D'.code.toByte() && head[2] == '3'.code.toByte()
        ) {
            return Format.MP3
        }
        val b0 = head[0].toInt() and 0xFF
        val b1 = head[1].toInt() and 0xFF
        if (b0 == 0xFF && (b1 and 0xE0) == 0xE0) return Format.MP3
        return Format.UNKNOWN
    }

    /** Parse a 16-bit PCM RIFF/WAVE file into mono float samples at its native rate. */
    private fun decodeWav(file: File): DecodedAudio {
        val len = file.length()
        if (len < 44) throw UnsupportedAudioFormatException("WAV file too small to be valid")
        val bytes = file.readBytes()

        var offset = 12
        var sampleRate = TARGET_SAMPLE_RATE
        var nChannels = 1
        var bitsPerSample = 16
        var dataStart = -1
        var dataLen = 0
        while (offset + 8 <= bytes.size) {
            val fourCC = String(bytes, offset, 4)
            val size = le32(bytes, offset + 4)
            if (fourCC == "fmt ") {
                sampleRate = le32(bytes, offset + 12)
                nChannels = le16(bytes, offset + 10)
                bitsPerSample = le16(bytes, offset + 22)
            } else if (fourCC == "data") {
                // dataLen may be stale; clamp to actual bytes remaining in file.
                dataStart = offset + 8
                val remaining = (bytes.size - dataStart).coerceAtLeast(0)
                dataLen = size.coerceAtMost(remaining)
                break
            }
            offset += 8 + size
        }
        if (dataStart < 0) throw UnsupportedAudioFormatException("WAV has no data chunk")
        if (bitsPerSample != 16) {
            throw UnsupportedAudioFormatException("Only 16-bit PCM WAV files are supported")
        }

        val frameBytes = nChannels * 2
        val frameCount = dataLen / frameBytes
        val mono = FloatArray(frameCount)
        var bi = dataStart
        for (i in 0 until frameCount) {
            mono[i] = if (nChannels >= 2) {
                // Downmix stereo to mono by averaging the channels.
                (sampleAt(bytes, bi) + sampleAt(bytes, bi + 2)) * 0.5f
            } else {
                sampleAt(bytes, bi)
            }
            bi += frameBytes
        }
        return DecodedAudio(mono, sampleRate)
    }

    /** Decode an MP3 file into mono float samples at its native sample rate via JLayer. */
    private fun decodeMp3(path: String): DecodedAudio {
        val input = FileInputStream(path)
        try {
            val bitstream = Bitstream(input)
            val decoder = Decoder()
            var header = bitstream.readFrame()
            if (header == null) throw UnsupportedAudioFormatException("Invalid or empty MP3 file")

            // Prime the output buffer from the first frame's header so interleaved
            // PCM arrives at the file's real sample rate and channel count.
            val sampleRate = header.frequency()
            val channels = if (header.mode() == Header.SINGLE_CHANNEL) 1 else 2
            decoder.setOutputBuffer(SampleBuffer(sampleRate, channels))

            val frames = ArrayList<ShortArray>()
            while (header != null) {
                val output = decoder.decodeFrame(header, bitstream) as SampleBuffer
                val buf = output.getBuffer()
                val n = output.getBufferLength()
                if (n > 0) frames.add(buf.copyOf(n))
                bitstream.closeFrame()
                header = bitstream.readFrame()
            }

            val total = frames.sumOf { it.size }
            val mono = FloatArray(total / channels)
            var i = 0
            for (frame in frames) {
                if (channels >= 2) {
                    var j = 0
                    while (j + 1 < frame.size) {
                        mono[i++] = ((frame[j] + frame[j + 1]) * 0.5f) / 32768f
                        j += 2
                    }
                } else {
                    for (s in frame) mono[i++] = s / 32768f
                }
            }
            return DecodedAudio(mono, sampleRate)
        } catch (e: javazoom.jl.decoder.JavaLayerException) {
            throw java.io.IOException("Failed to decode MP3: ${e.message}", e)
        } finally {
            input.close()
        }
    }

    /** Linearly resample [samples] to [TARGET_SAMPLE_RATE]; identity when already 16 kHz. */
    private fun resampleToMono16k(samples: FloatArray, sampleRate: Int): FloatArray {
        if (sampleRate == TARGET_SAMPLE_RATE || samples.isEmpty()) return samples
        val srcLen = samples.size
        val dstLen = maxOf(1, (srcLen.toDouble() * TARGET_SAMPLE_RATE / sampleRate).toInt())
        val out = FloatArray(dstLen)
        val step = sampleRate.toDouble() / TARGET_SAMPLE_RATE
        for (i in 0 until dstLen) {
            val pos = i * step
            val i0 = pos.toInt().coerceIn(0, srcLen - 1)
            val i1 = (i0 + 1).coerceIn(0, srcLen - 1)
            val frac = (pos - i0).toFloat()
            out[i] = samples[i0] * (1f - frac) + samples[i1] * frac
        }
        return out
    }

    /** Write [samples] as a 16-bit PCM mono WAV at [sampleRate]. */
    private fun writePcm16Wav(dst: File, samples: FloatArray, sampleRate: Int) {
        FileOutputStream(dst).use { out ->
            out.write(wavHeader(sampleRate, samples.size * 2))
            val bytes = ByteArray(samples.size * 2)
            for (i in samples.indices) {
                val v = (samples[i].coerceIn(-1f, 1f) * 32767).toInt()
                bytes[i * 2] = (v and 0xFF).toByte()
                bytes[i * 2 + 1] = ((v shr 8) and 0xFF).toByte()
            }
            out.write(bytes)
        }
    }

    /** 44-byte canonical PCM 16-bit mono WAV header (same shape [AudioRecorder] writes). */
    private fun wavHeader(sampleRate: Int, dataSize: Int): ByteArray {
        val b = ByteArray(44)
        b[0] = 'R'.code.toByte(); b[1] = 'I'.code.toByte(); b[2] = 'F'.code.toByte(); b[3] = 'F'.code.toByte()
        writeInt32Le(b, 4, 36 + dataSize)
        b[8] = 'W'.code.toByte(); b[9] = 'A'.code.toByte(); b[10] = 'V'.code.toByte(); b[11] = 'E'.code.toByte()
        b[12] = 'f'.code.toByte(); b[13] = 'm'.code.toByte(); b[14] = 't'.code.toByte(); b[15] = ' '.code.toByte()
        writeInt32Le(b, 16, 16)                 // fmt chunk size
        writeInt16Le(b, 20, 1)                  // PCM
        writeInt16Le(b, 22, 1)                  // mono
        writeInt32Le(b, 24, sampleRate)
        writeInt32Le(b, 28, sampleRate * 2)     // byte rate
        writeInt16Le(b, 32, 2)                  // block align
        writeInt16Le(b, 34, 16)                 // bits per sample
        b[36] = 'd'.code.toByte(); b[37] = 'a'.code.toByte(); b[38] = 't'.code.toByte(); b[39] = 'a'.code.toByte()
        writeInt32Le(b, 40, dataSize)
        return b
    }

    /** Read the 16-bit little-endian PCM sample starting at [off]. */
    private fun sampleAt(b: ByteArray, off: Int): Float {
        val raw = ((b[off + 1].toInt() and 0xFF) shl 8) or (b[off].toInt() and 0xFF)
        return raw.toShort().toFloat() / 32768f
    }

    private fun le16(b: ByteArray, off: Int): Int =
        (b[off].toInt() and 0xFF) or ((b[off + 1].toInt() and 0xFF) shl 8)

    private fun le32(b: ByteArray, off: Int): Int =
        (b[off].toInt() and 0xFF) or
            ((b[off + 1].toInt() and 0xFF) shl 8) or
            ((b[off + 2].toInt() and 0xFF) shl 16) or
            ((b[off + 3].toInt() and 0xFF) shl 24)

    private fun writeInt16Le(b: ByteArray, off: Int, v: Int) {
        b[off] = (v and 0xFF).toByte(); b[off + 1] = ((v shr 8) and 0xFF).toByte()
    }

    private fun writeInt32Le(b: ByteArray, off: Int, v: Int) {
        b[off] = (v and 0xFF).toByte(); b[off + 1] = ((v shr 8) and 0xFF).toByte()
        b[off + 2] = ((v shr 16) and 0xFF).toByte(); b[off + 3] = ((v shr 24) and 0xFF).toByte()
    }
}
