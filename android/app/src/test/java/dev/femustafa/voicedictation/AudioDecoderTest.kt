package dev.femustafa.voicedictation

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.io.FileOutputStream

/**
 * JVM tests for [AudioDecoder]: 16-bit PCM WAV (mono/stereo, any sample rate)
 * and MP3 (via the bundled JLayer decoder) are normalized to the canonical
 * 16 kHz mono PCM format, and unknown formats are rejected.
 */
class AudioDecoderTest {

    @Test
    fun monoWavDecodesAtNativeRate() {
        val src = temp("mono.wav")
        // 8 kHz mono: 4 samples with known values.
        writeWav(src, 8000, 1, shortArrayOf(0, 16384, -16384, 0))
        val decoded = AudioDecoder.decodeSamples(src.absolutePath)
        assertEquals(8000, decoded.sampleRate)
        assertEquals(4, decoded.samples.size)
        assertEquals(0f, decoded.samples[0], 1e-3f)
        assertEquals(0.5f, decoded.samples[1], 1e-3f)
        assertEquals(-0.5f, decoded.samples[2], 1e-3f)
    }

    @Test
    fun stereoWavIsDownmixedToMonoAverage() {
        val src = temp("stereo.wav")
        // Left channel 0.25, right channel 0.75 at frame 0 -> mono 0.5.
        writeWav(src, 44100, 2, shortArrayOf(8192, 24576, -8192, 8192))
        val decoded = AudioDecoder.decodeSamples(src.absolutePath)
        assertEquals(44100, decoded.sampleRate)
        assertEquals(2, decoded.samples.size)
        assertEquals(0.5f, decoded.samples[0], 1e-3f)
        assertEquals(0f, decoded.samples[1], 1e-3f)
    }

    @Test
    fun outputWavIsResampledTo16k() {
        val src = temp("rate44k.wav")
        val n = 44100 * 2 // 2 seconds at 44.1 kHz
        val samples = ShortArray(n)
        for (i in samples.indices) samples[i] = (i % 1000).toShort()
        writeWav(src, 44100, 1, samples)

        val out = temp("out16k.wav")
        AudioDecoder.decodeToMono16kWav(src.absolutePath, out.absolutePath)

        val decoded = AudioDecoder.decodeSamples(out.absolutePath)
        assertEquals(AudioDecoder.TARGET_SAMPLE_RATE, decoded.sampleRate)
        // ~2 seconds of audio at 16 kHz.
        assertTrue(decoded.samples.size in 31000..33000)
        // First and last samples are preserved (identity at the ends of the range).
        assertEquals(samples[0] / 32768f, decoded.samples[0], 0.01f)
        assertEquals(samples[n - 1] / 32768f, decoded.samples.last(), 0.01f)
    }

    @Test
    fun mp3FixtureIsDecoded() {
        val fixture = loadFixture("mp3_fixture.mp3")
        val decoded = AudioDecoder.decodeSamples(fixture.absolutePath)
        assertEquals(AudioDecoder.TARGET_SAMPLE_RATE, decoded.sampleRate)
        assertTrue("expected audible samples", decoded.samples.size > 1000)
        assertTrue(decoded.samples.any { kotlin.math.abs(it) > 0.01f })
    }

    @Test
    fun unknownFormatIsRejected() {
        val src = temp("bogus.txt")
        src.writeText("this is not audio")
        assertThrows(AudioDecoder.UnsupportedAudioFormatException::class.java) {
            AudioDecoder.decodeToMono16kWav(src.absolutePath, temp("out.wav").absolutePath)
        }
    }

    @Test
    fun non16BitWavIsRejected() {
        val src = temp("8bit.wav")
        // 8-bit PCM is not supported by the decoder.
        writeWav(src, 16000, 1, shortArrayOf(0, 1, 2, 3), bitsPerSample = 8)
        assertThrows(AudioDecoder.UnsupportedAudioFormatException::class.java) {
            AudioDecoder.decodeSamples(src.absolutePath)
        }
    }

    // ---- helpers ----

    private fun temp(name: String): File {
        val f = File(System.getProperty("java.io.tmpdir"), "audiodecoder-test-$name")
        f.deleteOnExit()
        return f
    }

    private fun loadFixture(name: String): File {
        val bytes = javaClass.getResourceAsStream("/$name")?.readBytes()
            ?: error("missing test resource: $name")
        return temp(name).also { it.writeBytes(bytes) }
    }

    /** Write a minimal PCM WAV. [bitsPerSample] must be 8 or 16. */
    private fun writeWav(file: File, sampleRate: Int, channels: Int, samples: ShortArray, bitsPerSample: Int = 16) {
        val bytesPerSample = bitsPerSample / 8
        val dataSize = samples.size * bytesPerSample
        FileOutputStream(file).use { out ->
            val h = ByteArray(44)
            h[0] = 'R'.code.toByte(); h[1] = 'I'.code.toByte(); h[2] = 'F'.code.toByte(); h[3] = 'F'.code.toByte()
            putInt(h, 4, 36 + dataSize)
            h[8] = 'W'.code.toByte(); h[9] = 'A'.code.toByte(); h[10] = 'V'.code.toByte(); h[11] = 'E'.code.toByte()
            h[12] = 'f'.code.toByte(); h[13] = 'm'.code.toByte(); h[14] = 't'.code.toByte(); h[15] = ' '.code.toByte()
            putInt(h, 16, 16)
            putShort(h, 20, 1.toShort())           // PCM
            putShort(h, 22, channels.toShort())
            putInt(h, 24, sampleRate)
            putInt(h, 28, sampleRate * channels * bytesPerSample)
            putShort(h, 32, (channels * bytesPerSample).toShort())
            putShort(h, 34, bitsPerSample.toShort())
            h[36] = 'd'.code.toByte(); h[37] = 'a'.code.toByte(); h[38] = 't'.code.toByte(); h[39] = 'a'.code.toByte()
            putInt(h, 40, dataSize)
            out.write(h)

            if (bitsPerSample == 16) {
                val data = ByteArray(dataSize)
                for (i in samples.indices) {
                    data[i * 2] = (samples[i].toInt() and 0xFF).toByte()
                    data[i * 2 + 1] = ((samples[i].toInt() shr 8) and 0xFF).toByte()
                }
                out.write(data)
            } else {
                val data = ByteArray(dataSize)
                for (i in samples.indices) {
                    data[i] = ((samples[i].toInt() + 32768) shr 8).toByte()
                }
                out.write(data)
            }
        }
    }

    private fun putShort(b: ByteArray, off: Int, v: Short) {
        b[off] = (v.toInt() and 0xFF).toByte(); b[off + 1] = ((v.toInt() shr 8) and 0xFF).toByte()
    }

    private fun putInt(b: ByteArray, off: Int, v: Int) {
        b[off] = (v and 0xFF).toByte(); b[off + 1] = ((v shr 8) and 0xFF).toByte()
        b[off + 2] = ((v shr 16) and 0xFF).toByte(); b[off + 3] = ((v shr 24) and 0xFF).toByte()
    }
}
