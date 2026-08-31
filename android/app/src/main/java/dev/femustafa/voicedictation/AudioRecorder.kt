package dev.femustafa.voicedictation

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.File
import java.io.FileOutputStream

/**
 * Captures the mic to a 16 kHz mono 16-bit PCM WAV file using [AudioRecord].
 *
 * No on-device ffmpeg: we capture at the same 16 kHz mono PCM the Termux prototype
 * produced, and the whisper AAR decodes/resamples as needed (spec / Phase-1 step).
 *
 * This is a hardware seam — verified on-device, not unit-tested. `read()` is
 * blocking, so run [start]'s capture on a background coroutine.
 */
class AudioRecorder {
    private var recorder: AudioRecord? = null
    private var capture: Thread? = null
    private var output: File? = null
    private var recording = false

    val isRecording: Boolean
        get() = recording

    /**
     * Start recording [destination]. Permission must already be granted. Captures on
     * a background thread so the caller isn't blocked by the read loop.
     */
    fun start(destination: File) {
        if (recording) return
        val sampleRate = 16000
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
        )
        val record = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBuf.coerceAtLeast(4096),
        )

        if (record.state != AudioRecord.STATE_INITIALIZED) {
            record.release()
            error("AudioRecord failed to initialize")
        }

        record.startRecording()
        recording = true
        this.recorder = record
        this.output = destination

        capture = Thread {
            writeWavHeader(destination)
            try {
                val buffer = ShortArray(minBuf / 2)
                val outputStream = FileOutputStream(destination, true)
                while (recording) {
                    val read = record.read(buffer, 0, buffer.size)
                    if (read > 0) {
                        // Write raw little-endian PCM samples.
                        val bytes = ByteArray(read * 2)
                        for (i in 0 until read) {
                            bytes[i * 2] = (buffer[i].toInt() and 0xFF).toByte()
                            bytes[i * 2 + 1] = ((buffer[i].toInt() shr 8) and 0xFF).toByte()
                        }
                        outputStream.write(bytes)
                    }
                }
                outputStream.flush()
                outputStream.close()
            } finally {
                record.stop()
                record.release()
                this.recorder = null
                finalizeWavHeader(destination)
            }
        }.apply { isDaemon = true; start() }
    }

    /** Stop capture and finalize the WAV file (header length + data size). */
    fun stop() {
        if (!recording) return
        recording = false
        capture?.join(2000)
        capture = null
    }

    private fun writeWavHeader(file: File) {
        FileOutputStream(file).use { out ->
            // Same header shape whisper.cpp expects: PCM 16-bit mono.
            writeHeader(out, 0)
        }
    }

    private fun finalizeWavHeader(file: File) {
        val dataSize = file.length() - 44
        java.io.RandomAccessFile(file, "rw").use { raf ->
            raf.seek(4); raf.writeInt(36 + dataSize.toInt())
            raf.seek(40); raf.writeInt(dataSize.toInt())
        }
    }

    private fun writeHeader(out: FileOutputStream, dataSize: Int) {
        val bytes = ByteArray(44)
        // "RIFF"
        bytes[0]='R'.code.toByte(); bytes[1]='I'.code.toByte(); bytes[2]='F'.code.toByte(); bytes[3]='F'.code.toByte()
        writeInt32Le(bytes, 4, 36 + dataSize)
        // "WAVE"
        bytes[8]='W'.code.toByte(); bytes[9]='A'.code.toByte(); bytes[10]='V'.code.toByte(); bytes[11]='E'.code.toByte()
        // "fmt "
        bytes[12]='f'.code.toByte(); bytes[13]='m'.code.toByte(); bytes[14]='t'.code.toByte(); bytes[15]=' '.code.toByte()
        writeInt32Le(bytes, 16, 16)              // subchunk1 size
        writeInt16Le(bytes, 20, 1)               // PCM
        writeInt16Le(bytes, 22, 1)               // mono
        writeInt32Le(bytes, 24, 16000)           // sample rate
        writeInt32Le(bytes, 28, 16000 * 2)       // byte rate
        writeInt16Le(bytes, 32, 2)               // block align
        writeInt16Le(bytes, 34, 16)              // bits per sample
        // "data"
        bytes[36]='d'.code.toByte(); bytes[37]='a'.code.toByte(); bytes[38]='t'.code.toByte(); bytes[39]='a'.code.toByte()
        writeInt32Le(bytes, 40, dataSize)
        out.write(bytes, 0, 44)
    }

    private fun writeInt16Le(b: ByteArray, off: Int, v: Int) {
        b[off] = (v and 0xFF).toByte(); b[off + 1] = ((v shr 8) and 0xFF).toByte()
    }

    private fun writeInt32Le(b: ByteArray, off: Int, v: Int) {
        b[off] = (v and 0xFF).toByte(); b[off + 1] = ((v shr 8) and 0xFF).toByte()
        b[off + 2] = ((v shr 16) and 0xFF).toByte(); b[off + 3] = ((v shr 24) and 0xFF).toByte()
    }
}
