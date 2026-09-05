package dev.femustafa.voicedictation

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.k2fsa.sherpa.onnx.SileroVadModelConfig
import com.k2fsa.sherpa.onnx.Vad
import com.k2fsa.sherpa.onnx.VadModelConfig
import dev.femustafa.voicedictation.WhisperEngine.*
import java.io.File
import java.io.FileNotFoundException
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.ShortBuffer
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * [WhisperEngine] backed directly by onnxruntime-android for the Dolphin attention
 * ONNX encoder+decoder pairs (`dolphin-attn` catalog; DataoceanAI models exported
 * by DakeQQ).
 *
 * Why a dedicated engine instead of reusing [DolphinCtcEngine]: the CTC family loads
 * through sherpa-onnx as a single greedy-decoded model and **ignores** the passed
 * language (auto-detect only, which leaks into Naagri for Urdu speech). The attention
 * engine decodes with an explicit `ur`/`PK` pin (`language_start`/`language_end`), so
 * it is the one that honors the app's language selection for Urdu.
 *
 * Decode is a KV-cached attention **beam search over the FULL logits output**
 * (`/output_layer/Gemm_output_0`) of the graph-surgeried decoder, ported from the
 * laptop verifier `tools/dolphin-onnx/scripts/verify_dakeqq_beam.py`. The beam MUST
 * use length-normalized selection (ticket 29 finding): the small decoder gives `<eos>`
 * an abnormally high probability right after the prefix, so without length
 * normalization it returns an empty/truncated transcript. Layer count and head/dim
 * are derived from the encoder output at load time (base: 6 layers; small: 12 layers),
 * never hardcoded. The fp16/arm tier keeps the whole KV cache in float16, so the cache
 * dtype is fixed to fp16 (fp32 fp32 tiers would degrade to fp16 here).
 *
 * Vocabulary: `units.txt` id→symbol list; decode is SentencePiece `DecodePieces`
 * (leading `▁` → space, concatenate, trim leading space) without needing `bpe.model`.
 *
 * Longer clips are split into per-utterance segments with a resident Silero VAD
 * (sherpa-onnx's `Vad`, model file `silero_vad.onnx` beside the encoder) and each
 * utterance decoded independently before joining — see [transcribeClip]. If the VAD
 * is unavailable or finds no speech, a single whole-clip decode is used instead.
 */
class DolphinAttnEngine(
    private val context: Context,
) : WhisperEngine {

    /**
     * A beach contains a partial hypothesis: the token ids decoded so far (including
     * the 5-token `<sos><ur><PK><asr><nots>` prefix), its cumulative log-prob, and the
     * fp16 decoder KV cache (state AFTER processing all but the last token — the next
     * decode step consumes exactly one more). Sharing the lists across siblings is fine:
     * they are treated as immutable.
     */
    private class Beam(
        val tokens: IntArray,
        val score: Double,
        val deKeys: List<ShortBuffer>,
        val deValues: List<ShortBuffer>,
    ) {
        /** A new beam extending this one with [token], appending its log-prob and
         * advancing to the decoder KV state returned by the step that produced [token]. */
        fun extend(token: Int, logProb: Double, deKeys: List<ShortBuffer>,
                   deValues: List<ShortBuffer>): Beam =
            Beam(tokens + token, score + logProb, deKeys, deValues)
    }

    /** Everything needed to decode with a loaded model pair. */
    class DolphinAttnModelRef(
        val env: OrtEnvironment,
        val encoder: OrtSession,
        val decoder: OrtSession,
        val tokenList: List<String>,
        val nl: Int,
        val headDim: Int,
        val dModel: Int,
        /** Load-cached constant decoder inputs, reused across every step's run. */
        val langStartTensor: OnnxTensor,
        val langEndTensor: OnnxTensor,
        val maskTensor: OnnxTensor,
        /**
         * Resident Silero VAD (sherpa-onnx), loaded once for [transcribe] to split long
         * clips into per-utterance decodes. Null when the silero_vad.onnx sibling is
         * absent/undersized → whole-clip decode fallback.
         */
        val vad: Vad?,
    ) : WhisperModelRef

    /** A single decoder run: fresh KV state + the raw full-logits row. */
    private class DecodeResult(
        val deKeys: List<ShortBuffer>,
        val deValues: List<ShortBuffer>,
        val fullLogits: FloatArray,
    )

    private val threads: Int
        get() = VoiceDictationApp.from(context).whisperThreads()

    override suspend fun load(modelPath: String): WhisperModelRef {
        // Identify the catalog entry whose encoder file matches the passed path so we
        // can derive the sibling decoder + shared vocab paths.
        val entry = ModelCatalog.dolphinAttnModels.firstOrNull { entry ->
            modelPath.endsWith(entry.model.fileName)
        } ?: throw FileNotFoundException("No Dolphin attention model entry for path: $modelPath")

        val dir = File(modelPath).parentFile
            ?: throw FileNotFoundException("no parent dir for $modelPath")
        val decoderPath = File(dir, ModelDownloader.dolphinAttnDecoderName(entry)).absolutePath
        val unitsPath = File(dir, ModelDownloader.dolphinAttnUnitsName()).absolutePath

        if (!File(decoderPath).exists() || !File(unitsPath).exists()) {
            val missing = buildList {
                if (!File(decoderPath).exists()) add("decoder")
                if (!File(unitsPath).exists()) add("units")
            }.joinToString(", ")
            Log.w(TAG, "Dolphin attention files missing for ${entry.model.id}: $missing")
            throw FileNotFoundException(
                "Dolphin attention files missing for ${entry.model.id} ($missing). " +
                    "Download the model in the Models screen first."
            )
        }

        Log.i(TAG, "loading Dolphin attention model: $modelPath (+ decoder, units)")

        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
        }
        val encoder = env.createSession(modelPath, options)
        val decoder = env.createSession(decoderPath, options)

        // Derived architecture: number of layers = number of `en_key_*` encoder outputs
        // (base 6, small 12); head/dim from the en_key_0 output shape [head, d_model, T].
        val encOutputs = encoder.getOutputInfo()
        val nl = encOutputs.keys.count { it.startsWith("en_key_") }
        val keyShape = (encOutputs.getValue("en_key_0").info as TensorInfo).shape
        val headDim = keyShape[0].toInt()
        val dModel = keyShape[1].toInt()
        Log.i(TAG, "Dolphin attention arch: nl=$nl headDim=$headDim dModel=$dModel")
        Log.i(TAG, "decoder outputs: ${decoder.getOutputInfo().keys.sorted()}")
        Log.i(TAG, "encoder outputs: ${encoder.getOutputInfo().keys.sorted()}")
        Log.i(TAG, "decoder inputs: " + decoder.getInputInfo().map { (name, node) ->
            // TensorInfo holds the declared type + shape of each graph input; log them
            // so a Kotlin-provided shape can be diffed against the model's expectation.
            val ti = (node.info as? ai.onnxruntime.TensorInfo)
            "$name=${ti?.type}:${ti?.shape?.toList()}"
        })

        val unitsFile = File(unitsPath)
        if (!unitsFile.exists() || unitsFile.length() == 0L) {
            throw FileNotFoundException("Dolphin attention vocab missing/empty: $unitsPath")
        }
        val tokenList = unitsFile.readLines().map { it.trim().substringBefore(' ') }

        // Resident Silero VAD: load once so long clips can be split into per-utterance
        // decodes. The model is the shared silero_vad.onnx sibling in the same dir as
        // the encoder; if it's missing/undersized (or fails to initialize) we degrade
        // to whole-clip decode rather than failing the load.
        val vadModelFile = File(dir, ModelDownloader.vadFileName())
        val vad: Vad? = if (vadModelFile.exists() &&
            vadModelFile.length() >= ModelDownloader.MIN_VAD_BYTES
        ) {
            try {
                val silero = SileroVadModelConfig()
                silero.model = vadModelFile.absolutePath
                silero.threshold = 0.5f
                silero.minSilenceDuration = 0.25f
                silero.minSpeechDuration = 0.5f
                silero.windowSize = VAD_WINDOW
                silero.maxSpeechDuration = 5.0f

                val vadConfig = VadModelConfig()
                vadConfig.sileroVadModelConfig = silero
                vadConfig.sampleRate = SAMPLE_RATE
                vadConfig.numThreads = threads
                vadConfig.provider = "cpu"

                Vad(null, vadConfig)
            } catch (e: Exception) {
                Log.w(TAG, "failed to init Silero VAD; whole-clip decode only: ${e.message}")
                null
            }
        } else {
            Log.w(TAG, "Silero VAD model missing/undersized at $vadModelFile; whole-clip decode only")
            null
        }

        return DolphinAttnModelRef(
            env, encoder, decoder, tokenList, nl, headDim, dModel,
            langStartTensor = OnnxTensor.createTensor(
                env, toDirectLongBuffer(intArrayOf(LANG_UR)), longArrayOf(1)
            ),
            langEndTensor = OnnxTensor.createTensor(
                env, toDirectLongBuffer(intArrayOf(REGION_PK)), longArrayOf(1)
            ),
            maskTensor = OnnxTensor.createTensor(
                env, ByteBuffer.allocateDirect(1).order(ByteOrder.nativeOrder()).also { it.put(0, 0) },
                longArrayOf(1), OnnxJavaType.INT8
            ),
            vad = vad,
        )
    }

    override suspend fun transcribe(
        model: WhisperModelRef,
        audioPath: String,
        languageMode: LanguageMode,
        language: String?,
    ): String {
        val ref = model as? DolphinAttnModelRef
            ?: throw IllegalArgumentException("unexpected model handle")
        // Unlike DolphinCtcEngine this engine pins Urdu (ur/PK) explicitly. Only the
        // ur/PK pin is verified (ticket 28/29); a different requested language falls
        // back to it with a warning.
        if (language != null && language != "ur") {
            Log.w(TAG, "language '$language' requested; only ur/PK pin verified, using it")
        }
        return try {
            val audio = readWaveShorts(audioPath)
                ?: throw IOException("Failed to read wave file: $audioPath")
            Log.i(TAG, "transcribing ${audio.size} samples (${audio.size / SAMPLE_RATE.toDouble()} s)")
            transcribeClip(ref, audio)
        } catch (e: Exception) {
            Log.e(TAG, "dolphin attention transcribe failed", e)
            throw e
        }
    }

    override suspend fun createSession(
        model: WhisperModelRef,
        languageMode: LanguageMode,
        language: String?,
    ): TranscriptionSession? {
        val ref = model as? DolphinAttnModelRef ?: return null
        if (language != null && language != "ur") {
            Log.w(TAG, "language '$language' requested; only ur/PK pin verified, using it")
        }
        val waveWriter = InMemoryWaveWriter(16000)
        // Use the resident VAD from the model ref, or create a new one if not available
        val vadLike: VadLike = ref.vad?.let { SherpaVad(it) } ?: run {
            Log.w(TAG, "No resident VAD available; cannot create streaming session")
            return null
        }
        return DolphinAttnSession(this, ref, waveWriter, context, vadLike)
    }

    override fun release(model: WhisperModelRef) {
        val ref = model as? DolphinAttnModelRef ?: return
        try {
            ref.vad?.release()
            ref.langStartTensor.close()
            ref.langEndTensor.close()
            ref.maskTensor.close()
            ref.decoder.close()
            ref.encoder.close()
            Log.i(TAG, "released Dolphin attention model")
        } catch (e: Exception) {
            Log.w(TAG, "error releasing Dolphin attention model: ${e.message}")
        }
    }

    /**
     * Transcribe [audio] with the resident [ref]. When a Silero VAD is loaded it splits
     * the clip into per-utterance segments, each decoded independently and joined — the
     * pattern sherpa-onnx itself uses for Dolphin (VAD → offline decode per utterance),
     * which avoids one monolithic decode of a long clip. Falls back to a single whole-clip
     * decode when VAD is unavailable, finds no speech, or every decoded segment is blank
     * (e.g. segmentation ran on noise) — preserving the pre-VAD behavior and the
     * "No speech detected" surface.
     */
    private fun transcribeClip(ref: DolphinAttnModelRef, audio: ShortArray): String {
        val vad = ref.vad
        if (vad == null) {
            Log.i(TAG, "no Silero VAD loaded; whole-clip decode")
            return decodeTokens(ref, beamSearch(ref, audio))
        }
        val segments = segmentWithVad(vad, audio)
        if (segments.isEmpty()) {
            Log.i(TAG, "Silero VAD found no speech in ${audio.size} samples; whole-clip decode")
            return decodeTokens(ref, beamSearch(ref, audio))
        }
        Log.i(TAG, "Silero VAD split ${audio.size} samples into ${segments.size} utterance(s)")
        val texts = segments.map { decodeTokens(ref, beamSearch(ref, it)) }.filter { it.isNotBlank() }
        if (texts.isEmpty()) {
            Log.w(TAG, "all VAD segments decoded blank; whole-clip decode")
            return decodeTokens(ref, beamSearch(ref, audio))
        }
        return texts.joinToString(" ")
    }

    /**
     * Feed [audio] to the sherpa Silero [vad] and return each utterance it emits as an
     * int16 ShortArray ready for [beamSearch]. Shorts are converted to the normalized
     * float [-1,1] input Silero expects and back to int16 for the fused STFT encoder.
     */
    private fun segmentWithVad(vad: Vad, audio: ShortArray): List<ShortArray> {
        val floats = FloatArray(audio.size) { audio[it] / 32768f }
        val utterances = segmentAudioWithVad(floats, SherpaVad(vad))
        return utterances.map { u ->
            ShortArray(u.samples.size) { (u.samples[it] * 32767f).toInt().toShort() }
        }
    }

    /**
     * Beam search over the decoder's FULL logits output. Faithful port of
     * `verify_dakeqq_beam.py::beam_search` with `use_sliced=False`:
     *
     *  1. Run the encoder (raw int16 in-graph) -> fp16 cross KV.
     *  2. Prefill `<sos><ur><PK><asr><nots>`.
     *  3. Loop: move EOS-terminated beams to `finished`; expand the rest by the top-k
     *     tokens of the full-logits log-softmax; keep the best [BEAM_SIZE] by cumulative
     *     score. Each decode passes exactly one new token with the parent's lazy KV.
     *  4. Pick the finished hypothesis with the best LENGTH-NORMALIZED score
     *     (score / #generated tokens), dropping prefix-only empties.
     */
    internal fun beamSearch(ref: DolphinAttnModelRef, audio: ShortArray): IntArray {
        val env = ref.env
        val audioTensor = OnnxTensor.createTensor(
            env, toDirectShortBuffer(audio), longArrayOf(1, 1, audio.size.toLong()),
            OnnxJavaType.INT16
        )

        // Reusable encoder cross-KV, kept open for the whole decode.
        val encResult = ref.encoder.run(mapOf("audio" to audioTensor))
        val enKeys = (0 until ref.nl).map { encResult.tensor("en_key_$it") }
        val enValues = (0 until ref.nl).map { encResult.tensor("en_value_$it") }
        try {

            val prefix = intArrayOf(SOS, LANG_UR, REGION_PK, ASR, NOTS)

            // Prefill with an empty decoder cache; the decoder consumes all 5 prefix
            // tokens at once and returns the KV for them + the full logits of the last.
            val emptyKeys = (0 until ref.nl).map { EMPTY_BUFFER }
            val emptyValues = (0 until ref.nl).map { EMPTY_BUFFER }
            val pre = runDecoder(ref, enKeys, enValues, prefix, 0, emptyKeys, emptyValues)
            Log.i(TAG, "prefill kv[0] remaining=${pre.deKeys[0].remaining()} " +
                "h=${pre.deKeys[0].remaining() / (ref.headDim * ref.dModel)}")
            val preLogProbs = logSoftmax(pre.fullLogits)

            var beams = topK(preLogProbs, BEAM_SIZE).map { idx ->
                val b = Beam(prefix + intArrayOf(idx), preLogProbs[idx], pre.deKeys, pre.deValues)
                b
            }

            val finished = mutableListOf<Beam>()
            // Cap generation by audio length instead of a flat cap: normal Urdu dictation
            // runs ~2-3 tok/s but degenerate loops (ticket 29 Phase C/D) can otherwise burn
            // all MAX_LEN steps. Budget worst-case tok/s of audio, plus small headroom.
            val maxLen = max(
                MIN_GEN_TOKENS,
                min(MAX_LEN, audio.size / SAMPLE_RATE * TOK_PER_SEC_WORST + HEADROOM)
            )
            val t0 = SystemClock.elapsedRealtime()
            var runs = 0L
            for (step in 0 until maxLen) {
                val active = mutableListOf<Beam>()
                for (b in beams) {
                    if (b.tokens.last() == EOS) finished.add(b) else active.add(b)
                }
                if (active.isEmpty()) break

                val candidates = mutableListOf<Beam>()
                for (b in active) {
                    val r = runDecoder(
                        ref, enKeys, enValues,
                        intArrayOf(b.tokens.last()),
                        historyLen = b.tokens.size - 1,
                        deKeys = b.deKeys, deValues = b.deValues,
                    )
                    runs++
                    val lp = logSoftmax(r.fullLogits)
                    for (idx in topK(lp, BEAM_SIZE)) {
                        candidates.add(b.extend(idx, lp[idx], r.deKeys, r.deValues))
                    }
                }
                candidates.sortByDescending { it.score }
                beams = candidates.take(BEAM_SIZE)
                // Hard-stop once the best-ranked beam has finished. With greedy this is the
                // natural decode end; it also short-circuits degenerate no-EOS loops.
                if (beams.first().tokens.last() == EOS) break
            }
            val decodeMs = SystemClock.elapsedRealtime() - t0
            Log.i(TAG, "beam search done: steps=${finished.size + 0}, runs=$runs, " +
                "decodeMs=$decodeMs, maxH=${beams.firstOrNull()?.tokens?.size ?: 0}")

            // Length-normalized selection (ticket 29): skip prefix-only hypotheses (an
            // empty transcript is never a valid dictation), then choose by cumulative
            // score divided by the number of generated content tokens.
            val prefLen = prefix.size
            val finishedPool = if (finished.isNotEmpty()) finished else beams
            var meaningful = finishedPool.filter { it.tokens.size > prefLen }
            if (meaningful.isEmpty()) meaningful = beams.filter { it.tokens.size > prefLen }
            return if (meaningful.isNotEmpty()) {
                meaningful.maxByOrNull { it.score / max(1.0, (it.tokens.size - prefLen).toDouble()) }!!.tokens
            } else {
                beams.first().tokens
            }
        } finally {
            audioTensor.close()
            encResult.close()
        }
    }

    /**
     * One decoder forward: given the fp16 cached decoder KV (all tokens except the last
     * of [inputIds]) and the encoder cross KV, decode [inputIds] (the prefill passes all
     * prefix tokens at once; streaming steps pass exactly one) and return the fresh
     * decoder KV + the full-logits row for the last position.
     */
    private fun runDecoder(
        ref: DolphinAttnModelRef,
        enKeys: List<OnnxTensor>,
        enValues: List<OnnxTensor>,
        inputIds: IntArray,
        historyLen: Int,
        deKeys: List<ShortBuffer>,
        deValues: List<ShortBuffer>,
    ): DecodeResult {
        val env = ref.env
        val n = ref.nl
        val inputs = HashMap<String, OnnxTensor>()
        // Per-call tensors we own; not the reused encoder cross KV. Closed in finally.
        val owned = mutableListOf<OnnxTensor>()
        try {
            for (i in 0 until n) {
                // remaining() is the TOTAL fp16 element count (head*dModel*historyLen);
                // fold it back to historyLen for the [head,dModel,h]/[head,h,dModel] feeds.
                val h = deKeys[i].remaining() / (ref.headDim * ref.dModel)
                val keyShape = longArrayOf(ref.headDim.toLong(), ref.dModel.toLong(), h.toLong())
                val valShape = longArrayOf(ref.headDim.toLong(), h.toLong(), ref.dModel.toLong())
                owned.add(OnnxTensor.createTensor(
                    env, deKeys[i].duplicate(), keyShape, OnnxJavaType.FLOAT16
                ).also { inputs["in_de_key_$i"] = it })
                owned.add(OnnxTensor.createTensor(
                    env, deValues[i].duplicate(), valShape, OnnxJavaType.FLOAT16
                ).also { inputs["in_de_value_$i"] = it })
                inputs["en_key_$i"] = enKeys[i]
                inputs["en_value_$i"] = enValues[i]
            }
            owned.add(OnnxTensor.createTensor(
                env, toDirectIntBuffer(inputIds), longArrayOf(1, inputIds.size.toLong())
            ).also { inputs["input_ids"] = it })
            owned.add(OnnxTensor.createTensor(
                env, toDirectLongBuffer(intArrayOf(historyLen)), longArrayOf(1)
            ).also { inputs["history_len"] = it })
            owned.add(OnnxTensor.createTensor(
                env, toDirectLongBuffer(intArrayOf(inputIds.size)), longArrayOf(1)
            ).also { inputs["ids_len"] = it })
            // Constants are created once at load() and reused across every step, cutting
            // per-run JNI tensor churn (the dominant fixed cost on the phone).
            inputs["language_start"] = ref.langStartTensor
            inputs["language_end"] = ref.langEndTensor
            inputs["attention_mask"] = ref.maskTensor

            val result = ref.decoder.run(inputs)
            try {
                val keys = (0 until n).map { result.tensor("out_de_key_$it").getShortBuffer() }
                val values = (0 until n).map { result.tensor("out_de_value_$it").getShortBuffer() }
                val logits = result.tensor(OUTPUT_LOGITS).getFloatBuffer()
                val row = FloatArray(logits.remaining())
                logits.get(row)
                return DecodeResult(keys, values, row)
            } finally {
                result.close()
            }
        } finally {
            owned.forEach { try { it.close() } catch (_: Exception) {} }
        }
    }

    private fun logSoftmax(logits: FloatArray): DoubleArray {
        var m = Double.NEGATIVE_INFINITY
        for (v in logits) m = max(m, v.toDouble())
        var sum = 0.0
        for (v in logits) sum += exp(v.toDouble() - m)
        val norm = m + ln(sum)
        return DoubleArray(logits.size) { logits[it].toDouble() - norm }
    }

    /** Indices of the k largest values (descending), i.e. argsort(-v)[:k]. */
    private fun topK(v: DoubleArray, k: Int): IntArray =
        IntArray(v.size) { it }
            .sortedWith(compareByDescending { v[it] })
            .take(k)
            .toIntArray()

    /**
     * OrtSession.Result is not a Map: `get(name)` returns
     * [java.util.Optional]. Unwrap it to the named tensor (encoder cross-KV outputs
     * or decoder out_{de_key,de_value}/logits outputs); throws if the model did not
     * produce the expected output.
     */
    private fun OrtSession.Result.tensor(name: String): OnnxTensor =
        get(name).orElseThrow {
            IllegalStateException("ONNX run did not produce output '$name'")
        } as OnnxTensor

    /** Convert content token ids (after the 5-token prefix) to text via units.txt. */
    internal fun decodeTokens(ref: DolphinAttnModelRef, tokens: IntArray): String {
        val content = tokens.drop(PREFIX_SIZE)
            .filter { it != 0 && it != SOS && it != EOS && it != NOTS }
            .filter { it < ref.tokenList.size }
            .map { ref.tokenList[it] }
        return decodeDolphinPieces(content)
    }

    // ---- Wave reading (16-bit PCM mono; the app's recorder output) ----

    private fun readWaveShorts(path: String): ShortArray? {
        val f = File(path)
        if (!f.exists() || f.length() < 44) return null
        val bytes = f.readBytes()
        if (bytes[0] != 'R'.code.toByte() || bytes[1] != 'I'.code.toByte() ||
            bytes[2] != 'F'.code.toByte() || bytes[3] != 'F'.code.toByte() ||
            bytes[8] != 'W'.code.toByte() || bytes[9] != 'A'.code.toByte() ||
            bytes[10] != 'V'.code.toByte() || bytes[11] != 'E'.code.toByte()
        ) return null

        var offset = 12
        var sampleRate = 16000
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
                dataStart = offset + 8
                dataLen = size.coerceAtMost(bytes.size - dataStart)
                break
            }
            offset += 8 + size
        }
        if (dataStart < 0 || bitsPerSample != 16) return null
        // The encoder's fused STFT expects 16 kHz input (like the app recorder).
        if (sampleRate != 16000) return null

        val frameBytes = nChannels * 2
        val sampleCount = dataLen / frameBytes
        if (sampleCount <= 0) return null

        val out = ShortArray(sampleCount)
        var bi = dataStart
        for (i in 0 until sampleCount) {
            out[i] = (((bytes[bi + 1].toInt() and 0xFF)) shl 8 or (bytes[bi].toInt() and 0xFF)).toShort()
            bi += frameBytes
        }
        return out
    }

    private fun le16(b: ByteArray, off: Int): Int =
        (b[off].toInt() and 0xFF) or ((b[off + 1].toInt() and 0xFF) shl 8)

    private fun le32(b: ByteArray, off: Int): Int =
        (b[off].toInt() and 0xFF) or
            ((b[off + 1].toInt() and 0xFF) shl 8) or
            ((b[off + 2].toInt() and 0xFF) shl 16) or
            ((b[off + 3].toInt() and 0xFF) shl 24)

    private fun toDirectShortBuffer(a: ShortArray): ShortBuffer =
        ByteBuffer.allocateDirect(a.size * 2).order(ByteOrder.nativeOrder())
            .asShortBuffer().apply { put(a); rewind() }

    private fun toDirectIntBuffer(a: IntArray) =
        ByteBuffer.allocateDirect(a.size * 4).order(ByteOrder.nativeOrder())
            .asIntBuffer().apply { put(a); rewind() }

    private fun toDirectLongBuffer(a: IntArray) =
        ByteBuffer.allocateDirect(a.size * 8).order(ByteOrder.nativeOrder())
            .asLongBuffer().apply { a.forEach { put(it.toLong()) }; rewind() }

    companion object {
        private const val TAG = "DolphinAttnEngine"

        // Language/task pin tokens (Dolphin ASR vocabulary).
        const val SOS = 39999
        const val EOS = 40000
        const val NOTS = 324
        const val ASR = 6
        const val LANG_UR = 136
        const val REGION_PK = 269

        /** The 5-token prefix length: `<sos> <ur> <PK> <asr> <notimestamp>`. */
        const val PREFIX_SIZE = 5

        /** Full-logits output name added by `add_logits_output.py` graph surgery. */
        const val OUTPUT_LOGITS = "/output_layer/Gemm_output_0"

        private const val BEAM_SIZE = 1
        private const val MAX_LEN = 60

        const val SAMPLE_RATE = 16000
        /** Worst-case new-token rate per second of audio (degenerate loops ran ~16 tok/s). */
        private const val TOK_PER_SEC_WORST = 14
        /** Headroom on top of the worst-case token budget. */
        private const val HEADROOM = 8
        /** Never allow fewer generated steps than this, even for very short clips. */
        private const val MIN_GEN_TOKENS = 8

        /** SentencePiece metasymbol for a leading space (U+2581). */
        const val META = "\u2581"

        /** Born with a zero capacity; reused for the empty decoder KV prefill. */
        private val EMPTY_BUFFER: ShortBuffer =
            ByteBuffer.allocateDirect(0).order(ByteOrder.nativeOrder()).asShortBuffer()
    }
}

/**
 * SentencePiece `DecodePieces` for units.txt symbols: a leading `▁` (U+2581) meta
 * symbol becomes a space (keeping the rest), pieces are concatenated, and the leading
 * space is trimmed. Matches `sp.DecodePieces` on the real Urdu BPE output (ticket 29),
 * so `bpe.model` is not needed on device for decoding.
 */
internal fun decodeDolphinPieces(pieces: List<String>): String {
    val sb = StringBuilder()
    for (p in pieces) {
        if (p.startsWith(DolphinAttnEngine.META)) {
            sb.append(' ').append(p.substring(1))
        } else {
            sb.append(p)
        }
    }
    return sb.toString().trimStart()
}