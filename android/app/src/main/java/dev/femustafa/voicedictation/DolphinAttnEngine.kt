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
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

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
 * never hardcoded. The KV cache dtype is derived from the encoder output at load time
 * (FLOAT16 for the fp16/arm tier, FLOAT for the fp32-derived int8 tier), then shared
 * across prefill and each decode step.
 *
 * Vocabulary: `units.txt` id→symbol list; decode is SentencePiece `DecodePieces`
 * (leading `▁` → space, concatenate, trim leading space) without needing `bpe.model`.
 *
 * The batch path decodes clips at or under the 14 s chunk cap as a single whole-clip
 * beam search (the highest quality path, and the one the laptop verifier
 * `verify_dakeqq_beam.py` validates). Longer clips are split by the resident Silero
 * VAD into merged chunks that are hard-split to <= 14 s each and decoded independently,
 * so the tail is never dropped.
 *
 * Two hard runtime limits shape the chunk size (verified empirically on ORT 1.27/1.30
 * for both base and small):
 *   - the decoder's fused SkipLayerNormalization kernel fails once the decoder KV
 *     history exceeds 72 tokens (base and small identical), so a single decode may
 *     emit at most ~66 content tokens after the 5-token ur/PK prefix;
 *   - normal Urdu dictation runs ~2-4 tok/s, so a <= 14 s chunk stays under that
 *     ceiling; the old flat 60-token generation cap silently truncated the tail of any
 *     clip that needed more. [tokenBudget] also hard-caps generation at the kernel
 *     ceiling as a belt-and-braces guard, trading a truncated tail for a crash.
 * The same resident VAD also drives the simulated-streaming session
 * ([DolphinAttnSession]), which splits a live clip into per-utterance segments.
 */

/**
 * Minimal surface of [DolphinAttnEngine]'s decode used by [DolphinAttnSession].
 * Extracting this seam lets unit tests drive the session with a fake decode (the ONNX
 * runtime isn't loadable on the JVM) without a real [android.content.Context] or engine.
 */
internal interface DolphinRecognizerLike {
    /** Beam-search decode [audio] into content token ids (prefix already prepended). */
    fun beamSearch(ref: DolphinAttnEngine.DolphinAttnModelRef, audio: ShortArray): IntArray

    /** Convert [tokens] (content ids after the prefix) to text via the units vocab. */
    fun decodeTokens(ref: DolphinAttnEngine.DolphinAttnModelRef, tokens: IntArray): String
}

open class DolphinAttnEngine(
    private val context: Context,
) : WhisperEngine, DolphinRecognizerLike {

    /**
     * A beach contains a partial hypothesis: the token ids decoded so far (including
     * the 5-token `<sos><ur><PK><asr><nots>` prefix), its cumulative log-prob, and the
     * decoder KV cache (ShortBuffer for fp16 / FloatBuffer for fp32 tiers; state AFTER
     * processing all but the last token — the next decode step consumes exactly one
     * more). Sharing the lists across siblings is fine: they are treated as immutable.
     */
    private class Beam(
        val tokens: IntArray,
        val score: Double,
        val deKeys: List<Any>,
        val deValues: List<Any>,
    ) {
        /** A new beam extending this one with [token], appending its log-prob and
         * advancing to the decoder KV state returned by the step that produced [token]. */
        fun extend(token: Int, logProb: Double, deKeys: List<Any>,
                   deValues: List<Any>): Beam =
            Beam(tokens + token, score + logProb, deKeys, deValues)
    }

    /** Everything needed to decode with a loaded model pair. */
    open class DolphinAttnModelRef(
        val env: OrtEnvironment?,
        val encoder: OrtSession?,
        val decoder: OrtSession?,
        val tokenList: List<String>,
        val nl: Int,
        val headDim: Int,
        val dModel: Int,
        /** KV cache dtype derived from the encoder output at load time. */
        val kvDtype: OnnxJavaType,
        /** Load-cached constant decoder inputs, reused across every step's run. */
        val langStartTensor: OnnxTensor?,
        val langEndTensor: OnnxTensor?,
        val maskTensor: OnnxTensor?,
        /**
         * Resident Silero VAD (sherpa-onnx), loaded once for [transcribe] to split long
         * clips into per-utterance decodes. Null when the silero_vad.onnx sibling is
         * absent/undersized → whole-clip decode fallback.
         */
        val vad: Vad?,
    ) : WhisperModelRef

    /** A single decoder run: fresh KV state + the raw full-logits row. */
    private class DecodeResult(
        val deKeys: List<Any>,
        val deValues: List<Any>,
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

        val kvDtype = (encOutputs.getValue("en_key_0").info as TensorInfo).type
        Log.i(TAG, "Dolphin attention arch: nl=$nl headDim=$headDim dModel=$dModel kvDtype=$kvDtype")
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
                val app = VoiceDictationApp.from(context)
                silero.threshold = app.vadThreshold()
                silero.minSilenceDuration = app.vadMinSilence()
                silero.minSpeechDuration = app.vadMinSpeech()
                silero.windowSize = VAD_WINDOW
                silero.maxSpeechDuration = app.vadMaxSpeech()

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
            kvDtype = kvDtype,
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
            // Clips at or under the 14 s chunk cap decode in a single beam search (the
            // highest-quality path). Longer clips are VAD-segmented, merged and hard-split
            // into <= 14 s chunks, and each chunk decoded independently so the tail is
            // never dropped; a longer chunk would cross the decoder's 66-token kernel
            // ceiling (SkipLayerNorm_0 fails at a 72-token KV history) and either
            // truncate or crash.
            if (audio.size <= DOLPHIN_MAX_SAMPLES) {
                decodeTokens(ref, beamSearch(ref, audio))
            } else {
                val chunks = chunkAudio(ref, audio)
                if (chunks.isEmpty()) {
                    Log.w(TAG, "no VAD chunks for long clip; falling back to whole-clip decode")
                    decodeTokens(ref, beamSearch(ref, audio))
                } else {
                    Log.i(
                        TAG,
                        "long clip ${"%.1f".format(audio.size / SAMPLE_RATE.toDouble())}s -> ${chunks.size} chunks"
                    )
                    var acc: String? = null
                    for (chunk in chunks) {
                        val t = decodeTokens(ref, beamSearch(ref, chunk)).trim()
                        if (t.isNotEmpty()) {
                            acc = if (acc == null) t else seamMerge(acc!!, t)
                        }
                    }
                    acc ?: ""
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "dolphin attention transcribe failed", e)
            throw e
        }
    }

    /**
     * Split [audio] into ordered chunks of at most [DOLPHIN_MAX_SAMPLES] samples
     * (~14 s), so each chunk's beam search stays under the decoder's 66-content-token
     * kernel ceiling. The VAD merge passes a single over-limit utterance through
     * untrimmed, so [splitChunksAtMost] hard-splits any piece that still exceeds the
     * cap into equal sub-pieces: boundaries may fall mid-word, but every sample is
     * decoded by exactly one piece, so nothing is dropped.
     * Uses the resident Silero VAD from [DolphinAttnModelRef.vad] (reset first — ticket 31
     * finding: the stateful VAD must not carry speech-start state across uses). Returns an
     * empty list when the VAD is unavailable or yields no segments (caller falls back to a
     * whole-clip decode).
     */
    private fun chunkAudio(ref: DolphinAttnModelRef, audio: ShortArray): List<ShortArray> {
        val vad = ref.vad ?: return emptyList()
        vad.reset()
        return segmentAudioWithVad(
            samples = audio.toFloatArray(SAMPLE_RATE),
            vad = SherpaVad(vad),
            padding = 0,
            maxChunkSamples = DOLPHIN_MAX_SAMPLES,
        ).map { seg ->
            ShortArray(seg.samples.size) { (seg.samples[it] * 32767f).toInt().toShort() }
        }.flatMap { splitChunksAtMost(it, DOLPHIN_MAX_SAMPLES) }
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
        // The resident VAD is shared across batch transcribes and every streaming session,
        // and silero is stateful (speech-start flag + buffered tail). Without a reset,
        // state left over from a previous dictation can swallow or merge the FIRST
        // utterance of this recording. Reset = clean slate per usage.
        val residentVad = ref.vad
        if (residentVad == null) {
            Log.w(TAG, "No resident VAD available; cannot create streaming session")
            return null
        }
        residentVad.reset()
        val vadLike = SherpaVad(residentVad)
        return DolphinAttnSession(this, ref, waveWriter, context, vadLike)
    }

    override fun release(model: WhisperModelRef) {
        val ref = model as? DolphinAttnModelRef ?: return
        try {
            ref.vad?.release()
            ref.langStartTensor?.close()
            ref.langEndTensor?.close()
            ref.maskTensor?.close()
            ref.decoder?.close()
            ref.encoder?.close()
            Log.i(TAG, "released Dolphin attention model")
        } catch (e: Exception) {
            Log.w(TAG, "error releasing Dolphin attention model: ${e.message}")
        }
    }

    /**
     * Beam search over the decoder's FULL logits output. Faithful port of
     * `verify_dakeqq_beam.py::beam_search` with `use_sliced=False`:
     *
     *  1. Run the encoder (raw int16 in-graph) -> fp16 cross KV.
     *  2. Prefill `<sos><ur><PK><asr><nots>`.
     *  3. Loop: move EOS-terminated beams to `finished`; expand the rest by the top-k
     *     tokens of the full-logits log-softmax; keep the best [beamSize] by cumulative
     *     score. Each decode passes exactly one new token with the parent's lazy KV.
     *     The loop ends only when every beam has EOS'd (or [maxLen] is hit) — it never
     *     hard-stops on the top-ranked beam, so a stray early `<eos>` can't drop the
     *     trailing sentences (ticket 29 EOS-bias; matches the verifier).
     *  4. Pick the finished hypothesis with the best LENGTH-NORMALIZED score
     *     (score / #generated tokens), dropping prefix-only empties.
     */
    override fun beamSearch(ref: DolphinAttnModelRef, audio: ShortArray): IntArray {
        val env = ref.env!!
        val audioTensor = OnnxTensor.createTensor(
            env, toDirectShortBuffer(audio), longArrayOf(1, 1, audio.size.toLong()),
            OnnxJavaType.INT16
        )

        // Reusable encoder cross-KV, kept open for the whole decode.
        val encResult = ref.encoder!!.run(mapOf("audio" to audioTensor))
        val enKeys = (0 until ref.nl).map { encResult.tensor("en_key_$it") }
        val enValues = (0 until ref.nl).map { encResult.tensor("en_value_$it") }
        try {

            val prefix = intArrayOf(SOS, LANG_UR, REGION_PK, ASR, NOTS)

            // Prefill with an empty decoder cache; the decoder consumes all 5 prefix
            // tokens at once and returns the KV for them + the full logits of the last.
            val emptyKeys = (0 until ref.nl).map { emptyKvBuffer(ref.kvDtype) }
            val emptyValues = (0 until ref.nl).map { emptyKvBuffer(ref.kvDtype) }
            val pre = runDecoder(ref, enKeys, enValues, prefix, 0, emptyKeys, emptyValues)
            val firstKey = pre.deKeys[0] as? ShortBuffer ?: (pre.deKeys[0] as FloatBuffer)
            Log.i(TAG, "prefill kv[0] remaining=${firstKey.remaining()} " +
                "h=${firstKey.remaining() / (ref.headDim * ref.dModel)}")
            val preLogProbs = logSoftmax(pre.fullLogits)

            val beamSize = VoiceDictationApp.from(context).dolphinBeamSize()
            var beams = topK(preLogProbs, beamSize).map { idx ->
                val b = Beam(prefix + intArrayOf(idx), preLogProbs[idx], pre.deKeys, pre.deValues)
                b
            }

            val finished = mutableListOf<Beam>()
            // Cap generation by audio length instead of a flat cap: normal Urdu dictation
            // runs ~2-3 tok/s but degenerate loops (ticket 29 Phase C/D) can otherwise burn
            // all budget steps. Budget worst-case tok/s of audio, plus small headroom, and
            // never exceed the decoder's 66-content-token kernel ceiling (SkipLayerNorm_0).
            val maxLen = tokenBudget(audio.size)
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
                    for (idx in topK(lp, beamSize)) {
                        candidates.add(b.extend(idx, lp[idx], r.deKeys, r.deValues))
                    }
                }
                candidates.sortByDescending { it.score }
                beams = candidates.take(beamSize)
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
     * One decoder forward: given the cached decoder KV (dtype derived from the model,
     * all tokens except the last of [inputIds]) and the encoder cross KV, decode
     * [inputIds] (the prefill passes all prefix tokens at once; streaming steps pass
     * exactly one) and return the fresh decoder KV + the full-logits row for the last
     * position.
     */
    private fun runDecoder(
        ref: DolphinAttnModelRef,
        enKeys: List<OnnxTensor>,
        enValues: List<OnnxTensor>,
        inputIds: IntArray,
        historyLen: Int,
        deKeys: List<Any>,
        deValues: List<Any>,
    ): DecodeResult {
        val env = ref.env!!
        val n = ref.nl
        val inputs = HashMap<String, OnnxTensor>()
        // Per-call tensors we own; not the reused encoder cross KV. Closed in finally.
        val owned = mutableListOf<OnnxTensor>()
        try {
            for (i in 0 until n) {
                // Create the key/value input tensors from the cached buffers, sized by
                // their dtype (ShortBuffer for fp16, FloatBuffer for fp32 tiers).
                val (kTensor, vTensor) = kvInputTensors(env, deKeys[i], deValues[i], ref)
                owned.add(kTensor.also { inputs["in_de_key_$i"] = it })
                owned.add(vTensor.also { inputs["in_de_value_$i"] = it })
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
            inputs["language_start"] = ref.langStartTensor!!
            inputs["language_end"] = ref.langEndTensor!!
            inputs["attention_mask"] = ref.maskTensor!!

            val result = ref.decoder!!.run(inputs)
            try {
                val keys = (0 until n).map { result.tensor("out_de_key_$it") }
                val values = (0 until n).map { result.tensor("out_de_value_$it") }
                val logits = result.tensor(OUTPUT_LOGITS).getFloatBuffer()
                val row = FloatArray(logits.remaining())
                logits.get(row)
                val dk = keys.map { kb ->
                    if (ref.kvDtype == OnnxJavaType.FLOAT16) kb.getShortBuffer()
                    else kb.getFloatBuffer()
                }
                val dv = values.map { vb ->
                    if (ref.kvDtype == OnnxJavaType.FLOAT16) vb.getShortBuffer()
                    else vb.getFloatBuffer()
                }
                return DecodeResult(dk, dv, row)
            } finally {
                result.close()
            }
        } finally {
            owned.forEach { try { it.close() } catch (_: Exception) {} }
        }
    }

    /**
     * Build the key/value input tensors for one decoder layer from the cached KV
     * buffers. The cache dtype (from [DolphinAttnModelRef.kvDtype]) selects the
     * buffer element type: ShortBuffer for FLOAT16, FloatBuffer for FLOAT (fp32)
     * tiers. `remaining()` is the TOTAL element count (head*dModel*historyLen);
     * fold it back to historyLen for the [head,dModel,h]/[head,h,dModel] feeds.
     */
    private fun kvInputTensors(
        env: OrtEnvironment,
        key: Any,
        value: Any,
        ref: DolphinAttnModelRef,
    ): Pair<OnnxTensor, OnnxTensor> {
        val head = ref.headDim.toLong()
        val dim = ref.dModel.toLong()
        return when (ref.kvDtype) {
            OnnxJavaType.FLOAT16 -> {
                val kb = key as ShortBuffer
                val vb = value as ShortBuffer
                val h = kb.remaining() / (ref.headDim * ref.dModel)
                OnnxTensor.createTensor(env, kb.duplicate(), longArrayOf(head, dim, h.toLong()), OnnxJavaType.FLOAT16) to
                    OnnxTensor.createTensor(env, vb.duplicate(), longArrayOf(head, h.toLong(), dim), OnnxJavaType.FLOAT16)
            }
            OnnxJavaType.FLOAT -> {
                val kb = key as FloatBuffer
                val vb = value as FloatBuffer
                val h = kb.remaining() / (ref.headDim * ref.dModel)
                // FLOAT tensors are built from a FloatBuffer with the 3-arg overload
                // (no explicit type variant exists for float32 buffers).
                OnnxTensor.createTensor(env, kb.duplicate(), longArrayOf(head, dim, h.toLong())) to
                    OnnxTensor.createTensor(env, vb.duplicate(), longArrayOf(head, h.toLong(), dim))
            }
            else -> throw IllegalStateException("Unsupported KV dtype: ${ref.kvDtype}")
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
    override fun decodeTokens(ref: DolphinAttnModelRef, tokens: IntArray): String {
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

        /**
         * Hard per-decode ceiling on generated content tokens: the decoder's fused
         * SkipLayerNormalization kernel rejects a KV history beyond 72 tokens (after the
         * 5-token prefix, that is 66 content tokens). Crossing it fails ORT with
         * `ORT_INVALID_ARGUMENT` on SkipLayerNorm_0. [DOLPHIN_MAX_MS] is sized so normal
         * speech never reaches 66, and [tokenBudget] never lets a decode exceed it.
         */
        private const val MAX_DECODER_CONTENT_TOKENS = 66

        const val SAMPLE_RATE = 16000
        /** Worst-case new-token rate per second of audio (degenerate loops ran ~16 tok/s). */
        private const val TOK_PER_SEC_WORST = 14
        /** Headroom on top of the worst-case token budget. */
        private const val HEADROOM = 8
        /** Never allow fewer generated steps than this, even for very short clips. */
        private const val MIN_GEN_TOKENS = 8

        /**
         * Dolphin batch chunks may span at most this many ms. 14 s × ~4.1 tok/s ≈ 57
         * content tokens, well under the 66-token decoder kernel ceiling, so a full
         * chunk never truncates and never trips SkipLayerNorm_0. The Dolphin repo's
         * 30 s `SPEECH_LENGTH` (ADR 0006) assumed no kernel ceiling; our 30 s chunks
         * would need ~60-90 tokens (base fp16 ~2-4 tok/s) and overflow it, so the cap
         * is 14 s instead. The ~4.1 tok/s worst case is measured from on-device Urdu:
         * a 16 s cap left zero margin (a 17.8 s unbroken utterance hit all 66 tokens).
         * [chunkAudio] hard-splits any oversized piece — preferring quiet pauses near the
         * cap so boundaries don't land mid-word — because the VAD merge passes a single
         * over-limit utterance through untrimmed.
         */
        private const val DOLPHIN_MAX_MS = 14_000

        /** The Dolphin chunk cap in samples at 16 kHz: 14 s × 16 kHz = 224,000. */
        internal const val DOLPHIN_MAX_SAMPLES = DOLPHIN_MAX_MS * SAMPLE_RATE / 1000

        /** Near-silence scan window for split points: 300 ms at 16 kHz. */
        private const val RMS_WINDOW_SAMPLES = 4_800

        /** Below this window RMS, a split landed in a pause rather than mid-speech. */
        private const val QUIET_RMS = 320f

        /**
         * Cap generated decode steps for [sampleCount] samples of audio. Keeps the
         * degenerate-loop bound (ticket 29 Phase C/D) proportional to audio length
         * (worst-case tok/s plus headroom), while never exceeding the decoder's
         * 66-content-token kernel ceiling ([MAX_DECODER_CONTENT_TOKENS]) — a decode is
         * capped rather than allowed to crash ORT's SkipLayerNorm_0.
         */
        internal fun tokenBudget(sampleCount: Int): Int =
            max(
                MIN_GEN_TOKENS,
                min(MAX_DECODER_CONTENT_TOKENS, sampleCount / SAMPLE_RATE * TOK_PER_SEC_WORST + HEADROOM),
            )

        /**
         * Split [samples] into consecutive pieces of at most [maxSamples], preferring a
         * split point inside a quiet (near-silent) window.
         *
         * [VadSegmentation.mergeSegments] passes a single over-limit utterance through
         * untrimmed (the resident VAD won't split within unbroken speech), so without
         * this the 14 s cap is only enforced between utterances and a dense long
         * dictation can still burn all 66 budget tokens and truncate. A bare equal-ish
         * split fixes that but can land mid-word on dense continuous speech, which makes
         * a chunk run out of real audio, hallucinate to burn its budget, and join
         * mid-sentence. So the last quarter of each candidate piece is scanned for a
         * near-silent window first, and only if none exists (uninterruptedly loud
         * audio) the split falls back to the hard cap. Consecutive samples are
         * preserved in order; a buffer already at or under the cap is returned
         * unchanged.
         */
        internal fun splitChunksAtMost(samples: ShortArray, maxSamples: Int): List<ShortArray> {
            if (samples.size <= maxSamples) return listOf(samples)
            val rms = windowedRms(samples)
            val out = ArrayList<ShortArray>()
            var start = 0
            while (samples.size - start > maxSamples) {
                val hi = start + maxSamples
                val lo = start + maxSamples * 3 / 4
                val split = quietSplit(rms, lo.coerceAtMost(samples.size), hi.coerceAtMost(samples.size))
                    ?: hi.coerceAtMost(samples.size)
                out.add(samples.copyOfRange(start, split))
                start = split
            }
            out.add(samples.copyOfRange(start, samples.size))
            return out
        }

        /** RMS of every [RMS_WINDOW_SAMPLES]-sample window, for finding quiet split points. */
        private fun windowedRms(samples: ShortArray): FloatArray {
            val n = samples.size
            val count = (n + RMS_WINDOW_SAMPLES - 1) / RMS_WINDOW_SAMPLES
            val rms = FloatArray(count)
            for (i in 0 until count) {
                val from = i * RMS_WINDOW_SAMPLES
                val to = min(from + RMS_WINDOW_SAMPLES, n)
                var sum = 0L
                for (j in from until to) {
                    val v = samples[j].toInt()
                    sum += v * v
                }
                rms[i] = sqrt(sum.toDouble() / (to - from)).toFloat()
            }
            return rms
        }

        /**
         * Sample offset of the quietest qualifying [RMS_WINDOW_SAMPLES]-sample window whose
         * end lands in `loIncl..hiIncl`, or null when every candidate window is louder than
         * [QUIET_RMS] (no real pause to rest the boundary on).
         */
        private fun quietSplit(rms: FloatArray, loIncl: Int, hiIncl: Int): Int? {
            var best = Int.MAX_VALUE
            var bestRms = QUIET_RMS
            val firstWin = loIncl / RMS_WINDOW_SAMPLES
            val lastWin = hiIncl / RMS_WINDOW_SAMPLES
            for (i in firstWin..lastWin) {
                val r = rms[i]
                if (r < bestRms) {
                    bestRms = r
                    best = ((i + 1) * RMS_WINDOW_SAMPLES).coerceIn(loIncl, hiIncl)
                }
            }
            return if (best == Int.MAX_VALUE) null else best
        }

        /** SentencePiece metasymbol for a leading space (U+2581). */
        const val META = "\u2581"

        /**
         * The empty decoder KV cache for the prefill, typed to match the KV cache
         * dtype: ShortBuffer (fp16 tier) or FloatBuffer (fp32/int8 tier).
         */
        private fun emptyKvBuffer(dtype: OnnxJavaType): Any = when (dtype) {
            OnnxJavaType.FLOAT16 -> EMPTY_SHORT_BUFFER
            OnnxJavaType.FLOAT -> EMPTY_FLOAT_BUFFER
            else -> throw IllegalStateException("Unsupported KV dtype: $dtype")
        }

        /** Born with a zero capacity; reused for the fp16 empty decoder KV prefill. */
        private val EMPTY_SHORT_BUFFER: ShortBuffer =
            ByteBuffer.allocateDirect(0).order(ByteOrder.nativeOrder()).asShortBuffer()

        /** Born with a zero capacity; reused for the fp32/int8 empty decoder KV prefill. */
        private val EMPTY_FLOAT_BUFFER: FloatBuffer = FloatBuffer.allocate(0)
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

/**
 * Merge adjacent segments by removing word-level overlaps at chunk boundaries.
 * Given two strings a (previous chunk) and b (next chunk), compares the last
 * `windowWords` words of a with the first `windowWords` words of b. If a
 * contiguous word-run of at least `minMatchWords` words is found, the matched
 * run is dropped from b (left wins). Otherwise the segments are joined unchanged.
 * Addresses the seam duplication artifact seen in long-audio streaming.
 */
internal fun seamMerge(
    a: String,
    b: String,
    windowWords: Int = 12,
    minMatchWords: Int = 4,
): String {
    val aw = a.trim().split(" ").filter { it.isNotEmpty() }
    val bw = b.trim().split(" ").filter { it.isNotEmpty() }

    var bestSize = 0
    var bestAi = -1
    var bestBj = -1

    val tail = aw.takeLast(windowWords)
    val head = bw.take(windowWords)

    for (i in tail.indices) {
        for (j in head.indices) {
            var k = 0
            while (i + k < tail.size && j + k < head.size && tail[i + k] == head[j + k]) {
                k++
            }
            if (k > bestSize) {
                bestSize = k
                bestAi = i
                bestBj = j
            }
        }
    }

    return if (bestSize >= minMatchWords) {
        val keepB = bw.take(bestBj) + bw.drop(bestBj + bestSize)
        (aw + keepB).joinToString(" ")
    } else {
        (aw + bw).joinToString(" ")
    }
}