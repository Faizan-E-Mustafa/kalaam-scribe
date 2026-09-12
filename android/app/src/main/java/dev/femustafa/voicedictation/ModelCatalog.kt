package dev.femustafa.voicedictation

/**
 * The Model catalogs from the spec (spec.Notes "Model catalog"). Two separate
 * catalogs, one per file [ModelFormat], because they are independent model sets
 * (CONTEXT.md "Model format"):
 *
 * - [ggmlModels] — quantized whisper.cpp GGML `.bin` models (q4_0, q8_0, q5_1…),
 *   loaded by [AarWhisperEngine].
 * - [onnxModels] — sherpa-onnx Whisper ONNX models. Organized by ONNX model
 *   identity (`tiny`, `tiny.en`, `base.en`, `small`, `roman-urdu`), each offered in two precision
 *   tiers: [ModelPrecision.FP32] (full) and [ModelPrecision.INT8] (quantized),
 *   loaded by [SherpaWhisperEngine]. Quantization is a GGML concept on the GGML
 *   side; on the ONNX side we offer int8 as a distinct tier.
 * - [dolphinCtcModels] — Dolphin CTC multilingual models (DataoceanAI), a separate
 *   ONNX model family from Whisper that supports 40+ Eastern languages including
 *   Urdu. Loaded by [DolphinCtcEngine] via sherpa-onnx's `OfflineDolphinModelConfig`.
 *   Files are a single `model.onnx`/`model.int8.onnx` + `tokens.txt` (no
 *   encoder/decoder split). sherpa-onnx auto-detects the language, so these are all
 *   [LanguageMode.Auto] and the language setting is ignored. This family is
 *   currently disabled: retained in code but never offered (see [onnxDefault]) —
 *   the Dolphin attention family below is the supported Dolphin path.
 * - [omnilingualModels] — Meta's OmniASR CTC (sherpa-onnx port), single
 *   `model.int8.onnx` + `tokens.txt`. Auto-detects from 1600+ zero-shot languages
 *   per utterance; sherpa-onnx exposes **no way to pin an output language** (the
 *   language-conditioned LLM variants are unsupported), so it is [LanguageMode.Auto]
 *   and, like the multilingual whisper tier, passes the language filter for every
 *   selected language — the picker labels it "auto-detect · 1600+ languages".
 * - [dolphinAttnModels] — Dolphin attention ASR (DataoceanAI) ONNX encoder+decoder
 *   pairs (dataocean-dolphin-asr), loaded by [DolphinAttnEngine] directly through
 *   onnxruntime-android. Unlike the CTC family, these honor an explicit `ur`/`PK`
 *   language pin via `language_start`/`language_end`, so they are the models that
 *   respect the app's language selection for Urdu. Files are an encoder.onnx +
 *   a graph-surgeried decoder.onnx (full-logits output) + a shared `units.txt`
 *   vocabulary (decision: decode from `units.txt` only; `bpe.model` is published
 *   but unnecessary for decoding). Precision tier is fp16/arm.
 *
 * Sources:
 * - Roman-Urdu are conversions of `cheetos18/whisper-small-roman-urdu` hosted on
 *   the project's own HuggingFace repo (see [RU_HF]). GGML (`ggml-*.bin`) plus ONNX
 *   (`roman-urdu-{encoder,decoder}.onnx` / `.int8.onnx` + shared `roman-urdu-tokens.txt`).
 * - GGML English/multilingual from `ggerganov/whisper.cpp` HuggingFace repo.
 * - ONNX Whisper from `csukuangfj/sherpa-onnx-whisper-*` HuggingFace repos (fp32
 *   `.onnx` and int8 `.int8.onnx` files).
 * - Dolphin CTC from `csukuangfj/sherpa-onnx-dolphin-*-ctc-multi-lang*` HuggingFace
 *   repos (`model.onnx`/`model.int8.onnx` + `tokens.txt` files).
 * - Dolphin attention from the project's own HuggingFace repo `dolphin-attn/`
 *   (encoder/decoder onnx pairs + shared units.txt vocabulary).
 * - Omnilingual from `csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12`
 *   HuggingFace repo (`model.int8.onnx` + `tokens.txt`).
 */
data class CatalogEntry(
    /** The domain [Model] this entry selects (id, fileName, languageMode). */
    val model: Model,
    /** Human-friendly name shown in the picker. */
    val displayName: String,
    /** Public download URL for GGML format, or null when not available. */
    val sourceUrl: String?,
    /** Public download URL for ONNX encoder (.onnx/.int8.onnx), or null. */
    val onnxSourceUrl: String?,
    /** Approximate size in MiB (0 = unknown). */
    val approxSizeMb: Long,
    /** Whether this is the default Model on first launch. */
    val isDefault: Boolean,
    /** ONNX precision tier; null for GGML entries. */
    val precision: ModelPrecision? = null,
)

object ModelCatalog {

    const val HF_WHISPER_CPP = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
    const val RU_HF = "https://huggingface.co/femustafa/voicedictation-models/resolve/main"
    /** sherpa-onnx pre-exported Whisper model repos (one per model size). */
    const val SHERPA_ONNX_TINY = "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny/resolve/main"
    const val SHERPA_ONNX_TINY_EN = "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en/resolve/main"
    const val SHERPA_ONNX_BASE_EN = "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base.en/resolve/main"
    const val SHERPA_ONNX_BASE = "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base/resolve/main"
    const val SHERPA_ONNX_SMALL = "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small/resolve/main"
    const val SHERPA_ONNX_SMALL_EN = "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small.en/resolve/main"

    /** Dolphin CTC multilingual model repos (one per size/precision). */
    const val SHERPA_ONNX_DOLPHIN_BASE =
        "https://huggingface.co/csukuangfj/sherpa-onnx-dolphin-base-ctc-multi-lang-2025-04-02/resolve/main"
    const val SHERPA_ONNX_DOLPHIN_BASE_INT8 =
        "https://huggingface.co/csukuangfj/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/resolve/main"
    const val SHERPA_ONNX_DOLPHIN_SMALL =
        "https://huggingface.co/csukuangfj/sherpa-onnx-dolphin-small-ctc-multi-lang-2025-04-02/resolve/main"
    const val SHERPA_ONNX_DOLPHIN_SMALL_INT8 =
        "https://huggingface.co/csukuangfj/sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02/resolve/main"

    /** Project-hosted Dolphin attention ASR ONNX pairs (encoder/decoder + units). */
    const val DOLPHIN_ATTN_HF = "$RU_HF/dolphin-attn"
    /** Project-hosted Dolphin attention int8 ONNX pairs (fp32 quantized to int8). */
    const val DOLPHIN_ATTN_INT8_HF = "$RU_HF/dolphin-attn-int8"

    /** sherpa-onnx-port of Meta's OmniASR CTC presentet (1600+ zero-shot languages). */
    const val OMNILINGUAL_HF =
        "https://huggingface.co/csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/resolve/main"

    /**
     * The languages the Dolphin attention models (DataoceanAI) can decode. These
     * are the language tokens present in the Dolphin ASR vocabulary (LANG_IDS in
     * `tools/dolphin-onnx/scripts/infer_onnx.py`): the model was trained on these
     * 16 languages and its decoder accepts their `<lang>` pin tokens. Only the
     * `ur`/`PK` pin is verified on device, so Dolphin is offered for Urdu only; the
     * other 15 languages remain unverified and are not advertised (see DolphinAttnEngine).
     */
    val DOLPHIN_LANGUAGES: Set<String> = setOf("ur")

    /** sherpa-onnx-hosted Silero VAD model (shared by all Dolphin attention models;
     *  bundled into the dolphin-attn download as `silero_vad.onnx`). */
    const val SHERPA_SILERO_VAD =
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

    /** GGML (whisper.cpp) catalog: quantized `.bin` models. */
    val ggmlModels: List<CatalogEntry> = listOf(
        CatalogEntry(
            model = Model(id = "roman-urdu-q4_0", fileName = "ggml-model-q4_0.bin", languageMode = LanguageMode.RomanUrdu),
            displayName = "Roman-Urdu · q4_0",
            sourceUrl = "$RU_HF/ggml-model-q4_0.bin",
            onnxSourceUrl = null,
            approxSizeMb = 139,
            isDefault = true,
        ),
        CatalogEntry(
            model = Model(id = "roman-urdu-f16", fileName = "ggml-model-f16.bin", languageMode = LanguageMode.RomanUrdu),
            displayName = "Roman-Urdu · f16",
            sourceUrl = "$RU_HF/ggml-model-f16.bin",
            onnxSourceUrl = null,
            approxSizeMb = 550,
            isDefault = false,
        ),
        CatalogEntry(
            model = Model(id = "english-full", fileName = "ggml-base.en.bin", languageMode = LanguageMode.English),
            displayName = "English · base.en · f16",
            sourceUrl = "$HF_WHISPER_CPP/ggml-base.en.bin",
            onnxSourceUrl = null,
            approxSizeMb = 148,
            isDefault = false,
        ),
        CatalogEntry(
            model = Model(id = "english-q8", fileName = "ggml-base.en-q8_0.bin", languageMode = LanguageMode.English),
            displayName = "English · base.en · q8_0",
            sourceUrl = "$HF_WHISPER_CPP/ggml-base.en-q8_0.bin",
            onnxSourceUrl = null,
            approxSizeMb = 82,
            isDefault = false,
        ),
        CatalogEntry(
            model = Model(id = "multilingual-small-q8", fileName = "ggml-small-q8_0.bin", languageMode = LanguageMode.Auto),
            displayName = "Multilingual · small · q8_0",
            sourceUrl = "$HF_WHISPER_CPP/ggml-small-q8_0.bin",
            onnxSourceUrl = null,
            approxSizeMb = 264,
            isDefault = false,
        ),
        CatalogEntry(
            model = Model(id = "multilingual-tiny", fileName = "ggml-tiny.bin", languageMode = LanguageMode.Auto),
            displayName = "Multilingual · tiny · f16",
            sourceUrl = "$HF_WHISPER_CPP/ggml-tiny.bin",
            onnxSourceUrl = null,
            approxSizeMb = 77,
            isDefault = false,
        ),
        CatalogEntry(
            model = Model(id = "english-tiny", fileName = "ggml-tiny.en.bin", languageMode = LanguageMode.English),
            displayName = "English · tiny.en · f16",
            sourceUrl = "$HF_WHISPER_CPP/ggml-tiny.en.bin",
            onnxSourceUrl = null,
            approxSizeMb = 75,
            isDefault = false,
        ),
        CatalogEntry(
            model = Model(id = "english-tiny-q8", fileName = "ggml-tiny.en-q8_0.bin", languageMode = LanguageMode.English),
            displayName = "English · tiny.en · q8_0",
            sourceUrl = "$HF_WHISPER_CPP/ggml-tiny.en-q8_0.bin",
            onnxSourceUrl = null,
            approxSizeMb = 42,
            isDefault = false,
        ),
        CatalogEntry(
            model = Model(id = "tiny-q5", fileName = "ggml-tiny-q5_1.bin", languageMode = LanguageMode.Auto),
            displayName = "Multilingual · tiny · q5_1",
            sourceUrl = "$HF_WHISPER_CPP/ggml-tiny-q5_1.bin",
            onnxSourceUrl = null,
            approxSizeMb = 32,
            isDefault = false,
        ),
        CatalogEntry(
            model = Model(id = "english-tiny-q5", fileName = "ggml-tiny.en-q5_1.bin", languageMode = LanguageMode.English),
            displayName = "English · tiny.en · q5_1",
            sourceUrl = "$HF_WHISPER_CPP/ggml-tiny.en-q5_1.bin",
            onnxSourceUrl = null,
            approxSizeMb = 32,
            isDefault = false,
        ),
    )

    /**
     * ONNX (sherpa-onnx) catalog, organized by ONNX model identity, each in two
     * precision tiers. The fileName is the encoder filename for that tier so the
     * app-private path and the WhisperManager file-exists check line up.
     *
     * The picker only offers [onnxModelsActive]: English and multilingual models
     * run their int8 tier, while Roman-Urdu ships both tiers (the fp32 reference
     * build and the practical int8 one). The English/multilingual fp32 tiers are
     * [onnxModelsDisabled] — retained in code but never offered.
     */
    val onnxModels: List<CatalogEntry> = listOf(
        // base.en (English) — fp32 + int8
        onnx(
            id = "base.en-fp32",
            displayName = "English · Balanced",
            languageMode = LanguageMode.English,
            precision = ModelPrecision.FP32,
            encoderUrl = "$SHERPA_ONNX_BASE_EN/base.en-encoder.onnx",
            approxSizeMb = 291,
        ),
        onnx(
            id = "base.en-int8",
            displayName = "English · Balanced",
            languageMode = LanguageMode.English,
            precision = ModelPrecision.INT8,
            encoderUrl = "$SHERPA_ONNX_BASE_EN/base.en-encoder.int8.onnx",
            approxSizeMb = 160,
        ),
        // base (multilingual) — fp32 + int8
        onnx(
            id = "base-fp32",
            displayName = "Multilingual · Balanced",
            languageMode = LanguageMode.Auto,
            precision = ModelPrecision.FP32,
            encoderUrl = "$SHERPA_ONNX_BASE/base-encoder.onnx",
            approxSizeMb = 291,
        ),
        onnx(
            id = "base-int8",
            displayName = "Multilingual · Balanced",
            languageMode = LanguageMode.Auto,
            precision = ModelPrecision.INT8,
            encoderUrl = "$SHERPA_ONNX_BASE/base-encoder.int8.onnx",
            approxSizeMb = 160,
        ),
        // tiny.en (English) — fp32 + int8
        onnx(
            id = "tiny.en-fp32",
            displayName = "English · Fast",
            languageMode = LanguageMode.English,
            precision = ModelPrecision.FP32,
            encoderUrl = "$SHERPA_ONNX_TINY_EN/tiny.en-encoder.onnx",
            approxSizeMb = 150,
        ),
        onnx(
            id = "tiny.en-int8",
            displayName = "English · Fast",
            languageMode = LanguageMode.English,
            precision = ModelPrecision.INT8,
            encoderUrl = "$SHERPA_ONNX_TINY_EN/tiny.en-encoder.int8.onnx",
            approxSizeMb = 103,
        ),
        // tiny (multilingual) — fp32 + int8
        onnx(
            id = "tiny-fp32",
            displayName = "Multilingual · Fast",
            languageMode = LanguageMode.Auto,
            precision = ModelPrecision.FP32,
            encoderUrl = "$SHERPA_ONNX_TINY/tiny-encoder.onnx",
            approxSizeMb = 152,
        ),
        onnx(
            id = "tiny-int8",
            displayName = "Multilingual · Fast",
            languageMode = LanguageMode.Auto,
            precision = ModelPrecision.INT8,
            encoderUrl = "$SHERPA_ONNX_TINY/tiny-encoder.int8.onnx",
            approxSizeMb = 103,
        ),
        // small (multilingual) — fp32 + int8
        onnx(
            id = "small-fp32",
            displayName = "Multilingual · High accuracy",
            languageMode = LanguageMode.Auto,
            precision = ModelPrecision.FP32,
            encoderUrl = "$SHERPA_ONNX_SMALL/small-encoder.onnx",
            approxSizeMb = 969,
        ),
        onnx(
            id = "small-int8",
            displayName = "Multilingual · High accuracy",
            languageMode = LanguageMode.Auto,
            precision = ModelPrecision.INT8,
            encoderUrl = "$SHERPA_ONNX_SMALL/small-encoder.int8.onnx",
            approxSizeMb = 375,
        ),
        // small.en (English) — fp32 + int8
        onnx(
            id = "small.en-fp32",
            displayName = "English · High accuracy",
            languageMode = LanguageMode.English,
            precision = ModelPrecision.FP32,
            encoderUrl = "$SHERPA_ONNX_SMALL_EN/small.en-encoder.onnx",
            approxSizeMb = 969,
        ),
        onnx(
            id = "small.en-int8",
            displayName = "English · High accuracy",
            languageMode = LanguageMode.English,
            precision = ModelPrecision.INT8,
            encoderUrl = "$SHERPA_ONNX_SMALL_EN/small.en-encoder.int8.onnx",
            approxSizeMb = 375,
        ),
        // Roman-Urdu (fine-tune of whisper-small) — fp32 + int8
        onnx(
            id = "roman-urdu-fp32",
            displayName = "Roman-Urdu · High accuracy",
            languageMode = LanguageMode.RomanUrdu,
            precision = ModelPrecision.FP32,
            encoderUrl = "$RU_HF/roman-urdu-encoder.onnx",
            approxSizeMb = 925,
        ),
        onnx(
            id = "roman-urdu-int8",
            displayName = "Roman-Urdu",
            languageMode = LanguageMode.RomanUrdu,
            precision = ModelPrecision.INT8,
            encoderUrl = "$RU_HF/roman-urdu-encoder.int8.onnx",
            approxSizeMb = 359,
        ),
    )

    /**
     * Dolphin CTC multilingual catalog (DataoceanAI). Each size is offered in two
     * precision tiers like Whisper, but the files are a single `model.onnx` /
     * `model.int8.onnx` plus `tokens.txt` (no encoder/decoder split). The fileName
     * is the model file; the tokens file is derived as `<id>-tokens.txt` by
     * [DolphinCtcEngine]/[ModelDownloader].
     */
    val dolphinCtcModels: List<CatalogEntry> = listOf(
        dolphinCtc(
            id = "dolphin-base-int8",
            displayName = "Dolphin base · int8",
            precision = ModelPrecision.INT8,
            modelUrl = "$SHERPA_ONNX_DOLPHIN_BASE_INT8/model.int8.onnx",
            approxSizeMb = 99,
        ),
        dolphinCtc(
            id = "dolphin-base-fp32",
            displayName = "Dolphin base · fp32",
            precision = ModelPrecision.FP32,
            modelUrl = "$SHERPA_ONNX_DOLPHIN_BASE/model.onnx",
            approxSizeMb = 303,
        ),
        dolphinCtc(
            id = "dolphin-small-int8",
            displayName = "Dolphin small · int8",
            precision = ModelPrecision.INT8,
            modelUrl = "$SHERPA_ONNX_DOLPHIN_SMALL_INT8/model.int8.onnx",
            approxSizeMb = 239,
        ),
        dolphinCtc(
            id = "dolphin-small-fp32",
            displayName = "Dolphin small · fp32",
            precision = ModelPrecision.FP32,
            modelUrl = "$SHERPA_ONNX_DOLPHIN_SMALL/model.onnx",
            approxSizeMb = 783,
        ),
    )

    /**
     * Omnilingual (Meta OmniASR CTC, sherpa-onnx port) catalog. A single
     * `model.int8.onnx` + `tokens.txt` (the 300M int8 tier, downloaded ~348 MB,
     * CPU RTF ~0.23). The fileName is the model file; the tokens file is derived
     * as `<id>-tokens.txt` by [OmnilingualEngine]/[ModelDownloader]. Auto-detects
     * the language per utterance; sherpa-onnx cannot pin an output language, so
     * the language setting is ignored (see [supportsLanguage]).
     */
    val omnilingualModels: List<CatalogEntry> = listOf(
        omnilingual(
            id = "omnilingual-300m-int8",
            displayName = "Omnilingual · 1600+ languages",
            modelUrl = "$OMNILINGUAL_HF/model.int8.onnx",
            approxSizeMb = 348,
        ),
    )

    private fun omnilingual(
        id: String,
        displayName: String,
        modelUrl: String,
        approxSizeMb: Long,
    ): CatalogEntry = CatalogEntry(
        // No way to pin the output language in sherpa-onnx, so it is always Auto
        // and the user language selection is not applied (the engine ignores it).
        model = Model(id = id, fileName = "$id-model.onnx", languageMode = LanguageMode.Auto),
        displayName = displayName,
        sourceUrl = null,
        onnxSourceUrl = modelUrl,
        approxSizeMb = approxSizeMb,
        isDefault = false,
        precision = ModelPrecision.INT8,
    )

    /**
     * Dolphin attention ASR catalog (DataoceanAI, exported by DakeQQ). These are
     * ONNX encoder + decoder.onnx pairs (the decoder is graph-surgeried to expose
     * the full logits output) + a shared `units.txt` vocabulary, loaded directly
     * through onnxruntime-android by [DolphinAttnEngine]. Decoding honors an
     * explicit `ur`/`PK` language pin, so unlike the CTC family these respect
     * the app's language setting for Urdu. Ships two precision tiers: fp16/arm
     * (dolphin-attn/) and fp32→int8 quantized (dolphin-attn-int8/). The decoder
     * filename is `<id>-decoder.onnx` and the shared vocab is `dolphin-attn-units.txt`.
     */
    val dolphinAttnModels: List<CatalogEntry> = listOf(
        dolphinAttn(
            id = "dolphin-attn-base",
            displayName = "Dolphin · Balanced",
            encoderUrl = "$DOLPHIN_ATTN_HF/base/encoder.onnx",
            approxSizeMb = 250,
        ),
        dolphinAttn(
            id = "dolphin-attn-small",
            displayName = "Dolphin · High accuracy",
            encoderUrl = "$DOLPHIN_ATTN_HF/small/encoder.onnx",
            approxSizeMb = 699,
        ),
        CatalogEntry(
            model = Model(id = "dolphin-attn-int8-base", fileName = "dolphin-attn-int8-base-encoder.onnx", languageMode = LanguageMode.Auto),
            displayName = "Dolphin · Balanced",
            sourceUrl = null,
            onnxSourceUrl = "$DOLPHIN_ATTN_INT8_HF/base/encoder.onnx",
            approxSizeMb = 263,
            isDefault = false,
            precision = ModelPrecision.INT8,
        ),
        CatalogEntry(
            model = Model(id = "dolphin-attn-int8-small", fileName = "dolphin-attn-int8-small-encoder.onnx", languageMode = LanguageMode.Auto),
            displayName = "Dolphin · High accuracy",
            sourceUrl = null,
            onnxSourceUrl = "$DOLPHIN_ATTN_INT8_HF/small/encoder.onnx",
            approxSizeMb = 748,
            isDefault = false,
            precision = ModelPrecision.INT8,
        ),
    )

    /** Build one Dolphin CTC [CatalogEntry] with a unique model + tokens filename. */
    private fun dolphinCtc(
        id: String,
        displayName: String,
        precision: ModelPrecision,
        modelUrl: String,
        approxSizeMb: Long,
    ): CatalogEntry = CatalogEntry(
        // Dolphin auto-detects language in sherpa-onnx, so it is always Auto and
        // the user language selection is not applied (the engine ignores it).
        model = Model(id = id, fileName = "$id-model.onnx", languageMode = LanguageMode.Auto),
        displayName = displayName,
        sourceUrl = null,
        onnxSourceUrl = modelUrl,
        approxSizeMb = approxSizeMb,
        isDefault = false,
        precision = precision,
    )

    /**
     * Build one Dolphin attention [CatalogEntry]. The language mode is Auto and
     * [Model.canOverrideLanguage] is true so the user's language selection is
     * honored: the engine uses it to drive the `ur`/`PK` pin (unlike CTC).
     */
    private fun dolphinAttn(
        id: String,
        displayName: String,
        encoderUrl: String,
        approxSizeMb: Long,
    ): CatalogEntry = CatalogEntry(
        model = Model(id = id, fileName = "$id-encoder.onnx", languageMode = LanguageMode.Auto),
        displayName = displayName,
        sourceUrl = null,
        onnxSourceUrl = encoderUrl,
        approxSizeMb = approxSizeMb,
        isDefault = false,
        precision = ModelPrecision.FP16,
    )

    /** Build one ONNX [CatalogEntry] with its tier-correct encoder filename. */
    private fun onnx(
        id: String,
        displayName: String,
        languageMode: LanguageMode,
        precision: ModelPrecision,
        encoderUrl: String,
        approxSizeMb: Long,
    ): CatalogEntry {
        val encoderFileName =
            if (precision == ModelPrecision.INT8) "$id-encoder.int8.onnx" else "$id-encoder.onnx"
        return CatalogEntry(
            model = Model(id = id, fileName = encoderFileName, languageMode = languageMode),
            displayName = displayName,
            sourceUrl = null,
            onnxSourceUrl = encoderUrl,
            approxSizeMb = approxSizeMb,
            isDefault = false,
            precision = precision,
        )
    }

    /**
     * The GGML catalog's default (Roman-Urdu q4_0, the spec default). GGML models
     * are currently disabled — retained in code but never offered to the user —
     * so this is not what the app starts with (see [onnxDefault]).
     */
    val ggmlDefault: CatalogEntry
        get() = ggmlModels.first { it.isDefault }

    /**
     * Whether an ONNX Whisper entry is offered in the picker: English and
     * multilingual models run their int8 tier; Roman-Urdu ships both tiers.
     */
    private fun isEnabledOnnx(entry: CatalogEntry): Boolean =
        entry.precision != ModelPrecision.FP32 || isRomanUrdu(entry.model.id)

    /** The ONNX entries the picker offers (see [isEnabledOnnx]). */
    val onnxModelsActive: List<CatalogEntry>
        get() = onnxModels.filter { isEnabledOnnx(it) }

    /** English/multilingual fp32 tiers — retained in code but never offered. */
    val onnxModelsDisabled: List<CatalogEntry>
        get() = onnxModels.filterNot { isEnabledOnnx(it) }

    /**
     * The catalog the picker shows and the app runs: every ONNX-backed model
     * family that is still offered. Disabled families are retained but never
     * offered: GGML (see [ggmlModels]/[ggmlDefault]), Dolphin CTC
     * (see [dolphinCtcModels]), and the English/multilingual fp32 tiers
     * (see [onnxModelsDisabled]). Dolphin attention and Omnilingual stay enabled.
     */
    val activeCatalog: List<CatalogEntry>
        get() = onnxModelsActive + omnilingualModels + dolphinAttnModels

    /**
     * The app's default model: the Roman-Urdu int8 ONNX conversion, the ONNX
     * successor of the disabled GGML default (`roman-urdu-q4_0`). Quantized for a
     * reasonable size while keeping accurate Urdu dictation.
     */
    val onnxDefault: CatalogEntry
        get() = byActiveId("roman-urdu-int8") ?: error("roman-urdu-int8 must resolve")

    /** Look up by model id across all five catalogs. */
    fun byId(id: String): CatalogEntry? =
        (ggmlModels + onnxModels + dolphinCtcModels + omnilingualModels + dolphinAttnModels).firstOrNull { it.model.id == id }

    /** Look up an active (ONNX-backed) catalog entry by id, or null. */
    fun byActiveId(id: String): CatalogEntry? = activeCatalog.firstOrNull { it.model.id == id }

    /** Whether [entry] is a Dolphin CTC model (loaded by [DolphinCtcEngine]). */
    fun isDolphinCtc(entry: CatalogEntry): Boolean =
        entry in dolphinCtcModels

    /** Whether the model file [fileName] belongs to a Dolphin CTC catalog entry. */
    fun isDolphinCtcFileName(fileName: String): Boolean =
        dolphinCtcModels.any { it.model.fileName == fileName }

    /**
     * Whether [entry] is a Dolphin attention model (loaded by [DolphinAttnEngine]
     * through onnxruntime-android, honoring the ur/PK pin).
     */
    fun isDolphinAttn(entry: CatalogEntry): Boolean =
        entry in dolphinAttnModels

    /** Whether the model file [fileName] belongs to a Dolphin attention catalog entry. */
    fun isDolphinAttnFileName(fileName: String): Boolean =
        dolphinAttnModels.any { it.model.fileName == fileName }

    /** Whether [entry] is an Omnilingual model (loaded by [OmnilingualEngine]). */
    fun isOmnilingual(entry: CatalogEntry): Boolean =
        entry in omnilingualModels

    /** Whether the model file [fileName] belongs to an Omnilingual catalog entry. */
    fun isOmnilingualFileName(fileName: String): Boolean =
        omnilingualModels.any { it.model.fileName == fileName }
}

/**
 * A technical engine+size name for the picker subtext, e.g. `whisper tiny.en`,
 * `whisper roman-urdu`, `dolphin base`, `omnilingual 300m`. Whisper ids carry a
 * `-fp32`/`-int8` precision suffix that is stripped; Dolphin ids carry an
 * `-attn-`/`-int8-` infix, so the engine prefix is applied explicitly.
 */
val CatalogEntry.modelName: String
    get() =
        when {
            model.id.startsWith("dolphin-attn") -> "dolphin " + model.id
                .removePrefix("dolphin-attn-int8-")
                .removePrefix("dolphin-attn-")
            ModelCatalog.isOmnilingual(this) ->
                "omnilingual " + model.id.removeSuffix("-int8").removePrefix("omnilingual-")
            else -> "whisper " + model.id.removeSuffix("-fp32").removeSuffix("-int8")
        }

/**
 * Whether this catalog entry can transcribe [code] (a [WhisperLanguages] code).
 * Drives the picker's language filter and first-run onboarding:
 * - Dolphin attention models support their [ModelCatalog.DOLPHIN_LANGUAGES].
 * - Omnilingual auto-detects from 1600+ zero-shot languages (no way to pin an
 *   output language in sherpa-onnx), so it passes every [WhisperLanguages] code.
 * - Roman-Urdu models transcribe Urdu (roman/Latin script), so `ur` only.
 * - English-only whisper models transcribe `en` only.
 * - Multilingual whisper models support every [WhisperLanguages] entry.
 */
fun CatalogEntry.supportsLanguage(code: String): Boolean = when {
    ModelCatalog.isDolphinAttn(this) -> code in ModelCatalog.DOLPHIN_LANGUAGES
    ModelCatalog.isOmnilingual(this) -> WhisperLanguages.supports(code)
    model.languageMode == LanguageMode.RomanUrdu -> code == "ur"
    model.languageMode == LanguageMode.English -> code == "en"
    else -> WhisperLanguages.supports(code)
}

/**
 * Filter a catalog to the entries that support [code]. A null [code] (the
 * user chose Auto-detect) shows the whole list unfiltered.
 */
fun List<CatalogEntry>.filterForLanguage(code: String?): List<CatalogEntry> =
    if (code == null) this else filter { it.supportsLanguage(code) }