package dev.femustafa.voicedictation

/**
 * The Model catalogs from the spec (spec.Notes "Model catalog"). Two separate
 * catalogs, one per file [ModelFormat], because they are independent model sets
 * (CONTEXT.md "Model format"):
 *
 * - [ggmlModels] — quantized whisper.cpp GGML `.bin` models (q4_0, q8_0, q5_1…),
 *   loaded by [AarWhisperEngine].
 * - [onnxModels] — sherpa-onnx ONNX models. Organized by ONNX model identity
 *   (`tiny`, `tiny.en`, `base.en`, `small`), each offered in two precision tiers:
 *   [ModelPrecision.FP32] (full) and [ModelPrecision.INT8] (quantized), loaded by
 *   [SherpaWhisperEngine]. Quantization is a GGML concept on the GGML side; on the
 *   ONNX side we offer int8 as a distinct tier.
 *
 * Sources:
 * - Roman-Urdu are conversions of `cheetos18/whisper-small-roman-urdu` hosted on
 *   the project's own HuggingFace repo (see [RU_HF]). GGML-only; no ONNX export.
 * - GGML English/multilingual from `ggerganov/whisper.cpp` HuggingFace repo.
 * - ONNX from `csukuangfj/sherpa-onnx-whisper-*` HuggingFace repos (fp32 `.onnx`
 *   and int8 `.int8.onnx` files).
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
    const val SHERPA_ONNX_SMALL = "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small/resolve/main"

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
     */
    val onnxModels: List<CatalogEntry> = listOf(
        // base.en (English) — fp32 + int8
        onnx(
            id = "base.en-fp32",
            displayName = "English base",
            languageMode = LanguageMode.English,
            precision = ModelPrecision.FP32,
            encoderUrl = "$SHERPA_ONNX_BASE_EN/base.en-encoder.onnx",
            approxSizeMb = 291,
        ),
        onnx(
            id = "base.en-int8",
            displayName = "English base (int8)",
            languageMode = LanguageMode.English,
            precision = ModelPrecision.INT8,
            encoderUrl = "$SHERPA_ONNX_BASE_EN/base.en-encoder.int8.onnx",
            approxSizeMb = 160,
        ),
        // tiny.en (English) — fp32 + int8
        onnx(
            id = "tiny.en-fp32",
            displayName = "English tiny (tiny.en)",
            languageMode = LanguageMode.English,
            precision = ModelPrecision.FP32,
            encoderUrl = "$SHERPA_ONNX_TINY_EN/tiny.en-encoder.onnx",
            approxSizeMb = 150,
        ),
        onnx(
            id = "tiny.en-int8",
            displayName = "English tiny (int8)",
            languageMode = LanguageMode.English,
            precision = ModelPrecision.INT8,
            encoderUrl = "$SHERPA_ONNX_TINY_EN/tiny.en-encoder.int8.onnx",
            approxSizeMb = 103,
        ),
        // tiny (multilingual) — fp32 + int8
        onnx(
            id = "tiny-fp32",
            displayName = "Multilingual tiny",
            languageMode = LanguageMode.Auto,
            precision = ModelPrecision.FP32,
            encoderUrl = "$SHERPA_ONNX_TINY/tiny-encoder.onnx",
            approxSizeMb = 152,
        ),
        onnx(
            id = "tiny-int8",
            displayName = "Multilingual tiny (int8)",
            languageMode = LanguageMode.Auto,
            precision = ModelPrecision.INT8,
            encoderUrl = "$SHERPA_ONNX_TINY/tiny-encoder.int8.onnx",
            approxSizeMb = 103,
        ),
        // small (multilingual) — fp32 + int8
        onnx(
            id = "small-fp32",
            displayName = "Multilingual small",
            languageMode = LanguageMode.Auto,
            precision = ModelPrecision.FP32,
            encoderUrl = "$SHERPA_ONNX_SMALL/small-encoder.onnx",
            approxSizeMb = 969,
        ),
        onnx(
            id = "small-int8",
            displayName = "Multilingual small (int8)",
            languageMode = LanguageMode.Auto,
            precision = ModelPrecision.INT8,
            encoderUrl = "$SHERPA_ONNX_SMALL/small-encoder.int8.onnx",
            approxSizeMb = 375,
        ),
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

    /** The default GGML Model (Roman-Urdu q4_0, the spec default). */
    val default: CatalogEntry
        get() = ggmlModels.first { it.isDefault }

    /** Look up by model id across both catalogs. */
    fun byId(id: String): CatalogEntry? =
        (ggmlModels + onnxModels).firstOrNull { it.model.id == id }
}