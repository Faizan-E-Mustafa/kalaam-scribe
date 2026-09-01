package dev.femustafa.voicedictation

/**
 * The Model catalog from the spec (spec.Notes "Model catalog"): each entry is a
 * GGML file plus its language mode, its public source, approximate size, and a
 * default flag. This is the ModelCatalog unit seam, so it is pure Kotlin (no
 * Android/device) and unit-tested.
 *
 * Sources:
 * - 1–2 Roman-Urdu are conversions of `cheetos18/whisper-small-roman-urdu`
 *   hosted on the project's own HuggingFace repo (see [RU_HF]).
 * - 3–6 are downloadable from the `ggerganov/whisper.cpp` HuggingFace repo.
 */
data class CatalogEntry(
    /** The domain [Model] this entry selects (id, fileName, languageMode). */
    val model: Model,
    /** Human-friendly name shown in the picker. */
    val displayName: String,
    /** Public download URL, or null when the file is not publicly hosted. */
    val sourceUrl: String?,
    /** Approximate size in MiB (0 = unknown). */
    val approxSizeMb: Long,
    /** Whether this is the default Model on first launch. */
    val isDefault: Boolean,
)

object ModelCatalog {

    const val HF_WHISPER_CPP = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
    const val RU_HF = "https://huggingface.co/femustafa/voicedictation-models/resolve/main"

    val models: List<CatalogEntry> = listOf(
        CatalogEntry(
            model = Model(id = "roman-urdu-q4_0", fileName = "ggml-model-q4_0.bin", languageMode = LanguageMode.RomanUrdu),
            displayName = "Roman-Urdu q4_0",
            sourceUrl = "$RU_HF/ggml-model-q4_0.bin",
            approxSizeMb = 139,
            isDefault = true,
        ),
        CatalogEntry(
            model = Model(id = "roman-urdu-f16", fileName = "ggml-model-f16.bin", languageMode = LanguageMode.RomanUrdu),
            displayName = "Roman-Urdu full (f16)",
            sourceUrl = "$RU_HF/ggml-model-f16.bin",
            approxSizeMb = 550,
            isDefault = false,
        ),
        CatalogEntry(
            model = Model(id = "english-full", fileName = "ggml-base.en.bin", languageMode = LanguageMode.English),
            displayName = "English full",
            sourceUrl = "$HF_WHISPER_CPP/ggml-base.en.bin",
            approxSizeMb = 148,
            isDefault = false,
        ),
        CatalogEntry(
            model = Model(id = "english-q8", fileName = "ggml-base.en-q8_0.bin", languageMode = LanguageMode.English),
            displayName = "English quantized (q8_0)",
            sourceUrl = "$HF_WHISPER_CPP/ggml-base.en-q8_0.bin",
            approxSizeMb = 82,
            isDefault = false,
        ),
        CatalogEntry(
            model = Model(id = "multilingual-small-q8", fileName = "ggml-small-q8_0.bin", languageMode = LanguageMode.Auto),
            displayName = "Multilingual small (q8_0)",
            sourceUrl = "$HF_WHISPER_CPP/ggml-small-q8_0.bin",
            approxSizeMb = 264,
            isDefault = false,
        ),
        CatalogEntry(
            model = Model(id = "multilingual-tiny", fileName = "ggml-tiny.bin", languageMode = LanguageMode.Auto),
            displayName = "Multilingual tiny (full)",
            sourceUrl = "$HF_WHISPER_CPP/ggml-tiny.bin",
            approxSizeMb = 77,
            isDefault = false,
        ),
        CatalogEntry(
            model = Model(id = "english-tiny", fileName = "ggml-tiny.en.bin", languageMode = LanguageMode.English),
            displayName = "English tiny (tiny.en)",
            sourceUrl = "$HF_WHISPER_CPP/ggml-tiny.en.bin",
            approxSizeMb = 75,
            isDefault = false,
        ),
        CatalogEntry(
            model = Model(id = "english-tiny-q8", fileName = "ggml-tiny.en-q8_0.bin", languageMode = LanguageMode.English),
            displayName = "English tiny q8_0",
            sourceUrl = "$HF_WHISPER_CPP/ggml-tiny.en-q8_0.bin",
            approxSizeMb = 42,
            isDefault = false,
        ),
        CatalogEntry(
            model = Model(id = "tiny-q5", fileName = "ggml-tiny-q5_1.bin", languageMode = LanguageMode.Auto),
            displayName = "Tiny q5_1 (multilingual)",
            sourceUrl = "$HF_WHISPER_CPP/ggml-tiny-q5_1.bin",
            approxSizeMb = 32,
            isDefault = false,
        ),
        CatalogEntry(
            model = Model(id = "english-tiny-q5", fileName = "ggml-tiny.en-q5_1.bin", languageMode = LanguageMode.English),
            displayName = "English tiny q5_1",
            sourceUrl = "$HF_WHISPER_CPP/ggml-tiny.en-q5_1.bin",
            approxSizeMb = 32,
            isDefault = false,
        ),
    )

    /** The default Model (Roman-Urdu q4_0, the spec default). */
    val default: CatalogEntry
        get() = models.first { it.isDefault }

    fun byId(id: String): CatalogEntry? = models.firstOrNull { it.model.id == id }
}
