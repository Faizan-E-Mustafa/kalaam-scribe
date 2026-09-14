package dev.femustafa.voicedictation

/**
 * How a [Model] should be transcribed: which language whisper is told to use.
 *
 * Maps to the whisper AAR's `WhisperConfig.language` string:
 * - [Auto] -> `"auto"` (whisper auto-detects).
 * - [English] -> `"en"` (fixed English, faster/more stable for English-only Models).
 * - [RomanUrdu] -> `"en"` (fixed English). Skips whisper's language-detection pass
 *   (~40% lower latency) but can leak English output on short clips; the seed
 *   prompt that counters this needs the engine-binding build (see issue tracker).
 *   Never `ur`, which leaks native-script tokens (see CONTEXT.md language).
 */
enum class LanguageMode {
    Auto,
    English,
    RomanUrdu,
}

/**
 * The model file format: GGML (whisper.cpp .bin, used by [AarWhisperEngine]) or
 * ONNX (sherpa-onnx .onnx + .tokens, used by [SherpaWhisperEngine]). GGML is
 * currently disabled: the app always runs ONNX (see [VoiceDictationApp.modelFormat]),
 * with the GGML branch retained in code but never offered.
 */
enum class ModelFormat {
    GGML,
    ONNX,
}

/**
 * How dictation turns captured audio into text.
 * - [Batch] decodes the entire clip as one pass after Stop: no text until the
 *   recording ends, simplest and works with every backend (including GGML).
 * - [SimulatedStreaming] opens a [TranscriptionSession]: live VAD splits audio
 *   into utterances that decode while still recording, growing the transcript
 *   in real time. Requires a streaming-capable backend; GGML falls back to batch.
 */
enum class TranscriptionMode {
    Batch,
    SimulatedStreaming,
}

/**
 * Numerical precision of an ONNX model file. sherpa-onnx auto-detects quantized
 * (int8) models from `int8` in the filename, so this only chooses which file to
 * point at: full-precision fp32, half-precision fp16 (fp16/arm tier), or
 * quantized int8.
 */
enum class ModelPrecision {
    FP32,
    FP16,
    INT8,
}

/** Human label for a precision tier, used in the model picker meta line. */
val ModelPrecision.label: String
    get() = when (this) {
        ModelPrecision.FP32 -> "fp32"
        ModelPrecision.FP16 -> "fp16"
        ModelPrecision.INT8 -> "int8"
    }

/**
 * A downloadable ASR model: a GGML/ONNX file plus the language mode it runs in.
 * A model is selected/switched independently of any dictation (CONTEXT.md "Model").
 * The file format (GGML vs ONNX) is chosen app-wide, not per model.
 */
data class Model(
    val id: String,
    val fileName: String,
    val languageMode: LanguageMode,
) {
    /**
     * Whether a user-picked language may override transcription for this Model.
     * Only allows this for multilingual (non-English) Models that are not one of
     * the Roman-Urdu conversions: Roman-Urdu runs a fixed `en` (never `ur`, which
     * leaks native-script tokens; see [LanguageMode.RomanUrdu]), and English-only
     * Models are fixed `en` (CONTEXT.md "Model" / language).
     */
    val canOverrideLanguage: Boolean
        get() = languageMode == LanguageMode.Auto && !isRomanUrdu(id)
}

/** The Roman-Urdu conversion Model ids. Their output must stay Roman/Latin (`en`), never `ur`. */
fun isRomanUrdu(modelId: String): Boolean =
    modelId == "roman-urdu-q4_0" || modelId == "roman-urdu-f16" ||
        modelId == "roman-urdu-fp32" || modelId == "roman-urdu-int8"

/**
 * The languages shown in the language dropdowns. Mirrors the whisper AAR's
 * `WhisperConfig.language` codes from `openai/whisper` `tokenizer.py` `LANGUAGES`
 * (99 entries): each is a 2-letter code plus its English name. Use these codes
 * directly; a null/blank selection means auto-detect (`"auto"`).
 *
 * One extra, synthetic entry is appended: [URDU_ROMAN] ("Urdu (Roman)"). It is
 * NOT a real whisper code — it only selects the Roman-Urdu catalog in the
 * dropdowns, never reaches an engine (Roman-Urdu models run their fixed-English
 * [LanguageMode.RomanUrdu] mode), and is excluded from [supports] so multilingual
 * /omnilingual models don't claim it. It still appears in [entries]/[nameOf] and
 * sorts next to "Urdu".
 */
object WhisperLanguages {
    /** Synthetic code for Roman-script Urdu; see the [WhisperLanguages] doc. */
    const val URDU_ROMAN = "ur-roman"

    /** Alphabetical by English name; used to fill the onboarding and picker dropdowns. */
    val entries: List<Pair<String, String>> = listOf(
        "en" to "English", "zh" to "Chinese", "de" to "German", "es" to "Spanish",
        "ru" to "Russian", "ko" to "Korean", "fr" to "French", "ja" to "Japanese",
        "pt" to "Portuguese", "tr" to "Turkish", "pl" to "Polish", "ca" to "Catalan",
        "nl" to "Dutch", "ar" to "Arabic", "sv" to "Swedish", "it" to "Italian",
        "id" to "Indonesian", "hi" to "Hindi", "fi" to "Finnish", "vi" to "Vietnamese",
        "he" to "Hebrew", "uk" to "Ukrainian", "el" to "Greek", "ms" to "Malay",
        "cs" to "Czech", "ro" to "Romanian", "da" to "Danish", "hu" to "Hungarian",
        "ta" to "Tamil", "no" to "Norwegian", "th" to "Thai", "ur" to "Urdu",
        "hr" to "Croatian", "bg" to "Bulgarian", "lt" to "Lithuanian", "la" to "Latin",
        "mi" to "Maori", "ml" to "Malayalam", "cy" to "Welsh", "sk" to "Slovak",
        "te" to "Telugu", "fa" to "Persian", "lv" to "Latvian", "bn" to "Bengali",
        "sr" to "Serbian", "az" to "Azerbaijani", "sl" to "Slovenian", "kn" to "Kannada",
        "et" to "Estonian", "mk" to "Macedonian", "br" to "Breton", "eu" to "Basque",
        "is" to "Icelandic", "hy" to "Armenian", "ne" to "Nepali", "mn" to "Mongolian",
        "bs" to "Bosnian", "kk" to "Kazakh", "sq" to "Albanian", "sw" to "Swahili",
        "gl" to "Galician", "mr" to "Marathi", "pa" to "Punjabi", "si" to "Sinhala",
        "km" to "Khmer", "sn" to "Shona", "yo" to "Yoruba", "so" to "Somali",
        "af" to "Afrikaans", "oc" to "Occitan", "ka" to "Georgian", "be" to "Belarusian",
        "tg" to "Tajik", "sd" to "Sindhi", "gu" to "Gujarati", "am" to "Amharic",
        "yi" to "Yiddish", "lo" to "Lao", "uz" to "Uzbek", "fo" to "Faroese",
        "ht" to "Haitian creole", "ps" to "Pashto", "tk" to "Turkmen", "nn" to "Nynorsk",
        "mt" to "Maltese", "sa" to "Sanskrit", "lb" to "Luxembourgish", "my" to "Myanmar",
        "bo" to "Tibetan", "tl" to "Tagalog", "mg" to "Malagasy", "as" to "Assamese",
        "tt" to "Tatar", "haw" to "Hawaiian", "ln" to "Lingala", "ha" to "Hausa",
        "ba" to "Bashkir", "jw" to "Javanese", "su" to "Sundanese", "yue" to "Cantonese",
        URDU_ROMAN to "Urdu (Roman)",
    ).sortedBy { it.second }

    private val byCode: Map<String, String> = entries.toMap()

    fun nameOf(code: String): String? = byCode[code]

    /** Whether [code] is a real whisper engine language code (not the synthetic [URDU_ROMAN]). */
    fun supports(code: String): Boolean = byCode.containsKey(code) && code != URDU_ROMAN
}
