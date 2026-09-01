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
 * A downloadable ASR model: a GGML file plus the language mode it runs in.
 * A model is selected/switched independently of any dictation (CONTEXT.md "Model").
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
    modelId == "roman-urdu-q4_0" || modelId == "roman-urdu-f16"

/**
 * The languages the whisper AAR supports for `WhisperConfig.language`.
 * Mirrors `openai/whisper` `tokenizer.py` `LANGUAGES` (99 entries): each is a
 * 2-letter code plus its English name. Use these codes directly; a null/blank
 * selection means auto-detect (`"auto"`).
 */
object WhisperLanguages {
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
    )

    private val byCode: Map<String, String> = entries.toMap()

    fun nameOf(code: String): String? = byCode[code]

    fun supports(code: String): Boolean = byCode.containsKey(code)
}
