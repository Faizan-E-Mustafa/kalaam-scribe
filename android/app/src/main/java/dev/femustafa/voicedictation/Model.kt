package dev.femustafa.voicedictation

/**
 * How a [Model] should be transcribed: which language whisper is told to use.
 *
 * Maps to the whisper AAR's `WhisperConfig.language` string:
 * - [Auto] -> `"auto"` (whisper auto-detects). Required for Roman-Urdu so it stays
 *   in Roman/Latin script; do NOT force `ur` (see CONTEXT.md language).
 * - [English] -> `"en"` (fixed English, faster/more stable for English-only Models).
 */
enum class LanguageMode {
    Auto,
    English,
}

/**
 * A downloadable ASR model: a GGML file plus the language mode it runs in.
 * A model is selected/switched independently of any dictation (CONTEXT.md "Model").
 */
data class Model(
    val id: String,
    val fileName: String,
    val languageMode: LanguageMode,
)
