package dev.femustafa.kalaamscribe

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for the domain [Model]/[WhisperLanguages] language-selection rules:
 * the supported code set and which Models may accept a user language override.
 */
class ModelTest {

    @Test
    fun whisperLanguagesHasAllSupportedCodes() {
        // whisper's LANGUAGES table (openai/whisper tokenizer.py) has 100 codes,
        // plus the synthetic (non-engine) "Urdu (Roman)" entry.
        assertEquals(101, WhisperLanguages.entries.size)
        // A few representative entries present.
        assertEquals("English", WhisperLanguages.nameOf("en"))
        assertEquals("German", WhisperLanguages.nameOf("de"))
        assertEquals("Hindi", WhisperLanguages.nameOf("hi"))
        assertEquals("Urdu", WhisperLanguages.nameOf("ur"))
    }

    @Test
    fun urduRomanIsSyntheticAndSortsNextToUrdu() {
        assertNull(WhisperLanguages.nameOf("xx"))
        // Synthetic entry exists for the dropdowns and names...
        assertEquals("Urdu (Roman)", WhisperLanguages.nameOf(WhisperLanguages.URDU_ROMAN))
        // ...but it is not a real whisper engine code (multilingual/omnilingual
        // models must not claim it, and it must never reach an engine).
        assertFalse(WhisperLanguages.supports(WhisperLanguages.URDU_ROMAN))
        // Sorts alphabetically right after "Urdu".
        val urdu = WhisperLanguages.entries.indexOfFirst { it.first == "ur" }
        val roman = WhisperLanguages.entries.indexOfFirst { it.first == WhisperLanguages.URDU_ROMAN }
        assertEquals(urdu + 1, roman)
    }

    @Test
    fun supportsCodes() {
        assertTrue(WhisperLanguages.supports("en"))
        assertTrue(WhisperLanguages.supports("de"))
        assertFalse(WhisperLanguages.supports("xx"))
        assertNull(WhisperLanguages.nameOf("xx"))
    }

    @Test
    fun multilingualModelsCanOverrideLanguage() {
        val multilingual = Model(id = "multilingual-tiny", fileName = "ggml-tiny.bin", languageMode = LanguageMode.Auto)
        val multilingualSmall = Model(id = "multilingual-small-q8", fileName = "ggml-small-q8_0.bin", languageMode = LanguageMode.Auto)
        assertTrue(multilingual.canOverrideLanguage)
        assertTrue(multilingualSmall.canOverrideLanguage)
    }

    @Test
    fun dolphinCtcModelsCanOverrideLanguageFlagButIgnoreIt() {
        // Dolphin CTC is LanguageMode.Auto, so at the Model level the override flag
        // is on — but the engine ignores the passed language (auto-detect only).
        for (e in ModelCatalog.dolphinCtcModels) {
            assertTrue(e.model.canOverrideLanguage)
            assertEquals(LanguageMode.Auto, e.model.languageMode)
        }
    }

    @Test
    fun englishAndRomanUrduCannotOverrideLanguage() {
        val english = Model(id = "english-q8", fileName = "ggml-base.en-q8_0.bin", languageMode = LanguageMode.English)
        val romanUrduQ4 = Model(id = "roman-urdu-q4_0", fileName = "ggml-model-q4_0.bin", languageMode = LanguageMode.RomanUrdu)
        val romanUrduF16 = Model(id = "roman-urdu-f16", fileName = "ggml-model-f16.bin", languageMode = LanguageMode.RomanUrdu)
        val romanUrduOnnx = Model(id = "roman-urdu-fp32", fileName = "roman-urdu-fp32-encoder.onnx", languageMode = LanguageMode.RomanUrdu)
        assertFalse(english.canOverrideLanguage)
        assertFalse(romanUrduQ4.canOverrideLanguage)
        assertFalse(romanUrduF16.canOverrideLanguage)
        assertFalse(romanUrduOnnx.canOverrideLanguage)
    }

    @Test
    fun isRomanUrduMatchesRomanUrduIds() {
        assertTrue(isRomanUrdu("roman-urdu-q4_0"))
        assertTrue(isRomanUrdu("roman-urdu-f16"))
        assertTrue(isRomanUrdu("roman-urdu-fp32"))
        assertTrue(isRomanUrdu("roman-urdu-int8"))
        assertFalse(isRomanUrdu("multilingual-tiny"))
        assertFalse(isRomanUrdu("english-q8"))
    }
}
