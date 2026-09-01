package dev.femustafa.voicedictation

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
        // whisper's LANGUAGES table (openai/whisper tokenizer.py) has 100 codes.
        assertEquals(100, WhisperLanguages.entries.size)
        // A few representative entries present.
        assertEquals("English", WhisperLanguages.nameOf("en"))
        assertEquals("German", WhisperLanguages.nameOf("de"))
        assertEquals("Hindi", WhisperLanguages.nameOf("hi"))
        assertEquals("Urdu", WhisperLanguages.nameOf("ur"))
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
    fun englishAndRomanUrduCannotOverrideLanguage() {
        val english = Model(id = "english-q8", fileName = "ggml-base.en-q8_0.bin", languageMode = LanguageMode.English)
        val romanUrduQ8 = Model(id = "roman-urdu-q8", fileName = "ggml-model-q8_0.bin", languageMode = LanguageMode.Auto)
        val romanUrduF16 = Model(id = "roman-urdu-f16", fileName = "ggml-model-f16.bin", languageMode = LanguageMode.Auto)
        assertFalse(english.canOverrideLanguage)
        assertFalse(romanUrduQ8.canOverrideLanguage)
        assertFalse(romanUrduF16.canOverrideLanguage)
    }

    @Test
    fun isRomanUrduMatchesBothConversionIds() {
        assertTrue(isRomanUrdu("roman-urdu-q8"))
        assertTrue(isRomanUrdu("roman-urdu-f16"))
        assertFalse(isRomanUrdu("multilingual-tiny"))
        assertFalse(isRomanUrdu("english-q8"))
    }
}
