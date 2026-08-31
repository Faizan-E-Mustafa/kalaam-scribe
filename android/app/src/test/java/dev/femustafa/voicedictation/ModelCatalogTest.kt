package dev.femustafa.voicedictation

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Unit tests for the ModelCatalog unit seam (spec "ModelCatalog maps six Models"). */
class ModelCatalogTest {

    @Test
    fun hasSevenModels() {
        assertEquals(7, ModelCatalog.models.size)
    }

    @Test
    fun defaultIsEnglishQ8() {
        val d = ModelCatalog.default
        assertEquals("english-q8", d.model.id)
        assertTrue(d.isDefault)
        assertEquals("ggml-base.en-q8_0.bin", d.model.fileName)
        assertEquals(LanguageMode.English, d.model.languageMode)
    }

    @Test
    fun romanUrduModelsAreAutoDetectAndNotPubliclyHosted() {
        val ruQ8 = ModelCatalog.byId("roman-urdu-q8")!!
        val ruF16 = ModelCatalog.byId("roman-urdu-f16")!!

        // Roman-Urdu must stay in Roman/Latin script -> auto-detect, never `ur`.
        assertEquals(LanguageMode.Auto, ruQ8.model.languageMode)
        assertEquals(LanguageMode.Auto, ruF16.model.languageMode)
        // Locally converted, not downloadable.
        assertNull(ruQ8.sourceUrl)
        assertNull(ruF16.sourceUrl)
    }

    @Test
    fun englishModelsAreFixedEn() {
        assertEquals(LanguageMode.English, ModelCatalog.byId("english-full")!!.model.languageMode)
        assertEquals(LanguageMode.English, ModelCatalog.byId("english-q8")!!.model.languageMode)
        assertEquals("ggml-base.en.bin", ModelCatalog.byId("english-full")!!.model.fileName)
        assertEquals("ggml-base.en-q8_0.bin", ModelCatalog.byId("english-q8")!!.model.fileName)
    }

    @Test
    fun publicModelsPointAtWhisperCppHF() {
        val publicOnes = ModelCatalog.models.filter { it.sourceUrl != null }
        // 5 of the 7 are publicly hosted (2 Roman-Urdu are local).
        assertEquals(5, publicOnes.size)
        for (e in publicOnes) {
            assertTrue(e.sourceUrl!!.startsWith(ModelCatalog.HF_WHISPER_CPP))
            assertTrue(e.sourceUrl!!.endsWith(e.model.fileName))
        }
        // default is now the hosted English q8 (will be Roman-Urdu again after upload)
        assertTrue(ModelCatalog.default.sourceUrl != null)
    }

    @Test
    fun everyEntryHasUniqueId() {
        val ids = ModelCatalog.models.map { it.model.id }.toSet()
        assertEquals(ModelCatalog.models.size, ids.size)
    }
}
