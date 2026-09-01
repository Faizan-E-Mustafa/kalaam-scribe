package dev.femustafa.voicedictation

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Unit tests for the ModelCatalog unit seam (spec "ModelCatalog maps six Models"). */
class ModelCatalogTest {

    @Test
    fun hasTenModels() {
        assertEquals(10, ModelCatalog.models.size)
    }

    @Test
    fun defaultIsRomanUrduQ4() {
        val d = ModelCatalog.default
        assertEquals("roman-urdu-q4_0", d.model.id)
        assertTrue(d.isDefault)
        assertEquals("ggml-model-q4_0.bin", d.model.fileName)
        assertEquals(LanguageMode.Auto, d.model.languageMode)
    }

    @Test
    fun romanUrduModelsAreAutoDetectAndHosted() {
        val ruQ4 = ModelCatalog.byId("roman-urdu-q4_0")!!
        val ruF16 = ModelCatalog.byId("roman-urdu-f16")!!

        // Roman-Urdu must stay in Roman/Latin script -> auto-detect, never `ur`.
        assertEquals(LanguageMode.Auto, ruQ4.model.languageMode)
        assertEquals(LanguageMode.Auto, ruF16.model.languageMode)
        // Downloadable from the project's Roman-Urdu HuggingFace repo.
        assertNotNull(ruQ4.sourceUrl)
        assertNotNull(ruF16.sourceUrl)
        assertTrue(ruQ4.sourceUrl!!.startsWith(ModelCatalog.RU_HF))
        assertTrue(ruQ4.sourceUrl!!.endsWith(ruQ4.model.fileName))
        assertTrue(ruF16.sourceUrl!!.endsWith(ruF16.model.fileName))
    }

    @Test
    fun englishModelsAreFixedEn() {
        assertEquals(LanguageMode.English, ModelCatalog.byId("english-full")!!.model.languageMode)
        assertEquals(LanguageMode.English, ModelCatalog.byId("english-q8")!!.model.languageMode)
        assertEquals("ggml-base.en.bin", ModelCatalog.byId("english-full")!!.model.fileName)
        assertEquals("ggml-base.en-q8_0.bin", ModelCatalog.byId("english-q8")!!.model.fileName)
    }

    @Test
    fun publicModelsPointAtTheirRespectiveHF() {
        val publicOnes = ModelCatalog.models.filter { it.sourceUrl != null }
        // All 10 are publicly hosted: 8 from whisper.cpp + 2 Roman-Urdu from RU_HF.
        assertEquals(10, publicOnes.size)
        val whisperCpp = publicOnes.filter { it.sourceUrl!!.startsWith(ModelCatalog.HF_WHISPER_CPP) }
        val ru = publicOnes.filter { it.sourceUrl!!.startsWith(ModelCatalog.RU_HF) }
        assertEquals(8, whisperCpp.size)
        assertEquals(2, ru.size)
        for (e in publicOnes) {
            assertTrue(e.sourceUrl!!.endsWith(e.model.fileName))
        }
        // default is the hosted Roman-Urdu q4_0.
        assertTrue(ModelCatalog.default.sourceUrl != null)
    }

    @Test
    fun everyEntryHasUniqueId() {
        val ids = ModelCatalog.models.map { it.model.id }.toSet()
        assertEquals(ModelCatalog.models.size, ids.size)
    }
}
