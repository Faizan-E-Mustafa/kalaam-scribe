package dev.femustafa.voicedictation

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Unit tests for the ModelCatalog unit seam (spec "Model catalog"). */
class ModelCatalogTest {

    @Test
    fun ggmlCatalogHasTenModels() {
        assertEquals(10, ModelCatalog.ggmlModels.size)
    }

    @Test
    fun onnxCatalogHasSixIdentitiesInTwoPrecisionTiers() {
        // 6 ONNX model identities (tiny, tiny.en, base.en, base, small, roman-urdu) × 2 tiers (fp32, int8).
        assertEquals(12, ModelCatalog.onnxModels.size)
        assertEquals(ModelPrecision.FP32, ModelCatalog.onnxModels.first().precision)
        assertTrue(ModelCatalog.onnxModels.all { it.precision != null })
        // Each identity is offered in both fp32 and int8.
        val ids = ModelCatalog.onnxModels.map { it.model.id }
        for (identity in listOf("tiny", "tiny.en", "base.en", "base", "small", "roman-urdu")) {
            assertTrue("missing fp32 for $identity", ids.contains("$identity-fp32"))
            assertTrue("missing int8 for $identity", ids.contains("$identity-int8"))
        }
    }

    @Test
    fun dolphinCtcCatalogHasTwoSizesInTwoPrecisionTiers() {
        // 2 Dolphin CTC sizes (base, small) × 2 tiers (fp32, int8).
        assertEquals(4, ModelCatalog.dolphinCtcModels.size)
        assertTrue(ModelCatalog.dolphinCtcModels.all { it.precision != null })
        val ids = ModelCatalog.dolphinCtcModels.map { it.model.id }
        for (size in listOf("base", "small")) {
            assertTrue("missing fp32 for $size", ids.contains("dolphin-$size-fp32"))
            assertTrue("missing int8 for $size", ids.contains("dolphin-$size-int8"))
        }
    }

    @Test
    fun dolphinCtcModelsUseAutoLanguageAndSingleModelFile() {
        for (e in ModelCatalog.dolphinCtcModels) {
            // Dolphin auto-detects language; no language override applies.
            assertEquals(LanguageMode.Auto, e.model.languageMode)
            assertTrue(e.model.canOverrideLanguage)
            // It's a single model.onnx + derived tokens file (no encoder/decoder).
            assertTrue(e.model.fileName.endsWith("-model.onnx"))
            assertEquals(DolphinCtcEngine.dolphinTokensName(e), "${e.model.id}-tokens.txt")
            // ONNX-hosted, no GGML source.
            assertNotNull(e.onnxSourceUrl)
            assertNull(e.sourceUrl)
            assertTrue(e.onnxSourceUrl!!.endsWith("model.onnx") || e.onnxSourceUrl!!.endsWith("model.int8.onnx"))
        }
    }

    @Test
    fun dolphinCtcEntriesRouteToDolphinEngineAndCatalog() {
        for (e in ModelCatalog.dolphinCtcModels) {
            assertTrue(ModelCatalog.isDolphinCtc(e))
            assertTrue(ModelCatalog.isDolphinCtcFileName(e.model.fileName))
            // byId resolves them across all catalogs.
            assertEquals(e.model.id, ModelCatalog.byId(e.model.id)!!.model.id)
        }
    }

    @Test
    fun dolphinAttnCatalogHasBaseAndSmallFp16Encoders() {
        // 2 Dolphin attention sizes (base, small), fp16/arm tier only.
        assertEquals(2, ModelCatalog.dolphinAttnModels.size)
        val ids = ModelCatalog.dolphinAttnModels.map { it.model.id }
        assertTrue(ids.contains("dolphin-attn-base"))
        assertTrue(ids.contains("dolphin-attn-small"))
        for (e in ModelCatalog.dolphinAttnModels) {
            // fp16/arm only (the verifier covered the kill-question "which tier ships").
            assertEquals(ModelPrecision.FP16, e.precision)
            // Language-mode Auto: the decoder pins ur/PK tokens itself.
            assertEquals(LanguageMode.Auto, e.model.languageMode)
            assertTrue(e.model.canOverrideLanguage)
            // An encoder + sibling decoder named after the id, sharing units.txt.
            assertTrue(e.model.fileName.endsWith("-encoder.onnx"))
            // The decoder URL is its sibling alongside the encoder.
            assertEquals("${e.model.id}-decoder.onnx", ModelDownloader.dolphinAttnDecoderName(e))
            val size = e.model.id.removePrefix("dolphin-attn-")
            assertEquals("${ModelCatalog.RU_HF}/dolphin-attn/$size/decoder.onnx",
                e.onnxSourceUrl!!.substringBefore("/encoder.onnx") + "/decoder.onnx")
            // A sibling units.txt URL one directory up.
            assertTrue(e.onnxSourceUrl!!.startsWith(ModelCatalog.DOLPHIN_ATTN_HF))
            // ONNX-hosted, no GGML source.
            assertNull(e.sourceUrl)
        }
    }

    @Test
    fun dolphinAttnEntriesRouteToDolphinAttnCatalog() {
        for (e in ModelCatalog.dolphinAttnModels) {
            assertTrue(ModelCatalog.isDolphinAttn(e))
            assertTrue(ModelCatalog.isDolphinAttnFileName(e.model.fileName))
            assertFalse(ModelCatalog.isDolphinCtc(e))
            // byId resolves them across all catalogs.
            assertEquals(e.model.id, ModelCatalog.byId(e.model.id)!!.model.id)
        }
    }

    @Test
    fun defaultIsRomanUrduQ4() {
        val d = ModelCatalog.default
        assertEquals("roman-urdu-q4_0", d.model.id)
        assertTrue(d.isDefault)
        assertEquals("ggml-model-q4_0.bin", d.model.fileName)
        assertEquals(LanguageMode.RomanUrdu, d.model.languageMode)
        // Default lives on the GGML catalog, not the ONNX one.
        assertTrue(d in ModelCatalog.ggmlModels)
        assertFalse(d in ModelCatalog.onnxModels)
    }

    @Test
    fun romanUrduModelsUseFixedEnglishAndHosted() {
        val ruQ4 = ModelCatalog.byId("roman-urdu-q4_0")!!
        val ruF16 = ModelCatalog.byId("roman-urdu-f16")!!

        // Roman-Urdu runs fixed `en` (faster; may leak English on short clips until
        // the seed-prompt engine build lands). Never `ur` or auto-detect.
        assertEquals(LanguageMode.RomanUrdu, ruQ4.model.languageMode)
        assertEquals(LanguageMode.RomanUrdu, ruF16.model.languageMode)
        // Downloadable from the project's Roman-Urdu HuggingFace repo (GGML-only).
        assertNotNull(ruQ4.sourceUrl)
        assertNotNull(ruF16.sourceUrl)
        assertNull(ruQ4.onnxSourceUrl)
        assertTrue(ruQ4.sourceUrl!!.startsWith(ModelCatalog.RU_HF))
        assertTrue(ruQ4.sourceUrl!!.endsWith(ruQ4.model.fileName))
        assertTrue(ruF16.sourceUrl!!.endsWith(ruF16.model.fileName))
    }

    @Test
    fun romanUrduOnnxModelsUseFixedEnglishAndProjectRepo() {
        val ruFp32 = ModelCatalog.byId("roman-urdu-fp32")!!
        val ruInt8 = ModelCatalog.byId("roman-urdu-int8")!!

        // Fixed `en`, never overridable — same rule as the GGML Roman-Urdu models.
        assertEquals(LanguageMode.RomanUrdu, ruFp32.model.languageMode)
        assertEquals(LanguageMode.RomanUrdu, ruInt8.model.languageMode)
        assertFalse(ruFp32.model.canOverrideLanguage)
        assertFalse(ruInt8.model.canOverrideLanguage)
        // ONNX-only: hosted in the project's Roman-Urdu HF repo, no GGML source.
        assertNull(ruFp32.sourceUrl)
        assertNull(ruInt8.sourceUrl)
        assertTrue(ruFp32.onnxSourceUrl!!.startsWith(ModelCatalog.RU_HF))
        assertTrue(ruInt8.onnxSourceUrl!!.startsWith(ModelCatalog.RU_HF))
        // Identity-based remote names shared across tiers (fp32: .onnx, int8: .int8.onnx).
        assertTrue(ruFp32.onnxSourceUrl!!.endsWith("roman-urdu-encoder.onnx"))
        assertTrue(ruInt8.onnxSourceUrl!!.endsWith("roman-urdu-encoder.int8.onnx"))
        assertEquals("roman-urdu-fp32-encoder.onnx", ruFp32.model.fileName)
        assertEquals("roman-urdu-int8-encoder.int8.onnx", ruInt8.model.fileName)
    }

    @Test
    fun englishModelsAreFixedEn() {
        assertEquals(LanguageMode.English, ModelCatalog.byId("english-full")!!.model.languageMode)
        assertEquals(LanguageMode.English, ModelCatalog.byId("english-q8")!!.model.languageMode)
        assertEquals("ggml-base.en.bin", ModelCatalog.byId("english-full")!!.model.fileName)
        assertEquals("ggml-base.en-q8_0.bin", ModelCatalog.byId("english-q8")!!.model.fileName)
    }

    @Test
    fun ggmlPublicModelsPointAtTheirRespectiveHF() {
        val publicOnes = ModelCatalog.ggmlModels.filter { it.sourceUrl != null }
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
    fun onnxEntriesPointAtSherpaReposWithTierCorrectFiles() {
        for (e in ModelCatalog.onnxModels) {
            assertNotNull(e.onnxSourceUrl)
            assertNull(e.sourceUrl)
            // GGML catalog and ONNX catalog are fully separate sets.
            assertFalse(e in ModelCatalog.ggmlModels)
            // Encoder URL points at the correct remote identity for the tier:
            // `<identity>-encoder.onnx` for fp32, `<identity>-encoder.int8.onnx` for int8.
            // The local fileName is per-tier (ids are unique), so it intentionally
            // differs from the shared remote filename.
            val identity = e.model.id.removeSuffix("-fp32").removeSuffix("-int8")
            val expectedSuffix = if (e.precision == ModelPrecision.INT8) {
                "$identity-encoder.int8.onnx"
            } else {
                "$identity-encoder.onnx"
            }
            assertTrue("encoder URL must end with $expectedSuffix, got ${e.onnxSourceUrl}", e.onnxSourceUrl!!.endsWith(expectedSuffix))
            // The GGML-format source stays null for ONNX-only entries.
            assertTrue(e.sourceUrl == null)
        }
    }

    @Test
    fun everyEntryHasUniqueId() {
        val all = ModelCatalog.ggmlModels + ModelCatalog.onnxModels +
            ModelCatalog.dolphinCtcModels + ModelCatalog.dolphinAttnModels
        val ids = all.map { it.model.id }.toSet()
        assertEquals(all.size, ids.size)
    }
}