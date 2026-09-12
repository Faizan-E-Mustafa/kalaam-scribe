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
    fun onnxCatalogHasSevenIdentitiesInTwoPrecisionTiers() {
        // 7 ONNX model identities (tiny, tiny.en, base.en, small, small.en, base, roman-urdu) × 2 tiers (fp32, int8).
        assertEquals(14, ModelCatalog.onnxModels.size)
        assertEquals(ModelPrecision.FP32, ModelCatalog.onnxModels.first().precision)
        assertTrue(ModelCatalog.onnxModels.all { it.precision != null })
        // Each identity is offered in both fp32 and int8.
        val ids = ModelCatalog.onnxModels.map { it.model.id }
        for (identity in listOf("tiny", "tiny.en", "base.en", "small", "small.en", "base", "roman-urdu")) {
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
    fun dolphinAttnCatalogHasBaseAndSmallInFp16AndInt8Tiers() {
        // 4 Dolphin attention entries: base fp16, small fp16, int8-base, int8-small.
        assertEquals(4, ModelCatalog.dolphinAttnModels.size)
        val ids = ModelCatalog.dolphinAttnModels.map { it.model.id }
        for (id in listOf(
            "dolphin-attn-base", "dolphin-attn-small",
            "dolphin-attn-int8-base", "dolphin-attn-int8-small",
        )) {
            assertTrue("missing $id", ids.contains(id))
        }
        // fp16/arm entries are the first two; int8 entries follow.
        assertEquals(ModelPrecision.FP16, ModelCatalog.dolphinAttnModels[0].precision)
        assertEquals(ModelPrecision.FP16, ModelCatalog.dolphinAttnModels[1].precision)
        assertEquals(ModelPrecision.INT8, ModelCatalog.dolphinAttnModels[2].precision)
        assertEquals(ModelPrecision.INT8, ModelCatalog.dolphinAttnModels[3].precision)
        for (e in ModelCatalog.dolphinAttnModels) {
            // Language-mode Auto: the decoder pins ur/PK tokens itself.
            assertEquals(LanguageMode.Auto, e.model.languageMode)
            assertTrue(e.model.canOverrideLanguage)
            // An encoder + sibling decoder named after the id, sharing units.txt.
            assertTrue(e.model.fileName.endsWith("-encoder.onnx"))
            // The decoder URL is its sibling alongside the encoder.
            assertEquals("${e.model.id}-decoder.onnx", ModelDownloader.dolphinAttnDecoderName(e))
            // The base path and size dir differ between the fp16 and int8 tiers.
            val (hall, sizeDir, expectedBase) = if (e.precision == ModelPrecision.FP16) {
                Triple(ModelCatalog.DOLPHIN_ATTN_HF, e.model.id.removePrefix("dolphin-attn-"),
                    "${ModelCatalog.RU_HF}/dolphin-attn")
            } else {
                Triple(ModelCatalog.DOLPHIN_ATTN_INT8_HF, e.model.id.removePrefix("dolphin-attn-int8-"),
                    "${ModelCatalog.RU_HF}/dolphin-attn-int8")
            }
            assertEquals("$expectedBase/$sizeDir/decoder.onnx",
                e.onnxSourceUrl!!.substringBefore("/encoder.onnx") + "/decoder.onnx")
            // Hosted in the tier's own top-level HF directory.
            assertTrue("int8 URLs must live under $hall", e.onnxSourceUrl!!.startsWith(hall))
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
    fun ggmlDefaultIsRomanUrduQ4() {
        val d = ModelCatalog.ggmlDefault
        assertEquals("roman-urdu-q4_0", d.model.id)
        assertTrue(d.isDefault)
        assertEquals("ggml-model-q4_0.bin", d.model.fileName)
        assertEquals(LanguageMode.RomanUrdu, d.model.languageMode)
        // Default lives on the GGML catalog, not the ONNX one.
        assertTrue(d in ModelCatalog.ggmlModels)
        assertFalse(d in ModelCatalog.onnxModels)
    }

    @Test
    fun onnxDefaultIsRomanUrduInt8() {
        val d = ModelCatalog.onnxDefault
        assertEquals("roman-urdu-int8", d.model.id)
        assertEquals(ModelPrecision.INT8, d.precision)
        assertEquals(LanguageMode.RomanUrdu, d.model.languageMode)
        // It is an ONNX-family model, so it must be part of the active catalog.
        assertTrue(d in ModelCatalog.onnxModels)
        assertTrue(d in ModelCatalog.activeCatalog)
        // The disabled GGML default is not the app's active one.
        assertFalse(ModelCatalog.ggmlModels.any { it.model.id == d.model.id })
    }

    @Test
    fun onnxActiveIsInt8EnglishAndMultilingualPlusBothRomanUrduTiers() {
        val activeIds = ModelCatalog.onnxModelsActive.map { it.model.id }.toSet()
        assertEquals(8, activeIds.size)
        assertEquals(
            setOf(
                "tiny.en-int8", "base.en-int8", "small.en-int8",
                "tiny-int8", "base-int8", "small-int8",
                "roman-urdu-fp32", "roman-urdu-int8",
            ),
            activeIds,
        )
        // The disabled fp32 tiers are exactly the English/multilingual ones.
        val disabledIds = ModelCatalog.onnxModelsDisabled.map { it.model.id }.toSet()
        assertEquals(
            setOf(
                "tiny.en-fp32", "base.en-fp32", "small.en-fp32",
                "tiny-fp32", "base-fp32", "small-fp32",
            ),
            disabledIds,
        )
        assertTrue(ModelCatalog.onnxModelsActive.all { it in ModelCatalog.onnxModels })
        assertTrue(ModelCatalog.onnxModelsDisabled.all { it in ModelCatalog.onnxModels })
    }

    @Test
    fun modelNameIsEnginePrefixedStem() {
        assertEquals("whisper tiny.en", ModelCatalog.byId("tiny.en-int8")!!.modelName)
        assertEquals("whisper base.en", ModelCatalog.byId("base.en-int8")!!.modelName)
        assertEquals("whisper small.en", ModelCatalog.byId("small.en-int8")!!.modelName)
        assertEquals("whisper tiny", ModelCatalog.byId("tiny-int8")!!.modelName)
        assertEquals("whisper small", ModelCatalog.byId("small-fp32")!!.modelName)
        assertEquals("whisper roman-urdu", ModelCatalog.byId("roman-urdu-int8")!!.modelName)
        assertEquals("dolphin base", ModelCatalog.byId("dolphin-attn-base")!!.modelName)
        assertEquals("dolphin base", ModelCatalog.byId("dolphin-attn-int8-base")!!.modelName)
        assertEquals("dolphin small", ModelCatalog.byId("dolphin-attn-small")!!.modelName)
        assertEquals("dolphin small", ModelCatalog.byId("dolphin-attn-int8-small")!!.modelName)
    }

    @Test
    fun activeCatalogIsTheEnabledOnnxBackedFamilies() {
        val ids = ModelCatalog.activeCatalog.map { it.model.id }
        assertEquals(
            ModelCatalog.onnxModelsActive.size + ModelCatalog.omnilingualModels.size + ModelCatalog.dolphinAttnModels.size,
            ids.size,
        )
        // A representative of every enabled family is present...
        assertTrue(ids.contains("roman-urdu-int8"))
        assertTrue(ids.contains("small.en-int8"))
        assertTrue(ids.contains("dolphin-attn-small"))
        assertTrue(ids.contains("omnilingual-300m-int8"))
        // ...and the disabled families (GGML, Dolphin CTC, fp32 tiers) are not.
        assertFalse(ids.contains("roman-urdu-q4_0"))
        assertFalse(ids.contains("english-full"))
        assertFalse(ids.contains("dolphin-base-int8"))
        assertFalse(ids.contains("dolphin-small-int8"))
        assertFalse(ids.contains("base.en-fp32"))
        assertFalse(ids.contains("base-fp32"))
        assertFalse(ids.contains("tiny.en-fp32"))
        assertFalse(ids.contains("tiny-fp32"))
        assertFalse(ids.contains("small-fp32"))
        assertFalse(ids.contains("small.en-fp32"))
        // byActiveId resolves only active ids.
        assertNull(ModelCatalog.byActiveId("english-full"))
        assertNull(ModelCatalog.byActiveId("dolphin-small-fp32"))
        assertNull(ModelCatalog.byActiveId("base.en-fp32"))
        assertEquals("roman-urdu-int8", ModelCatalog.byActiveId("roman-urdu-int8")?.model?.id)
        assertEquals("omnilingual-300m-int8", ModelCatalog.byActiveId("omnilingual-300m-int8")?.model?.id)
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
        assertTrue(ModelCatalog.ggmlDefault.sourceUrl != null)
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
            ModelCatalog.dolphinCtcModels + ModelCatalog.omnilingualModels + ModelCatalog.dolphinAttnModels
        val ids = all.map { it.model.id }.toSet()
        assertEquals(all.size, ids.size)
    }

    @Test
    fun omnilingualModelIsAutoDetectAndPassesAllLanguageFilters() {
        val omni = ModelCatalog.byId("omnilingual-300m-int8")!!
        assertEquals(LanguageMode.Auto, omni.model.languageMode)
        assertTrue(omni.model.canOverrideLanguage)
        assertTrue(omni.model.fileName.endsWith("-model.onnx"))
        assertEquals(OmnilingualEngine.omnilingualTokensName(omni), "${omni.model.id}-tokens.txt")
        assertNotNull(omni.onnxSourceUrl)
        assertNull(omni.sourceUrl)
        assertTrue(omni.onnxSourceUrl!!.endsWith("model.int8.onnx"))
        assertEquals(ModelPrecision.INT8, omni.precision)
        // Omnilingual auto-detects from 1600+ languages; no way to pin output language.
        assertTrue(omni.supportsLanguage("en"))
        assertTrue(omni.supportsLanguage("ur"))
        assertTrue(omni.supportsLanguage("hi"))
        assertTrue(omni.supportsLanguage("haw"))
        assertFalse(omni.supportsLanguage("zz"))
    }

    @Test
    fun omnilingualEntryRoutesCorrectly() {
        val omni = ModelCatalog.byId("omnilingual-300m-int8")!!
        assertTrue(ModelCatalog.isOmnilingual(omni))
        assertTrue(ModelCatalog.isOmnilingualFileName(omni.model.fileName))
        assertFalse(ModelCatalog.isDolphinCtc(omni))
        assertFalse(ModelCatalog.isDolphinAttn(omni))
        assertEquals(omni.model.id, ModelCatalog.byId(omni.model.id)!!.model.id)
        assertEquals(omni.model.id, ModelCatalog.byActiveId(omni.model.id)!!.model.id)
    }

    @Test
    fun englishModelsSupportOnlyEnglish() {
        val en = ModelCatalog.byId("small.en-int8")!!
        assertTrue(en.supportsLanguage("en"))
        assertFalse(en.supportsLanguage("ur"))
        assertFalse(en.supportsLanguage("hi"))
    }

    @Test
    fun multilingualModelsSupportEveryWhisperLanguage() {
        val ml = ModelCatalog.byId("small-int8")!!
        assertTrue(ml.supportsLanguage("en"))
        assertTrue(ml.supportsLanguage("ur"))
        assertTrue(ml.supportsLanguage("hi"))
        assertTrue(ml.supportsLanguage("haw"))
        assertFalse(ml.supportsLanguage("zz"))
    }

    @Test
    fun romanUrduModelsSupportOnlyUrdu() {
        val ru = ModelCatalog.byId("roman-urdu-int8")!!
        assertTrue(ru.supportsLanguage("ur"))
        assertFalse(ru.supportsLanguage("en"))
        assertFalse(ru.supportsLanguage("hi"))
    }

    @Test
    fun dolphinAttnModelsAreAdvertisedForUrduOnly() {
        val dol = ModelCatalog.byId("dolphin-attn-small")!!
        for (code in ModelCatalog.DOLPHIN_LANGUAGES) {
            assertTrue("dolphin must support $code", dol.supportsLanguage(code))
        }
        // Only the ur/PK pin is verified on device; the other 15 vocab languages
        // are not advertised.
        assertEquals(setOf("ur"), ModelCatalog.DOLPHIN_LANGUAGES)
        assertTrue(dol.supportsLanguage("ur"))
        assertFalse(dol.supportsLanguage("en"))
        assertFalse(dol.supportsLanguage("de"))
        assertFalse(dol.supportsLanguage("fr"))
        assertFalse(dol.supportsLanguage("hi"))
    }

    @Test
    fun filterForLanguageRestrictsActiveCatalog() {
        val urduIds = ModelCatalog.activeCatalog.filterForLanguage("ur").map { it.model.id }.toSet()
        assertTrue(urduIds.containsAll(setOf("roman-urdu-int8", "roman-urdu-fp32", "small-int8", "dolphin-attn-small", "omnilingual-300m-int8")))
        // No English-only whisper model is shown for Urdu.
        assertFalse(urduIds.any { it.contains(".en") })

        val englishIds = ModelCatalog.activeCatalog.filterForLanguage("en").map { it.model.id }.toSet()
        assertTrue(englishIds.contains("small.en-int8"))
        assertTrue(englishIds.contains("small-int8"))
        assertTrue(englishIds.contains("omnilingual-300m-int8"))
        assertFalse(englishIds.any { it.startsWith("roman-urdu") })
        // Dolphin is advertised for Urdu only, so it must not appear for English.
        assertFalse(englishIds.any { it.startsWith("dolphin-") })

        // A language without dedicated models still shows the multilingual tier + omnilingual.
        val hawaiianIds = ModelCatalog.activeCatalog.filterForLanguage("haw").map { it.model.id }.toSet()
        assertEquals(setOf("tiny-int8", "base-int8", "small-int8", "omnilingual-300m-int8"), hawaiianIds)

        // Auto-detect (null) shows the whole active catalog.
        assertEquals(ModelCatalog.activeCatalog.size, ModelCatalog.activeCatalog.filterForLanguage(null).size)
    }
}