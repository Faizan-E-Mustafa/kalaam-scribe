package dev.femustafa.voicedictation

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Unit tests for the WhisperManager resident-model lifecycle (ADR 0004), using a
 * fake [WhisperEngine]. These run on the dev machine without a device.
 */
class WhisperManagerTest {

    private val baseDir = File("/tmp/fake-model-dir")

    private val english = Model(id = "english-q8", fileName = "ggml-base.en-q8_0.bin", languageMode = LanguageMode.English)
    private val romanUrdu = Model(id = "roman-urdu-q4_0", fileName = "ggml-model-q4_0.bin", languageMode = LanguageMode.RomanUrdu)

    private class FakeHandle : WhisperModelRef

    private class FakeSession : TranscriptionSession {
        override fun accept(samples: ShortArray) = Unit
        override val partials = kotlinx.coroutines.flow.MutableSharedFlow<String>()
        override suspend fun flush(): String = "SESSION_TEXT"
        override fun close() = Unit
    }

    private class FakeEngine : WhisperEngine {
        var loadCount = 0
        val released = mutableListOf<WhisperModelRef>()
        val transcribeCalls = mutableListOf<Triple<String, LanguageMode, String?>>()
        val createSessionCalls = mutableListOf<Triple<WhisperModelRef, LanguageMode, String?>>()

        /** When set, load()/transcribe() wait here once the caller signals entry. */
        var gate: CompletableDeferred<Unit>? = null

        /** Signal, usable for awaiting an actual suspension point (avoid busy-wait). */
        val transcribeEntered = CompletableDeferred<Unit>()

        /** When true, [createSession] returns a fake streaming session. */
        var sessionSupport = false

        /** Number of times transcribe() was entered (before any gate). */
        var transcribeEntryCount = 0

        override suspend fun load(modelPath: String): WhisperModelRef {
            gate?.await()
            loadCount++
            return FakeHandle()
        }

        override suspend fun transcribe(
            model: WhisperModelRef,
            audioPath: String,
            languageMode: LanguageMode,
            language: String?,
        ): String {
            transcribeEntryCount++
            if (!transcribeEntered.isCompleted) transcribeEntered.complete(Unit)
            gate?.await()
            transcribeCalls += Triple(audioPath, languageMode, language)
            return "text"
        }

        override suspend fun createSession(
            model: WhisperModelRef,
            languageMode: LanguageMode,
            language: String?,
        ): TranscriptionSession? {
            createSessionCalls += Triple(model, languageMode, language)
            return if (sessionSupport) FakeSession() else null
        }

        override fun release(model: WhisperModelRef) {
            released += model
        }
    }

    @Test
    fun repeatedTranscribeReusesResidentModel() = runTest {
        val engine = FakeEngine()
        val manager = WhisperManager(baseDir, engine)
        manager.switchTo(english)

        assertEquals(1, engine.loadCount)

        manager.transcribe("/data/audio/1.wav")
        manager.transcribe("/data/audio/2.wav")
        manager.transcribe("/data/audio/3.wav")

        // Model loaded once and never reloaded per inference.
        assertEquals(1, engine.loadCount)
        assertEquals(3, engine.transcribeCalls.size)
    }

    @Test
    fun switchToUnloadsCurrentAndLoadsNew() = runTest {
        val engine = FakeEngine()
        val manager = WhisperManager(baseDir, engine)

        manager.switchTo(english)
        assertEquals(1, engine.loadCount)
        val firstHandle = manager.residentHandle()

        manager.switchTo(romanUrdu)
        assertEquals(2, engine.loadCount)
        // The previous model was released.
        assertEquals(listOf(firstHandle), engine.released)
    }

    @Test
    fun appliesLanguageModeOnTranscribe() = runTest {
        val engine = FakeEngine()
        val manager = WhisperManager(baseDir, engine)
        manager.switchTo(english)
        manager.transcribe("/data/audio/a.wav")

        val (audio1, mode1, lang1) = engine.transcribeCalls.single()
        assertEquals("/data/audio/a.wav", audio1)
        assertEquals(LanguageMode.English, mode1)
        // No user override for a fixed-English Model.
        assertNull(lang1)

        // Roman-Urdu (fixed en) is applied as RomanUrdu, never overridden to `ur`.
        manager.switchTo(romanUrdu)
        manager.transcribe("/data/audio/b.wav")
        val (_, mode2, lang2) = engine.transcribeCalls.last()
        assertEquals(LanguageMode.RomanUrdu, mode2)
        assertNull(lang2)
    }

    @Test
    fun userLanguageAppliesOnlyToMultilingualModels() = runTest {
        val engine = FakeEngine()
        val manager = WhisperManager(baseDir, engine)
        val multilingual = Model(id = "multilingual-tiny", fileName = "ggml-tiny.bin", languageMode = LanguageMode.Auto)

        manager.switchTo(multilingual)
        manager.setLanguageCode("de")
        manager.transcribe("/data/audio/a.wav")

        val (_, mode, lang) = engine.transcribeCalls.single()
        assertEquals(LanguageMode.Auto, mode)
        assertEquals("de", lang)
    }

    @Test
    fun userLanguageDoesNotApplyToEnglishOrRomanUrdu() = runTest {
        val engine = FakeEngine()
        val manager = WhisperManager(baseDir, engine)

        // A user override is set, but it must NOT reach a fixed-English or Roman-Urdu Model.
        manager.setLanguageCode("de")
        manager.switchTo(english)
        manager.transcribe("/data/audio/a.wav")
        assertNull(engine.transcribeCalls.last().third)

        manager.switchTo(romanUrdu)
        manager.transcribe("/data/audio/b.wav")
        assertNull(engine.transcribeCalls.last().third)
    }

    @Test
    fun blankLanguageMeansAuto() = runTest {
        val engine = FakeEngine()
        val manager = WhisperManager(baseDir, engine)
        val multilingual = Model(id = "multilingual-tiny", fileName = "ggml-tiny.bin", languageMode = LanguageMode.Auto)

        manager.switchTo(multilingual)
        manager.setLanguageCode("")
        manager.transcribe("/data/audio/a.wav")

        val (_, _, lang) = engine.transcribeCalls.single()
        assertNull(lang)
    }

    @Test
    fun translationIsSerializedNoConcurrentStart() = runTest {
        val engine = FakeEngine()
        val manager = WhisperManager(baseDir, engine)
        manager.switchTo(english)

        // Block transcription so we can observe that a second call waits.
        val gate = CompletableDeferred<Unit>()
        engine.gate = gate

        val first = launch { manager.transcribe("/data/audio/1.wav") }
        // Real suspension point: wait until the first transcribe entered the engine.
        engine.transcribeEntered.await()
        assertEquals(1, engine.transcribeEntryCount)
        assertTrue(engine.transcribeCalls.isEmpty())

        val second = launch { manager.transcribe("/data/audio/2.wav") }
        kotlinx.coroutines.yield()

        // The second dictation must NOT have started while the first is running:
        // the engine's transcribe has been entered exactly once.
        assertEquals(1, engine.transcribeEntryCount)

        gate.complete(Unit)
        first.join()
        second.join()

        // Both eventually run, model still loaded once.
        assertEquals(2, engine.transcribeEntryCount)
        assertEquals(2, engine.transcribeCalls.size)
        assertEquals(1, engine.loadCount)
    }

    @Test
    fun createSessionReturnsNullForNonStreamingEngine() = runTest {
        val engine = FakeEngine()
        val manager = WhisperManager(baseDir, engine)
        manager.switchTo(english)

        val session = manager.createSession()

        assertNull(session)
        // Model loaded once for the session attempt, reused.
        assertEquals(1, engine.loadCount)
    }

    @Test
    fun createSessionWithoutModelReturnsNull() = runTest {
        val engine = FakeEngine()
        val manager = WhisperManager(baseDir, engine)
        engine.sessionSupport = true

        val session = manager.createSession()

        assertNull(session)
        assertEquals(0, engine.loadCount)
    }

    @Test
    fun createSessionReturnsStreamingSessionFromEngine() = runTest {
        val engine = FakeEngine()
        val manager = WhisperManager(baseDir, engine)
        engine.sessionSupport = true
        manager.switchTo(english)

        val session = manager.createSession()

        assertEquals("SESSION_TEXT", session?.flush())
        assertEquals(1, engine.createSessionCalls.size)
        // Model still resident, not reloaded.
        assertEquals(1, engine.loadCount)
    }

    @Test
    fun createSessionAppliesLanguageResolutionLikeTranscribe() = runTest {
        val engine = FakeEngine()
        val manager = WhisperManager(baseDir, engine)
        engine.sessionSupport = true
        val multilingual = Model(id = "multilingual-tiny", fileName = "ggml-tiny.bin", languageMode = LanguageMode.Auto)
        manager.switchTo(multilingual)
        manager.setLanguageCode("de")

        val first = manager.createSession()

        val (_, mode, lang) = engine.createSessionCalls.single()
        assertEquals(LanguageMode.Auto, mode)
        assertEquals("de", lang)
        // Close the open session so the resident model is free again.
        first?.close()

        // Fixed-English models never receive the user override.
        manager.switchTo(english)
        val second = manager.createSession()
        val (_, mode2, lang2) = engine.createSessionCalls.last()
        assertEquals(LanguageMode.English, mode2)
        assertNull(lang2)
        second?.close()
    }
}
