package dev.femustafa.kalaamscribe

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for the `DolphinAttnEngine` batch-decode guards (no ORT/native libs
 * needed on the JVM — pure companion functions).
 *
 * The original bug was a flat 60-token generation cap that silently dropped the tail of
 * any clip needing more. The fix: split long clips into <= 14 s chunks and cap each
 * decode at 66 content tokens (a 72-token decoder KV history is the fused
 * SkipLayerNormalization kernel wall, verified on ORT 1.27/1.30 for base and small).
 *
 * Two distinct guards, both under test here:
 *  - [tokenBudget]: the per-decode step cap — length-scaled, but never above 66.
 *  - [splitChunksAtMost]: hard-splits an over-cap piece (e.g. one unbroken 17.8 s
 *    utterance) into pieces at most 14 s, preferring a quiet pause as the boundary
 *    near the cap, because the VAD merge passes an over-limit utterance through
 *    untrimmed and would otherwise let one chunk burn all 66 budget tokens and
 *    truncate its tail.
 */
class DolphinAttnEngineTest {

    @Test
    fun `a full 14s chunk clears the old 60-token flat cap`() {
        // 14 s at the measured on-device worst (~4.1 tok/s) ≈ 57 tokens; the 66 ceiling
        // leaves ~9 tokens of margin instead of the 16 s cap's zero margin.
        assertEquals(66, DolphinAttnEngine.tokenBudget(14 * 16_000))
        assertTrue(DolphinAttnEngine.tokenBudget(14 * 16_000) > 60)
    }

    @Test
    fun `long clips stay under the 66-token kernel ceiling`() {
        assertEquals(66, DolphinAttnEngine.tokenBudget(60 * 16_000))
        assertEquals(66, DolphinAttnEngine.tokenBudget(120 * 16_000))
        assertEquals(66, DolphinAttnEngine.tokenBudget(30 * 16_000))
    }

    @Test
    fun `short clips get at least the minimum token budget`() {
        // Sub-second audio: budget formula gives 0×14+8 = 8 → MIN_GEN_TOKENS floor.
        assertEquals(8, DolphinAttnEngine.tokenBudget(10_000))
        assertEquals(8, DolphinAttnEngine.tokenBudget(0))
    }

    @Test
    fun `budget grows with audio length until it hits the kernel ceiling`() {
        val tiny = DolphinAttnEngine.tokenBudget(1 * 16_000)    // 1 s → 22
        val short = DolphinAttnEngine.tokenBudget(5 * 16_000)   // 5 s → 78, capped to 66
        val medium = DolphinAttnEngine.tokenBudget(15 * 16_000) // 15 s → 218, capped to 66
        assertTrue(tiny < short)
        assertEquals(short, medium)
    }

    @Test
    fun `pieces at or under the cap pass through unchanged`() {
        val data = shortArrayOf(1, 2, 3, 4, 5)
        val pieces = DolphinAttnEngine.splitChunksAtMost(data, 224_000)
        assertEquals(1, pieces.size)
        assertTrue(data.contentEquals(pieces[0]))
    }

    @Test
    fun `an oversized single utterance is split so every sample is covered`() {
        // The 17.8 s case from the field: 284,630 samples, cap 14 s (= 224,000 samples).
        val data = ShortArray(284_630) { (it % 100).toShort() }
        val pieces = DolphinAttnEngine.splitChunksAtMost(data, 224_000)
        assertTrue("must be split", pieces.size > 1)
        assertTrue("every piece fits the cap", pieces.all { it.size <= 224_000 })
        assertEquals("no samples lost or duplicated", data.size, pieces.sumOf { it.size })
        val rejoin = pieces.reduce { acc, p -> acc + p }
        assertTrue("order preserved", data.contentEquals(rejoin))
    }

    @Test
    fun `prefers a silent pause as the split point`() {
        // A constant signal just over the cap; the only quiet region is a zeroed run
        // covering whole RMS windows in the allowed last quarter (WIN=4800).
        val max = 224_000
        val data = ShortArray(max + 1000) { 999 }
        for (i in 170_000 until 215_000) data[i] = 0
        val pieces = DolphinAttnEngine.splitChunksAtMost(data, max)
        assertEquals(2, pieces.size)
        // Boundary lands at the end of the first fully-quiet window (window 36),
        // not at the hard cap: 37 × 4800 = 177,600.
        assertEquals(177_600, pieces[0].size)
        assertTrue(pieces[0].copyOfRange(170_000, pieces[0].size).all { it == 0.toShort() })
    }

    @Test
    fun `falls back to the hard cap when there is no pause`() {
        // Loud everywhere (rms 400 > QUIET_RMS 320) → no quiet window qualifies.
        val max = 224_000
        val data = ShortArray(max + 1000) { 400 }
        val pieces = DolphinAttnEngine.splitChunksAtMost(data, max)
        assertEquals(2, pieces.size)
        assertEquals(max, pieces[0].size)
    }

    @Test
    fun `greedily splits an oversized buffer with every piece at most the cap`() {
        val data = ShortArray(1_000_000) { 7 }
        val pieces = DolphinAttnEngine.splitChunksAtMost(data, 224_000)
        assertTrue(pieces.all { it.size <= 224_000 })
        assertEquals(data.size, pieces.sumOf { it.size })
        assertTrue(pieces.size >= 4)
    }
}