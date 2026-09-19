package dev.femustafa.kalaamscribe

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Unit tests for the seam word-merge post-processing that removes chunk/segment
 * seam word-duplication when a chunk ends mid-sentence and the next chunk begins
 * with a repeat.
 *
 * These tests validate the pure text merge helper before it is wired into batch
 * and streaming joins. The helper operates word-wise: compare the last N words
 * of the left segment (a) with the first N words of the right segment (b); if
 * ≥4 words match contiguously, the matched run is dropped from b (left wins).
 * This matches the device artifact observed when the streaming guard fires:
 * "...بہت عرصہ ہوا آپ سے کسی **عرصہ ہوا آپ سے کسی** بات نہیں ہوئی..."
 */
class SeamMergeTest {

    private fun seamMerge(
        a: String,
        b: String,
        windowWords: Int = 12,
        minMatchWords: Int = 4,
    ): String {
        val aw = a.trim().split(" ").filter { it.isNotEmpty() }
        val bw = b.trim().split(" ").filter { it.isNotEmpty() }

        var bestSize = 0
        var bestAi = -1
        var bestBj = -1

        val tail = aw.takeLast(windowWords)
        val head = bw.take(windowWords)

        for (i in tail.indices) {
            for (j in head.indices) {
                var k = 0
                while (i + k < tail.size && j + k < head.size && tail[i + k] == head[j + k]) {
                    k++
                }
                if (k > bestSize) {
                    bestSize = k
                    bestAi = i
                    bestBj = j
                }
            }
        }

        return if (bestSize >= minMatchWords) {
            val keepB = bw.take(bestBj) + bw.drop(bestBj + bestSize)
            (aw + keepB).joinToString(" ")
        } else {
            (aw + bw).joinToString(" ")
        }
    }

    /** Device artifact: 6-word overlap collapsed (mid-sentence chunk split). */
    @Test
    fun `collapses 6-word duplicate at seam`() {
        val a = "بہت عرصہ ہوا آپ سے کوئی بات نہیں ہوئی"
        val b = "عرصہ ہوا آپ سے کوئی بات نہیں ہوئی اس نے ان اس نے"
        val result = seamMerge(a, b)
        assertEquals(
            "left unchanged, right dropped the 8-word run",
            "بہت عرصہ ہوا آپ سے کوئی بات نہیں ہوئی اس نے ان اس نے",
            result,
        )
    }

    /** Genuine repeat phrase preserved: only 3-word overlap, below min threshold. */
    @Test
    fun `preserves genuine 3-word repeat (below min threshold)`() {
        val a = "بہت عرصہ ہوا آپ سے کوئی بات نہیں ہوئی"
        val b = "بات نہیں ہوئی اس نے کچھ کہا"
        val result = seamMerge(a, b)
        assertEquals(
            "3-word repeat preserved, both parts kept",
            "بہت عرصہ ہوا آپ سے کوئی بات نہیں ہوئی بات نہیں ہوئی اس نے کچھ کہا",
            result,
        )
    }

    /** Single-word seam: untouched (below threshold). */
    @Test
    fun `does not merge single-word seam`() {
        val a = "something something"
        val b = "extra something something"
        val result = seamMerge(a, b)
        assertEquals(
            "single-word overlap unchanged",
            "something something extra something something",
            result,
        )
    }

    /** No overlap: identity join. */
    @Test
    fun `joins non-overlapping segments unchanged`() {
        val a = "one two three"
        val b = "four five six"
        val result = seamMerge(a, b)
        assertEquals("no overlap, identity join", "one two three four five six", result)
    }

    /** 4-word overlap (≥ threshold) — entire dup run dropped from b. */
    @Test
    fun `collapses 4-word duplicate from b prefix`() {
        val a = "alpha beta gamma delta epsilon"
        val b = "beta gamma delta epsilon zeta"
        val result = seamMerge(a, b)
        assertEquals(
            "4-word prefix duplicate dropped from b",
            "alpha beta gamma delta epsilon zeta",
            result,
        )
    }

    /** Overlap in middle of b: non-dup prefix kept, dup run dropped. */
    @Test
    fun `collapses middle-of-b duplicate, keeps b prefix`() {
        val a = "the quick brown fox"
        val b = "unusual brown fox jumps high"
        val result = seamMerge(a, b)
        assertEquals(
            "2-word overlap < 4 threshold → no merge (preserved)",
            "the quick brown fox unusual brown fox jumps high",
            result,
        )
    }

    /** 3-word overlap in middle of b is below threshold — no merge. */
    @Test
    fun `preserves 3-word overlap in middle of b (below threshold)`() {
        val a = "the quick brown fox jumps"
        val b = "unusual brown fox jumps high"
        val result = seamMerge(a, b)
        assertEquals(
            "3-word overlap below threshold → identity join",
            "the quick brown fox jumps unusual brown fox jumps high",
            result,
        )
    }

    /** Real Urdu example from device streaming guard firing (14.3s -> 2 chunks). */
    @Test
    fun `collapses the exact device artifact observed in streaming round 2`() {
        val a = "اسلحہ مولیکن کیا حال ہے آپ کا امید ہے آپ حریت سے ہوگئے بہت عرصہ ہوا آپ سے کوئی"
        val b = "عرصہ ہوا آپ سے کوئی بات نہیں ہوئی اس نے ان اس نے"
        val result = seamMerge(a, b)
        // overlap is 5 words ("عرصہ ہوا آپ سے کوئی"), so "بہت" stays in a;
        // b's leading run dropped, rest preserved.
        assertEquals(
            "device artifact collapsed (5-word duplicate)",
            "اسلحہ مولیکن کیا حال ہے آپ کا امید ہے آپ حریت سے ہوگئے بہت عرصہ ہوا آپ سے کوئی بات نہیں ہوئی اس نے ان اس نے",
            result,
        )
    }

    /** Minimum threshold: exactly 4 words merges. */
    @Test
    fun `merges when exactly 4 words overlap`() {
        val a = "a b c d e f"
        val b = "c d e f g"
        val result = seamMerge(a, b)
        assertEquals("4-word overlap collapsed", "a b c d e f g", result)
    }

    /** 3-word overlap (below threshold) — no merge. */
    @Test
    fun `does not merge when exactly 3 words overlap`() {
        val a = "a b c d e"
        val b = "c d e f g"
        val result = seamMerge(a, b)
        assertEquals(
            "3-word overlap below threshold → identity join",
            "a b c d e c d e f g",
            result,
        )
    }

    /** Empty left segment: right segment returned unchanged. */
    @Test
    fun `empty left segment returns right as-is`() {
        assertEquals("right side", seamMerge("", "right side"))
    }

    /** Empty right segment: left segment returned unchanged. */
    @Test
    fun `empty right segment returns left as-is`() {
        assertEquals("left side", seamMerge("left side", ""))
    }

    /** Both empty: empty result. */
    @Test
    fun `both empty returns empty`() {
        assertEquals("", seamMerge("", ""))
    }

    /** Whitespace-only segments handled gracefully. */
    @Test
    fun `whitespace-only segments handled gracefully`() {
        assertEquals("some text repeated", seamMerge("some text", "   repeated"))
        assertEquals("some text", seamMerge("some text", "     "))
    }
}