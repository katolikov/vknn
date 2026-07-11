package com.vknn.chat

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

// Validates the pure-Kotlin BPE with LLAMA3_PATTERN against HuggingFace `tokenizers` reference ids
// (generated from the same Llama-3.2 vocab/merges the app downloads with the model). Runs on the JVM
// (./gradlew :app:testDebugUnitTest) — no device needed — so a tokenizer regression is caught before it
// can garble on-device chat. Also checks encode->decode round-trips.
class LlamaTokenizerTest {
    private fun load(): Tokenizer = Tokenizer(
        File("src/test/resources/llama3/vocab.json").readText(),
        File("src/test/resources/llama3/merges.txt").readText(),
        Tokenizer.LLAMA3_PATTERN,
    )

    @Test
    fun matchesReferenceIds() {
        val tk = load()
        val cases = listOf(
            // The prompt the device run answered ("The capital of France is Paris.").
            "What is the capital of France?" to intArrayOf(3923, 374, 279, 6864, 315, 9822, 30),
            "The capital of France is Paris." to intArrayOf(791, 6864, 315, 9822, 374, 12366, 13),
            "Hello, world! 123" to intArrayOf(9906, 11, 1917, 0, 220, 4513),
            // \p{N}{1,3} groups digits in threes: "1234567" -> "123","456","7".
            "year 2026 and 1234567" to intArrayOf(3236, 220, 2366, 21, 323, 220, 4513, 10961, 22),
            "def reverse(head):\n    prev = None" to intArrayOf(755, 10134, 26408, 997, 262, 8031, 284, 2290),
            "  spaces   and\ttabs\n" to intArrayOf(220, 12908, 256, 323, 3324, 3518, 198),
        )
        for ((text, expected) in cases) {
            assertArrayEquals("encode mismatch for: ${text.replace("\n", "\\n")}", expected, tk.encode(text))
        }
    }

    @Test
    fun roundTrips() {
        val tk = load()
        for (s in listOf("What is the capital of France?", "Hello, world! 123", "def f(x): return x*2")) {
            val ids = tk.encode(s)
            val dec = Tokenizer.StreamDecoder(tk)
            val sb = StringBuilder()
            for (id in ids) sb.append(dec.add(id))
            assertEquals(s, sb.toString())
        }
    }
}
