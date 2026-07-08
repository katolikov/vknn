package com.vknn.chat

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

// Validates the pure-Kotlin BPE against HuggingFace `tokenizers` reference ids (generated from the same
// Qwen2.5 vocab). Runs on the JVM (./gradlew :app:testDebugUnitTest) — no device needed — so a tokenizer
// regression is caught before it can garble on-device chat. Also checks encode->decode round-trips.
class TokenizerTest {
    private fun load(): Tokenizer {
        val vocab = File("src/main/assets/vocab.json").readText()
        val merges = File("src/main/assets/merges.txt").readText()
        return Tokenizer(vocab, merges)
    }

    @Test
    fun matchesReferenceIds() {
        val tk = load()
        val cases = listOf(
            "Write a Python function to reverse a linked list." to intArrayOf(7985, 264, 13027, 729, 311, 9931, 264, 10592, 1140, 13),
            "def reverse(head):\n    prev = None" to intArrayOf(750, 9931, 25340, 982, 262, 7872, 284, 2240),
            "Hello, world! 123" to intArrayOf(9707, 11, 1879, 0, 220, 16, 17, 18),
            "  spaces   and\ttabs\n" to intArrayOf(220, 12621, 256, 323, 3244, 3435, 198),
        )
        for ((text, expected) in cases) {
            assertArrayEquals("encode mismatch for: ${text.replace("\n", "\\n")}", expected, tk.encode(text))
        }
    }

    @Test
    fun roundTrips() {
        val tk = load()
        for (s in listOf("Write a Python function to reverse a linked list.", "Hello, world! 123", "def f(x): return x*2")) {
            val ids = tk.encode(s)
            val dec = Tokenizer.StreamDecoder(tk)
            val sb = StringBuilder()
            for (id in ids) sb.append(dec.add(id))
            assertEquals(s, sb.toString())
        }
    }
}
