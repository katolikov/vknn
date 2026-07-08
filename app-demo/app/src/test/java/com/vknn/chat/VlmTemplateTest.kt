package com.vknn.chat

import com.vknn.chat.vlm.VlmTemplate
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

// Validates the SmolVLM2 tokenizer pipeline (Digits pre-split + original GPT-2 regex) against
// HuggingFace `tokenizers` reference ids, the prompt-template structure, and — once the ground-truth
// files land in <repo>/vlm_gate/ — the full prompt expansion byte-for-byte against the HF processor.
// Runs on the JVM: ./gradlew :app:testDebugUnitTest.
class VlmTemplateTest {
    private fun load(): Tokenizer = VlmTemplate.tokenizer(
        File("src/test/resources/smolvlm2/vocab.json").readText(),
        File("src/test/resources/smolvlm2/merges.txt").readText(),
    )

    @Test
    fun matchesReferenceIds() {
        val tk = load()
        val cases = listOf(
            "User:" to intArrayOf(11126, 42),
            "\nAssistant:" to intArrayOf(198, 9519, 9531, 42),
            " What is in this picture? Answer in 12 words." to
                intArrayOf(1812, 314, 281, 451, 4177, 47, 19842, 281, 216, 33, 34, 1924, 30),
            "abc 123 def" to intArrayOf(25276, 216, 33, 34, 35, 753),
            "Hello, world! 123" to intArrayOf(19556, 28, 905, 17, 216, 33, 34, 35),
            "I'm CAN'T don't" to intArrayOf(57, 5248, 27190, 23, 68, 1326, 982),
            "  spaces   and\ttabs\n" to intArrayOf(216, 5600, 256, 284, 197, 100, 7366, 198),
        )
        for ((text, expected) in cases) {
            assertArrayEquals("encode mismatch for: ${text.replace("\n", "\\n")}", expected, tk.encode(text))
        }
    }

    @Test
    fun promptStructure() {
        val tk = load()
        val ids = VlmTemplate.promptIds(tk, "Describe this.")
        assertEquals(VlmTemplate.IM_START.toLong(), ids[0])
        assertEquals(VlmTemplate.IMAGE_ROWS, ids.count { it == VlmTemplate.IMAGE.toLong() })
        // <fake_token_around_image> brackets the image block on both sides.
        assertEquals(2, ids.count { it == VlmTemplate.FAKE_IMAGE.toLong() })
        assertEquals(1, ids.count { it == VlmTemplate.GLOBAL_IMG.toLong() })
        assertEquals(1, ids.count { it == VlmTemplate.END_OF_UTTERANCE.toLong() })
        // The tail is "\nAssistant:".
        val tail = tk.encode("\nAssistant:")
        for (i in tail.indices) {
            assertEquals(tail[i].toLong(), ids[ids.size - tail.size + i])
        }
    }

    // Ground truth from the host reference: vlm_gate/prompt_ids.json (exact HF processor ids for one
    // question) and vlm_gate/ref_tokens.json (eos id + the question). Skipped until the files land.
    @Test
    fun matchesHostReferencePromptIds() {
        val gate = findVlmGate()
        assumeTrue("vlm_gate/ not present yet", gate != null)
        val refTokens = JSONObject(File(gate, "ref_tokens.json").readText())
        val question = refTokens.getString("question")
        if (refTokens.has("eos")) {
            assertEquals(VlmTemplate.EOS.toLong(), refTokens.getLong("eos"))
        }
        val expected = parseIds(File(gate, "prompt_ids.json").readText())
        val got = VlmTemplate.promptIds(load(), question)
        assertArrayEquals(expected, got)
    }

    // Accepts a bare JSON array or an object wrapping it under "ids"/"prompt_ids".
    private fun parseIds(json: String): LongArray {
        val arr: JSONArray = if (json.trimStart().startsWith("[")) {
            JSONArray(json)
        } else {
            val o = JSONObject(json)
            o.optJSONArray("ids") ?: o.getJSONArray("prompt_ids")
        }
        return LongArray(arr.length()) { arr.getLong(it) }
    }

    // The gate directory sits at the repo root; the test runs with the app module as its working dir.
    private fun findVlmGate(): File? {
        var dir: File? = File(".").absoluteFile
        repeat(5) {
            val cand = File(dir, "vlm_gate")
            if (File(cand, "prompt_ids.json").exists() && File(cand, "ref_tokens.json").exists()) return cand
            dir = dir?.parentFile ?: return null
        }
        return null
    }
}
