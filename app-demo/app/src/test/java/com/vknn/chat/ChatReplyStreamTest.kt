package com.vknn.chat

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

// Replays sampled token streams through the exact token->text path ChatViewModel.send runs — one
// Tokenizer.StreamDecoder, fed a token at a time, appending each returned piece — and pins the head of
// the reply. The head is where a byte-level BPE stream decoder can silently lose text: the first token
// may be a lone leading space, only the lead bytes of a multi-byte UTF-8 character, or a newline.
//
// The two recorded streams are the greedy continuations Qwen2.5-Coder-0.5B produces for the given
// prompts; the first is byte-for-byte what build-host/vknn_chat emits from the compiled .vxm.
// Runs on the JVM: ./gradlew :app:testDebugUnitTest.
class ChatReplyStreamTest {
    private fun tokenizer(): Tokenizer = Tokenizer(
        File("src/main/assets/vocab.json").readText(),
        File("src/main/assets/merges.txt").readText(),
    )

    /** The decode loop's text assembly, verbatim: `val piece = dec.add(next); if (piece.isNotEmpty()) sb.append(piece)`. */
    private fun replyFrom(tokenizer: Tokenizer, sampled: IntArray): String {
        val streamDecoder = Tokenizer.StreamDecoder(tokenizer)
        val reply = StringBuilder()
        for (id in sampled) {
            val piece = streamDecoder.add(id)
            if (piece.isNotEmpty()) reply.append(piece)
        }
        return reply.toString()
    }

    /** Greedy continuation of "def add(a, b):" — the first sampled token (198) is a newline. */
    @Test
    fun newlineFirstTokenReachesTheReply() {
        val reply = replyFrom(tokenizer(), intArrayOf(198, 262, 1173, 955, 1, 52478, 314, 64))
        assertTrue("leading newline dropped: ${reply.take(4)}", reply.startsWith("\n"))
        assertEquals("\n    print(f\"ADDING {a", reply)
    }

    /** Greedy continuation of "Write a function to reverse a list" — the first token (304) is " in". */
    @Test
    fun leadingSpaceFirstTokenReachesTheReply() {
        val reply = replyFrom(tokenizer(), intArrayOf(304, 13027, 13, 22555, 0, 5692))
        assertTrue("leading space dropped: ${reply.take(4)}", reply.startsWith(" "))
        assertEquals(" in Python. Sure! Here", reply)
    }

    // A first token carrying only the lead bytes of a multi-byte character is held back until the
    // continuation token arrives — held back, never dropped.
    @Test
    fun firstTokenThatIsOnlyAUtf8LeadIsHeldBackNotDropped() {
        val tokenizer = tokenizer()

        // 75360 = f0 9f 98 (three of the four bytes of U+1F600), 222 = 80 (the last byte).
        val emojiDecoder = Tokenizer.StreamDecoder(tokenizer)
        assertEquals("", emojiDecoder.add(75360))
        assertEquals("😀", emojiDecoder.add(222))
        assertEquals("😀", replyFrom(tokenizer, intArrayOf(75360, 222)))

        // 158 = e2 (the lead byte of U+20AC), 7492 = 82 ac.
        val euroDecoder = Tokenizer.StreamDecoder(tokenizer)
        assertEquals("", euroDecoder.add(158))
        assertEquals("€", euroDecoder.add(7492))
        assertEquals("€", replyFrom(tokenizer, intArrayOf(158, 7492)))
    }

    // Head-of-stream guard over the whole vocabulary: every token whose bytes are a complete UTF-8
    // sequence must come straight back out of a fresh decoder. Nothing valid is ever withheld.
    @Test
    fun noCompleteFirstTokenIsWithheld() {
        val tokenizer = tokenizer()
        var checked = 0
        for (id in 0 until 151643) {
            val bytes = tokenizer.tokenBytes(id)
            if (bytes.isEmpty()) continue
            val text = String(bytes, Charsets.UTF_8)
            if (!text.toByteArray(Charsets.UTF_8).contentEquals(bytes)) continue // an incomplete lead
            assertEquals("token $id withheld as the first token", text, Tokenizer.StreamDecoder(tokenizer).add(id))
            checked++
        }
        assertTrue("expected most of the vocabulary to be complete UTF-8, got $checked", checked > 150_000)
    }
}
