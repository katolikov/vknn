package com.vknn.chat

import com.vknn.chat.vlm.PromptBudgetVerdict
import com.vknn.chat.vlm.VlmPromptBudget
import com.vknn.chat.vlm.VlmTemplate
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

// The hard token-budget guard behind the coach-prompt editor. The decoder's prefill window bounds the
// WHOLE prompt — 81 image-token rows plus the chat template plus the question — so the question a user
// may type is much shorter than the window suggests. A question that overruns the window fails the turn
// at native prefill; these tests pin the arithmetic that stops it from being saved.
class VlmPromptBudgetTest {
    private fun tokenizer(): Tokenizer = VlmTemplate.tokenizer(
        File("src/test/resources/smolvlm2/vocab.json").readText(),
        File("src/test/resources/smolvlm2/merges.txt").readText(),
    )

    // Each " word" is one token in this vocabulary, so an n-word question costs exactly n tokens and the
    // whole prompt costs PROMPT_OVERHEAD_TOKENS + n. Verified by wordQuestionCostsOneTokenPerWord.
    private fun wordQuestion(tokens: Int): String = (1..tokens).joinToString(" ") { "word" }

    private val window = 128

    @Test
    fun wordQuestionCostsOneTokenPerWord() {
        val tk = tokenizer()
        val overhead = VlmPromptBudget.measure(tk, "", window).promptTokenCount
        for (n in listOf(1, 5, 20, 36)) {
            assertEquals(
                "an $n-word question should cost $n tokens on top of the template",
                overhead + n,
                VlmPromptBudget.measure(tk, wordQuestion(n), window).promptTokenCount,
            )
        }
    }

    // The prompt is far more than the question: the 81 image rows dominate it.
    @Test
    fun promptCountIncludesTheImageRowsAndTemplate() {
        val tk = tokenizer()
        val measurement = VlmPromptBudget.measure(tk, "Describe this.", window)
        assertEquals(VlmTemplate.promptIds(tk, "Describe this.").size, measurement.promptTokenCount)
        assertTrue(
            "prompt (${measurement.promptTokenCount}) must exceed the ${VlmTemplate.IMAGE_ROWS} image rows",
            measurement.promptTokenCount > VlmTemplate.IMAGE_ROWS,
        )
    }

    @Test
    fun shippedDefaultQuestionIsWellUnderTheWindow() {
        val measurement = VlmPromptBudget.measure(tokenizer(), VlmTemplate.QUESTION, window)
        assertEquals(109, measurement.promptTokenCount)
        assertEquals("109 / 128 tokens", measurement.counterText())
        assertTrue(measurement.fitsPrefillWindow)
        assertEquals(PromptBudgetVerdict.WITHIN_BUDGET, measurement.verdict)
        assertNull(measurement.rejectionReason())
    }

    // The native prefill refuses only a prompt LONGER than the window, so a prompt of exactly the window
    // is accepted.
    @Test
    fun aQuestionExactlyAtTheCapIsAccepted() {
        val tk = tokenizer()
        val overhead = VlmPromptBudget.measure(tk, "", window).promptTokenCount
        val measurement = VlmPromptBudget.measure(tk, wordQuestion(window - overhead), window)
        assertEquals(window, measurement.promptTokenCount)
        assertTrue(measurement.fitsPrefillWindow)
        assertNull(measurement.rejectionReason())
        assertEquals(PromptBudgetVerdict.NEAR_LIMIT, measurement.verdict)
    }

    @Test
    fun aQuestionOneTokenOverTheCapIsRejected() {
        val tk = tokenizer()
        val overhead = VlmPromptBudget.measure(tk, "", window).promptTokenCount
        val measurement = VlmPromptBudget.measure(tk, wordQuestion(window - overhead + 1), window)
        assertEquals(window + 1, measurement.promptTokenCount)
        assertFalse(measurement.fitsPrefillWindow)
        assertEquals(PromptBudgetVerdict.OVER_BUDGET, measurement.verdict)
        assertNotNull(measurement.rejectionReason())
        assertTrue(measurement.rejectionReason()!!.contains("128-token prefill window"))
    }

    // The 131-token question that overran the window on device.
    @Test
    fun aLongQuestionIsRejected() {
        val tk = tokenizer()
        val overhead = VlmPromptBudget.measure(tk, "", window).promptTokenCount
        val measurement = VlmPromptBudget.measure(tk, wordQuestion(131 - overhead), window)
        assertEquals(131, measurement.promptTokenCount)
        assertFalse(measurement.fitsPrefillWindow)
    }

    @Test
    fun theWarningBandOpensExactlyAtTheHeadroom() {
        val tk = tokenizer()
        val overhead = VlmPromptBudget.measure(tk, "", window).promptTokenCount
        val lastQuiet = window - VlmPromptBudget.WARNING_HEADROOM_TOKENS
        assertEquals(
            PromptBudgetVerdict.WITHIN_BUDGET,
            VlmPromptBudget.measure(tk, wordQuestion(lastQuiet - overhead), window).verdict,
        )
        assertEquals(
            PromptBudgetVerdict.NEAR_LIMIT,
            VlmPromptBudget.measure(tk, wordQuestion(lastQuiet - overhead + 1), window).verdict,
        )
    }

    // The window is the loaded model's, not a constant: the same question flips verdict when the model
    // reports a wider prefill bucket.
    @Test
    fun theVerdictFollowsTheModelsWindow() {
        val tk = tokenizer()
        val overLong = wordQuestion(60)
        assertFalse(VlmPromptBudget.measure(tk, overLong, 128).fitsPrefillWindow)
        assertTrue(VlmPromptBudget.measure(tk, overLong, 256).fitsPrefillWindow)
        assertEquals("152 / 256 tokens", VlmPromptBudget.measure(tk, overLong, 256).counterText())
    }
}
