package com.vknn.chat

import com.vknn.chat.ui.PromptCounterTone
import com.vknn.chat.ui.validateCoachQuestion
import com.vknn.chat.ui.validatePromptTemplate
import com.vknn.chat.vlm.VlmPromptBudget
import com.vknn.chat.vlm.VlmPromptMeasurement
import com.vknn.chat.vlm.VlmTemplate
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

// What the prompt-editor dialog does with the draft in the field: whether Save is offered, and how the
// counter under the field reads. These are the user-visible half of the token-budget guard.
class PromptEditorValidationTest {
    private val window = 128

    private fun tokenizer(): Tokenizer = VlmTemplate.tokenizer(
        File("src/test/resources/smolvlm2/vocab.json").readText(),
        File("src/test/resources/smolvlm2/merges.txt").readText(),
    )

    private fun measurer(): (String) -> VlmPromptMeasurement {
        val tk = tokenizer()
        return { question -> VlmPromptBudget.measure(tk, question, window) }
    }

    private fun wordQuestion(tokens: Int): String = (1..tokens).joinToString(" ") { "word" }

    private fun questionTokensFor(promptTokens: Int): String {
        val overhead = measurer()("").promptTokenCount
        return wordQuestion(promptTokens - overhead)
    }

    @Test
    fun theDefaultQuestionSavesQuietly() {
        val validation = validateCoachQuestion(VlmTemplate.QUESTION, measurer())
        assertTrue(validation.canSave)
        assertEquals("109 / 128 tokens", validation.counterText)
        assertEquals(PromptCounterTone.NEUTRAL, validation.tone)
    }

    @Test
    fun aQuestionAtTheCapStillSavesButWarns() {
        val validation = validateCoachQuestion(questionTokensFor(window), measurer())
        assertTrue(validation.canSave)
        assertEquals("128 / 128 tokens", validation.counterText)
        assertEquals(PromptCounterTone.WARNING, validation.tone)
    }

    @Test
    fun aQuestionOverTheCapCannotBeSaved() {
        val validation = validateCoachQuestion(questionTokensFor(window + 1), measurer())
        assertFalse(validation.canSave)
        assertEquals("129 / 128 tokens", validation.counterText)
        assertEquals(PromptCounterTone.ERROR, validation.tone)
        assertNotNull(validation.rejectionReason)
        assertTrue(validation.rejectionReason!!.contains("128-token prefill window"))
    }

    @Test
    fun anEmptyQuestionCannotBeSaved() {
        val validation = validateCoachQuestion("   ", measurer())
        assertFalse(validation.canSave)
    }

    // No model loaded: nothing to measure against, so no counter and no refusal.
    @Test
    fun anUnmeasurableQuestionCarriesNoCounter() {
        val validation = validateCoachQuestion("Describe this.") { null }
        assertTrue(validation.canSave)
        assertEquals("", validation.counterText)
    }

    @Test
    fun anEmptyChatTemplateIsRawCompletion() {
        val validation = validatePromptTemplate("")
        assertTrue(validation.canSave)
        assertTrue(validation.counterText.contains("continues"))
    }

    @Test
    fun theInstructPresetIsSaveable() {
        val validation = validatePromptTemplate(ChatPromptTemplate.INSTRUCT_PRESET)
        assertTrue(validation.canSave)
        assertTrue(validation.counterText.contains("answers"))
    }

    @Test
    fun aChatTemplateWithoutThePlaceholderCannotBeSaved() {
        val validation = validatePromptTemplate("You are a helpful assistant.")
        assertFalse(validation.canSave)
        assertTrue(validation.rejectionReason!!.contains(ChatPromptTemplate.PLACEHOLDER))
    }
}
