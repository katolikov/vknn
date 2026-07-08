package com.vknn.chat.vlm

import com.vknn.chat.Tokenizer

/** Where a candidate coach question sits against the decoder's prefill window. */
enum class PromptBudgetVerdict { WITHIN_BUDGET, NEAR_LIMIT, OVER_BUDGET }

/**
 * One coach turn measured against the model's prefill window.
 *
 * [promptTokenCount] is the whole sequence the decoder receives — the image-token rows and the chat
 * template as well as the question — because the window bounds all of it, not the question alone.
 * [prefillWindowTokens] comes from the loaded model (`nativeVlmInfo` index 5), so it tracks whatever
 * prefill bucket the .vxm was compiled with.
 */
data class VlmPromptMeasurement(
    val promptTokenCount: Int,
    val prefillWindowTokens: Int,
) {
    /** The native prefill refuses a prompt longer than the window, so the window itself still fits. */
    val fitsPrefillWindow: Boolean get() = promptTokenCount <= prefillWindowTokens

    val verdict: PromptBudgetVerdict
        get() = when {
            !fitsPrefillWindow -> PromptBudgetVerdict.OVER_BUDGET
            promptTokenCount > prefillWindowTokens - VlmPromptBudget.WARNING_HEADROOM_TOKENS -> PromptBudgetVerdict.NEAR_LIMIT
            else -> PromptBudgetVerdict.WITHIN_BUDGET
        }

    /** The counter under the field, e.g. "109 / 128 tokens". */
    fun counterText(): String = "$promptTokenCount / $prefillWindowTokens tokens"

    /** Why the question cannot be saved, or null when it fits. */
    fun rejectionReason(): String? =
        if (fitsPrefillWindow) null else "Too long for this model's $prefillWindowTokens-token prefill window"
}

object VlmPromptBudget {
    /** Within this many tokens of the window the counter warns: one more word can overrun it. */
    const val WARNING_HEADROOM_TOKENS = 8

    /**
     * Measure the full prompt [VlmTemplate.promptIds] builds for [question] — the single source of the
     * template, so the count is what the decoder is actually handed.
     */
    fun measure(tokenizer: Tokenizer, question: String, prefillWindowTokens: Int): VlmPromptMeasurement =
        VlmPromptMeasurement(VlmTemplate.promptIds(tokenizer, question).size, prefillWindowTokens)
}
