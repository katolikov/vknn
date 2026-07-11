package com.vknn.chat.model

import android.content.Context
import android.content.SharedPreferences
import com.vknn.chat.ChatPromptTemplate
import com.vknn.chat.vlm.VlmTemplate
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

// The two user-editable prompts, persisted in SharedPreferences — the same settings store as
// BackendSetting. Both take effect on the next turn: an in-flight generation keeps the prompt it
// started with.
class PromptSettings(context: Context) {
    private val preferences: SharedPreferences =
        context.getSharedPreferences("vknn_settings", Context.MODE_PRIVATE)

    private val _vlmQuestion = MutableStateFlow(
        preferences.getString(KEY_VLM_QUESTION, null) ?: VlmTemplate.QUESTION,
    )

    /** The question the camera coach asks about the captured photo. */
    val vlmQuestion: StateFlow<String> = _vlmQuestion.asStateFlow()

    private val _chatTemplateOverrides = MutableStateFlow(loadChatTemplateOverrides())

    /**
     * The user's saved chat templates, keyed by [ChatPromptTemplate.Dialect.id]. A dialect absent from
     * the map has no override and falls back to its instruct preset — see [chatTemplate]. Kept per
     * dialect because a template's control tokens belong to one family: a Qwen ChatML template applied
     * with Llama-3 ids would tokenize to garbage.
     */
    val chatTemplateOverrides: StateFlow<Map<String, String>> = _chatTemplateOverrides.asStateFlow()

    /**
     * The active chat template for [dialect]: the user's saved override, else the dialect's instruct
     * preset. The template carries a `{prompt}` placeholder for the message; empty means raw completion
     * — see [ChatPromptTemplate].
     */
    fun chatTemplate(dialect: ChatPromptTemplate.Dialect): String =
        _chatTemplateOverrides.value[dialect.id] ?: dialect.instructPreset

    fun setVlmQuestion(question: String) = put(KEY_VLM_QUESTION, question, _vlmQuestion)

    fun setChatTemplate(dialect: ChatPromptTemplate.Dialect, template: String) {
        preferences.edit().putString(chatTemplateKey(dialect.id), template).apply()
        _chatTemplateOverrides.value = _chatTemplateOverrides.value + (dialect.id to template)
    }

    private fun loadChatTemplateOverrides(): Map<String, String> =
        ChatPromptTemplate.DIALECTS.mapNotNull { dialect ->
            preferences.getString(chatTemplateKey(dialect.id), null)?.let { dialect.id to it }
        }.toMap()

    // Resetting to a default is a Save of that default: the editor restores it into the draft field.
    private fun put(key: String, value: String, target: MutableStateFlow<String>) {
        preferences.edit().putString(key, value).apply()
        target.value = value
    }

    // The Qwen dialect keeps the pre-1.3 bare key so an existing customization carries over unchanged;
    // every other family appends its dialect id.
    private fun chatTemplateKey(dialectId: String): String =
        if (dialectId == ChatPromptTemplate.QWEN.id) KEY_CHAT_PROMPT_TEMPLATE else "$KEY_CHAT_PROMPT_TEMPLATE.$dialectId"

    private companion object {
        const val KEY_VLM_QUESTION = "vlm_question"
        const val KEY_CHAT_PROMPT_TEMPLATE = "chat_prompt_template"
    }
}
