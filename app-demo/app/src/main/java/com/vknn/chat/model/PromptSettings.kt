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

    private val _chatPromptTemplate = MutableStateFlow(
        preferences.getString(KEY_CHAT_PROMPT_TEMPLATE, null) ?: ChatPromptTemplate.DEFAULT,
    )

    /**
     * The chat prompt template, with a `{prompt}` placeholder for the message. Empty is the default
     * and means raw completion — see [ChatPromptTemplate].
     */
    val chatPromptTemplate: StateFlow<String> = _chatPromptTemplate.asStateFlow()

    fun setVlmQuestion(question: String) = put(KEY_VLM_QUESTION, question, _vlmQuestion)

    fun setChatPromptTemplate(template: String) = put(KEY_CHAT_PROMPT_TEMPLATE, template, _chatPromptTemplate)

    // Resetting to a default is a Save of that default: the editor restores it into the draft field.
    private fun put(key: String, value: String, target: MutableStateFlow<String>) {
        preferences.edit().putString(key, value).apply()
        target.value = value
    }

    private companion object {
        const val KEY_VLM_QUESTION = "vlm_question"
        const val KEY_CHAT_PROMPT_TEMPLATE = "chat_prompt_template"
    }
}
