package com.vknn.chat

/**
 * The optional prompt template wrapped around a chat message.
 *
 * The shipped Qwen model is the instruct variant, tuned to answer inside ChatML turns, so [DEFAULT]
 * is the ChatML template [INSTRUCT_PRESET]. Clearing the template falls back to raw completion: the
 * message is fed to the decoder untouched and the model continues it instead of answering.
 *
 * Templates carry ChatML control tokens. Those ids live in the tokenizer's `added_tokens` table rather
 * than the bundled `vocab.json`, so [Tokenizer.encode] cannot produce them and [Tokenizer.tokenBytes]
 * yields no bytes for them. [encodeTurn] therefore splits the template on the control-token literals
 * and splices each one in by id. The split runs over the *template*, never over the message: a message
 * that happens to contain `<|im_end|>` encodes as ordinary text and cannot forge a turn boundary.
 */
object ChatPromptTemplate {
    /** Stands for the message inside a template. Substituted only into the template's text spans. */
    const val PLACEHOLDER = "{prompt}"

    // Qwen2 control-token ids (the tokenizer's added_tokens table). The bundled vocab.json ends at
    // 151642, so every id here is outside it.
    const val END_OF_TEXT = 151643 // <|endoftext|> — end-of-stream outside a ChatML turn
    const val IM_START = 151644 // <|im_start|>
    const val IM_END = 151645 // <|im_end|> — ends a ChatML turn

    /** ChatML, the template the instruct model is tuned on. */
    const val INSTRUCT_PRESET =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n$PLACEHOLDER<|im_end|>\n<|im_start|>assistant\n"

    /** The shipped model is instruction-tuned, so ChatML is the default; clear it for raw completion. */
    const val DEFAULT = INSTRUCT_PRESET

    private val CONTROL_TOKEN_IDS: Map<String, Int> = linkedMapOf(
        "<|im_start|>" to IM_START,
        "<|im_end|>" to IM_END,
        "<|endoftext|>" to END_OF_TEXT,
    )

    /** True when [template] wraps the message rather than leaving it to raw completion. */
    fun isActive(template: String): Boolean = template.isNotBlank()

    /**
     * True when [template] can be applied: it either is inactive, or it marks where the message goes.
     * An active template without [PLACEHOLDER] would drop the message entirely.
     */
    fun isApplicable(template: String): Boolean = !isActive(template) || template.contains(PLACEHOLDER)

    /**
     * Ids the decoder stops on. [END_OF_TEXT] always ends a reply; a template that closes its turns
     * with `<|im_end|>` stops there too, because the base model's own end-of-stream never arrives
     * inside a ChatML turn.
     */
    fun stopTokensFor(template: String): Set<Int> {
        val stops = mutableSetOf(END_OF_TEXT)
        if (isActive(template) && template.contains("<|im_end|>")) stops.add(IM_END)
        return stops
    }

    /**
     * Ids for one turn. With no template the message continues the running context, a leading newline
     * separating it from the previous turn ([continuingContext]). With a template the rendered prompt
     * replaces that, control tokens spliced in by id.
     */
    fun encodeTurn(tokenizer: Tokenizer, template: String, message: String, continuingContext: Boolean): IntArray {
        if (!isActive(template)) {
            return tokenizer.encode(if (continuingContext) "\n$message" else message)
        }
        val ids = ArrayList<Int>(64)
        for (span in splitOnControlTokens(template)) {
            when (span) {
                is TemplateSpan.Control -> ids.add(span.id)
                // The message lands here, inside a text span, so it is byte-level BPE like any other text.
                is TemplateSpan.Text -> for (id in tokenizer.encode(span.text.replace(PLACEHOLDER, message))) ids.add(id)
            }
        }
        return ids.toIntArray()
    }

    private sealed interface TemplateSpan {
        data class Text(val text: String) : TemplateSpan
        data class Control(val id: Int) : TemplateSpan
    }

    // Template -> alternating text and control-token spans, taking the earliest literal at each step.
    private fun splitOnControlTokens(template: String): List<TemplateSpan> {
        val spans = ArrayList<TemplateSpan>()
        var cursor = 0
        while (cursor < template.length) {
            var literalAt = -1
            var literal = ""
            for (candidate in CONTROL_TOKEN_IDS.keys) {
                val at = template.indexOf(candidate, cursor)
                if (at >= 0 && (literalAt < 0 || at < literalAt)) {
                    literalAt = at
                    literal = candidate
                }
            }
            if (literalAt < 0) {
                spans.add(TemplateSpan.Text(template.substring(cursor)))
                break
            }
            if (literalAt > cursor) spans.add(TemplateSpan.Text(template.substring(cursor, literalAt)))
            spans.add(TemplateSpan.Control(CONTROL_TOKEN_IDS.getValue(literal)))
            cursor = literalAt + literal.length
        }
        return spans
    }
}
