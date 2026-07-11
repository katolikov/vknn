package com.vknn.chat

/**
 * The optional prompt template wrapped around a chat message, and the per-family [Dialect] that gives
 * a template its tokenizer regex, control-token ids, and stop tokens.
 *
 * A chat model is instruction-tuned, so the default template is its family's instruct preset
 * ([Dialect.instructPreset]); clearing the template falls back to raw completion, where the message is
 * fed to the decoder untouched and the model continues it instead of answering.
 *
 * Templates carry control tokens (Qwen ChatML `<|im_start|>`, Llama-3 `<|start_header_id|>`, ...). Those
 * ids live in the tokenizer's `added_tokens` table rather than the byte-level `vocab.json`, so
 * [Tokenizer.encode] cannot produce them and [Tokenizer.tokenBytes] yields no bytes for them.
 * [encodeTurn] therefore splits the template on the control-token literals of the selected dialect and
 * splices each one in by id. The split runs over the *template*, never over the message: a message that
 * happens to contain a control literal encodes as ordinary text and cannot forge a turn boundary.
 */
object ChatPromptTemplate {
    /** Stands for the message inside a template. Substituted only into the template's text spans. */
    const val PLACEHOLDER = "{prompt}"

    // Qwen2 control-token ids (the tokenizer's added_tokens table). The bundled vocab.json ends at
    // 151642, so every id here is outside it.
    const val END_OF_TEXT = 151643 // <|endoftext|> — end-of-stream outside a ChatML turn
    const val IM_START = 151644 // <|im_start|>
    const val IM_END = 151645 // <|im_end|> — ends a ChatML turn

    /** ChatML, the template the Qwen instruct model is tuned on. */
    const val INSTRUCT_PRESET =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n$PLACEHOLDER<|im_end|>\n<|im_start|>assistant\n"

    /** The Qwen instruct preset is the default chat template; clear it for raw completion. */
    const val DEFAULT = INSTRUCT_PRESET

    // Llama-3 control-token ids (the tokenizer's added_tokens table, ids 128000+). The byte-level
    // vocab.json stops at 127999, so every id here is outside it.
    const val LLAMA_BEGIN_OF_TEXT = 128000 // <|begin_of_text|>
    const val LLAMA_END_OF_TEXT = 128001 // <|end_of_text|> — end-of-stream outside a chat turn
    const val LLAMA_START_HEADER = 128006 // <|start_header_id|>
    const val LLAMA_END_HEADER = 128007 // <|end_header_id|>
    const val LLAMA_EOT = 128009 // <|eot_id|> — ends a chat turn

    /**
     * The Llama-3 instruct template with a static system line. The role words, the `\n\n` separators,
     * and the message all encode as ordinary byte-level BPE text; only the five control literals splice
     * in by id. Deliberately static (not the transformers default, which injects a dynamic cutoff/date
     * system block) so the app's rendered prompt is reproducible.
     */
    const val LLAMA_INSTRUCT_PRESET =
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>" +
            "<|start_header_id|>user<|end_header_id|>\n\n$PLACEHOLDER<|eot_id|>" +
            "<|start_header_id|>assistant<|end_header_id|>\n\n"

    /**
     * A chat family's prompt conventions, selected per model so each template tokenizes and terminates
     * correctly.
     *
     * @property id stable key: names the dialect in [PromptSettings] and in a catalogue entry's `chatDialectId`.
     * @property presetLabel the editor's one-tap preset button label.
     * @property pattern the [Tokenizer] pre-tokenizer regex for this family.
     * @property tokenizerFromAssets true when the vocab/merges ship in the app assets (Qwen); false when
     *   they download alongside the model (Llama), read from the model's companion files.
     * @property instructPreset the default instruct template, with a [PLACEHOLDER] for the message.
     * @property controlTokenIds each control literal mapped to the id it splices in as.
     * @property endOfText the id that ends a reply in every case (raw completion's end-of-stream).
     * @property turnStopLiteral the literal a template uses to close a turn.
     * @property turnStopToken the id also stopped on when an active template closes turns with [turnStopLiteral].
     */
    data class Dialect(
        val id: String,
        val presetLabel: String,
        val pattern: String,
        val tokenizerFromAssets: Boolean,
        val instructPreset: String,
        val controlTokenIds: Map<String, Int>,
        val endOfText: Int,
        val turnStopLiteral: String,
        val turnStopToken: Int,
    )

    val QWEN = Dialect(
        id = "qwen",
        presetLabel = "Instruct (ChatML)",
        pattern = Tokenizer.QWEN2_PATTERN,
        tokenizerFromAssets = true,
        instructPreset = INSTRUCT_PRESET,
        controlTokenIds = linkedMapOf(
            "<|im_start|>" to IM_START,
            "<|im_end|>" to IM_END,
            "<|endoftext|>" to END_OF_TEXT,
        ),
        endOfText = END_OF_TEXT,
        turnStopLiteral = "<|im_end|>",
        turnStopToken = IM_END,
    )

    val LLAMA3 = Dialect(
        id = "llama3",
        presetLabel = "Instruct (Llama 3)",
        pattern = Tokenizer.LLAMA3_PATTERN,
        tokenizerFromAssets = false,
        instructPreset = LLAMA_INSTRUCT_PRESET,
        controlTokenIds = linkedMapOf(
            "<|begin_of_text|>" to LLAMA_BEGIN_OF_TEXT,
            "<|start_header_id|>" to LLAMA_START_HEADER,
            "<|end_header_id|>" to LLAMA_END_HEADER,
            "<|eot_id|>" to LLAMA_EOT,
            "<|end_of_text|>" to LLAMA_END_OF_TEXT,
        ),
        endOfText = LLAMA_END_OF_TEXT,
        turnStopLiteral = "<|eot_id|>",
        turnStopToken = LLAMA_EOT,
    )

    val DIALECTS: List<Dialect> = listOf(QWEN, LLAMA3)

    /** The dialect named by [id], or the ChatML default (Qwen) for a null or unknown id. */
    fun dialect(id: String?): Dialect = DIALECTS.firstOrNull { it.id == id } ?: QWEN

    /** True when [template] wraps the message rather than leaving it to raw completion. */
    fun isActive(template: String): Boolean = template.isNotBlank()

    /**
     * True when [template] can be applied: it either is inactive, or it marks where the message goes.
     * An active template without [PLACEHOLDER] would drop the message entirely.
     */
    fun isApplicable(template: String): Boolean = !isActive(template) || template.contains(PLACEHOLDER)

    /**
     * Ids the decoder stops on. [Dialect.endOfText] always ends a reply; a template that closes its
     * turns with the dialect's [Dialect.turnStopLiteral] stops there too, because the base model's own
     * end-of-stream never arrives inside a chat turn.
     */
    fun stopTokensFor(template: String, dialect: Dialect = QWEN): Set<Int> {
        val stops = mutableSetOf(dialect.endOfText)
        if (isActive(template) && template.contains(dialect.turnStopLiteral)) stops.add(dialect.turnStopToken)
        return stops
    }

    /**
     * Ids for one turn. With no template the message continues the running context, a leading newline
     * separating it from the previous turn ([continuingContext]). With a template the rendered prompt
     * replaces that, the dialect's control tokens spliced in by id.
     */
    fun encodeTurn(
        tokenizer: Tokenizer,
        template: String,
        message: String,
        continuingContext: Boolean,
        dialect: Dialect = QWEN,
    ): IntArray {
        if (!isActive(template)) {
            return tokenizer.encode(if (continuingContext) "\n$message" else message)
        }
        val ids = ArrayList<Int>(64)
        for (span in splitOnControlTokens(template, dialect.controlTokenIds)) {
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
    private fun splitOnControlTokens(template: String, controlTokenIds: Map<String, Int>): List<TemplateSpan> {
        val spans = ArrayList<TemplateSpan>()
        var cursor = 0
        while (cursor < template.length) {
            var literalAt = -1
            var literal = ""
            for (candidate in controlTokenIds.keys) {
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
            spans.add(TemplateSpan.Control(controlTokenIds.getValue(literal)))
            cursor = literalAt + literal.length
        }
        return spans
    }
}
