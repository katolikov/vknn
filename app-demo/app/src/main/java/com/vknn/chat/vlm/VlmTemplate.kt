package com.vknn.chat.vlm

import com.vknn.chat.Tokenizer

// SmolVLM2 prompt construction: the special-token ids and the chat-template expansion that must
// reproduce the HF processor's ids exactly (validated byte-for-byte against vlm_gate/prompt_ids.json
// in VlmTemplateTest). Every template constant lives here.
object VlmTemplate {
    // Special-token ids from the SmolVLM2 tokenizer (vocab + added_tokens).
    const val IM_START = 1              // <|im_start|>
    const val FAKE_IMAGE = 49189        // <fake_token_around_image>
    const val GLOBAL_IMG = 49152        // <global-img>
    const val IMAGE = 49190             // <image>  (each occurrence takes one vision row)
    const val END_OF_UTTERANCE = 49279  // <end_of_utterance> — ends the user turn AND stops generation
    const val EOS = END_OF_UTTERANCE
    const val PAD = END_OF_UTTERANCE    // prefill-window pad id (pad rows are masked, never folded)

    const val IMAGE_ROWS = 81           // vision rows per 384x384 tile (processor image_seq_len)

    // The default coach question, and the value the in-app prompt editor resets to. The user's edited
    // question is held by PromptSettings; VlmViewModel passes whichever is active to promptIds.
    // Naming what is in the frame before advising anchors the answer to THIS photo: a question that asks
    // only for advice draws generic photography platitudes, since the model can answer it without
    // consulting the image at all. The whole prompt — 81 image-token rows, the chat template, and this
    // text — must fit the decoder's prefill window (109 of 128 tokens here); a longer question overruns
    // it and the turn fails, which is what VlmPromptBudget measures and the editor enforces.
    const val QUESTION = "Say what is in this photo, then give 2 specific tips to improve it."

    /**
     * Ids for one single-image user turn:
     * <|im_start|>User:<fake_token_around_image><global-img> + 81x<image> + <fake_token_around_image>
     * + question + <end_of_utterance>\nAssistant:
     * Text spans between special tokens run through byte-level BPE; special tokens map atomically.
     */
    fun promptIds(tk: Tokenizer, question: String = QUESTION): LongArray {
        val ids = ArrayList<Int>(IMAGE_ROWS + 32)
        ids.add(IM_START)
        for (t in tk.encode("User:")) ids.add(t)
        ids.add(FAKE_IMAGE)
        ids.add(GLOBAL_IMG)
        repeat(IMAGE_ROWS) { ids.add(IMAGE) }
        ids.add(FAKE_IMAGE)
        for (t in tk.encode(question)) ids.add(t)
        ids.add(END_OF_UTTERANCE)
        for (t in tk.encode("\nAssistant:")) ids.add(t)
        return LongArray(ids.size) { ids[it].toLong() }
    }

    /** The SmolVLM2 tokenizer over downloaded vocab/merges (GPT-2 regex + isolated digits). */
    fun tokenizer(vocabJson: String, mergesText: String): Tokenizer =
        Tokenizer(vocabJson, mergesText, Tokenizer.GPT2_PATTERN, splitDigits = true)
}
