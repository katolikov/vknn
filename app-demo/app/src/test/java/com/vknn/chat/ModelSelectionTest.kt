package com.vknn.chat

import com.vknn.chat.model.ModelCatalog
import com.vknn.chat.model.ModelSelection
import org.junit.Assert.assertEquals
import org.junit.Test

// Validates the pure selection-key rules the variant picker persists through: catalogue keys must
// stay within their mode, local keys pass through, and cache file names are variant-scoped and
// filesystem-safe. Runs on the JVM: ./gradlew :app:testDebugUnitTest.
class ModelSelectionTest {

    @Test
    fun persistedCatalogueKeyOfTheSameModeIsKept() {
        assertEquals(
            ModelCatalog.QWEN_INT4_PREFILL.id,
            ModelSelection.validKey(ModelCatalog.QWEN_INT4_PREFILL.id, ModelCatalog.QWEN),
        )
    }

    @Test
    fun missingOrForeignKeysFallBackToTheModeDefault() {
        assertEquals(ModelCatalog.QWEN.id, ModelSelection.validKey(null, ModelCatalog.QWEN))
        assertEquals(ModelCatalog.QWEN.id, ModelSelection.validKey("removed_model", ModelCatalog.QWEN))
        // A VLM catalogue id persisted under the chat mode never selects a VLM model for Chat.
        assertEquals(ModelCatalog.QWEN.id, ModelSelection.validKey(ModelCatalog.SMOLVLM2.id, ModelCatalog.QWEN))
    }

    @Test
    fun unnamedLocalKeysPassThroughUnderChat() {
        // A file whose name reveals no family is a generic text decoder: allowed in Chat.
        val key = ModelSelection.LOCAL_PREFIX + "fresh-build.vxm"
        assertEquals(key, ModelSelection.validKey(key, ModelCatalog.QWEN))
    }

    @Test
    fun aQwenLocalFileIsNotKeptAsTheVlmSelection() {
        // The reported bug: a Qwen text .vxm sitting in the app dir was offered in (and stuck as) the
        // VLM tab's model, where it has no vision bucket and its load fails. It must fall back to the
        // VLM default instead.
        val qwenLocal = ModelSelection.LOCAL_PREFIX + "qwen2.5-coder-0.5b-decode-c1024.vxm"
        assertEquals(ModelCatalog.SMOLVLM2.id, ModelSelection.validKey(qwenLocal, ModelCatalog.SMOLVLM2))
        // ...but the same file is a valid Chat selection.
        assertEquals(qwenLocal, ModelSelection.validKey(qwenLocal, ModelCatalog.QWEN))
        // A SmolVLM local file is valid for VLM and rejected for Chat.
        val vlmLocal = ModelSelection.LOCAL_PREFIX + "smolvlm2-2.2b-fp16.vxm"
        assertEquals(vlmLocal, ModelSelection.validKey(vlmLocal, ModelCatalog.SMOLVLM2))
        assertEquals(ModelCatalog.QWEN.id, ModelSelection.validKey(vlmLocal, ModelCatalog.QWEN))
    }

    @Test
    fun modeFromFileNameIdentifiesFamilies() {
        assertEquals(ModelCatalog.QWEN.mode, ModelSelection.modeFromFileName("qwen2.5-coder-0.5b-decode-c1024.vxm"))
        assertEquals(ModelCatalog.SMOLVLM2.mode, ModelSelection.modeFromFileName("smolvlm2-2.2b-i4.vxm"))
        assertEquals(ModelCatalog.DL3DV.mode, ModelSelection.modeFromFileName("encoder8_fp16.vxm"))
        assertEquals(null, ModelSelection.modeFromFileName("mystery-model.vxm"))
    }

    @Test
    fun variantReadsFromTheFileName() {
        assertEquals("int4", ModelSelection.variantFromFileName("qwen-c1024-prefill256-INT4.vxm"))
        assertEquals("fp16", ModelSelection.variantFromFileName("qwen-c1024-prefill256.vxm"))
    }

    @Test
    fun cacheFileNamesAreVariantScopedAndFilesystemSafe() {
        // The pre-variant release cached the chat model as qwen.cache; the id-derived name keeps it.
        assertEquals("qwen.cache", ModelSelection.cacheFileName(ModelCatalog.QWEN))
        assertEquals("qwen_int4_prefill.cache", ModelSelection.cacheFileName(ModelCatalog.QWEN_INT4_PREFILL))
        assertEquals("smolvlm2.cache", ModelSelection.cacheFileName(ModelCatalog.SMOLVLM2))
        val local = ModelSelection.LOCAL_PREFIX + "fresh build.vxm"
        val spec = ModelCatalog.QWEN.copy(id = local)
        assertEquals("local_fresh_build.vxm.cache", ModelSelection.cacheFileName(spec))
    }
}
